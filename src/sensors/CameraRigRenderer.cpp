#include "sensors/CameraRigRenderer.hpp"
#include "core/VulkanUtils.hpp"

#include <cstring>
#include <fstream>

namespace cybercab {

namespace {

struct alignas(16) ScenePushConstants {
    Mat4 viewProj;
    Vec4 sunDirIntensity;
    Vec4 fogColorDensity;
};

struct alignas(16) IspPushConstants {
    uint32_t camIndex;
    uint32_t width;
    uint32_t height;
    uint32_t soiled;
    Vec4 dist;
    Vec4 intr;
    float renderScale;
    float timeSec;
};

} // namespace

CameraRigRenderer::~CameraRigRenderer() {
    destroy();
}

void CameraRigRenderer::init(const VulkanContext& ctx, const SimConfig& cfg, const WorldEngine& world) {
    destroy();
    ctx_ = &ctx;
    camWidth_ = (cfg.initialResMode == "full") ? cfg.fullCamWidth : cfg.camWidth;
    camHeight_ = (cfg.initialResMode == "full") ? cfg.fullCamHeight : cfg.camHeight;

    createRenderPassAndGraphicsPipeline();
    uploadStaticWorld(world);

    // Reserve dynamic buffers generously (up to ~65k verts / ~200k indices)
    dynamicVBCapacity_ = 65536 * sizeof(SceneVertex);
    dynamicIBCapacity_ = 262144 * sizeof(uint32_t);
    dynamicVB_.create(*ctx_, dynamicVBCapacity_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    dynamicIB_.create(*ctx_, dynamicIBCapacity_, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    ispPipeline_.create(*ctx_, "shaders/cam_isp_soil.comp.spv",
                        {
                            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                        },
                        sizeof(IspPushConstants));
    for (int c = 0; c < kNumCams; ++c) {
        ispSets_[c] = ispPipeline_.allocateDescriptorSet();
    }

    createCameraTargets(camWidth_, camHeight_);
}

void CameraRigRenderer::resize(int camWidth, int camHeight) {
    if (!ctx_ || (camWidth == camWidth_ && camHeight == camHeight_)) return;
    vkDeviceWaitIdle(ctx_->device());
    camWidth_ = camWidth;
    camHeight_ = camHeight;
    createCameraTargets(camWidth_, camHeight_);
}

void CameraRigRenderer::destroyCameraTargets() {
    if (!ctx_) return;
    for (int c = 0; c < kNumCams; ++c) {
        if (framebuffers_[c] != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(ctx_->device(), framebuffers_[c], nullptr);
            framebuffers_[c] = VK_NULL_HANDLE;
        }
        rawColorImages_[c].destroy();
        rawDepthImages_[c].destroy();
        distortedImages_[c].destroy();
    }
    camTensorSSBO_.destroy();
}

void CameraRigRenderer::destroy() {
    if (!ctx_) return;
    vkDeviceWaitIdle(ctx_->device());
    destroyCameraTargets();
    ispPipeline_.destroy();
    staticVB_.destroy();
    staticIB_.destroy();
    dynamicVB_.destroy();
    dynamicIB_.destroy();
    if (gfxPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx_->device(), gfxPipeline_, nullptr);
        gfxPipeline_ = VK_NULL_HANDLE;
    }
    if (gfxLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx_->device(), gfxLayout_, nullptr);
        gfxLayout_ = VK_NULL_HANDLE;
    }
    if (offscreenPass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(ctx_->device(), offscreenPass_, nullptr);
        offscreenPass_ = VK_NULL_HANDLE;
    }
    ctx_ = nullptr;
}

void CameraRigRenderer::createRenderPassAndGraphicsPipeline() {
    VkFormat depthFmt = ctx_->findDepthFormat();

    VkAttachmentDescription atts[2]{};
    atts[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
    atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    atts[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    atts[1].format = depthFmt;
    atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
    atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 2;
    rpci.pAttachments = atts;
    rpci.subpassCount = 1;
