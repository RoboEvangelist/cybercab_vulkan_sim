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
    rpci.pSubpasses = &sub;
    rpci.dependencyCount = 2;
    rpci.pDependencies = deps;
    VK_CHECK(vkCreateRenderPass(ctx_->device(), &rpci, nullptr, &offscreenPass_));

    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ScenePushConstants)};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(ctx_->device(), &plci, nullptr, &gfxLayout_));

    auto vertCode = readSpirvFile(resolveAssetPath("shaders/scene.vert.spv"));
    auto fragCode = readSpirvFile(resolveAssetPath("shaders/scene.frag.spv"));
    VkShaderModule vertMod = createShaderModule(ctx_->device(), vertCode);
    VkShaderModule fragMod = createShaderModule(ctx_->device(), fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{0, sizeof(SceneVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription, 5> attrs{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneVertex, px)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneVertex, nx)},
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(SceneVertex, r)},
        {3, 0, VK_FORMAT_R32_SFLOAT, offsetof(SceneVertex, emissive)},
        {4, 0, VK_FORMAT_R32_SFLOAT, offsetof(SceneVertex, semId)},
    }};

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cbAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cbAtt;

    VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dyn;
    gpci.layout = gfxLayout_;
    gpci.renderPass = offscreenPass_;
    gpci.subpass = 0;

    VK_CHECK(vkCreateGraphicsPipelines(ctx_->device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &gfxPipeline_));

    vkDestroyShaderModule(ctx_->device(), vertMod, nullptr);
    vkDestroyShaderModule(ctx_->device(), fragMod, nullptr);
}

void CameraRigRenderer::createCameraTargets(int w, int h) {
    destroyCameraTargets();
    VkFormat depthFmt = ctx_->findDepthFormat();

    for (int c = 0; c < kNumCams; ++c) {
        rawColorImages_[c].create(*ctx_, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT, true, VK_FILTER_LINEAR);
        rawDepthImages_[c].create(*ctx_, w, h, depthFmt,
                                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                  VK_IMAGE_ASPECT_DEPTH_BIT, false);
        distortedImages_[c].create(*ctx_, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                        VK_IMAGE_ASPECT_COLOR_BIT, true, VK_FILTER_LINEAR);
        distortedImages_[c].transitionLayoutImmediate(VK_IMAGE_LAYOUT_GENERAL);

        VkImageView views[2] = {rawColorImages_[c].view(), rawDepthImages_[c].view()};
        VkFramebufferCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass = offscreenPass_;
        fci.attachmentCount = 2;
        fci.pAttachments = views;
        fci.width = static_cast<uint32_t>(w);
        fci.height = static_cast<uint32_t>(h);
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(ctx_->device(), &fci, nullptr, &framebuffers_[c]));
    }

    VkDeviceSize tensorBytes = static_cast<VkDeviceSize>(kNumCams) * w * h * 4 * sizeof(float);
    camTensorSSBO_.create(*ctx_, tensorBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    updateIspDescriptorSets();
}

void CameraRigRenderer::updateIspDescriptorSets() {
    for (int c = 0; c < kNumCams; ++c) {
        VkDescriptorImageInfo rawInfo{rawColorImages_[c].sampler(), rawColorImages_[c].view(),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo outInfo{VK_NULL_HANDLE, distortedImages_[c].view(),
                                      VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorBufferInfo bufInfo = camTensorSSBO_.descriptorInfo();

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = ispSets_[c];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &rawInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = ispSets_[c];
        writes[1].dstBinding = 1;
