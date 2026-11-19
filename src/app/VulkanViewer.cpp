#include "app/VulkanViewer.hpp"
#include "app/BitmapFont.hpp"
#include "core/VulkanPipeline.hpp"
#include "core/VulkanUtils.hpp"

#include <SDL2/SDL_vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace cybercab {

namespace {

struct alignas(16) ScenePC {
    Mat4 viewProj;
    Vec4 sunDirAmbient;
    Vec4 fogColorDensity;
};

struct alignas(16) LinesPC {
    Mat4 viewProj;
};

struct alignas(16) UiPC {
    Vec2 screenSize;
};

void appendCube(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                Vec3 center, Vec3 size, float yaw, Vec3 color, float alpha, float emissive) {
    const float hx = size.x * 0.5f, hy = size.y * 0.5f, hz = size.z * 0.5f;
    const float c = std::cos(yaw), s = std::sin(yaw);
    auto xf = [&](Vec3 p) -> Vec3 {
        return {center.x + p.x * c + p.z * s, center.y + p.y, center.z - p.x * s + p.z * c};
    };
    const Vec3 normals[6] = {{0,0,1},{0,0,-1},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}};
    const Vec3 pts[6][4] = {
        {{-hx,-hy, hz},{ hx,-hy, hz},{ hx, hy, hz},{-hx, hy, hz}},
        {{ hx,-hy,-hz},{-hx,-hy,-hz},{-hx, hy,-hz},{ hx, hy,-hz}},
        {{ hx,-hy, hz},{ hx,-hy,-hz},{ hx, hy,-hz},{ hx, hy, hz}},
        {{-hx,-hy,-hz},{-hx,-hy, hz},{-hx, hy, hz},{-hx, hy,-hz}},
        {{-hx, hy, hz},{ hx, hy, hz},{ hx, hy,-hz},{-hx, hy,-hz}},
        {{-hx,-hy,-hz},{ hx,-hy,-hz},{ hx,-hy, hz},{-hx,-hy, hz}},
    };
    for (int f = 0; f < 6; ++f) {
        const uint32_t base = static_cast<uint32_t>(verts.size());
        const Vec3 n = normals[f];
        const Vec3 wn{n.x * c + n.z * s, n.y, -n.x * s + n.z * c};
        for (int i = 0; i < 4; ++i) {
            const Vec3 wp = xf(pts[f][i]);
            verts.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, color.x, color.y, color.z, alpha, emissive, 0, 0, 0});
        }
        idx.push_back(base + 0); idx.push_back(base + 1); idx.push_back(base + 2);
        idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 3);
    }
}

void appendBoxLines(std::vector<LineVertex>& out, Vec3 center, Vec3 he, float yaw, Vec4 col) {
    const float c = std::cos(yaw), s = std::sin(yaw);
    auto xf = [&](float lx, float ly, float lz) -> Vec3 {
        return {center.x + lx * c + lz * s, center.y + ly, center.z - lx * s + lz * c};
    };
    Vec3 v[8] = {
        xf(-he.x, -he.y, -he.z), xf( he.x, -he.y, -he.z),
        xf( he.x,  he.y, -he.z), xf(-he.x,  he.y, -he.z),
        xf(-he.x, -he.y,  he.z), xf( he.x, -he.y,  he.z),
        xf( he.x,  he.y,  he.z), xf(-he.x,  he.y,  he.z),
    };
    const int e[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    for (int i = 0; i < 12; ++i) {
        const Vec3& a = v[e[i][0]];
        const Vec3& b = v[e[i][1]];
        out.push_back({a.x, a.y, a.z, col.x, col.y, col.z, col.w});
        out.push_back({b.x, b.y, b.z, col.x, col.y, col.z, col.w});
    }
}

Vec3 semanticColor(SemanticClass sem) {
    switch (sem) {
        case SemanticClass::Vehicle:    return {0.220f, 0.741f, 0.973f};
        case SemanticClass::Pedestrian: return {0.957f, 0.247f, 0.369f};
        case SemanticClass::SignalSign: return {0.984f, 0.749f, 0.141f};
        case SemanticClass::Overhang:   return {0.753f, 0.518f, 0.988f};
        case SemanticClass::Curb:       return {0.392f, 0.455f, 0.545f};
        case SemanticClass::OccShadow:  return {0.706f, 0.325f, 0.035f};
        default:                        return {0.580f, 0.639f, 0.722f};
    }
}

} // namespace

VulkanViewer::~VulkanViewer() {
    destroy();
}

void VulkanViewer::init(SimConfig& cfg, WorldEngine& world, CameraRigRenderer& rig,
                        PerceptionPipeline& pipeline, VulkanContext& ctx) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }
    window_ = SDL_CreateWindow(
        "Tesla Cybercab — 8-Camera Vulkan Deep Perception, Occupancy & SAT Physics",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1600, 920,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window_) {
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }

    ctx.init(cfg.useVulkanValidation, window_);
    ctx_ = &ctx;

    rig.init(ctx, cfg, world);
    pipeline.init(ctx, rig);

    createSwapchain();
    createMainRenderPass();
    createPipelines();
    createSyncObjects();

    // Font atlas
    const auto fontPixels = BitmapFontAtlas::buildAtlasRGBA8();
    fontAtlas_.create(*ctx_, BitmapFontAtlas::kAtlasW, BitmapFontAtlas::kAtlasH, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, true, VK_FILTER_NEAREST);
    fontAtlas_.uploadRGBA8(fontPixels.data(), BitmapFontAtlas::kAtlasW, BitmapFontAtlas::kAtlasH);

    for (int c = 0; c < 8; ++c) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                       ctx_->descriptorPool(), 1, &uiSetLayout_};
        VK_CHECK(vkAllocateDescriptorSets(ctx_->device(), &ai, &uiSets_[c]));
    }
    refreshCameraDescriptors(rig);

    // Pre-allocate dynamic GPU vertex/index buffers
    dynWorldVB_.create(*ctx_, 65536 * sizeof(SceneVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    dynWorldIB_.create(*ctx_, 262144 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    voxelVB_.create(*ctx_, 120000 * sizeof(SceneVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    voxelIB_.create(*ctx_, 200000 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    lineVB_.create(*ctx_, 32768 * sizeof(LineVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uiVB_.create(*ctx_, 65536 * sizeof(UiVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uiIB_.create(*ctx_, 131072 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    dynVertsScratch_.reserve(32768);
    dynIdxScratch_.reserve(65536);
    voxelVertsScratch_.reserve(100000);
    voxelIdxScratch_.reserve(160000);
    lineVertsScratch_.reserve(16384);
    uiVertsScratch_.reserve(32768);
    uiIdxScratch_.reserve(65536);
    uiBatchesScratch_.reserve(16);
    uiButtonsScratch_.reserve(64);
    uiPanelsScratch_.reserve(8);
}

void VulkanViewer::refreshCameraDescriptors(const CameraRigRenderer& rig) {
    if (!ctx_) return;
    for (int c = 0; c < 8; ++c) {
        VkDescriptorImageInfo camInfo{rig.distortedImage(c).sampler(), rig.distortedImage(c).view(),
                                      VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo fontInfo{fontAtlas_.sampler(), fontAtlas_.view(),
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet writes[2]{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, uiSets_[c], 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &camInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, uiSets_[c], 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &fontInfo, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx_->device(), 2, writes, 0, nullptr);
    }
}

void VulkanViewer::destroy() {
    if (!ctx_) return;
    vkDeviceWaitIdle(ctx_->device());
    if (imageAvailSem_ != VK_NULL_HANDLE) vkDestroySemaphore(ctx_->device(), imageAvailSem_, nullptr);
    if (renderDoneSem_ != VK_NULL_HANDLE) vkDestroySemaphore(ctx_->device(), renderDoneSem_, nullptr);
    if (inFlightFence_ != VK_NULL_HANDLE) vkDestroyFence(ctx_->device(), inFlightFence_, nullptr);

    dynWorldVB_.destroy(); dynWorldIB_.destroy();
    voxelVB_.destroy(); voxelIB_.destroy();
    lineVB_.destroy();
    uiVB_.destroy(); uiIB_.destroy();
    fontAtlas_.destroy();

    if (uiPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_->device(), uiPipeline_, nullptr);
    if (uiLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_->device(), uiLayout_, nullptr);
    if (uiSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx_->device(), uiSetLayout_, nullptr);
    if (linesPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_->device(), linesPipeline_, nullptr);
    if (linesLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_->device(), linesLayout_, nullptr);
    if (scenePipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx_->device(), scenePipeline_, nullptr);
    if (sceneLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx_->device(), sceneLayout_, nullptr);

    cleanupSwapchain();
    if (mainPass_ != VK_NULL_HANDLE) vkDestroyRenderPass(ctx_->device(), mainPass_, nullptr);

    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
    ctx_ = nullptr;
}

void VulkanViewer::createSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_->physicalDevice(), ctx_->surface(), &caps));

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_->physicalDevice(), ctx_->surface(), &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_->physicalDevice(), ctx_->surface(), &fmtCount, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    swapFormat_ = chosen.format;

    int winW = 1600, winH = 920;
    SDL_Vulkan_GetDrawableSize(window_, &winW, &winH);
    swapExtent_ = {
        std::clamp(static_cast<uint32_t>(std::max(1, winW)), caps.minImageExtent.width, caps.maxImageExtent.width),
        std::clamp(static_cast<uint32_t>(std::max(1, winH)), caps.minImageExtent.height, caps.maxImageExtent.height)
    };
    if (caps.currentExtent.width != UINT32_MAX) {
        swapExtent_ = caps.currentExtent;
    }

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imgCount > caps.maxImageCount) imgCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = ctx_->surface();
    sci.minImageCount = imgCount;
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = swapExtent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE;
    VK_CHECK(vkCreateSwapchainKHR(ctx_->device(), &sci, nullptr, &swapchain_));

    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(ctx_->device(), swapchain_, &actualCount, nullptr);
    swapImages_.resize(actualCount);
    vkGetSwapchainImagesKHR(ctx_->device(), swapchain_, &actualCount, swapImages_.data());

    swapViews_.resize(actualCount);
    for (uint32_t i = 0; i < actualCount; ++i) {
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = swapImages_[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = swapFormat_;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(ctx_->device(), &ivci, nullptr, &swapViews_[i]));
    }

    depthImage_.create(*ctx_, swapExtent_.width, swapExtent_.height, ctx_->findDepthFormat(),
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, false);
}

void VulkanViewer::createMainRenderPass() {
    if (mainPass_ == VK_NULL_HANDLE) {
        VkAttachmentDescription atts[2]{};
        atts[0].format = swapFormat_;
        atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
        atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        atts[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        atts[1].format = ctx_->findDepthFormat();
        atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
        atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference cRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &cRef;
        sub.pDepthStencilAttachment = &dRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, nullptr, 0, 2, atts, 1, &sub, 1, &dep};
        VK_CHECK(vkCreateRenderPass(ctx_->device(), &rpci, nullptr, &mainPass_));
    }

    swapFramebuffers_.resize(swapViews_.size());
    for (size_t i = 0; i < swapViews_.size(); ++i) {
        VkImageView views[2] = {swapViews_[i], depthImage_.view()};
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, nullptr, 0, mainPass_,
                                    2, views, swapExtent_.width, swapExtent_.height, 1};
        VK_CHECK(vkCreateFramebuffer(ctx_->device(), &fci, nullptr, &swapFramebuffers_[i]));
    }
}

void VulkanViewer::cleanupSwapchain() {
    for (auto fb : swapFramebuffers_) {
        if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(ctx_->device(), fb, nullptr);
    }
    swapFramebuffers_.clear();
    depthImage_.destroy();
    for (auto v : swapViews_) {
        if (v != VK_NULL_HANDLE) vkDestroyImageView(ctx_->device(), v, nullptr);
    }
    swapViews_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(ctx_->device(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanViewer::recreateSwapchain() {
    vkDeviceWaitIdle(ctx_->device());
    cleanupSwapchain();
    createSwapchain();
    createMainRenderPass();
}

void VulkanViewer::createPipelines() {
    // 1. Scene Pipeline (with alpha blending for voxels & ghosts)
    {
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ScenePC)};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VK_CHECK(vkCreatePipelineLayout(ctx_->device(), &plci, nullptr, &sceneLayout_));

        auto vc = readSpirvFile(resolveAssetPath("shaders/scene.vert.spv"));
        auto fc = readSpirvFile(resolveAssetPath("shaders/scene.frag.spv"));
        VkShaderModule vm = createShaderModule(ctx_->device(), vc);
        VkShaderModule fm = createShaderModule(ctx_->device(), fc);

        VkPipelineShaderStageCreateInfo stages[2]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vm, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fm, "main", nullptr},
        };
        VkVertexInputBindingDescription bind{0, sizeof(SceneVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 5> attrs{{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneVertex, px)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneVertex, nx)},
            {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(SceneVertex, r)},
            {3, 0, VK_FORMAT_R32_SFLOAT, offsetof(SceneVertex, emissive)},
            {4, 0, VK_FORMAT_R32_SFLOAT, offsetof(SceneVertex, semId)},
        }};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &bind, 5, attrs.data()};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
                                                  VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE};
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS, VK_FALSE, VK_FALSE, {}, {}, 0, 1};

        VkPipelineColorBlendAttachmentState cba{};
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
        cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba, {0,0,0,0}};

        VkDynamicState dynS[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dynS};
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, nullptr, 0, 2, stages, &vi, &ia, nullptr, &vp, &rs, &ms, &ds, &cb, &dyn, sceneLayout_, mainPass_, 0, VK_NULL_HANDLE, -1};
        VK_CHECK(vkCreateGraphicsPipelines(ctx_->device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &scenePipeline_));
        vkDestroyShaderModule(ctx_->device(), vm, nullptr);
        vkDestroyShaderModule(ctx_->device(), fm, nullptr);
    }

    // 2. 3D Lines Pipeline
    {
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(LinesPC)};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcr};
        VK_CHECK(vkCreatePipelineLayout(ctx_->device(), &plci, nullptr, &linesLayout_));

        auto vc = readSpirvFile(resolveAssetPath("shaders/lines3d.vert.spv"));
        auto fc = readSpirvFile(resolveAssetPath("shaders/lines3d.frag.spv"));
        VkShaderModule vm = createShaderModule(ctx_->device(), vc);
        VkShaderModule fm = createShaderModule(ctx_->device(), fc);

        VkPipelineShaderStageCreateInfo stages[2]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vm, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fm, "main", nullptr},
        };
        VkVertexInputBindingDescription bind{0, sizeof(LineVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[2]{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LineVertex, x)},
            {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(LineVertex, r)},
        };
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &bind, 2, attrs};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_FALSE};
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
                                                  VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 2.0f};
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE};
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_TRUE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL, VK_FALSE, VK_FALSE, {}, {}, 0, 1};
        VkPipelineColorBlendAttachmentState cba{VK_TRUE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD,
                                               VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, 0xF};
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba, {0,0,0,0}};
        VkDynamicState dynS[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dynS};
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, nullptr, 0, 2, stages, &vi, &ia, nullptr, &vp, &rs, &ms, &ds, &cb, &dyn, linesLayout_, mainPass_, 0, VK_NULL_HANDLE, -1};
        VK_CHECK(vkCreateGraphicsPipelines(ctx_->device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &linesPipeline_));
        vkDestroyShaderModule(ctx_->device(), vm, nullptr);
        vkDestroyShaderModule(ctx_->device(), fm, nullptr);
    }

    // 3. 2D UI & Camera Matrix Pipeline
    {
        VkDescriptorSetLayoutBinding bindings[2]{
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, bindings};
        VK_CHECK(vkCreateDescriptorSetLayout(ctx_->device(), &dslci, nullptr, &uiSetLayout_));

        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(UiPC)};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &uiSetLayout_, 1, &pcr};
        VK_CHECK(vkCreatePipelineLayout(ctx_->device(), &plci, nullptr, &uiLayout_));

        auto vc = readSpirvFile(resolveAssetPath("shaders/ui.vert.spv"));
        auto fc = readSpirvFile(resolveAssetPath("shaders/ui.frag.spv"));
        VkShaderModule vm = createShaderModule(ctx_->device(), vc);
        VkShaderModule fm = createShaderModule(ctx_->device(), fc);

        VkPipelineShaderStageCreateInfo stages[2]{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vm, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fm, "main", nullptr},
        };
        VkVertexInputBindingDescription bind{0, sizeof(UiVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[4]{
            {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(UiVertex, x)},
            {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(UiVertex, u)},
            {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(UiVertex, r)},
            {3, 0, VK_FORMAT_R32_SFLOAT, offsetof(UiVertex, mode)},
        };
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &bind, 4, attrs};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
                                                  VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE};
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS, VK_FALSE, VK_FALSE, {}, {}, 0, 1};
        VkPipelineColorBlendAttachmentState cba{VK_TRUE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD,
                                               VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, 0xF};
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cba, {0,0,0,0}};
        VkDynamicState dynS[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, dynS};
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, nullptr, 0, 2, stages, &vi, &ia, nullptr, &vp, &rs, &ms, &ds, &cb, &dyn, uiLayout_, mainPass_, 0, VK_NULL_HANDLE, -1};
        VK_CHECK(vkCreateGraphicsPipelines(ctx_->device(), VK_NULL_HANDLE, 1, &gpci, nullptr, &uiPipeline_));
        vkDestroyShaderModule(ctx_->device(), vm, nullptr);
        vkDestroyShaderModule(ctx_->device(), fm, nullptr);
    }
}

void VulkanViewer::createSyncObjects() {
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
    VK_CHECK(vkCreateSemaphore(ctx_->device(), &sci, nullptr, &imageAvailSem_));
    VK_CHECK(vkCreateSemaphore(ctx_->device(), &sci, nullptr, &renderDoneSem_));
    VK_CHECK(vkCreateFence(ctx_->device(), &fci, nullptr, &inFlightFence_));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                     ctx_->commandPool(), VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK_CHECK(vkAllocateCommandBuffers(ctx_->device(), &cbai, &frameCmd_));
}

void VulkanViewer::applyUiAction(UiAction action, int param, SimConfig& cfg, WorldEngine& world,
                                 CameraRigRenderer& rig, PerceptionPipeline& pipeline,
                                 ViewerToggleState& toggles, std::string& toastMsg) {
    switch (action) {
        case UiAction::SetCamOverlay: {
            toggles.camOverlay = static_cast<CamOverlayMode>(std::clamp(param, 0, 4));
            const char* names[5] = {
                "Cam Overlay: 2D Neural Boxes + ByteTrack",
                "Cam Overlay: Raw Distorted RGB",
                "Cam Overlay: Intrinsic Calib Grid",
                "Cam Overlay: Soil / Feature Heatmap",
                "Cam Overlay: 3D Voxel Reprojection"
            };
            toastMsg = names[static_cast<int>(toggles.camOverlay)];
            break;
        }
        case UiAction::SelectCam: {
            toggles.selectedCam = std::clamp(param, 0, 7);
            toastMsg = "Inspecting " + cfg.cameras[toggles.selectedCam].id;
            break;
        }
        case UiAction::SetViewMode: {
            toggles.viewMode = static_cast<ViewMode>(std::clamp(param, 0, 2));
            toastMsg = (toggles.viewMode == ViewMode::Chase3D) ? "Viewport: 3D Twin"
                     : (toggles.viewMode == ViewMode::FsdVoxel) ? "Viewport: FSD Voxel" : "Viewport: BEV Tactical";
            break;
        }
        case UiAction::ToggleRes: {
            const bool toFull = (rig.camWidth() == cfg.camWidth);
            rig.resize(toFull ? cfg.fullCamWidth : cfg.camWidth, toFull ? cfg.fullCamHeight : cfg.camHeight);
            pipeline.onCameraResized(rig);
            refreshCameraDescriptors(rig);
            toastMsg = toFull ? "8x 640x480 Full Res" : "8x 320x240 Half Res";
            break;
        }
        case UiAction::ToggleDriveMode: {
            cfg.driveMode = (cfg.driveMode == "auto") ? "manual" : "auto";
            toastMsg = (cfg.driveMode == "manual") ? "Manual WASD Drive Active" : "Autonomous FSD Engaged";
            break;
        }
        case UiAction::ToggleAeb: {
            cfg.aebEnabled = !cfg.aebEnabled;
            toastMsg = cfg.aebEnabled ? "36Hz RSS/AEB Shield ON" : "AEB Shield OFF (Raw SAT Collisions)";
            break;
        }
        case UiAction::ToggleSoilSelected: {
            cfg.cameras[toggles.selectedCam].soiled = !cfg.cameras[toggles.selectedCam].soiled;
            toastMsg = cfg.cameras[toggles.selectedCam].id + (cfg.cameras[toggles.selectedCam].soiled ? " SOILED" : " CLEANED");
            break;
        }
        case UiAction::ToggleLayerVoxels:
            toggles.showVoxels = !toggles.showVoxels;
            toastMsg = toggles.showVoxels ? "3D Occupancy Voxels ON" : "3D Occupancy Voxels OFF";
            break;
        case UiAction::ToggleLayerShadows:
            toggles.showShadows = !toggles.showShadows;
            toastMsg = toggles.showShadows ? "Occlusion Shadows ON" : "Occlusion Shadows OFF";
            break;
        case UiAction::ToggleLayerFlow:
            toggles.showFlow = !toggles.showFlow;
            toastMsg = toggles.showFlow ? "4D Occupancy Flow Vectors ON" : "4D Occupancy Flow Vectors OFF";
            break;
        case UiAction::ToggleLayerGhosts:
            toggles.showWorldGhosts = !toggles.showWorldGhosts;
            toastMsg = toggles.showWorldGhosts ? "World Model Rollout Ghosts ON" : "World Model Rollout Ghosts OFF";
            break;
        case UiAction::ToggleLayerRss:
            toggles.showRss = !toggles.showRss;
            toastMsg = toggles.showRss ? "RSS Uncertainty Carpet ON" : "RSS Uncertainty Carpet OFF";
            break;
        case UiAction::ToggleLayerColliders:
            toggles.showColliders = !toggles.showColliders;
            toastMsg = toggles.showColliders ? "SAT Colliders & Track Boxes ON" : "SAT Colliders & Track Boxes OFF";
            break;
        case UiAction::ToggleLayerFrustums:
            toggles.showFrustums = !toggles.showFrustums;
            toastMsg = toggles.showFrustums ? "8-Camera 3D Frustums ON" : "8-Camera 3D Frustums OFF";
            break;
        case UiAction::ToggleLayerPlanner:
            toggles.showPlanner = !toggles.showPlanner;
            toastMsg = toggles.showPlanner ? "Quintic Lattice Planner ON" : "Quintic Lattice Planner OFF";
            break;
        case UiAction::ScenarioJaywalk:
            world.triggerJaywalkerCutIn();
            toastMsg = "Scenario: Jaywalker Cut-In 11.5m Ahead";
            break;
        case UiAction::ScenarioOccluded:
            world.triggerOccludedDartOut();
            toastMsg = "Scenario: Occluded Dart-Out Behind Truck";
            break;
        case UiAction::ScenarioHardBrake:
            world.triggerLeadHardBrake();
            toastMsg = "Scenario: Lead Hard Brake (-7.5 m/s2)";
            break;
        case UiAction::ScenarioSoilFront:
            cfg.cameras[0].soiled = true;
            cfg.cameras[2].soiled = true;
            cfg.cameras[3].soiled = true;
            toastMsg = "Scenario: Front-Left Cameras Soiled";
            break;
        case UiAction::ToggleForcedRed: {
            const bool next = !world.forcedRedSignal();
            world.setForcedRedSignal(next);
            toastMsg = next ? "Traffic Signal LOCKED RED" : "Traffic Signal Automatic Cycling";
            break;
        }
        case UiAction::ScenarioReset:
            for (auto& cam : cfg.cameras) cam.soiled = false;
            world.reset();
            pipeline.reset();
            toastMsg = "Reset Nominal Traffic & Clean Lenses";
            break;
        case UiAction::None:
        default:
            break;
    }
}

bool VulkanViewer::pollEvents(SimConfig& cfg, WorldEngine& world, CameraRigRenderer& rig,
                              PerceptionPipeline& pipeline, ManualInput& manual, ViewerToggleState& toggles,
                              std::string& toastMsg) {
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        if (ev.type == SDL_QUIT) return false;
        if (ev.type == SDL_WINDOWEVENT && ev.window.event == SDL_WINDOWEVENT_RESIZED) {
            framebufferResized_ = true;
        } else if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT) {
            const float mx = static_cast<float>(ev.button.x);
            const float my = static_cast<float>(ev.button.y);
            bool hitUi = false;
            for (int i = static_cast<int>(uiButtonsScratch_.size()) - 1; i >= 0; --i) {
                if (uiButtonsScratch_[i].contains(mx, my)) {
                    applyUiAction(uiButtonsScratch_[i].action, uiButtonsScratch_[i].param,
                                  cfg, world, rig, pipeline, toggles, toastMsg);
                    hitUi = true;
                    break;
                }
            }
            if (!hitUi) {
                for (const auto& pan : uiPanelsScratch_) {
                    if (mx >= pan.x && mx <= pan.x + pan.w && my >= pan.y && my <= pan.y + pan.h) {
                        hitUi = true;
                        break;
                    }
                }
            }
            if (!hitUi) {
                dragging_ = true;
                lastMouseX_ = ev.button.x;
                lastMouseY_ = ev.button.y;
            }
        } else if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT) {
            dragging_ = false;
        } else if (ev.type == SDL_MOUSEMOTION && dragging_) {
            const int dx = ev.motion.x - lastMouseX_;
            const int dy = ev.motion.y - lastMouseY_;
            lastMouseX_ = ev.motion.x;
            lastMouseY_ = ev.motion.y;
            orbitYaw_ -= static_cast<float>(dx) * 0.0065f;
            orbitPitch_ = clampf(orbitPitch_ + static_cast<float>(dy) * 0.0055f, 0.08f, 1.38f);
        } else if (ev.type == SDL_MOUSEWHEEL) {
            orbitDist_ = clampf(orbitDist_ - static_cast<float>(ev.wheel.y) * 1.1f, 5.5f, 48.0f);
        } else if (ev.type == SDL_KEYDOWN && ev.key.repeat == 0) {
            const SDL_Keycode k = ev.key.keysym.sym;
            if (k == SDLK_ESCAPE) return false;
            if (k == SDLK_1) applyUiAction(UiAction::SetCamOverlay, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_2) applyUiAction(UiAction::SetCamOverlay, 1, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_3) applyUiAction(UiAction::SetCamOverlay, 2, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_4) applyUiAction(UiAction::SetCamOverlay, 3, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_5) applyUiAction(UiAction::SetCamOverlay, 4, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_v) applyUiAction(UiAction::SetViewMode, (static_cast<int>(toggles.viewMode) + 1) % 3, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_SPACE) applyUiAction(UiAction::ToggleDriveMode, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_b) applyUiAction(UiAction::ToggleAeb, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_r) applyUiAction(UiAction::ToggleRes, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_TAB) applyUiAction(UiAction::SelectCam, (toggles.selectedCam + 1) % 8, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_c) applyUiAction(UiAction::ToggleSoilSelected, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_F1) applyUiAction(UiAction::ScenarioJaywalk, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_F2) applyUiAction(UiAction::ScenarioOccluded, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_F3) applyUiAction(UiAction::ScenarioHardBrake, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_F4) applyUiAction(UiAction::ScenarioSoilFront, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_F5) applyUiAction(UiAction::ScenarioReset, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_F6) applyUiAction(UiAction::ToggleForcedRed, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_o) applyUiAction(UiAction::ToggleLayerVoxels, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_h) applyUiAction(UiAction::ToggleLayerShadows, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_f) applyUiAction(UiAction::ToggleLayerFlow, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_m) applyUiAction(UiAction::ToggleLayerGhosts, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_e) applyUiAction(UiAction::ToggleLayerRss, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_k) applyUiAction(UiAction::ToggleLayerColliders, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_g) applyUiAction(UiAction::ToggleLayerFrustums, 0, cfg, world, rig, pipeline, toggles, toastMsg);
            else if (k == SDLK_p) applyUiAction(UiAction::ToggleLayerPlanner, 0, cfg, world, rig, pipeline, toggles, toastMsg);
        }
    }

    const Uint8* ks = SDL_GetKeyboardState(nullptr);
    manual.forward  = ks[SDL_SCANCODE_W] || ks[SDL_SCANCODE_UP];
    manual.backward = ks[SDL_SCANCODE_S] || ks[SDL_SCANCODE_DOWN];
    manual.left     = ks[SDL_SCANCODE_A] || ks[SDL_SCANCODE_LEFT];
    manual.right    = ks[SDL_SCANCODE_D] || ks[SDL_SCANCODE_RIGHT];
    return true;
}

void VulkanViewer::buildVoxelAndGhostGeometry(const WorldEngine& world, const PerceptionFrameOutput& perc,
                                              const ViewerToggleState& toggles) {
    voxelVertsScratch_.clear();
    voxelIdxScratch_.clear();

    if (toggles.showVoxels) {
        int count = 0;
        for (const auto& v : perc.activeVoxels) {
            if (count++ >= 3400) break;
            const float scale = 0.55f + 0.45f * v.confidence;
            const float bright = (v.confidence > 0.85f) ? 1.0f : (0.55f * v.confidence + 0.25f);
            const Vec3 col = semanticColor(v.sem) * bright;
            appendCube(voxelVertsScratch_, voxelIdxScratch_, {v.wx, v.wy, v.wz},
                       {0.84f * scale, 0.42f * scale, 0.84f * scale}, world.ego().yaw, col, 0.86f, 0.15f);
        }
    }
    if (toggles.showShadows) {
        int count = 0;
        for (const auto& v : perc.shadowVoxels) {
            if (count++ >= 800) break;
            appendCube(voxelVertsScratch_, voxelIdxScratch_, {v.wx, v.wy, v.wz},
                       {0.58f, 0.28f, 0.58f}, world.ego().yaw, semanticColor(SemanticClass::OccShadow), 0.72f, 0.08f);
        }
    }
    if (toggles.showWorldGhosts) {
        for (const auto& g : perc.worldGhosts) {
            const float sc = g.worldPosScale.w;
            appendCube(voxelVertsScratch_, voxelIdxScratch_,
                       {g.worldPosScale.x, g.worldPosScale.y, g.worldPosScale.z},
                       {0.76f * sc, 0.36f * sc, 0.76f * sc}, g.yawMeta.x,
                       {g.colorAlpha.x, g.colorAlpha.y, g.colorAlpha.z}, g.colorAlpha.w, 0.4f);
        }
    }

    // RSS Safety Carpet
    if (toggles.showRss) {
        const EgoState& ego = world.ego();
        const float envLen = std::max(1.2f, std::min(ego.rssSafeDist, ego.minObstacleDist + 0.5f));
        const float sinE = std::sin(ego.yaw), cosE = std::cos(ego.yaw);
        Vec3 rssCol{0.024f, 0.714f, 0.831f};
        float alpha = 0.24f;
        if (ego.aebActive || ego.behaviorState == "COLLISION_CONTACT") {
            rssCol = {0.957f, 0.247f, 0.369f}; alpha = 0.45f;
        } else if (ego.behaviorState == "YIELDING_PEDESTRIAN" || ego.behaviorState == "RED_LIGHT_HOLD" ||
                   ego.behaviorState == "STOP_SIGN_HOLD" || ego.behaviorState == "FOLLOWING_LEAD") {
            rssCol = {0.961f, 0.620f, 0.043f}; alpha = 0.32f;
        }
        const float centerFwd = 3.45f + envLen * 0.5f;
        appendCube(voxelVertsScratch_, voxelIdxScratch_,
                   {ego.pos.x + sinE * centerFwd, 0.08f, ego.pos.z + cosE * centerFwd},
                   {2.15f, 0.04f, envLen}, ego.yaw, rssCol, alpha, 0.35f);
        const float barFwd = 3.45f + envLen;
        appendCube(voxelVertsScratch_, voxelIdxScratch_,
                   {ego.pos.x + sinE * barFwd, 0.12f, ego.pos.z + cosE * barFwd},
                   {2.35f, 0.14f, 0.22f}, ego.yaw, rssCol, 0.88f, 0.75f);
    }

    if (!voxelVertsScratch_.empty() && voxelVertsScratch_.size() <= 120000 && voxelIdxScratch_.size() <= 200000) {
        voxelVB_.upload(voxelVertsScratch_.data(), voxelVertsScratch_.size() * sizeof(SceneVertex));
        voxelIB_.upload(voxelIdxScratch_.data(), voxelIdxScratch_.size() * sizeof(uint32_t));
    }
}

void VulkanViewer::build3DLines(const SimConfig& cfg, const WorldEngine& world, const PerceptionFrameOutput& perc,
                                const PlannerCommand& cmd, const ViewerToggleState& toggles) {
    lineVertsScratch_.clear();

    if (toggles.viewMode == ViewMode::FsdVoxel) {
        const Vec4 gridCol{0.12f, 0.17f, 0.25f, 0.55f};
        for (float g = -100.0f; g <= 100.0f; g += 4.0f) {
            lineVertsScratch_.push_back({g, 0.02f, -100.0f, gridCol.x, gridCol.y, gridCol.z, gridCol.w});
            lineVertsScratch_.push_back({g, 0.02f,  100.0f, gridCol.x, gridCol.y, gridCol.z, gridCol.w});
            lineVertsScratch_.push_back({-100.0f, 0.02f, g, gridCol.x, gridCol.y, gridCol.z, gridCol.w});
            lineVertsScratch_.push_back({ 100.0f, 0.02f, g, gridCol.x, gridCol.y, gridCol.z, gridCol.w});
        }
    }

    if (toggles.showPlanner) {
        for (size_t c = 0; c < cmd.candidateTrajectories.size(); ++c) {
            const auto& pts = cmd.candidateTrajectories[c];
            const bool isBest = (static_cast<int>(c) == cmd.bestCandIdx);
            const Vec4 col = isBest ? Vec4{0.063f, 0.725f, 0.506f, 0.98f}
                           : cmd.candidateBlocked[c] ? Vec4{0.957f, 0.247f, 0.369f, 0.55f}
                           : Vec4{0.392f, 0.455f, 0.545f, 0.38f};
            const float yOff = isBest ? 0.06f : 0.0f;
            for (size_t s = 1; s < pts.size(); ++s) {
                lineVertsScratch_.push_back({pts[s - 1].x, pts[s - 1].y + yOff, pts[s - 1].z, col.x, col.y, col.z, col.w});
                lineVertsScratch_.push_back({pts[s].x,     pts[s].y + yOff,     pts[s].z,     col.x, col.y, col.z, col.w});
            }
        }
    }

    if (toggles.showColliders) {
        const OBB egoOBB = world.getEgoWorldOBB();
        appendBoxLines(lineVertsScratch_, {egoOBB.cx, egoOBB.cy, egoOBB.cz}, {egoOBB.hx + 0.02f, egoOBB.hy + 0.02f, egoOBB.hz + 0.02f},
                       egoOBB.yaw, {0.133f, 0.827f, 0.933f, 0.75f});
        for (const auto& v : world.vehicles()) {
            appendBoxLines(lineVertsScratch_, {v.pos.x, v.halfExtents.y, v.pos.z}, v.halfExtents, v.yaw, {0.22f, 0.74f, 0.97f, 0.45f});
        }
        // Also draw ByteTrack perceived tracks + uncertainty halos in bright lime/yellow!
        for (const auto& tr : perc.tracks) {
            const Vec4 col = tr.matchedLowConf ? Vec4{0.984f, 0.749f, 0.141f, 0.9f} : Vec4{0.518f, 0.922f, 0.322f, 0.9f};
            appendBoxLines(lineVertsScratch_, {tr.worldPos.x, tr.halfExtents.y, tr.worldPos.z},
                           {tr.halfExtents.x + tr.posSigmaM * 0.25f, tr.halfExtents.y, tr.halfExtents.z + tr.posSigmaM * 0.25f},
                           tr.yaw, col);
        }
    }

    if (toggles.showFlow) {
        for (const auto& tr : perc.tracks) {
            const float spd = std::hypot(tr.worldVel.x, tr.worldVel.z);
            if (spd < 0.3f) continue;
            const Vec3 p0{tr.worldPos.x, 1.45f, tr.worldPos.z};
            const Vec3 p1{tr.worldPos.x + tr.worldVel.x * 0.8f, 1.45f, tr.worldPos.z + tr.worldVel.z * 0.8f};
            lineVertsScratch_.push_back({p0.x, p0.y, p0.z, 0.91f, 0.47f, 0.98f, 0.95f});
            lineVertsScratch_.push_back({p1.x, p1.y, p1.z, 0.91f, 0.47f, 0.98f, 0.95f});
            // Arrowhead chevron at p1
            const Vec3 dir = (p1 - p0).normalized();
            const Vec3 perp{-dir.z, 0.0f, dir.x};
            const Vec3 hL = p1 - dir * 0.32f + perp * 0.22f;
            const Vec3 hR = p1 - dir * 0.32f - perp * 0.22f;
            lineVertsScratch_.push_back({p1.x, p1.y, p1.z, 0.95f, 0.65f, 1.0f, 0.95f});
            lineVertsScratch_.push_back({hL.x, hL.y, hL.z, 0.95f, 0.65f, 1.0f, 0.95f});
            lineVertsScratch_.push_back({p1.x, p1.y, p1.z, 0.95f, 0.65f, 1.0f, 0.95f});
            lineVertsScratch_.push_back({hR.x, hR.y, hR.z, 0.95f, 0.65f, 1.0f, 0.95f});
        }

        // Dynamic occupancy voxel velocity vectors
        int flowCount = 0;
        for (const auto& v : perc.activeVoxels) {
            if (flowCount >= 140) break;
            const float spd = std::hypot(v.vx, v.vz);
            if (spd < 0.6f) continue;
            ++flowCount;
            const Vec3 p0{v.wx, v.wy + 0.15f, v.wz};
            const Vec3 p1{v.wx + v.vx * 0.35f, v.wy + 0.15f, v.wz + v.vz * 0.35f};
            lineVertsScratch_.push_back({p0.x, p0.y, p0.z, 0.22f, 0.88f, 0.97f, 0.80f});
            lineVertsScratch_.push_back({p1.x, p1.y, p1.z, 0.22f, 0.88f, 0.97f, 0.80f});
        }
    }

    if (world.contactFlashAlpha() > 0.01f) {
        const float a = world.contactFlashAlpha();
        const Vec3 center = world.contactFlashPos();
        const float radius = 0.85f + (1.0f - a) * 1.45f;
        const int segs = 28;
        for (int i = 0; i < segs; ++i) {
            const float t0 = (2.0f * kPi * static_cast<float>(i)) / static_cast<float>(segs);
            const float t1 = (2.0f * kPi * static_cast<float>(i + 1)) / static_cast<float>(segs);
            lineVertsScratch_.push_back({center.x + std::cos(t0) * radius, 0.24f, center.z + std::sin(t0) * radius, 0.98f, 0.18f, 0.28f, a});
            lineVertsScratch_.push_back({center.x + std::cos(t1) * radius, 0.24f, center.z + std::sin(t1) * radius, 0.98f, 0.18f, 0.28f, a});
        }
    }

    if (toggles.showFrustums) {
        const Mat4 worldFromSprung = world.getEgoSprungWorldMatrix();
        for (int c = 0; c < 8; ++c) {
            const auto& spec = cfg.cameras[c];
            const auto intr = spec.intrinsics(cfg.camWidth, cfg.camHeight);
            const Mat4 wfc = worldFromSprung * spec.extrinsicVehicleLocalFromCam();
            const float fl = std::min(spec.maxRangeM * 0.38f, 16.0f);
            const float hw = std::tan(intr.hfovRad * 0.5f) * fl;
            const float hh = std::tan(intr.vfovRad * 0.5f) * fl;
            const Vec3 o = wfc.transformPoint({0, 0, 0});
            const Vec3 f[4] = {
                wfc.transformPoint({-hw, -hh, fl}),
                wfc.transformPoint({ hw, -hh, fl}),
                wfc.transformPoint({ hw,  hh, fl}),
                wfc.transformPoint({-hw,  hh, fl}),
            };
            const float alpha = spec.soiled ? 0.12f : (c == toggles.selectedCam ? 0.9f : 0.45f);
            const Vec4 col{spec.uiColor.x, spec.uiColor.y, spec.uiColor.z, alpha};
            for (int i = 0; i < 4; ++i) {
                lineVertsScratch_.push_back({o.x, o.y, o.z, col.x, col.y, col.z, col.w});
                lineVertsScratch_.push_back({f[i].x, f[i].y, f[i].z, col.x, col.y, col.z, col.w});
                lineVertsScratch_.push_back({f[i].x, f[i].y, f[i].z, col.x, col.y, col.z, col.w});
                lineVertsScratch_.push_back({f[(i + 1) % 4].x, f[(i + 1) % 4].y, f[(i + 1) % 4].z, col.x, col.y, col.z, col.w});
            }
        }
    }

    if (!lineVertsScratch_.empty() && lineVertsScratch_.size() <= 32768) {
        lineVB_.upload(lineVertsScratch_.data(), lineVertsScratch_.size() * sizeof(LineVertex));
    }
}

void VulkanViewer::uiRect(float x, float y, float w, float h, Vec4 col) {
    const uint32_t b = static_cast<uint32_t>(uiVertsScratch_.size());
    uiVertsScratch_.push_back({x,     y,     0, 0, col.x, col.y, col.z, col.w, 0.0f});
    uiVertsScratch_.push_back({x + w, y,     1, 0, col.x, col.y, col.z, col.w, 0.0f});
    uiVertsScratch_.push_back({x + w, y + h, 1, 1, col.x, col.y, col.z, col.w, 0.0f});
    uiVertsScratch_.push_back({x,     y + h, 0, 1, col.x, col.y, col.z, col.w, 0.0f});
    uiIdxScratch_.push_back(b + 0); uiIdxScratch_.push_back(b + 1); uiIdxScratch_.push_back(b + 2);
    uiIdxScratch_.push_back(b + 0); uiIdxScratch_.push_back(b + 2); uiIdxScratch_.push_back(b + 3);
}

void VulkanViewer::uiRectOutline(float x, float y, float w, float h, float t, Vec4 col) {
    uiRect(x, y, w, t, col);
    uiRect(x, y + h - t, w, t, col);
    uiRect(x, y, t, h, col);
    uiRect(x + w - t, y, t, h, col);
}

void VulkanViewer::uiLine(float x0, float y0, float x1, float y1, float t, Vec4 col) {
    const float dx = x1 - x0, dy = y1 - y0;
    const float len = std::hypot(dx, dy);
    if (len < 0.5f) return;
    const float nx = -dy / len * (t * 0.5f);
    const float ny =  dx / len * (t * 0.5f);
    const uint32_t b = static_cast<uint32_t>(uiVertsScratch_.size());
    uiVertsScratch_.push_back({x0 + nx, y0 + ny, 0, 0, col.x, col.y, col.z, col.w, 0.0f});
    uiVertsScratch_.push_back({x1 + nx, y1 + ny, 1, 0, col.x, col.y, col.z, col.w, 0.0f});
    uiVertsScratch_.push_back({x1 - nx, y1 - ny, 1, 1, col.x, col.y, col.z, col.w, 0.0f});
    uiVertsScratch_.push_back({x0 - nx, y0 - ny, 0, 1, col.x, col.y, col.z, col.w, 0.0f});
    uiIdxScratch_.push_back(b + 0); uiIdxScratch_.push_back(b + 1); uiIdxScratch_.push_back(b + 2);
    uiIdxScratch_.push_back(b + 0); uiIdxScratch_.push_back(b + 2); uiIdxScratch_.push_back(b + 3);
}

void VulkanViewer::uiText(float x, float y, float scale, Vec4 col, const std::string& text) {
    float cx = x;
    const float adv = 6.0f * scale;
    const float cw = 7.0f * scale;
    const float ch = 7.0f * scale;
    for (char chRaw : text) {
        if (chRaw == '\n') { cx = x; y += 9.0f * scale; continue; }
        const int code = (chRaw >= 32 && chRaw <= 126) ? (chRaw - 32) : 0;
        const int gx = (code % BitmapFontAtlas::kCols) * BitmapFontAtlas::kCellW;
        const int gy = (code / BitmapFontAtlas::kCols) * BitmapFontAtlas::kCellH;
        const float u0 = static_cast<float>(gx) / BitmapFontAtlas::kAtlasW;
        const float v0 = static_cast<float>(gy) / BitmapFontAtlas::kAtlasH;
        const float u1 = static_cast<float>(gx + 7) / BitmapFontAtlas::kAtlasW;
        const float v1 = static_cast<float>(gy + 7) / BitmapFontAtlas::kAtlasH;

        const uint32_t b = static_cast<uint32_t>(uiVertsScratch_.size());
        uiVertsScratch_.push_back({cx,      y,      u0, v0, col.x, col.y, col.z, col.w, 2.0f});
        uiVertsScratch_.push_back({cx + cw, y,      u1, v0, col.x, col.y, col.z, col.w, 2.0f});
        uiVertsScratch_.push_back({cx + cw, y + ch, u1, v1, col.x, col.y, col.z, col.w, 2.0f});
        uiVertsScratch_.push_back({cx,      y + ch, u0, v1, col.x, col.y, col.z, col.w, 2.0f});
        uiIdxScratch_.push_back(b + 0); uiIdxScratch_.push_back(b + 1); uiIdxScratch_.push_back(b + 2);
        uiIdxScratch_.push_back(b + 0); uiIdxScratch_.push_back(b + 2); uiIdxScratch_.push_back(b + 3);
        cx += adv;
    }
}

void VulkanViewer::uiTexQuad(float x, float y, float w, float h, Vec4 tint) {
    const uint32_t b = static_cast<uint32_t>(uiVertsScratch_.size());
    uiVertsScratch_.push_back({x,     y,     0, 0, tint.x, tint.y, tint.z, tint.w, 1.0f});
    uiVertsScratch_.push_back({x + w, y,     1, 0, tint.x, tint.y, tint.z, tint.w, 1.0f});
    uiVertsScratch_.push_back({x + w, y + h, 1, 1, tint.x, tint.y, tint.z, tint.w, 1.0f});
    uiVertsScratch_.push_back({x,     y + h, 0, 1, tint.x, tint.y, tint.z, tint.w, 1.0f});
    uiIdxScratch_.push_back(b + 0); uiIdxScratch_.push_back(b + 1); uiIdxScratch_.push_back(b + 2);
    uiIdxScratch_.push_back(b + 0); uiIdxScratch_.push_back(b + 2); uiIdxScratch_.push_back(b + 3);
}

void VulkanViewer::buildUiOverlay(const SimConfig& cfg, const WorldEngine& world, const CameraRigRenderer& rig,
                                  const PerceptionFrameOutput& perc, const PlannerCommand& cmd,
                                  const ViewerToggleState& toggles, const std::string& toastMsg) {
    (void)cmd;
    uiVertsScratch_.clear();
    uiIdxScratch_.clear();
    uiBatchesScratch_.clear();
    uiButtonsScratch_.clear();
    uiPanelsScratch_.clear();

    const float W = static_cast<float>(swapExtent_.width);
    const float H = static_cast<float>(swapExtent_.height);
    const float pad = 10.0f;
    const float topBarH = 166.0f;

    auto flushBatch = [&](int camIdx) {
        const uint32_t endIdx = static_cast<uint32_t>(uiIdxScratch_.size());
        uint32_t start = 0;
        if (!uiBatchesScratch_.empty()) {
            const auto& prev = uiBatchesScratch_.back();
            start = prev.indexOffset + prev.indexCount;
        }
        if (endIdx > start) {
            uiBatchesScratch_.push_back({start, endIdx - start, camIdx});
        }
    };

    auto addChipButton = [&](float bx, float by, float bw, float bh, const std::string& label,
                             bool active, Vec4 activeBg, Vec4 activeBorder, UiAction act, int param = 0) {
        const Vec4 bg = active ? activeBg : Vec4{0.08f, 0.12f, 0.19f, 0.92f};
        const Vec4 bd = active ? activeBorder : Vec4{0.24f, 0.30f, 0.40f, 0.90f};
        const Vec4 fg = active ? Vec4{0.98f, 0.99f, 1.0f, 1.0f} : Vec4{0.76f, 0.82f, 0.89f, 1.0f};
        uiRect(bx, by, bw, bh, bg);
        uiRectOutline(bx, by, bw, bh, 1.0f, bd);
        uiText(bx + 6.0f, by + (bh - 8.5f) * 0.5f, 1.15f, fg, label);
        uiButtonsScratch_.push_back({bx, by, bw, bh, act, param});
    };

    auto addCheckbox = [&](float cxBox, float cyBox, float cwBox, const std::string& label,
                           bool checked, Vec4 dotCol, UiAction act) {
        const float ch = 18.0f;
        uiRect(cxBox, cyBox, cwBox, ch, checked ? Vec4{0.06f, 0.14f, 0.21f, 0.78f} : Vec4{0.05f, 0.08f, 0.13f, 0.65f});
        uiRectOutline(cxBox + 2.0f, cyBox + 3.0f, 12.0f, 12.0f, 1.0f,
                      checked ? dotCol : Vec4{0.35f, 0.42f, 0.52f, 0.9f});
        if (checked) {
            uiRect(cxBox + 4.5f, cyBox + 5.5f, 7.0f, 7.0f, dotCol);
        }
        uiText(cxBox + 18.0f, cyBox + 4.5f, 1.15f,
               checked ? Vec4{0.93f, 0.96f, 1.0f, 1.0f} : Vec4{0.58f, 0.65f, 0.74f, 1.0f}, label);
        uiButtonsScratch_.push_back({cxBox, cyBox, cwBox, ch, act, 0});
    };

    // 1. Top glass bar background, title & clickable 2D sensor overlay pills
    uiPanelsScratch_.push_back({pad, pad, W - 2.0f * pad, topBarH});
    uiRect(pad, pad, W - 2.0f * pad, topBarH, {0.035f, 0.051f, 0.086f, 0.88f});
    uiRectOutline(pad, pad, W - 2.0f * pad, topBarH, 1.0f, {0.20f, 0.25f, 0.33f, 0.85f});
    const bool aebWarn = world.ego().aebActive || world.ego().behaviorState == "COLLISION_CONTACT";
    uiRect(pad + 10.0f, pad + 10.0f, 8.0f, 8.0f, aebWarn ? Vec4{0.96f, 0.25f, 0.37f, 1.0f} : Vec4{0.20f, 0.83f, 0.60f, 1.0f});
    uiText(pad + 24.0f, pad + 8.0f, 1.45f, {0.95f, 0.97f, 1.0f, 1.0f},
           "TESLA CYBERCAB FSD VISION + SAT PHYSICS");

    char resBadge[48];
    std::snprintf(resBadge, sizeof(resBadge), "8x %dx%d @ %.0fHz", rig.camWidth(), rig.camHeight(), cfg.controlHz);
    uiRect(pad + 440.0f, pad + 5.0f, 150.0f, 18.0f, {0.08f, 0.14f, 0.22f, 0.92f});
    uiRectOutline(pad + 440.0f, pad + 5.0f, 150.0f, 18.0f, 1.0f, {0.20f, 0.35f, 0.50f, 0.8f});
    uiText(pad + 447.0f, pad + 9.0f, 1.15f, {0.40f, 0.86f, 0.98f, 1.0f}, resBadge);

    const char* ovNames[5] = {"[1]Neural+Box", "[2]Raw RGB", "[3]Calib Grid", "[4]Heat Map", "[5]Vox Reproj"};
    const float pillW = 104.0f;
    const float pillGap = 5.0f;
    float pillX = W - pad - 10.0f - (5.0f * pillW + 4.0f * pillGap);
    if (pillX > pad + 600.0f) {
        for (int m = 0; m < 5; ++m) {
            addChipButton(pillX, pad + 5.0f, pillW, 19.0f, ovNames[m],
                          static_cast<int>(toggles.camOverlay) == m,
                          {0.03f, 0.45f, 0.65f, 0.95f}, {0.22f, 0.78f, 0.98f, 1.0f},
                          UiAction::SetCamOverlay, m);
            pillX += pillW + pillGap;
        }
    }
    flushBatch(0);

    // 2. 8-Camera Matrix Cards (each draws its own texture quad!)
    const float cardGap = 6.0f;
    const float availW = W - 2.0f * pad - 20.0f;
    const float cardW = (availW - 7.0f * cardGap) / 8.0f;
    const float camImgH = std::min(106.0f, cardW * 0.75f);
    const float cardY = pad + 29.0f;

    for (int c = 0; c < 8; ++c) {
        const float cx = pad + 10.0f + static_cast<float>(c) * (cardW + cardGap);
        uiTexQuad(cx, cardY + 14.0f, cardW, camImgH, {1, 1, 1, 1});
        flushBatch(c);
    }

    // 3. Camera Overlays, Clickable Card Registration, Titles & Soil Badges
    for (int c = 0; c < 8; ++c) {
        const float cx = pad + 10.0f + static_cast<float>(c) * (cardW + cardGap);
        const float cy = cardY;
        const float iy = cy + 14.0f;
        const bool sel = (c == toggles.selectedCam);
        uiButtonsScratch_.push_back({cx, cy, cardW, camImgH + 14.0f, UiAction::SelectCam, c});

        uiRect(cx, cy, cardW, 14.0f, sel ? Vec4{0.06f, 0.22f, 0.34f, 0.96f} : Vec4{0.02f, 0.04f, 0.08f, 0.95f});
        uiText(cx + 4.0f, cy + 3.0f, 1.15f, {0.88f, 0.92f, 0.96f, 1.0f},
               cfg.cameras[c].shortName + " " + std::to_string(static_cast<int>(cfg.cameras[c].hfovDeg)) + "d");

        const float wCam = perc.soilCameras[c].tokenWeight;
        if (wCam < 0.70f || cfg.cameras[c].soiled) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "SOIL w=%.2f", wCam);
            uiRect(cx + cardW - 76.0f, iy + 3.0f, 73.0f, 12.0f, {0.91f, 0.35f, 0.05f, 0.92f});
            uiText(cx + cardW - 73.0f, iy + 5.0f, 1.0f, {1, 1, 1, 1}, buf);
        }

        if (toggles.camOverlay == CamOverlayMode::Neural) {
            for (const auto& d : perc.detections3D) {
                if (d.camIndex != c) continue;
                const auto& b2 = d.det2d;
                const float bx = cx + (b2.cxNorm - b2.wNorm * 0.5f) * cardW;
                const float by = iy + (b2.cyNorm - b2.hNorm * 0.5f) * camImgH;
                const float bw = b2.wNorm * cardW;
                const float bh = b2.hNorm * camImgH;
                const Vec3 sc = semanticColor(d.semantic);
                uiRectOutline(bx, by, bw, bh, 1.5f, {sc.x, sc.y, sc.z, 0.95f});
                char tag[16];
                const char* clsCh = (d.semantic == SemanticClass::Pedestrian) ? "P"
                                  : (d.semantic == SemanticClass::SignalSign) ? "S" : "V";
                std::snprintf(tag, sizeof(tag), "%s:%.0fm", clsCh, d.rangeM);
                uiRect(bx, std::max(iy, by - 10.0f), 36.0f, 10.0f, {0.03f, 0.05f, 0.09f, 0.82f});
                uiText(bx + 2.0f, std::max(iy, by - 9.0f), 0.95f, {sc.x, sc.y, sc.z, 1.0f}, tag);
