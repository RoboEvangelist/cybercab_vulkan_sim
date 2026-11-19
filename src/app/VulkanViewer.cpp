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
