#include "sensors/CameraRigRenderer.hpp"
#include "core/VulkanUtils.hpp"
#include "sensors/PhotonModel.hpp"

#include <cstring>
#include <fstream>

namespace cybercab {

namespace {

struct alignas(16) ScenePushConstants {
    Mat4 viewProj;
    Vec4 sunDirIntensity;
    Vec4 fogColorDensity;
    Vec4 hdrParams; // x = max output radiance
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
    uint32_t linearOut;
    float fullWell;
    float noiseA;
    float noiseB;
    uint32_t noiseSeed;
    uint32_t noiseEnable;
  uint32_t rccbEnable;
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};
static_assert(sizeof(IspPushConstants) == 96,
              "IspPushConstants must match cam_isp_soil.comp layout");

constexpr float kDaySkyR = 0.545f;
constexpr float kDaySkyG = 0.722f;
constexpr float kDaySkyB = 0.910f;
constexpr float kDayAmbient = 0.48f;
constexpr float kDayFogDens = 0.003f;
constexpr float kDayDefaultDiffGain = 0.0f;
constexpr float kDayDefaultEmisGain = 0.0f;
constexpr float kNightSkyR = 0.015f;
constexpr float kNightSkyG = 0.022f;
constexpr float kNightSkyB = 0.048f;
constexpr float kNightAmbient = 0.055f;
constexpr float kNightDiffGain = 0.14f;
constexpr float kNightEmisGain = 5.50f;
constexpr float kNightFogDens = 0.006f;

} // namespace

CameraRigRenderer::~CameraRigRenderer() {
    destroy();
}

void CameraRigRenderer::init(const VulkanContext& ctx, const SimConfig& cfg, const WorldEngine& world) {
    destroy();
    ctx_ = &ctx;
    hdr_ = cfg.isV15();
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
                            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
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
    camLinearSSBO_.destroy();
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
    atts[0].format = rawColorFormat();
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
        rawColorImages_[c].create(*ctx_, w, h, rawColorFormat(),
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
    camLinearSSBO_.create(*ctx_, hdr_ ? tensorBytes : 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
        VkDescriptorBufferInfo linInfo = camLinearSSBO_.descriptorInfo();

        VkWriteDescriptorSet writes[4]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = ispSets_[c];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &rawInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = ispSets_[c];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &outInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = ispSets_[c];
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &bufInfo;

        writes[3] = writes[2];
        writes[3].dstBinding = 3;
        writes[3].pBufferInfo = &linInfo;

        vkUpdateDescriptorSets(ctx_->device(), 4, writes, 0, nullptr);
    }
}

void CameraRigRenderer::uploadStaticWorld(const WorldEngine& world) {
    std::vector<SceneVertex> verts;
    std::vector<uint32_t> indices;
    world.buildStaticWorldGeometry(verts, indices);
    staticIndexCount_ = static_cast<uint32_t>(indices.size());

    staticVB_.create(*ctx_, verts.size() * sizeof(SceneVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    staticVB_.uploadVector(verts);

    staticIB_.create(*ctx_, indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    staticIB_.uploadVector(indices);
}

void CameraRigRenderer::uploadDynamicWorld(const WorldEngine& world, bool includeEgo) {
    dynVertsScratch_.clear();
    dynIndicesScratch_.clear();
    world.appendDynamicGeometry(dynVertsScratch_, dynIndicesScratch_, false, includeEgo);
    dynamicIndexCount_ = static_cast<uint32_t>(dynIndicesScratch_.size());
    if (dynVertsScratch_.empty() || dynIndicesScratch_.empty()) return;

    const VkDeviceSize vbBytes = dynVertsScratch_.size() * sizeof(SceneVertex);
    const VkDeviceSize ibBytes = dynIndicesScratch_.size() * sizeof(uint32_t);
    if (vbBytes > dynamicVBCapacity_) {
        dynamicVBCapacity_ = vbBytes * 2;
        dynamicVB_.create(*ctx_, dynamicVBCapacity_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (ibBytes > dynamicIBCapacity_) {
        dynamicIBCapacity_ = ibBytes * 2;
        dynamicIB_.create(*ctx_, dynamicIBCapacity_, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    dynamicVB_.upload(dynVertsScratch_.data(), vbBytes);
    dynamicIB_.upload(dynIndicesScratch_.data(), ibBytes);
}

void CameraRigRenderer::renderAllCameras(const WorldEngine& world, const SimConfig& cfg,
                                         float timeSec, GpuProfiler* profiler) {
    uploadDynamicWorld(world);
    const Mat4 worldFromSprung = world.getEgoSprungWorldMatrix();

    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    if (profiler) {
        profiler->reset(cmd);
        profiler->beginStage(cmd, "1_cam_render", VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    }

    VkViewport vp{0.0f, 0.0f, static_cast<float>(camWidth_), static_cast<float>(camHeight_), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {static_cast<uint32_t>(camWidth_), static_cast<uint32_t>(camHeight_)}};

    for (int c = 0; c < kNumCams; ++c) {
        const auto& spec = cfg.cameras.at(c);
        CameraIntrinsics intr = spec.intrinsics(camWidth_, camHeight_);
        // Overscan wide/barrel cameras so Brown-Conrady distortion pulls outer scene into edges cleanly
        float renderScale = spec.renderOverscan();

        constexpr float kKnockEpsDeg = 1e-6f;
        const bool hasKnock =
          std::abs(cfg.cameraKnockRpyDeg.x) > kKnockEpsDeg ||
          std::abs(cfg.cameraKnockRpyDeg.y) > kKnockEpsDeg ||
          std::abs(cfg.cameraKnockRpyDeg.z) > kKnockEpsDeg;
        const Mat4 knockRot = hasKnock
          ? (Mat4::rotY(degToRad(cfg.cameraKnockRpyDeg.z)) *
             Mat4::rotX(degToRad(cfg.cameraKnockRpyDeg.y)) *
             Mat4::rotZ(degToRad(cfg.cameraKnockRpyDeg.x)))
          : Mat4{};
        Mat4 worldFromCam = hasKnock
          ? (worldFromSprung * knockRot * spec.extrinsicVehicleLocalFromCam())
          : (worldFromSprung * spec.extrinsicVehicleLocalFromCam());
        Mat4 camFromWorld = worldFromCam.inverseRigid();
        Mat4 projRender = Mat4::vulkanPerspectiveCV(
            intr.fx / renderScale, intr.fy / renderScale, intr.cx, intr.cy,
            static_cast<float>(camWidth_), static_cast<float>(camHeight_), 0.25f, spec.maxRangeM);
        Mat4 projNominal = Mat4::vulkanPerspectiveCV(
            intr.fx, intr.fy, intr.cx, intr.cy,
            static_cast<float>(camWidth_), static_cast<float>(camHeight_), 0.25f, spec.maxRangeM);

        camMats_[c] = {worldFromCam, camFromWorld, projNominal, projNominal * camFromWorld,
                       intr, spec.distortion, renderScale};

        const bool night = cfg.isNightGlare();
        const float skyR = night ? kNightSkyR : kDaySkyR;
        const float skyG = night ? kNightSkyG : kDaySkyG;
        const float skyB = night ? kNightSkyB : kDaySkyB;
        const float amb = night ? kNightAmbient : kDayAmbient;
        const float fogD = night ? kNightFogDens : kDayFogDens;
        const float diffG = night ? kNightDiffGain : kDayDefaultDiffGain;
        const float emisG = night ? kNightEmisGain : kDayDefaultEmisGain;

        VkClearValue clears[2]{};
        clears[0].color = {{skyR, skyG, skyB, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpbi{};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = offscreenPass_;
        rpbi.framebuffer = framebuffers_[c];
        rpbi.renderArea = sc;
        rpbi.clearValueCount = 2;
        rpbi.pClearValues = clears;

        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxPipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        ScenePushConstants pc{};
        pc.viewProj = projRender * camFromWorld;
        pc.sunDirIntensity = Vec4{Vec3{65.0f, 95.0f, 45.0f}.normalized(), amb};
        pc.fogColorDensity = Vec4{skyR, skyG, skyB, fogD};
        pc.hdrParams =
          Vec4{hdr_ ? photon::kFullWellRadiance : 1.0f, diffG, emisG, 0.0f};
        vkCmdPushConstants(cmd, gfxLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

        VkDeviceSize off = 0;
        VkBuffer svb = staticVB_.handle();
        vkCmdBindVertexBuffers(cmd, 0, 1, &svb, &off);
        vkCmdBindIndexBuffer(cmd, staticIB_.handle(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, staticIndexCount_, 1, 0, 0, 0);

        if (dynamicIndexCount_ > 0) {
            VkBuffer dvb = dynamicVB_.handle();
            vkCmdBindVertexBuffers(cmd, 0, 1, &dvb, &off);
            vkCmdBindIndexBuffer(cmd, dynamicIB_.handle(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, dynamicIndexCount_, 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
        rawColorImages_[c].setLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    if (profiler) {
        profiler->endStage(cmd, "1_cam_render", VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        profiler->beginStage(cmd, "1b_isp_soil", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    uint32_t gx = (camWidth_ + 15) / 16;
    uint32_t gy = (camHeight_ + 15) / 16;
    // Per-frame noise key: deterministic in (seed, timeSec) so runs are reproducible.
    uint32_t frameNoiseKey = 0;
    std::memcpy(&frameNoiseKey, &timeSec, sizeof(frameNoiseKey));
    frameNoiseKey *= 2654435761u;
    for (int c = 0; c < kNumCams; ++c) {
        VulkanImage::cmdTransitionLayout(cmd, distortedImages_[c].handle(), VK_IMAGE_ASPECT_COLOR_BIT,
                                         distortedImages_[c].layout(), VK_IMAGE_LAYOUT_GENERAL);
        distortedImages_[c].setLayout(VK_IMAGE_LAYOUT_GENERAL);

        const auto& spec = cfg.cameras.at(c);
        const auto& intr = camMats_[c].intr;
        const IspPushConstants ipc{
            static_cast<uint32_t>(c),
            static_cast<uint32_t>(camWidth_),
            static_cast<uint32_t>(camHeight_),
            spec.soiled ? 1u : 0u,
            Vec4{spec.distortion[0], spec.distortion[1], spec.distortion[2], spec.distortion[3]},
            Vec4{intr.fx, intr.fy, intr.cx, intr.cy},
            camMats_[c].renderScale,
            timeSec,
            hdr_ ? 1u : 0u,
            photon::kFullWellRadiance,
            cfg.photonShotNoiseA,
            cfg.photonReadNoiseB,
            cfg.photonNoiseSeed ^ frameNoiseKey,
            (hdr_ && cfg.photonNoiseEnabled) ? 1u : 0u,
            (hdr_ && cfg.photonRccbMosaic) ? 1u : 0u,
            0u,
            0u,
            0u
        };
        ispPipeline_.bindAndDispatch(cmd, ispSets_[c], gx, gy, 1, &ipc, sizeof(ipc));
    }
    cmdComputeBarrier(cmd);

    if (profiler) {
        profiler->endStage(cmd, "1b_isp_soil", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    ctx_->endSingleTimeCommands(cmd);
}

bool CameraRigRenderer::projectWorldToPixel(int camIdx, const Vec3& worldPt, Vec2& outPx, float& outDepth) const {
    const auto& cm = camMats_.at(camIdx);
    const Vec3 pc = cm.camFromWorld.transformPoint(worldPt);
    outDepth = pc.z;
    if (pc.z <= 0.1f) return false;
    const float xn = pc.x / pc.z;
    const float yn = pc.y / pc.z;
    const Vec2 dist = distortNormalized(xn, yn, cm.distortion);
    outPx.x = cm.intr.fx * dist.x + cm.intr.cx;
    outPx.y = cm.intr.fy * dist.y + cm.intr.cy;
    return true;
}

std::vector<uint8_t> CameraRigRenderer::downloadCameraRGBA(int camIdx) const {
    return distortedImages_.at(camIdx).downloadRGBA8();
}

void CameraRigRenderer::dumpCameraGridPPM(const std::string& path) const {
    const int cols = 4;
    const int rows = 2;
    const int totalW = cols * camWidth_;
    const int totalH = rows * camHeight_;
    std::vector<uint8_t> rgb(static_cast<size_t>(totalW) * totalH * 3, 0);

    for (int c = 0; c < kNumCams; ++c) {
        auto pixels = downloadCameraRGBA(c);
        int gx = c % cols;
        int gy = c / cols;
        for (int y = 0; y < camHeight_; ++y) {
            for (int x = 0; x < camWidth_; ++x) {
                size_t si = (static_cast<size_t>(y) * camWidth_ + x) * 4;
                size_t di = (static_cast<size_t>(gy * camHeight_ + y) * totalW + (gx * camWidth_ + x)) * 3;
                rgb[di + 0] = pixels[si + 0];
                rgb[di + 1] = pixels[si + 1];
                rgb[di + 2] = pixels[si + 2];
            }
        }
    }

    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << totalW << " " << totalH << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
}

void CameraRigRenderer::dumpEgoShowcaseBMP(const WorldEngine& world, const std::string& path) {
    uploadDynamicWorld(world, true);

    const EgoState& ego = world.ego();
    const float sinY = std::sin(ego.yaw), cosY = std::cos(ego.yaw);
    const Vec3 fwd{sinY, 0.0f, cosY};
    const Vec3 right{-cosY, 0.0f, sinY};
    const Vec3 carCenter = ego.pos + fwd * 1.25f + Vec3{0.0f, 0.72f, 0.0f};

    // View 0: Pure side profile matching Cybercab studio side shot (car faces +X right)
    // View 1: Elevated front 3/4 view showing windshield, DRL lightbar, splitter & amber markers
    // View 2: Elevated 3/4 rear chase perspective matching 3D Twin viewport
    const Vec3 eyes[3] = {
        carCenter + right * 4.9f + Vec3{0.0f, 0.06f, 0.0f},
        carCenter + fwd * 4.6f + right * 3.0f + Vec3{0.0f, 1.45f, 0.0f},
        carCenter - fwd * 4.8f - right * 3.1f + Vec3{0.0f, 1.85f, 0.0f}
    };

    const float fy = static_cast<float>(camHeight_) / (2.0f * std::tan(degToRad(46.0f) * 0.5f));
    const float fx = fy;
    const Mat4 proj = Mat4::vulkanPerspectiveCV(
        fx, fy, static_cast<float>(camWidth_) * 0.5f, static_cast<float>(camHeight_) * 0.5f,
        static_cast<float>(camWidth_), static_cast<float>(camHeight_), 0.25f, 200.0f);

    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    VkViewport vp{0.0f, 0.0f, static_cast<float>(camWidth_), static_cast<float>(camHeight_), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {static_cast<uint32_t>(camWidth_), static_cast<uint32_t>(camHeight_)}};

    for (int vIdx = 0; vIdx < 3; ++vIdx) {
        VkClearValue clears[2]{};
        clears[0].color = {{0.545f, 0.722f, 0.910f, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo rpbi{};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = offscreenPass_;
        rpbi.framebuffer = framebuffers_[vIdx];
        rpbi.renderArea = sc;
        rpbi.clearValueCount = 2;
        rpbi.pClearValues = clears;
        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfxPipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        const Mat4 view = Mat4::lookAtVulkan(eyes[vIdx], carCenter);
        ScenePushConstants pc{};
        pc.viewProj = proj * view;
        pc.sunDirIntensity = Vec4{Vec3{65.0f, 95.0f, 45.0f}.normalized(), 0.52f};
        pc.fogColorDensity = Vec4{0.545f, 0.722f, 0.910f, 0.002f};
        pc.hdrParams = Vec4{1.0f, 0.0f, 0.0f, 0.0f}; // showcase BMP is a display path
        vkCmdPushConstants(cmd, gfxLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

        const VkDeviceSize off = 0;
        const VkBuffer svb = staticVB_.handle();
        vkCmdBindVertexBuffers(cmd, 0, 1, &svb, &off);
        vkCmdBindIndexBuffer(cmd, staticIB_.handle(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, staticIndexCount_, 1, 0, 0, 0);

        if (dynamicIndexCount_ > 0) {
            const VkBuffer dvb = dynamicVB_.handle();
            vkCmdBindVertexBuffers(cmd, 0, 1, &dvb, &off);
            vkCmdBindIndexBuffer(cmd, dynamicIB_.handle(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, dynamicIndexCount_, 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
        rawColorImages_[vIdx].setLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    ctx_->endSingleTimeCommands(cmd);

    const std::array<std::vector<uint8_t>, 3> imgs = {
        rawColorImages_[0].downloadRGBA8(),
        rawColorImages_[1].downloadRGBA8(),
        rawColorImages_[2].downloadRGBA8()
    };
    const int outW = camWidth_ * 3;
    const int outH = camHeight_;
    const int rowStride = ((outW * 3 + 3) / 4) * 4;
    std::vector<uint8_t> bmp(static_cast<size_t>(54 + rowStride * outH), 0);

    const uint32_t fileSize = static_cast<uint32_t>(bmp.size());
    const uint32_t imgBytes = static_cast<uint32_t>(rowStride * outH);
    bmp[0] = 'B'; bmp[1] = 'M';
    std::memcpy(&bmp[2], &fileSize, 4);
    const uint32_t dataOff = 54;
    std::memcpy(&bmp[10], &dataOff, 4);
    const uint32_t hdrSz = 40;
    std::memcpy(&bmp[14], &hdrSz, 4);
    std::memcpy(&bmp[18], &outW, 4);
    std::memcpy(&bmp[22], &outH, 4);
    const uint16_t planes = 1, bpp = 24;
    std::memcpy(&bmp[26], &planes, 2);
    std::memcpy(&bmp[28], &bpp, 2);
    std::memcpy(&bmp[34], &imgBytes, 4);

    for (int y = 0; y < outH; ++y) {
        const int dstY = outH - 1 - y; // BMP bottom-up rows
        for (int side = 0; side < 3; ++side) {
            const auto& src = imgs[static_cast<size_t>(side)];
            for (int x = 0; x < camWidth_; ++x) {
                const size_t si = (static_cast<size_t>(y) * camWidth_ + x) * 4;
                const size_t di = static_cast<size_t>(54 + dstY * rowStride + (side * camWidth_ + x) * 3);
                bmp[di + 0] = src[si + 2];
                bmp[di + 1] = src[si + 1];
                bmp[di + 2] = src[si + 0];
            }
        }
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bmp.data()), static_cast<std::streamsize>(bmp.size()));
}

} // namespace cybercab
