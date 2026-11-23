#pragma once

#include "config/SimConfig.hpp"
#include "core/GpuProfiler.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanImage.hpp"
#include "core/VulkanPipeline.hpp"
#include "world/WorldEngine.hpp"
#include <array>
#include <string>
#include <vector>

namespace cybercab {

struct CameraMatrices {
    Mat4 worldFromCam;
    Mat4 camFromWorld;
    Mat4 proj;
    Mat4 viewProj;
    CameraIntrinsics intr;
    std::array<float, 4> distortion{};
    float renderScale = 1.0f;
};

class CameraRigRenderer {
public:
    CameraRigRenderer() = default;
    ~CameraRigRenderer();

    void init(const VulkanContext& ctx, const SimConfig& cfg, const WorldEngine& world);
    void resize(int camWidth, int camHeight);
    void destroy();

    // Renders all 8 cameras into raw attachments and executes cam_isp_soil.comp
    // writing both displayable RGBA8 images and normalized GPU RGB tensor SSBO.
    void renderAllCameras(const WorldEngine& world, const SimConfig& cfg,
                          float timeSec, GpuProfiler* profiler = nullptr);

    // Projects a 3D world point into camera pixel coordinates (returns false if behind camera).
    bool projectWorldToPixel(int camIdx, const Vec3& worldPt, Vec2& outPx, float& outDepth) const;

    // Readback helper for tests & headless snapshot dumps
    std::vector<uint8_t> downloadCameraRGBA(int camIdx) const;
    void dumpCameraGridPPM(const std::string& path) const;
    void dumpEgoShowcaseBMP(const WorldEngine& world, const std::string& path);

    const CameraMatrices& cameraMatrices(int camIdx) const { return camMats_.at(camIdx); }
    const VulkanImage& distortedImage(int camIdx) const { return distortedImages_.at(camIdx); }
    const VulkanBuffer& cameraTensorSSBO() const { return camTensorSSBO_; }
    // v15 ISP-bypass tensor: 8 * H * W vec4, rgb = 14-bit linear code / 16383 (x kFullWellRadiance = radiance).
    // Allocated only when cfg.isV15(); otherwise a 16-byte placeholder.
    const VulkanBuffer& cameraLinearSSBO() const { return camLinearSSBO_; }
    bool linearHdrEnabled() const { return hdr_; }
    const VulkanBuffer& staticVertexBuffer() const { return staticVB_; }
    const VulkanBuffer& staticIndexBuffer() const { return staticIB_; }
    uint32_t staticIndexCount() const { return staticIndexCount_; }
    const VulkanBuffer& dynamicVertexBuffer() const { return dynamicVB_; }
    const VulkanBuffer& dynamicIndexBuffer() const { return dynamicIB_; }
    uint32_t dynamicIndexCount() const { return dynamicIndexCount_; }
    int camWidth() const { return camWidth_; }
    int camHeight() const { return camHeight_; }

private:
    const VulkanContext* ctx_ = nullptr;
  int camWidth_ = kLegacyHalfCamW;
  int camHeight_ = kLegacyHalfCamH;
    static constexpr int kNumCams = 8;

    VkRenderPass offscreenPass_ = VK_NULL_HANDLE;
    VkPipelineLayout gfxLayout_ = VK_NULL_HANDLE;
    VkPipeline gfxPipeline_ = VK_NULL_HANDLE;

    std::array<VulkanImage, kNumCams> rawColorImages_;
    std::array<VulkanImage, kNumCams> rawDepthImages_;
    std::array<VulkanImage, kNumCams> distortedImages_;
    std::array<VkFramebuffer, kNumCams> framebuffers_{};
    std::array<CameraMatrices, kNumCams> camMats_{};

    VulkanComputePipeline ispPipeline_;
    std::array<VkDescriptorSet, kNumCams> ispSets_{};
    VulkanBuffer camTensorSSBO_;
    VulkanBuffer camLinearSSBO_;
    bool hdr_ = false; // v15: float16 scene-referred raw render + linear photon tensor

    VkFormat rawColorFormat() const { return hdr_ ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM; }

    VulkanBuffer staticVB_;
    VulkanBuffer staticIB_;
    uint32_t staticIndexCount_ = 0;

    VulkanBuffer dynamicVB_;
    VulkanBuffer dynamicIB_;
    uint32_t dynamicIndexCount_ = 0;
    VkDeviceSize dynamicVBCapacity_ = 0;
    VkDeviceSize dynamicIBCapacity_ = 0;
    std::vector<SceneVertex> dynVertsScratch_;
    std::vector<uint32_t> dynIndicesScratch_;

    void createRenderPassAndGraphicsPipeline();
    void createCameraTargets(int w, int h);
    void destroyCameraTargets();
    void updateIspDescriptorSets();
    void uploadStaticWorld(const WorldEngine& world);
    void uploadDynamicWorld(const WorldEngine& world, bool includeEgo = false);
};

} // namespace cybercab
