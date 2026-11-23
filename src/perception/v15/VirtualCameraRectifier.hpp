#pragma once
// FSD v15 Module 2b: Canonical virtual pinhole rectifier & dual-branch (VRU + Elevated) views.
// Reuses CameraSpec intrinsics/extrinsics + brownConradyDistort to warp 8 physical cameras into
// 10 distortion-free 160x120 views compensated by online pitch/roll calibration R_calib.
#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace cybercab::v15 {

struct VirtualViewSpec {
    int srcCamIdx = 0;
    std::string name;
    CameraIntrinsics intr{};
    Vec3 posIso{};
    Vec3 rpyDeg{};
    Mat4 uprightVehFromVirtCam = Mat4::identity();
    Mat4 physCamFromVirtCam = Mat4::identity();
};

struct alignas(16) VirtViewSpecGpu {
    Mat4 physCamFromVirtCam;
    Vec4 virtIntr;
    Vec4 physIntr;
    Vec4 physDist;
    uint32_t srcCamIdx = 0;
    uint32_t srcW = 0;
    uint32_t srcH = 0;
    uint32_t renderOverscanBits = 0;
};

class VirtualCameraRectifier {
public:
    static constexpr int kVirtW = 160; // divisible by 5 for M3 5x5 patch tokenizer
    static constexpr int kVirtH = 120;
    static constexpr int kNumViews = 10;
    static constexpr int kViewVru = 8;
    static constexpr int kViewElevated = 9;

    VirtualCameraRectifier() = default;
    ~VirtualCameraRectifier();
    VirtualCameraRectifier(const VirtualCameraRectifier&) = delete;
    VirtualCameraRectifier& operator=(const VirtualCameraRectifier&) = delete;

    void init(const VulkanContext& ctx, const SimConfig& cfg, const VulkanBuffer& linearIn,
              const VulkanBuffer& bandsIn, int srcW, int srcH);
    void rebind(const SimConfig& cfg, const VulkanBuffer& linearIn, const VulkanBuffer& bandsIn,
                int srcW, int srcH);
    void updateViews(const SimConfig& cfg, int srcW, int srcH, const Mat4& vehicleCalibMatrix);
    void recordDispatch(VkCommandBuffer cmd) const;
    void destroy();

    // Pure CPU helper building canonical + dual-branch view specs for testing & GPU upload.
    static std::array<VirtualViewSpec, kNumViews> buildViewSpecs(
        const SimConfig& cfg, int srcW, int srcH, const Mat4& vehicleCalibMatrix,
        std::array<VirtViewSpecGpu, kNumViews>* outGpu = nullptr);

    const VulkanBuffer& virtLinearSSBO() const { return virtLinear_; }
    const VulkanBuffer& virtBandsSSBO() const { return virtBands_; }
    const VulkanBuffer& specsSSBO() const { return specsSSBO_; }
    const VirtualViewSpec& viewSpec(int v) const { return views_.at(v); }
    const VirtViewSpecGpu& gpuSpec(int v) const { return gpuSpecs_.at(v); }

private:
    struct RectPC { uint32_t virtW, virtH, numViews, pad; };

    const VulkanContext* ctx_ = nullptr;
    VulkanComputePipeline pipe_;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VulkanBuffer specsSSBO_, virtLinear_, virtBands_;
    std::array<VirtualViewSpec, kNumViews> views_{};
    std::array<VirtViewSpecGpu, kNumViews> gpuSpecs_{};
};

} // namespace cybercab::v15
