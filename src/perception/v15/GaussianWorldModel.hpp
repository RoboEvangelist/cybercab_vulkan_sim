#pragma once
// FSD v15 Module 6: 3D/4D Gaussian World Model, GaussianFormer-2 occupancy & EWA Reality View.
#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/v15/PatchTokenizer.hpp"
#include "perception/v15/StaticSpatialMemory.hpp"
#include <cstdint>
#include <vector>

namespace cybercab::v15 {

using Gaussian3DGpu = ::cybercab::V15Gaussian3D;

class GaussianWorldModel {
public:
    static constexpr int kMaxGaussians = 1024;
    static constexpr int kRealityW = 320;
    static constexpr int kRealityH = 180;

    GaussianWorldModel() = default;
    ~GaussianWorldModel();
    GaussianWorldModel(const GaussianWorldModel&) = delete;
    GaussianWorldModel& operator=(const GaussianWorldModel&) = delete;

    void initSplatGpu(const VulkanContext& ctx);
    void reset();
    void destroy();
    void updateFromPerception(const SimConfig& cfg, const EgoState& ego,
                              const StaticSpatialMemory& staticMem,
                              const PatchTokenizer& patches,
                              PerceptionFrameOutput& out);
    void dispatchRealitySplat(VkCommandBuffer cmd, const SimConfig& cfg, const EgoState& ego);
    void dispatchRealitySplat(VkCommandBuffer cmd, const Mat4& camFromWorld, const CameraIntrinsics& intr);
    void downloadRealityViewRGBA(std::vector<uint8_t>& outRgba) const;

    // Zwicker/Kerbl EWA Jacobian projection: Sigma2D = J W Sigma3D W^T J^T + 0.3 I_2 (xx, xy, yy).
    static void projectEwa2D(const Gaussian3DGpu& g, const Mat4& camFromWorld,
                             const CameraIntrinsics& intr, Vec2& uvOut,
                             float cov2dOut[3], float& depthOut);

  // GaussianFormer-2 P(x, t) = 1 - prod_i (1 - alpha_i f_i(x, t)).
  static float queryOccupancyProb(const std::vector<V15Gaussian3D>& gaussians,
                                  const Vec3& worldPt, float horizonSec = 0.0f,
                                  bool roadBlockingOnly = false);
  float queryOccupancyProb(const Vec3& worldPt, float horizonSec = 0.0f) const;
    void rollout4D(float horizonSec, std::vector<Gaussian3DGpu>& outPred) const;
    void setExternalGaussians(const std::vector<Gaussian3DGpu>& src);
    void setGaussiansForTest(std::vector<Gaussian3DGpu> gaussians) { gaussians_ = std::move(gaussians); }
    const std::vector<Gaussian3DGpu>& gaussians() const { return gaussians_; }

private:
    struct alignas(16) SplatPC { Mat4 camFromWorld; Vec4 intr; uint32_t width, height, numGaussians, pad; };
    const VulkanContext* ctx_ = nullptr;
    VulkanComputePipeline splatPipe_;
    VkDescriptorSet splatSet_ = VK_NULL_HANDLE;
    VulkanBuffer gaussSSBO_, rgbaSSBO_;
    std::vector<Gaussian3DGpu> gaussians_;
    mutable std::vector<Gaussian3DGpu> sortedScratch_;
    mutable std::vector<StaticCellGpu> cellScratch_;
    mutable std::vector<PatchTokenGpu> vruScratch_;
};

} // namespace cybercab::v15
