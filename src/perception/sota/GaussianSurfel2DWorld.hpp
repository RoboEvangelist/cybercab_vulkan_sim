#pragma once
// Post-v15 SOTA interface skeleton: 2D Gaussian Surfels / RaDe-GS world
// surface representation (declarations only; no definitions yet).
#include "config/SimConfig.hpp"
#include "core/Math.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/v15/PatchTokenizer.hpp"
#include <cstdint>
#include <vector>

namespace cybercab::sota {

struct alignas(16) GaussianSurfel2D {
  Vec4 centerOpacity;  // wx, wy, wz, alpha
  Vec4 tangentU;       // ux, uy, uz, radiusU
  Vec4 tangentV;       // vx, vy, vz, radiusV
  Vec4 colorSem;       // r, g, b, semIdF
};

struct SurfelWorldConfig {
  static constexpr uint32_t kDefaultMaxSurfels = 2048u;
  static constexpr float kDefaultDepthSmoothWeight = 0.25f;
  static constexpr float kDefaultNormalConsWeight = 0.50f;
  static constexpr float kDefaultMinSurfelOpacity = 0.05f;
  static constexpr float kDefaultDiskFlattenRatio = 0.05f;

  uint32_t maxSurfels = kDefaultMaxSurfels;
  float depthSmoothWeight = kDefaultDepthSmoothWeight;
  float normalConsWeight = kDefaultNormalConsWeight;
  float minSurfelOpacity = kDefaultMinSurfelOpacity;
  float diskFlattenRatio = kDefaultDiskFlattenRatio;
};

class GaussianSurfel2DWorld {
 public:
  GaussianSurfel2DWorld();
  ~GaussianSurfel2DWorld();
  GaussianSurfel2DWorld(const GaussianSurfel2DWorld&) = delete;
  GaussianSurfel2DWorld& operator=(const GaussianSurfel2DWorld&) = delete;

  void init(const VulkanContext& ctx, const SurfelWorldConfig& cfg);
  void reset();
  void destroy();
  void updateFromPatchTokens(const SimConfig& simCfg, const EgoState& ego,
                             const v15::PatchTokenizer& patches,
                             PerceptionFrameOutput& ioOut);
  void recordRasterDispatch(VkCommandBuffer cmd, const Mat4& camFromWorld,
                            const CameraIntrinsics& intr) const;
  float querySurfaceOccupancy(const Vec3& worldPt, float horizonSec) const;
  const std::vector<GaussianSurfel2D>& surfels() const;
};

}  // namespace cybercab::sota
