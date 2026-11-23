#pragma once
// Pure grounded on-road Static-class voxel filter shared by FSD v15
// safety shield and autoregressive rollout scoring (zero allocations).

#include "perception/PerceptionTypes.hpp"
#include "world/WorldEngine.hpp"
#include <cmath>

namespace cybercab::v15 {

inline constexpr float kStaticVoxMinYM = 0.35f;
inline constexpr float kStaticVoxMaxYM = 1.85f;
inline constexpr float kStaticVoxMinConf = 0.78f;
inline constexpr float kStaticVoxMinOccSigma = 0.10f;
inline constexpr float kStaticVoxMaxRoadLatM = 3.1f;

inline bool isGroundedStaticVoxel(const ActiveVoxel& v) {
  return v.wy >= kStaticVoxMinYM && v.wy <= kStaticVoxMaxYM &&
         v.confidence > kStaticVoxMinConf &&
         v.epistemicSigma >= kStaticVoxMinOccSigma;
}

inline float voxelRoadLateralM(const WorldEngine& world,
                               const ActiveVoxel& v) {
  const float loopU = world.findClosestLoopU({v.wx, 0.0f, v.wz});
  const auto& cl = world.centerline();
  const Vec3 tan = cl.getTangentAt(loopU);
  const Vec3 left{tan.z, 0.0f, -tan.x};
  const Vec3 rel = Vec3{v.wx, 0.0f, v.wz} - cl.getPointAt(loopU);
  return rel.dot(left);
}

inline bool isOnRoadBlockingStaticVoxel(const WorldEngine& world,
                                        const ActiveVoxel& v) {
  if (!isGroundedStaticVoxel(v)) {
    return false;
  }
  return std::abs(voxelRoadLateralM(world, v)) <= kStaticVoxMaxRoadLatM;
}

}  // namespace cybercab::v15
