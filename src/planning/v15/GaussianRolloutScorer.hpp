#pragma once
// FSD v15 Module 7: Box-free 4D Gaussian occupancy evaluation & multi-rollout
// comfort/jerk scoring. Zero per-frame heap allocations.
#include "config/V15PolicyConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/RoadCurve.hpp"
#include "world/WorldTypes.hpp"
#include <string>
#include <vector>

namespace cybercab::v15 {

inline constexpr float kDefaultLaneCenterLatM = -1.95f;
inline constexpr float kUnitSpeedScale = 1.0f;
inline constexpr float kUnboundedGapM = 1e4f;

struct ComfortRolloutChoice {
  float targetLatM = kDefaultLaneCenterLatM;
  float speedScale = kUnitSpeedScale;
  float gaussianCost = 0.0f;
  float comfortJerkCost = 0.0f;
  float minGaussianGapM = kUnboundedGapM;
};

/// Evaluates road-blocking 4D Gaussian occupancy P(x, t) by delegating to
/// GaussianWorldModel::queryOccupancyProb(..., /*roadBlockingOnly=*/true).
float queryGaussianFieldProb(const std::vector<V15Gaussian3D>& gaussians,
                             const Vec3& worldPt, float horizonSec);

/// Selects the optimal (lateral_nudge, speed_scale) autoregressive sub-rollout
/// minimizing 4D Gaussian occupancy risk, lateral jerk, and lane-center drift.
ComfortRolloutChoice selectComfortSubRollout(
  float baseLaneLat, float currentLat, float prevTargetLat, const EgoState& ego,
  const ClosedCatmullRomCurve3& centerline, float loopU, float lookaheadDist,
  float stopHorizonM, const std::vector<V15Gaussian3D>& gaussians,
  const V15PolicyConfig& p);

/// Evaluates 4D Gaussians along the front-bumper RSS corridor to trigger smooth
/// stops or close-range AEB on unboxed hazards.
void evaluateCorridorGaussians(const std::vector<V15Gaussian3D>& gaussians,
                               const std::vector<Vec3>& corrPts,
                               const std::vector<float>& corrArc, float rssDist,
                               const EgoState& ego, const V15PolicyConfig& p,
                               float& ioMinObsDist, float& ioDesiredSpeed,
                               bool& ioAeb, std::string& ioBehavior);

}  // namespace cybercab::v15
