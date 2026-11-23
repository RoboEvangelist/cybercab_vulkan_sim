#pragma once
// Post-v15 SOTA interface skeleton: DiffusionDrive truncated anchor-conditioned
// diffusion trajectory planner (declarations only; no definitions yet).
#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldTypes.hpp"
#include <cstdint>
#include <vector>

namespace cybercab::sota {

struct DiffusionPolicyConfig {
  static constexpr uint32_t kDefaultAnchorCount = 6u;
  static constexpr uint32_t kDefaultDiffusionSteps = 2u;
  static constexpr uint32_t kDefaultHorizonSteps = 24u;
  static constexpr float kDefaultHorizonDtSec = 0.15f;
  static constexpr float kDefaultNoiseScheduleEnd = 0.12f;
  static constexpr float kDefaultCollisionTemp = 2.0f;

  uint32_t anchorCount = kDefaultAnchorCount;
  uint32_t diffusionSteps = kDefaultDiffusionSteps;
  uint32_t horizonSteps = kDefaultHorizonSteps;
  float horizonDtSec = kDefaultHorizonDtSec;
  float noiseScheduleEnd = kDefaultNoiseScheduleEnd;
  float collisionTemp = kDefaultCollisionTemp;
};

struct TrajectoryAnchorMode {
  uint32_t modeId = 0u;
  float preferredLatM = 0.0f;
  float targetSpeedMs = 0.0f;
  float priorWeight = 0.0f;
};

struct DiffusionPlanOutput {
  int bestModeIdx = 0;
  uint32_t denoiseStepsRun = 0u;
  float scoreConfidence = 0.0f;
  std::vector<Vec3> waypoints;
  std::vector<float> candidateCosts;
};

class TruncatedDiffusionPolicy {
 public:
  TruncatedDiffusionPolicy();
  ~TruncatedDiffusionPolicy();
  TruncatedDiffusionPolicy(const TruncatedDiffusionPolicy&) = delete;
  TruncatedDiffusionPolicy& operator=(const TruncatedDiffusionPolicy&) = delete;

  void configure(const SimConfig& simCfg, const DiffusionPolicyConfig& cfg);
  void reset();
  DiffusionPlanOutput proposeTrajectories(
    const EgoState& ego, const PerceptionFrameOutput& perc,
    const std::vector<TrajectoryAnchorMode>& anchors);
  void denoiseStep(uint32_t stepIdx, const PerceptionFrameOutput& perc,
                   std::vector<Vec3>& ioWaypoints) const;
  int selectBestMode(const DiffusionPlanOutput& candidates) const;
};

}  // namespace cybercab::sota
