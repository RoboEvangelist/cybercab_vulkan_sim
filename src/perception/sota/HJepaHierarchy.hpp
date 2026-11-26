#pragma once
// Stage-3 H-JEPA: Multi-timescale hierarchical macro subgoal + deterministic CEM planner.
#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/sota/LeWmLatentModel.hpp"
#include "perception/sota/LeWmTypes.hpp"
#include "world/WorldEngine.hpp"
#include <cstdint>

namespace cybercab::sota {

struct MacroSubgoalResult {
  MacroSubgoal code = MacroSubgoal::KeepLane;
  float targetLatM = -1.95f;
  float targetSpeedMs = 10.5f;
  float macroEnergy = 0.0f;
  bool valid = true;
};

struct CemPlanResult {
  float recommendedLatM = -1.95f;
  float recommendedSpeedScale = 1.0f;
  float iter0BestCost = 0.0f;
  float finalBestCost = 0.0f;
  bool valid = true;
};

class HJepaHierarchy {
public:
  void reset();

  static MacroVec8 projectMacro(const LatentVec16& z, const LeWmWeights& w);

  static MacroSubgoalResult planMacroSubgoal(const SimConfig& cfg,
                                             const EgoState& ego,
                                             const LatentVec16& z,
                                             const LeWmLatentModel& model,
                                             bool redOrStop,
                                             bool sirenYield);

  static CemPlanResult optimizeActionCem(const SimConfig& cfg,
                                         const EgoState& ego,
                                         const LatentVec16& z,
                                         const MacroSubgoalResult& macro,
                                         const LeWmLatentModel& model,
                                         float prevRecLatM,
                                         float prevRecSpeedScale,
                                         uint32_t& faultMaskOut);

  void runHierarchy(const SimConfig& cfg, const WorldEngine& world,
                    const PerceptionFrameOutput& perc,
                    const LeWmLatentModel& model, const LatentVec16& z,
                    WorldModelSotaSummary& ioSummary);

private:
  float prevRecLatM_ = -1.95f;
  float prevRecSpeedScale_ = 1.0f;
};

}  // namespace cybercab::sota
