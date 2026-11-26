#pragma once
// Bounded Stage-2/3 LeWM & H-JEPA planner guidance (always downstream of hard safety shields).
#include "config/SimConfig.hpp"
#include "core/Math.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/sota/LeWmTypes.hpp"
#include <algorithm>
#include <cmath>

namespace cybercab::sota {

inline float candidateGuidanceCost(const SimConfig& cfg,
                                   const PerceptionFrameOutput& perc,
                                   float candTargetLatM,
                                   int candLaneIdx,
                                   int curLaneIdx,
                                   bool sirenYield) {
  const auto& wm = perc.worldModelSota;
  if (!cfg.usesSotaWorldModel() || !wm.active || wm.fmeaFallbackTriggered || sirenYield) {
    return 0.0f;
  }
  if (!std::isfinite(wm.recommendedLatM) || !std::isfinite(candTargetLatM)) {
    return 0.0f;
  }
  float gCost = std::min(48.0f, std::abs(candTargetLatM - wm.recommendedLatM) * 14.0f);
  if (wm.hierarchical) {
    const auto subgoal = static_cast<MacroSubgoal>(wm.macroSubgoalCode);
    if (curLaneIdx == 0 && candLaneIdx == 1 && subgoal != MacroSubgoal::OvertakeLeft) {
      gCost += 34.0f;
    } else if (curLaneIdx == 0 && candLaneIdx == 0 && subgoal == MacroSubgoal::OvertakeLeft) {
      gCost += 22.0f;
    } else if (curLaneIdx == 1 && candLaneIdx == 1 && subgoal == MacroSubgoal::ReturnRight) {
      gCost += 20.0f;
    }
  }
  return clampf(gCost, 0.0f, 82.0f);
}

inline float guidedSpeedScale(const SimConfig& cfg, const PerceptionFrameOutput& perc) {
  const auto& wm = perc.worldModelSota;
  if (!cfg.usesSotaWorldModel() || !wm.active || wm.fmeaFallbackTriggered) {
    return 1.0f;
  }
  if (!std::isfinite(wm.recommendedSpeedScale)) return 1.0f;
  return clampf(wm.recommendedSpeedScale, 0.55f, 1.0f);
}

}  // namespace cybercab::sota
