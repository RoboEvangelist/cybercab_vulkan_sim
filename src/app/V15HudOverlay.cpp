#include "app/V15HudOverlay.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cybercab::v15_hud {
namespace {
constexpr float kPercentScale = 100.0f;

float finiteOrZero(float v) { return std::isfinite(v) ? v : 0.0f; }
}  // namespace

V15HudTelemetryLines formatV15HudTelemetry(
  const EgoState& ego, const PerceptionFrameOutput& perc,
  const PlannerCommand& cmd) {
  V15HudTelemetryLines out{};
  const char* gearLbl = (ego.gear < 0 || cmd.gear < 0)
                          ? "R"
                          : ((ego.gear == 0 || cmd.gear == 0) ? "N" : "D");
  const char* alertTag = (cmd.behaviorState == "SENSOR_SOILED_MRM")
                           ? " | MRM"
                           : (cmd.aebActive ? " | AEB" : "");
  const char* sirenLbl = perc.v15.sirenDetected ? "YIELD" : "CLEAR";
  const float latOffsetM = finiteOrZero(ego.currentLateralOffset);
  if (perc.gaussianFormer3D.active) {
    std::snprintf(
      out.header.data(), out.header.size(),
      "FSD v15 REALITY | GEAR:%s | LAT:%+.2fm | GAUSS:%zu | GF3D:%u%s",
      gearLbl, latOffsetM, perc.v15Gaussians.size(),
      perc.gaussianFormer3D.gaussianCount, alertTag);
  } else {
    std::snprintf(
      out.header.data(), out.header.size(),
      "FSD v15 REALITY | GEAR:%s | LAT:%+.2fm | GAUSS:%zu%s", gearLbl,
      latOffsetM, perc.v15Gaussians.size(), alertTag);
  }
  std::snprintf(
    out.calib.data(), out.calib.size(),
    "CALIB P:%+.2fd R:%+.2fd | PREF:%+.2fm | REAR:%.1fm",
    finiteOrZero(perc.v15.calibPitchDeg), finiteOrZero(perc.v15.calibRollDeg),
    finiteOrZero(perc.v15.preferredLaneOffsetM),
    finiteOrZero(perc.v15.rearClearanceM));
  std::snprintf(
    out.audio.data(), out.audio.size(),
    "SIREN:%s (%.2f) %+.0fHz | NOV:%.2f (+%.2f)", sirenLbl,
    finiteOrZero(perc.v15.sirenConfidence),
    finiteOrZero(perc.v15.sirenDopplerHz),
    finiteOrZero(perc.v15.latentNovelty),
    finiteOrZero(perc.v15.surpriseSigmaDelta));
  std::snprintf(
    out.compute.data(), out.compute.size(),
    "SKIP:%.0f%% | SPARSE:%.0f%% | KV:%.0f%% | SPD:%.2fx",
    finiteOrZero(perc.v15.staticSkipRatio) * kPercentScale,
    finiteOrZero(perc.v15.tokenSparsityRatio) * kPercentScale,
    finiteOrZero(perc.v15.kvCacheReuseRatio) * kPercentScale,
    std::max(1.0f, finiteOrZero(perc.v15.effectiveSpeedupX)));
  return out;
}

}  // namespace cybercab::v15_hud
