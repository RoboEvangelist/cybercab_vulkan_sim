#include "app/V15HudOverlay.hpp"
#include <algorithm>
#include <cstdio>

namespace cybercab::v15_hud {
namespace {
constexpr float kPercentScale = 100.0f;
}  // namespace

V15HudTelemetryLines formatV15HudTelemetry(
  const EgoState& ego, const PerceptionFrameOutput& perc,
  const PlannerCommand& cmd) {
  V15HudTelemetryLines out{};
  const char* gearLbl = (ego.gear < 0 || cmd.gear < 0) ? "R" : "D";
  const char* sirenLbl = perc.v15.sirenDetected ? "YIELD" : "CLEAR";
  if (perc.gaussianFormer3D.active) {
    std::snprintf(
      out.header.data(), out.header.size(),
      "FSD v15 REALITY | GEAR:%s | LAT:%+.2fm | GAUSS:%zu | GF3D:%u",
      gearLbl, ego.currentLateralOffset, perc.v15Gaussians.size(),
      perc.gaussianFormer3D.gaussianCount);
  } else {
    std::snprintf(
      out.header.data(), out.header.size(),
      "FSD v15 REALITY | GEAR:%s | LAT:%+.2fm | GAUSS:%zu", gearLbl,
      ego.currentLateralOffset, perc.v15Gaussians.size());
  }
  std::snprintf(
    out.calib.data(), out.calib.size(),
    "CALIB P:%+.2fd R:%+.2fd | PREF:%+.2fm | REAR:%.1fm",
    perc.v15.calibPitchDeg, perc.v15.calibRollDeg,
    perc.v15.preferredLaneOffsetM, perc.v15.rearClearanceM);
  std::snprintf(
    out.audio.data(), out.audio.size(),
    "SIREN:%s (%.2f) %+.0fHz | NOV:%.2f (+%.2f)", sirenLbl,
    perc.v15.sirenConfidence, perc.v15.sirenDopplerHz, perc.v15.latentNovelty,
    perc.v15.surpriseSigmaDelta);
  std::snprintf(
    out.compute.data(), out.compute.size(),
    "SKIP:%.0f%% | SPARSE:%.0f%% | KV:%.0f%% | SPD:%.2fx",
    perc.v15.staticSkipRatio * kPercentScale,
    perc.v15.tokenSparsityRatio * kPercentScale,
    perc.v15.kvCacheReuseRatio * kPercentScale,
    std::max(1.0f, perc.v15.effectiveSpeedupX));
  return out;
}

}  // namespace cybercab::v15_hud
