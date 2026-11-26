#include "app/V15HudOverlay.hpp"
#include "perception/sota/LeWmTypes.hpp"
#include <algorithm>
#include <array>
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
  const char* wmTag = "";
  if (perc.worldModelSota.active) {
    wmTag = perc.worldModelSota.fmeaFallbackTriggered
              ? " | WM:FB"
              : (perc.worldModelSota.hierarchical ? " | HJEPA" : " | LEWM");
  }
  const char* sirenLbl = perc.v15.sirenDetected ? "YIELD" : "CLEAR";
  const float latOffsetM = finiteOrZero(ego.currentLateralOffset);
  if (perc.gaussianFormer3D.active) {
    std::snprintf(
      out.header.data(), out.header.size(),
      "FSD v15 REALITY | GEAR:%s | LAT:%+.2fm | GAUSS:%zu | GF3D:%u%s%s",
      gearLbl, latOffsetM, perc.v15Gaussians.size(),
      perc.gaussianFormer3D.gaussianCount, wmTag, alertTag);
  } else {
    std::snprintf(
      out.header.data(), out.header.size(),
      "FSD v15 REALITY | GEAR:%s | LAT:%+.2fm | GAUSS:%zu%s%s", gearLbl,
      latOffsetM, perc.v15Gaussians.size(), wmTag, alertTag);
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
  if (perc.worldModelSota.active) {
    const char* fb = perc.worldModelSota.fmeaFallbackTriggered ? "[FB]" : "";
    const float surp = finiteOrZero(perc.worldModelSota.predSurprise);
    const float cpuMs = finiteOrZero(perc.worldModelSota.stageCpuMs);
    if (perc.worldModelSota.hierarchical) {
      const char* sub =
        sota::macroSubgoalName(perc.worldModelSota.macroSubgoalCode);
      std::snprintf(
        out.worldModel.data(), out.worldModel.size(),
        "WM:HJEPA%s %s | SURP:%.2f | %.2fms",
        fb, sub, surp, cpuMs);
    } else {
      const float lat = finiteOrZero(perc.worldModelSota.recommendedLatM);
      const float spdPct =
        finiteOrZero(perc.worldModelSota.recommendedSpeedScale) * kPercentScale;
      std::snprintf(
        out.worldModel.data(), out.worldModel.size(),
        "WM:LEWM%s LAT:%+.2fm SPD:%.0f%% | SURP:%.2f | %.2fms",
        fb, lat, spdPct, surp, cpuMs);
    }
  }
  return out;
}

WorldModelHudLines formatWorldModelHudLines(
  const PerceptionFrameOutput& perc) {
  WorldModelHudLines lines{};
  const auto& gf = perc.gaussianFormer3D;
  const auto& wm = perc.worldModelSota;
  std::array<char, 32> repBuf{};
  if (gf.active) {
    std::snprintf(repBuf.data(), repBuf.size(), "GF3D(%u)", gf.gaussianCount);
  } else {
    std::snprintf(repBuf.data(), repBuf.size(), "BEV2D");
  }
  const char* wmMode = !wm.active ? "LEGACY"
    : (wm.hierarchical ? "HJEPA" : "LEWM");
  const char* fb = (wm.active && wm.fmeaFallbackTriggered) ? "[FB]" : "";
  std::array<char, 24> goalBuf{};
  if (wm.active && wm.hierarchical) {
    std::snprintf(
      goalBuf.data(), goalBuf.size(), " (%s)",
      sota::macroSubgoalName(wm.macroSubgoalCode));
  }
  std::snprintf(
    lines.line1.data(), lines.line1.size(),
    "REP:%s | WM:%s%s%s",
    repBuf.data(), wmMode, fb, goalBuf.data());
  if (wm.active) {
    std::snprintf(
      lines.line2.data(), lines.line2.size(),
      "SURP:%.2f SR:%.2f LAT:%+.2fm SPD:%.0f%% %.2fms",
      finiteOrZero(wm.predSurprise),
      finiteOrZero(wm.sigregLoss),
      finiteOrZero(wm.recommendedLatM),
      finiteOrZero(wm.recommendedSpeedScale) * kPercentScale,
      finiteOrZero(wm.stageCpuMs));
  } else {
    std::snprintf(
      lines.line2.data(), lines.line2.size(),
      "WM LEGACY: PREDICTIVE ROLLOUT OFF");
  }
  return lines;
}

}  // namespace cybercab::v15_hud
