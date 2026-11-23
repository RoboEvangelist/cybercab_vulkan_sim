#pragma once
// FSD v15 live Reality View layout constants and zero-allocation HUD formatter.
#include "perception/PerceptionTypes.hpp"
#include "world/WorldTypes.hpp"
#include <array>
#include <cstddef>

namespace cybercab::v15_hud {

inline constexpr int kRealityTexW = 320;
inline constexpr int kRealityTexH = 180;
inline constexpr int kRealityRgbaChannels = 4;
inline constexpr size_t kRealityExpectedBytes =
  static_cast<size_t>(kRealityTexW) * static_cast<size_t>(kRealityTexH) *
  static_cast<size_t>(kRealityRgbaChannels);
inline constexpr size_t kHudLineBufLen = 96u;
inline constexpr int kUiCameraSetCount = 8;
inline constexpr int kRealityUiSetIdx = 8;
inline constexpr int kTotalUiSetCount = 9;
inline constexpr int kMaxViewModeIdx = 3;
inline constexpr int kTotalViewModes = 4;

struct V15HudTelemetryLines {
  std::array<char, kHudLineBufLen> header{};
  std::array<char, kHudLineBufLen> calib{};
  std::array<char, kHudLineBufLen> audio{};
  std::array<char, kHudLineBufLen> compute{};
};

V15HudTelemetryLines formatV15HudTelemetry(
  const EgoState& ego, const PerceptionFrameOutput& perc,
  const PlannerCommand& cmd);

}  // namespace cybercab::v15_hud
