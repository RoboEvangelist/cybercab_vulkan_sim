#pragma once

#include "perception/PerceptionTypes.hpp"
#include "world/WorldTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cybercab::gaussian_overlay {

inline constexpr uint32_t kEllipsoidRings = 3u;
inline constexpr uint32_t kEllipsoidSlices = 8u;
inline constexpr uint32_t kVertsPerEllipsoid = 26u;
inline constexpr uint32_t kIndicesPerEllipsoid = 144u;
inline constexpr float kDefaultMinScaleM = 0.10f;
inline constexpr float kDefaultMaxScaleM = 6.0f;
inline constexpr float kDefaultMinOpacity = 0.05f;

struct GaussianOverlayConfig {
  float sigmaVisualScale = 1.0f;
  float minOpacity = kDefaultMinOpacity;
  float alphaScale = 0.72f;
  float minAlpha = 0.18f;
  float maxAlpha = 0.88f;
  float emissiveStatic = 0.16f;
  float emissiveDynamic = 0.34f;
  uint32_t maxGaussians = 512u;
};

struct GaussianOverlayStats {
  uint32_t inputCount = 0u;
  uint32_t renderedCount = 0u;
  uint32_t skippedInvalid = 0u;
  uint32_t truncatedByBudget = 0u;
  uint32_t verticesAdded = 0u;
  uint32_t indicesAdded = 0u;
};

/// Toggles Gaussian ellipsoid visualization and formats user toast feedback.
std::string toggleGaussianOverlay(bool& showGaussians,
                                  size_t activeGaussianCount);

/// Appends smooth-shaded CCW ellipsoid triangles for valid 3D Gaussians within
/// vertex, index, and primitive budgets.
GaussianOverlayStats appendGaussianEllipsoids(
  const std::vector<V15Gaussian3D>& gaussians,
  std::vector<SceneVertex>& outVerts,
  std::vector<uint32_t>& outIdx,
  size_t maxVerts,
  size_t maxIndices,
  const GaussianOverlayConfig& cfg = {});

}  // namespace cybercab::gaussian_overlay
