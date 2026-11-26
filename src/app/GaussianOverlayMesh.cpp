#include "app/GaussianOverlayMesh.hpp"
#include "perception/sota/Gaussian3DTypes.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace cybercab::gaussian_overlay {
namespace {

constexpr float kPiLocal = 3.14159265358979323846f;
constexpr float kTwoPiLocal = 6.28318530717958647692f;
constexpr int kMaxSemanticId = 7;
constexpr float kMinValidRgbSum = 0.01f;
constexpr float kNormalEps = 1e-6f;

std::array<Vec3, kVertsPerEllipsoid> buildUnitSphereDirs() {
  std::array<Vec3, kVertsPerEllipsoid> dirs{};
  dirs[0] = {0.0f, 1.0f, 0.0f};
  constexpr std::array<float, kEllipsoidRings> kRingPhis = {
    0.25f * kPiLocal, 0.0f, -0.25f * kPiLocal};
  size_t cursor = 1u;
  for (uint32_t r = 0u; r < kEllipsoidRings; ++r) {
    const float sinPhi = std::sin(kRingPhis[r]);
    const float cosPhi = std::cos(kRingPhis[r]);
    for (uint32_t s = 0u; s < kEllipsoidSlices; ++s) {
      const float theta =
        (kTwoPiLocal * static_cast<float>(s)) /
        static_cast<float>(kEllipsoidSlices);
      dirs[cursor++] = {
        cosPhi * std::cos(theta), sinPhi, cosPhi * std::sin(theta)};
    }
  }
  dirs[cursor] = {0.0f, -1.0f, 0.0f};
  return dirs;
}

std::array<uint32_t, kIndicesPerEllipsoid> buildUnitSphereIndices() {
  std::array<uint32_t, kIndicesPerEllipsoid> idx{};
  size_t cursor = 0u;
  auto pushTri = [&](uint32_t a, uint32_t b, uint32_t c) {
    idx[cursor++] = a;
    idx[cursor++] = b;
    idx[cursor++] = c;
  };
  constexpr uint32_t kTop = 0u;
  constexpr uint32_t kRing0 = 1u;
  constexpr uint32_t kRingLast = 1u + (kEllipsoidRings - 1u) * kEllipsoidSlices;
  constexpr uint32_t kBot = kVertsPerEllipsoid - 1u;
  for (uint32_t s = 0u; s < kEllipsoidSlices; ++s) {
    const uint32_t next = (s + 1u) % kEllipsoidSlices;
    pushTri(kTop, kRing0 + next, kRing0 + s);
  }
  for (uint32_t r = 0u; r + 1u < kEllipsoidRings; ++r) {
    const uint32_t uBase = 1u + r * kEllipsoidSlices;
    const uint32_t lBase = 1u + (r + 1u) * kEllipsoidSlices;
    for (uint32_t s = 0u; s < kEllipsoidSlices; ++s) {
      const uint32_t next = (s + 1u) % kEllipsoidSlices;
      pushTri(uBase + s, uBase + next, lBase + s);
      pushTri(uBase + next, lBase + next, lBase + s);
    }
  }
  for (uint32_t s = 0u; s < kEllipsoidSlices; ++s) {
    const uint32_t next = (s + 1u) % kEllipsoidSlices;
    pushTri(kBot, kRingLast + s, kRingLast + next);
  }
  return idx;
}

bool isValidGaussian(const V15Gaussian3D& g, float minOpacity) {
  if (!std::isfinite(g.meanOpacity.x) || !std::isfinite(g.meanOpacity.y) ||
      !std::isfinite(g.meanOpacity.z) || !std::isfinite(g.meanOpacity.w)) {
    return false;
  }
  if (!std::isfinite(g.scaleSem.x) || !std::isfinite(g.scaleSem.y) ||
      !std::isfinite(g.scaleSem.z)) {
    return false;
  }
  if (g.meanOpacity.w < minOpacity) return false;
  return g.scaleSem.x > 0.0f && g.scaleSem.y > 0.0f && g.scaleSem.z > 0.0f;
}

Vec3 resolveColor(const V15Gaussian3D& g) {
  if (std::isfinite(g.colorConf.x) && std::isfinite(g.colorConf.y) &&
      std::isfinite(g.colorConf.z) &&
      g.colorConf.x >= 0.0f && g.colorConf.y >= 0.0f && g.colorConf.z >= 0.0f &&
      (g.colorConf.x + g.colorConf.y + g.colorConf.z) >= kMinValidRgbSum) {
    return {clampf(g.colorConf.x, 0.0f, 1.0f),
            clampf(g.colorConf.y, 0.0f, 1.0f),
            clampf(g.colorConf.z, 0.0f, 1.0f)};
  }
  const int rawSem = std::isfinite(g.scaleSem.w)
    ? static_cast<int>(std::lround(g.scaleSem.w))
    : static_cast<int>(SemanticClass::Static);
  const int semIdx = std::clamp(rawSem, 0, kMaxSemanticId);
  return sota::semanticDefaultColor(static_cast<SemanticClass>(semIdx));
}

}  // namespace

std::string toggleGaussianOverlay(bool& showGaussians,
                                  size_t activeGaussianCount) {
  showGaussians = !showGaussians;
  if (!showGaussians) return "3D Gaussians OFF";
  if (activeGaussianCount == 0u) {
    return "3D Gaussians ON (0 active: use v15 or --spatial-rep gaussianformer3d)";
  }
  return "3D Gaussians ON (" + std::to_string(activeGaussianCount) +
         " ellipsoids)";
}

GaussianOverlayStats appendGaussianEllipsoids(
  const std::vector<V15Gaussian3D>& gaussians,
  std::vector<SceneVertex>& outVerts,
  std::vector<uint32_t>& outIdx,
  size_t maxVerts,
  size_t maxIndices,
  const GaussianOverlayConfig& cfg) {
  static const std::array<Vec3, kVertsPerEllipsoid> kUnitDirs =
    buildUnitSphereDirs();
  static const std::array<uint32_t, kIndicesPerEllipsoid> kUnitIndices =
    buildUnitSphereIndices();

  GaussianOverlayStats stats{};
  stats.inputCount = static_cast<uint32_t>(gaussians.size());
  const float visScale =
    (std::isfinite(cfg.sigmaVisualScale) && cfg.sigmaVisualScale > 0.0f)
    ? cfg.sigmaVisualScale
    : 1.0f;

  for (const auto& g : gaussians) {
    if (!isValidGaussian(g, cfg.minOpacity)) {
      ++stats.skippedInvalid;
      continue;
    }
    if (stats.renderedCount >= cfg.maxGaussians ||
        outVerts.size() + kVertsPerEllipsoid > maxVerts ||
        outIdx.size() + kIndicesPerEllipsoid > maxIndices) {
      ++stats.truncatedByBudget;
      continue;
    }

    const float sx = clampf(g.scaleSem.x * visScale,
                            kDefaultMinScaleM, kDefaultMaxScaleM);
    const float sy = clampf(g.scaleSem.y * visScale,
                            kDefaultMinScaleM, kDefaultMaxScaleM);
    const float sz = clampf(g.scaleSem.z * visScale,
                            kDefaultMinScaleM, kDefaultMaxScaleM);
    const float yaw = std::isfinite(g.velYaw.z) ? g.velYaw.z : 0.0f;
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    const float alpha = clampf(g.meanOpacity.w * cfg.alphaScale,
                               cfg.minAlpha, cfg.maxAlpha);
    const Vec3 col = resolveColor(g);
    const bool isDyn = std::isfinite(g.velYaw.w) && g.velYaw.w > 0.5f;
    const float emissive = isDyn ? cfg.emissiveDynamic : cfg.emissiveStatic;
    const float semId = std::isfinite(g.scaleSem.w) ? g.scaleSem.w : 0.0f;

    const uint32_t baseVert = static_cast<uint32_t>(outVerts.size());
    const float invSx = 1.0f / sx;
    const float invSy = 1.0f / sy;
    const float invSz = 1.0f / sz;
    for (const Vec3& u : kUnitDirs) {
      const float lx = u.x * sx;
      const float ly = u.y * sy;
      const float lz = u.z * sz;
      const float nlx = u.x * invSx;
      const float nly = u.y * invSy;
      const float nlz = u.z * invSz;
      const float nSq = nlx * nlx + nly * nly + nlz * nlz;
      const float invN = (nSq > kNormalEps) ? (1.0f / std::sqrt(nSq)) : 1.0f;
      outVerts.push_back({
        g.meanOpacity.x + lx * c + lz * s,
        g.meanOpacity.y + ly,
        g.meanOpacity.z - lx * s + lz * c,
        (nlx * c + nlz * s) * invN,
        nly * invN,
        (-nlx * s + nlz * c) * invN,
        col.x, col.y, col.z, alpha, emissive, semId, 0.0f, 0.0f});
    }
    for (const uint32_t localIdx : kUnitIndices) {
      outIdx.push_back(baseVert + localIdx);
    }
    ++stats.renderedCount;
    stats.verticesAdded += kVertsPerEllipsoid;
    stats.indicesAdded += kIndicesPerEllipsoid;
  }
  return stats;
}

}  // namespace cybercab::gaussian_overlay
