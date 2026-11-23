#pragma once
// Shared multi-camera corroboration helpers for v14 and v15 perception.
#include "perception/PerceptionTypes.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cybercab::multicam {

inline constexpr int kMaxCamIdx = 7;
inline constexpr int kMinCorrobCams = 1;
inline constexpr float kCorrobScoreBoost = 0.04f;
inline constexpr float kMaxCorrobScore = 0.99f;
inline constexpr float kCorrobSigmaFactor = 0.92f;
inline constexpr float kMinCorrobSigmaFloor = 0.80f;
inline constexpr float kMinCorrobCamWeight = 0.45f;

inline uint8_t camBit(int camIdx) {
  if (camIdx < 0 || camIdx > kMaxCamIdx) return 0u;
  return static_cast<uint8_t>(1u << static_cast<unsigned>(camIdx));
}

inline int popcountMask(uint8_t mask) {
  int cnt = 0;
  for (uint8_t m = mask; m != 0u; m &= static_cast<uint8_t>(m - 1u)) ++cnt;
  return cnt;
}

inline uint8_t resolveCamMask(uint8_t mask, int camIdx) {
  return (mask != 0u) ? mask : camBit(camIdx);
}

inline float corroborationNoiseScale(int corrobCount) {
  const int extra = std::max(0, corrobCount - kMinCorrobCams);
  return std::max(
    kMinCorrobSigmaFloor,
    std::pow(kCorrobSigmaFactor, static_cast<float>(extra)));
}

inline void applyDetectionCorroboration(Detection3D& keep,
                                        const Detection3D& cand) {
  if (cand.semantic != keep.semantic || cand.camWeight < kMinCorrobCamWeight) {
    return;
  }
  const uint8_t baseMask =
    resolveCamMask(keep.corroboratingCamMask, keep.camIndex);
  const uint8_t addMask =
    resolveCamMask(cand.corroboratingCamMask, cand.camIndex);
  const uint8_t merged = static_cast<uint8_t>(baseMask | addMask);
  const int newCount = std::max(kMinCorrobCams, popcountMask(merged));
  keep.corroboratingCamMask = merged;
  if (newCount > keep.corroboratingCamCount) {
    keep.corroboratingCamCount = newCount;
    keep.score =
      std::min(kMaxCorrobScore, keep.score + kCorrobScoreBoost * cand.score);
    keep.epistemicSigma = std::max(
      keep.epistemicSigma * kMinCorrobSigmaFloor,
      keep.epistemicSigma * kCorrobSigmaFactor);
  }
}

inline void applyTrackCorroboration(Track3D& track, const Detection3D& det) {
  if (det.semantic != track.semantic || det.camWeight < kMinCorrobCamWeight) {
    return;
  }
  const uint8_t merged = static_cast<uint8_t>(
    track.corroboratingCamMask |
    resolveCamMask(det.corroboratingCamMask, det.camIndex));
  const int newCount = std::max(kMinCorrobCams, popcountMask(merged));
  track.corroboratingCamMask = merged;
  if (newCount > track.corroboratingCamCount) {
    track.corroboratingCamCount = newCount;
    track.score =
      std::min(kMaxCorrobScore, track.score + kCorrobScoreBoost * det.score);
  }
}

}  // namespace cybercab::multicam
