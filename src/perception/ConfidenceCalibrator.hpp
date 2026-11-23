#pragma once
// Per-condition logit-temperature confidence calibration for FSD v15.
// Monotonic transform with zero heap allocations on hot perception path.

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace cybercab::calib {

struct CalibParams {
  float temperature = 1.0f;
  float bias = 0.0f;
};

struct CalibSample {
  float rawScore = 0.5f;
  bool correct = false;
};

inline constexpr float kMinProbEps = 1e-4f;
inline constexpr double kNllEps = 1e-6;
inline constexpr double kInitBestNll = 1e18;
inline constexpr float kMinTemp = 0.25f;
inline constexpr float kMinCalibOut = 0.01f;
inline constexpr float kMaxCalibOut = 0.99f;
inline constexpr float kFitTempMin = 0.5f;
inline constexpr float kFitTempMax = 4.5f;
inline constexpr float kCoarseTempStep = 0.125f;
inline constexpr int kCoarseTempSteps = 32;
inline constexpr float kFitBiasMin = -1.5f;
inline constexpr float kFitBiasMax = 1.5f;
inline constexpr float kCoarseBiasStep = 0.10f;
inline constexpr int kCoarseBiasSteps = 30;
inline constexpr int kFineHalfSteps = 10;
inline constexpr float kFineTempStep = 0.015f;
inline constexpr float kFineBiasStep = 0.012f;
inline constexpr CalibParams kNightCalib{4.20f, 0.05f};
inline constexpr CalibParams kDayCalib{1.65f, 0.18f};
inline constexpr CalibParams kSoiledCalib{2.20f, -0.08f};
inline constexpr float kSoilBlendStart = 0.15f;
inline constexpr float kSoilBlendSpan = 0.60f;
inline constexpr int kMinCorrobCams = 2;
inline constexpr int kMaxCorrobBonus = 2;
inline constexpr float kCorrobBiasStep = 0.08f;
inline constexpr int kNumCams = 8;

inline float logit(float p) {
  const float pc = clampf(p, kMinProbEps, 1.0f - kMinProbEps);
  return std::log(pc / (1.0f - pc));
}

inline float sigmoid(float z) {
  if (z >= 0.0f) {
    const float ez = std::exp(-z);
    return 1.0f / (1.0f + ez);
  }
  const float ez = std::exp(z);
  return ez / (1.0f + ez);
}

inline float calibrateScore(float rawScore, const CalibParams& p) {
  const float t = std::max(kMinTemp, p.temperature);
  const float z = logit(rawScore) / t + p.bias;
  return clampf(sigmoid(z), kMinCalibOut, kMaxCalibOut);
}

inline double evalNll(const std::vector<CalibSample>& s,
                      const CalibParams& p) {
  if (s.empty()) {
    return 0.0;
  }
  double nll = 0.0;
  for (const auto& it : s) {
    const double q = static_cast<double>(calibrateScore(it.rawScore, p));
    const double qc = std::max(kNllEps, std::min(1.0 - kNllEps, q));
    nll -= it.correct ? std::log(qc) : std::log(1.0 - qc);
  }
  return nll / static_cast<double>(s.size());
}

inline CalibParams searchGridNll(const std::vector<CalibSample>& samples,
                                 float t0, float tStep, int tMinI, int tMaxI,
                                 float b0, float bStep, int bMinI, int bMaxI,
                                 double& bestNll, CalibParams best) {
  for (int it = tMinI; it <= tMaxI; ++it) {
    const float t =
      clampf(t0 + tStep * static_cast<float>(it), kFitTempMin, kFitTempMax);
    for (int ib = bMinI; ib <= bMaxI; ++ib) {
      const float b =
        clampf(b0 + bStep * static_cast<float>(ib), kFitBiasMin, kFitBiasMax);
      const CalibParams cand{t, b};
      const double n = evalNll(samples, cand);
      if (n < bestNll) {
        bestNll = n;
        best = cand;
      }
    }
  }
  return best;
}

inline CalibParams fitTemperatureNll(const std::vector<CalibSample>& s) {
  if (s.empty()) {
    return {};
  }
  double bestNll = kInitBestNll;
  const CalibParams coarse =
    searchGridNll(s, kFitTempMin, kCoarseTempStep, 0, kCoarseTempSteps,
                  kFitBiasMin, kCoarseBiasStep, 0, kCoarseBiasSteps, bestNll,
                  CalibParams{});
  return searchGridNll(s, coarse.temperature, kFineTempStep, -kFineHalfSteps,
                       kFineHalfSteps, coarse.bias, kFineBiasStep,
                       -kFineHalfSteps, kFineHalfSteps, bestNll, coarse);
}

inline CalibParams paramsForCondition(bool isNightGlare, float camSoilProb,
                                      int corrobCamCount) {
  CalibParams p = isNightGlare ? kNightCalib : kDayCalib;
  const float sw =
    clampf((camSoilProb - kSoilBlendStart) / kSoilBlendSpan, 0.0f, 1.0f);
  p.temperature = lerpf(p.temperature, kSoiledCalib.temperature, sw);
  p.bias = lerpf(p.bias, kSoiledCalib.bias, sw);
  if (corrobCamCount >= kMinCorrobCams) {
    const int bonusCnt = std::min(kMaxCorrobBonus, corrobCamCount - 1);
    p.bias += kCorrobBiasStep * static_cast<float>(bonusCnt);
  }
  return p;
}

inline float trackMinSoilProb(
    const Track3D& tr,
    const std::array<SoilCameraSummary, kNumCams>& soilCams) {
  if (tr.corroboratingCamMask == 0u) {
    return soilCams[0].meanSoilProb;
  }
  float minSp = 1.0f;
  bool any = false;
  for (int c = 0; c < kNumCams; ++c) {
    const uint8_t bit = static_cast<uint8_t>(1u << static_cast<unsigned>(c));
    if ((tr.corroboratingCamMask & bit) != 0u) {
      minSp = std::min(minSp, soilCams[static_cast<size_t>(c)].meanSoilProb);
      any = true;
    }
  }
  return any ? minSp : soilCams[0].meanSoilProb;
}

inline void calibrateFrame(const SimConfig& cfg, PerceptionFrameOutput& out) {
  if (!cfg.isV15()) {
    return;
  }
  const bool night = cfg.isNightGlare();
  for (auto& d : out.detections3D) {
    const int ci = std::clamp(d.camIndex, 0, kNumCams - 1);
    const float sp = out.soilCameras[static_cast<size_t>(ci)].meanSoilProb;
    d.score = calibrateScore(
      d.score, paramsForCondition(night, sp, d.corroboratingCamCount));
  }
  for (auto& tr : out.tracks) {
    const float sp = trackMinSoilProb(tr, out.soilCameras);
    tr.score = calibrateScore(
      tr.score, paramsForCondition(night, sp, tr.corroboratingCamCount));
  }
}

}  // namespace cybercab::calib
