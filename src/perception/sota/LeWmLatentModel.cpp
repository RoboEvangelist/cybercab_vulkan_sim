#include "perception/sota/LeWmLatentModel.hpp"
#include "perception/sota/LeWmSceneEncoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace cybercab::sota {
namespace {

constexpr float kDefaultLatM = -1.95f;
constexpr float kLeftLatM = 1.95f;
constexpr float kRefDtSec = 0.25f;
constexpr float kStateClip = 2.0f;
constexpr float kVarEps = 1e-5f;

float sq(float v) { return v * v; }

}  // namespace

void LeWmLatentModel::init(const SimConfig& cfg) {
  reset();
  initFaultMask_ = kWmFaultNone;
  weightsLoadedFromFile_ = false;
  if (!validateWorldModelSotaParams(cfg.wmSota)) {
    initFaultMask_ |= kWmFaultInvalidConfig;
  }
  if (!cfg.wmSota.weightsPath.empty()) {
    if (LeWmWeights::loadBinary(cfg.wmSota.weightsPath, weights_)) {
      weightsLoadedFromFile_ = true;
    } else {
      initFaultMask_ |= kWmFaultWeightLoad;
      weights_ = LeWmWeights::defaultAnalytical();
    }
  } else {
    weights_ = LeWmWeights::defaultAnalytical();
  }
}

void LeWmLatentModel::reset() {
  latentRing_ = {};
  rawRing_ = {};
  ringHead_ = 0u;
  ringCount_ = 0u;
  hasPrevPred_ = false;
  prevPredZ_ = {};
  lastLatent_ = {};
  prevRecLatM_ = kDefaultLatM;
  prevRecSpeedScale_ = 1.0f;
}

float LeWmLatentModel::latentNormSq(const LatentVec16& v) {
  float s = 0.0f;
  for (float x : v) s += x * x;
  return s;
}

bool LeWmLatentModel::encodeScene(const SimConfig& cfg, const WorldEngine& world,
                                  const PerceptionFrameOutput& perc,
                                  LatentVec16& zOut, uint32_t& faultMaskOut,
                                  LatentVec16* rawFeaturesOut) const {
  return encodeSceneLatent16(
    cfg, world, perc, weights_, zOut, faultMaskOut, rawFeaturesOut);
}

SigRegMetrics LeWmLatentModel::updateSigReg(const LatentVec16& z, float targetVar) {
  latentRing_[ringHead_] = z;
  ringHead_ = (ringHead_ + 1u) % kLeWmSigRegWindow;
  if (ringCount_ < kLeWmSigRegWindow) ++ringCount_;

  SigRegMetrics m{};
  m.meanSqNorm = latentNormSq(z) / static_cast<float>(kLeWmLatentDim);
  if (!std::isfinite(m.meanSqNorm)) {
    m.collapsed = true;
    return m;
  }
  if (ringCount_ < 2u) {
    m.varDeviation = targetVar * static_cast<float>(kLeWmLatentDim);
    m.sigregLoss = m.varDeviation;
    return m;
  }
  LatentVec16 mean{};
  const float invN = 1.0f / static_cast<float>(ringCount_);
  for (uint32_t k = 0u; k < ringCount_; ++k) {
    for (size_t i = 0; i < kLeWmLatentDim; ++i) mean[i] += latentRing_[k][i] * invN;
  }
  float maxVar = 0.0f, offDiag = 0.0f;
  for (size_t i = 0; i < kLeWmLatentDim; ++i) {
    float v = 0.0f;
    for (uint32_t k = 0u; k < ringCount_; ++k) v += sq(latentRing_[k][i] - mean[i]);
    v *= invN;
    maxVar = std::max(maxVar, v);
    m.varDeviation += sq(v - targetVar);
    for (size_t j = i + 1u; j < 6u; ++j) {
      float cov = 0.0f;
      for (uint32_t k = 0u; k < ringCount_; ++k) {
        cov += (latentRing_[k][i] - mean[i]) * (latentRing_[k][j] - mean[j]);
      }
      offDiag += sq(cov * invN);
    }
  }
  m.offDiagCovNorm = std::sqrt(offDiag);
  m.sigregLoss = m.varDeviation * 0.0625f + m.offDiagCovNorm;
  m.collapsed = (ringCount_ >= 8u && maxVar < kVarEps);
  return m;
}

bool LeWmLatentModel::predictNext(const LatentVec16& z, const LeWmAction& act,
                                  float dtSec, LatentVec16& zNextOut) const {
  if (!std::isfinite(act.accelMs2) || !std::isfinite(act.latVelMs) ||
      !std::isfinite(dtSec) || dtSec <= 0.0f || !weights_.isFiniteValid()) {
    return false;
  }
  const float hScale = clampf(dtSec / kRefDtSec, 0.05f, 3.0f);
  for (size_t i = 0; i < kLeWmLatentDim; ++i) {
    if (!std::isfinite(z[i])) return false;
    float delta = 0.0f;
    for (size_t j = 0; j < kLeWmLatentDim; ++j) {
      const float aEff = (i == j) ? (weights_.dynA[i][j] - 1.0f) : weights_.dynA[i][j];
      delta += aEff * z[j];
    }
    delta += weights_.dynB[i][0] * act.accelMs2 + weights_.dynB[i][1] * act.latVelMs +
             weights_.dynBias[i];
    const float nextVal = clampf(z[i] + hScale * delta, -kStateClip, kStateClip);
    if (!std::isfinite(nextVal)) return false;
    zNextOut[i] = nextVal;
  }
  return std::sqrt(latentNormSq(zNextOut)) <= kLeWmMaxLatentNorm;
}

float LeWmLatentModel::stateRiskEnergy(const LatentVec16& z, const LeWmAction& act) {
  const float spd = std::max(0.0f, z[0]);
  const float leftWeight = clampf((z[1] + 1.0f) * 0.5f, 0.0f, 1.0f);
  const float rightWeight = 1.0f - leftWeight;
  const float sameRisk = (sq(std::max(0.0f, z[4])) + 0.85f * sq(std::max(0.0f, z[6]))) *
                         (0.4f + 0.8f * spd);
  const float adjRisk = (sq(std::max(0.0f, z[8])) + 1.6f * sq(std::max(0.0f, z[9])) +
                         1.2f * sq(std::max(0.0f, z[10]))) * (0.4f + 0.8f * spd);
  const float vruSig = (sq(std::max(0.0f, z[12])) + 0.9f * sq(std::max(0.0f, z[13]))) *
                       (0.3f + 0.9f * spd);
  const float closing = sq(std::max(0.0f, z[5])) * 0.65f * rightWeight;
  const float ctrl = 0.015f * sq(act.accelMs2) + 0.03f * sq(act.latVelMs);
  return rightWeight * sameRisk + leftWeight * adjRisk + vruSig + closing + ctrl;
}

bool LeWmLatentModel::rolloutEnergy(const LatentVec16& z0, const LeWmAction& act,
                                    uint32_t steps, float dtSec, float& energyOut,
                                    LatentVec16* terminalZOut) const {
  if (steps == 0u || steps > kLeWmMaxRollout) return false;
  LatentVec16 cur = z0, nxt{};
  float sumE = 0.0f;
  for (uint32_t s = 0u; s < steps; ++s) {
    if (!predictNext(cur, act, dtSec, nxt)) return false;
    const float w = 1.0f / (1.0f + 0.12f * static_cast<float>(s));
    sumE += w * stateRiskEnergy(nxt, act);
    cur = nxt;
  }
  energyOut = sumE / static_cast<float>(steps);
  if (terminalZOut) *terminalZOut = cur;
  return std::isfinite(energyOut);
}

float LeWmLatentModel::computeRawRingVariance() const {
  if (ringCount_ < 8u) return 1.0f;
  LatentVec16 mean{};
  const float invN = 1.0f / static_cast<float>(ringCount_);
  for (uint32_t k = 0u; k < ringCount_; ++k) {
    for (size_t i = 0; i < kLeWmLatentDim; ++i) mean[i] += rawRing_[k][i] * invN;
  }
  float maxV = 0.0f;
  for (size_t i = 0; i < kLeWmLatentDim; ++i) {
    float v = 0.0f;
    for (uint32_t k = 0u; k < ringCount_; ++k) v += sq(rawRing_[k][i] - mean[i]);
    maxV = std::max(maxV, v * invN);
  }
  return maxV;
}

void LeWmLatentModel::stepFrame(const SimConfig& cfg, const WorldEngine& world,
                                const PerceptionFrameOutput& perc, float dtSec,
                                WorldModelSotaSummary& outSummary) {
  const auto t0 = std::chrono::steady_clock::now();
  outSummary = {};
  outSummary.active = true;
  outSummary.hierarchical = cfg.isHJepa();
  outSummary.weightsLoadedFromFile = weightsLoadedFromFile_;
  uint32_t fault = initFaultMask_;

  LatentVec16 z{}, raw{};
  const uint32_t rawSlot = ringHead_;
  uint32_t encFault = kWmFaultNone;
  const bool encOk = encodeScene(cfg, world, perc, z, encFault, &raw);
  fault |= encFault;
  rawRing_[rawSlot] = raw;

  const SigRegMetrics sig = updateSigReg(z, cfg.wmSota.sigregTargetVar);
  const float zNorm = std::sqrt(latentNormSq(z));
  const float rawVar = computeRawRingVariance();
  if (!std::isfinite(zNorm) || (sig.collapsed && (rawVar > 1e-4f || zNorm < 1e-6f))) {
    fault |= kWmFaultLatentCollapse;
  }

  float predSurp = 0.0f;
  if (hasPrevPred_ && encOk) {
    float d2 = 0.0f;
    for (size_t i = 0; i < kLeWmLatentDim; ++i) d2 += sq(z[i] - prevPredZ_[i]);
    predSurp = std::sqrt(d2 / static_cast<float>(kLeWmLatentDim));
  }
  outSummary.predSurprise = std::isfinite(predSurp) ? predSurp : 0.0f;
  const float exSurp = std::max(0.0f, outSummary.predSurprise - cfg.wmSota.surpriseBaseline);
  outSummary.surpriseSigmaDelta = clampf(cfg.wmSota.surpriseSigmaGain * exSurp, 0.0f, 0.25f);
  outSummary.sigregLoss = std::isfinite(sig.sigregLoss) ? sig.sigregLoss : 0.0f;
  outSummary.latentNorm = std::isfinite(zNorm) ? zNorm : 0.0f;

  // Flat LeWM candidate scoring over drivable lane centers (when !H-JEPA; H-JEPA overwrites with macro+CEM)
  const float maxCruise = std::max(1.0f, cfg.targetCruiseSpeedMs);
  float bestE = 1e6f, recLat = (raw[1] >= 0.0f) ? kLeftLatM : kDefaultLatM, recScale = 1.0f;
  for (float candLat : {kDefaultLatM, kLeftLatM}) {
    const float latV = clampf((candLat - raw[1] * 1.95f) * 0.8f, -1.6f, 1.6f);
    for (float candAccel : {-2.5f, 0.0f}) {
      float e = 0.0f;
      if (!rolloutEnergy(z, {candAccel, latV}, cfg.wmSota.rolloutHorizonSteps,
                         cfg.wmSota.stepDtSec, e)) {
        fault |= kWmFaultRolloutDivergence;
        continue;
      }
      const float tot = cfg.wmSota.energyCostWeight * e +
                        0.04f * std::abs(candLat - kDefaultLatM) +
                        (candLat > 0.5f && z[9] > 0.25f ? 2.5f * cfg.wmSota.oncomingSafetyWeight : 0.0f);
      if (tot < bestE) {
        bestE = tot;
        recLat = candLat;
        const bool tgtBlocked = (candLat < 0.0f) ? (raw[4] > 0.45f || raw[6] > 0.45f)
                                                 : (raw[8] > 0.45f || raw[9] > 0.25f);
        recScale = (candAccel < -1.0f && tgtBlocked) ? 0.85f : 1.0f;
      }
    }
  }
  lastLatent_ = z;
  const LeWmAction curAct{clampf(world.ego().accel, -8.5f, 4.0f),
                          clampf((recLat - raw[1] * 1.95f) * 0.5f, -1.5f, 1.5f)};
  hasPrevPred_ = predictNext(z, curAct, std::max(0.02f, dtSec), prevPredZ_);

  outSummary.bestRolloutEnergy = std::isfinite(bestE) && bestE < 1e5f ? bestE : 0.0f;
  outSummary.macroSubgoalLatM = clampf(recLat, -2.1f, kLeftLatM);
  outSummary.macroSubgoalSpeedMs = maxCruise * recScale;
  outSummary.recommendedLatM = outSummary.macroSubgoalLatM;
  outSummary.recommendedSpeedScale = recScale;
  outSummary.fmeaFaultMask = fault;
  if (cfg.wmSota.enableFmeaFallback &&
      (fault & (kWmFaultNanInput | kWmFaultLatentCollapse |
                kWmFaultRolloutDivergence | kWmFaultInvalidConfig)) != 0u) {
    outSummary.fmeaFallbackTriggered = true;
    outSummary.recommendedLatM = kDefaultLatM;
    outSummary.recommendedSpeedScale = 1.0f;
    outSummary.surpriseSigmaDelta = 0.0f;
  } else {
    prevRecLatM_ = outSummary.recommendedLatM;
    prevRecSpeedScale_ = outSummary.recommendedSpeedScale;
  }
  const auto t1 = std::chrono::steady_clock::now();
  outSummary.stageCpuMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
}

}  // namespace cybercab::sota
