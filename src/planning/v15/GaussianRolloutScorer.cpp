#include "planning/v15/GaussianRolloutScorer.hpp"
#include "perception/v15/GaussianWorldModel.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace cybercab::v15 {
namespace {
constexpr int kSubRolloutSteps = 18, kNumNudges = 3, kNumSpeedScales = 2;
constexpr int kMinSecondDiffStep = 2, kMinJerkDiffStep = 3;
constexpr float kMaxPredHorizSec = 1.80f, kMinCorrScanS = 0.35f;
constexpr float kCloseGaussAebFrac = 0.85f, kMinAebSpeedMs = 0.35f;
constexpr float kMinLoopLenM = 1.0f, kMinRolloutStepSpdMs = 1.0f;
constexpr float kMinPredHorizSpdMs = 4.0f, kWaypointSampleHeightM = 0.65f;
constexpr float kCorridorSampleHeightM = 0.70f, kFrontHullExtentFrac = 0.85f;
constexpr float kFiniteDiffEps = 1e-4f, kSecondDiffCenterCoeff = 2.0f;
constexpr float kLatJerkNormGain = 0.15f, kInitBestScore = 1e12f;
constexpr float kFoundGapSentinelM = 1e3f, kCorridorWarnExtraM = 3.0f;
constexpr float kDynamicFlagThresh = 0.5f, kPedMatchRadiusPadM = 1.10f;
constexpr float kVehMatchRadiusPadM = 1.20f, kStaticGaussRampGain = 1.40f;

struct CorridorHitClass {
  bool pedHit = false, dynamicVehHit = false;
  float leadSpeedMs = 0.0f;
};

CorridorHitClass classifyCorridorHit(
  const std::vector<V15Gaussian3D>& gaussians, const Vec3& pt, float tPred,
  float sinEgo, float cosEgo) {
  CorridorHitClass hit{};
  const int pedSem = static_cast<int>(SemanticClass::Pedestrian);
  const int vehSem = static_cast<int>(SemanticClass::Vehicle);
  for (const auto& g : gaussians) {
    const int sem = static_cast<int>(std::lround(g.scaleSem.w));
    const bool isDyn = (g.velYaw.w > kDynamicFlagThresh);
    const float gx = g.meanOpacity.x + (isDyn ? g.velYaw.x * tPred : 0.0f);
    const float gz = g.meanOpacity.z + (isDyn ? g.velYaw.y * tPred : 0.0f);
    const float dXz = std::hypot(pt.x - gx, pt.z - gz);
    if (sem == pedSem && dXz < g.scaleSem.x + kPedMatchRadiusPadM)
      hit.pedHit = true;
    if (isDyn && sem == vehSem &&
        dXz < std::max(g.scaleSem.x, g.scaleSem.z) + kVehMatchRadiusPadM) {
      hit.dynamicVehHit = true;
      const float pv = sinEgo * g.velYaw.x + cosEgo * g.velYaw.y;
      hit.leadSpeedMs = std::max(hit.leadSpeedMs, std::max(0.0f, pv));
    }
  }
  return hit;
}
}  // namespace

float queryGaussianFieldProb(const std::vector<V15Gaussian3D>& gaussians,
                             const Vec3& worldPt, float horizonSec) {
  return GaussianWorldModel::queryOccupancyProb(
    gaussians, worldPt, horizonSec, true);
}

ComfortRolloutChoice selectComfortSubRollout(
  float baseLaneLat, float currentLat, float prevTargetLat,
  const EgoState& ego, const ClosedCatmullRomCurve3& centerline, float loopU,
  float lookaheadDist, float stopHorizonM,
  const std::vector<V15Gaussian3D>& gaussians, const V15PolicyConfig& p) {
  const float laneSign = (baseLaneLat >= 0.0f) ? 1.0f : -1.0f;
  const float centerLat =
    laneSign * clampf(std::abs(baseLaneLat), p.minLaneAbsM, p.maxLaneAbsM);
  const std::array<float, kNumNudges> nudges = {
    -p.nudgeOffsetM, 0.0f, p.nudgeOffsetM};
  const std::array<float, kNumSpeedScales> spds = {
    kUnitSpeedScale, p.speedBackoffRatio};
  const float totalLen = std::max(kMinLoopLenM, centerline.getLength());
  const float ds = lookaheadDist / static_cast<float>(kSubRolloutSteps - 1);
  const float baseSpd = std::max(std::abs(ego.speed), p.rolloutMinSpeedMs);

  std::array<Vec3, kSubRolloutSteps> pts{};
  std::array<float, kSubRolloutSteps> lats{};
  ComfortRolloutChoice best{};
  best.targetLatM = centerLat;
  float bestScore = kInitBestScore;

  for (float nudge : nudges) {
    const float candLat = laneSign * clampf(std::abs(centerLat + nudge),
                                            p.minLaneAbsM, p.maxLaneAbsM);
    for (float spScale : spds) {
      const float dt = ds / std::max(kMinRolloutStepSpdMs, baseSpd * spScale);
      float lat = currentLat, latJerkSum = 0.0f, prevD2 = 0.0f;
      for (int s = 0; s < kSubRolloutSteps; ++s) {
        const float uRaw = loopU + static_cast<float>(s) * ds / totalLen;
        float u = std::fmod(uRaw, 1.0f);
        if (u < 0.0f) u += 1.0f;
        const Vec3 cPt = centerline.getPointAt(u);
        const Vec3 tan = centerline.getTangentAt(u);
        pts[s] = cPt + Vec3{tan.z, 0.0f, -tan.x} * lat;
        pts[s].y = kWaypointSampleHeightM;
        lats[s] = lat;
        lat += clampf((candLat - lat) * p.latGainPerS, -p.maxLatSpeedMs,
                      p.maxLatSpeedMs) * dt;
        if (s >= kMinSecondDiffStep) {
          const float d2 =
            (lats[s] - kSecondDiffCenterCoeff * lats[s - 1] + lats[s - 2]) /
            std::max(kFiniteDiffEps, dt * dt);
          if (s >= kMinJerkDiffStep) latJerkSum += std::abs(d2 - prevD2) * dt;
          prevD2 = d2;
        }
      }
      float gaussCost = 0.0f, minGap = kUnboundedGapM;
      if (!gaussians.empty()) {
        const float fOff =
          ego.centerOffsetForward + ego.hz * kFrontHullExtentFrac;
        const float predSpd = std::max(kMinPredHorizSpdMs, baseSpd * spScale);
        for (int s = 1; s < kSubRolloutSteps; ++s) {
          const float distAlong = static_cast<float>(s) * ds;
          if (distAlong > stopHorizonM) break;
          const Vec3 seg = pts[s] - pts[s - 1];
          const float segL = std::max(kFiniteDiffEps, std::hypot(seg.x, seg.z));
          const Vec3 wpTan{seg.x / segL, 0.0f, seg.z / segL};
          const float tPred = std::min(kMaxPredHorizSec, distAlong / predSpd);
          const float pObb = queryGaussianFieldProb(
            gaussians, pts[s] + wpTan * ego.centerOffsetForward, tPred);
          const float pFr = queryGaussianFieldProb(
            gaussians, pts[s] + wpTan * fOff, tPred);
          const float prob = std::max(pObb, pFr);
          if (prob > p.gaussianOccThresh) {
            gaussCost += (prob - p.gaussianOccThresh) * p.gaussianOccCostGain;
            minGap = std::min(minGap, distAlong);
          }
        }
      }
      const float jerkC = latJerkSum * kLatJerkNormGain * p.jerkCostWeight +
                          std::abs(candLat - prevTargetLat) * p.jerkCostWeight;
      const float nudgeC = std::abs(candLat - centerLat) * p.nudgeCostWeight;
      const float total = gaussCost + jerkC + nudgeC +
                          (kUnitSpeedScale - spScale) * p.speedBackoffCost;
      if (total < bestScore) {
        bestScore = total;
        best = {candLat, spScale, gaussCost, jerkC, minGap};
      }
    }
  }
  return best;
}

void evaluateCorridorGaussians(
  const std::vector<V15Gaussian3D>& gaussians, const std::vector<Vec3>& corrPts,
  const std::vector<float>& corrArc, float rssDist, const EgoState& ego,
  const V15PolicyConfig& p, float& ioMinObsDist, float& ioDesiredSpeed,
  bool& ioAeb, std::string& ioBehavior) {
  if (gaussians.empty() || corrPts.size() < 2u ||
      corrPts.size() != corrArc.size()) return;
  const float sinEgo = std::sin(ego.yaw), cosEgo = std::cos(ego.yaw);
  const float scanSpd = std::max(kMinPredHorizSpdMs, std::abs(ego.speed));
  float minProbFwd = kUnboundedGapM;
  CorridorHitClass hit{};
  for (size_t i = 1; i < corrPts.size(); ++i) {
    const float sD = corrArc[i];
    if (sD < kMinCorrScanS || sD > rssDist + kCorridorWarnExtraM) continue;
    const Vec3 samplePt{corrPts[i].x, kCorridorSampleHeightM, corrPts[i].z};
    const float tPred = std::min(kMaxPredHorizSec, sD / scanSpd);
    if (queryGaussianFieldProb(gaussians, samplePt, tPred) >=
        p.gaussianSafetyStopProb) {
      minProbFwd = std::min(minProbFwd, sD);
      hit = classifyCorridorHit(gaussians, samplePt, tPred, sinEgo, cosEgo);
      break;
    }
  }
  if (minProbFwd >= kFoundGapSentinelM) return;

  ioMinObsDist = std::min(ioMinObsDist, minProbFwd);
  if (hit.dynamicVehHit && !hit.pedHit) {
    if (minProbFwd <= rssDist) {
      ioBehavior = "FOLLOWING_LEAD";
      if (hit.leadSpeedMs > ego.speed + kMinAebSpeedMs &&
          minProbFwd >= p.minHardAebDistM) {
        ioDesiredSpeed = std::min(ioDesiredSpeed, hit.leadSpeedMs);
      } else {
        ioDesiredSpeed = 0.0f;
        if (minProbFwd <= rssDist * kCloseGaussAebFrac &&
            ego.speed > hit.leadSpeedMs + p.aebRelSpeedMinMs) ioAeb = true;
      }
    } else {
      const float gap = std::max(0.0f, minProbFwd - rssDist);
      ioDesiredSpeed = std::min(
        ioDesiredSpeed, hit.leadSpeedMs + gap * p.followSpeedGain);
    }
    return;
  }
  if (minProbFwd <= rssDist) {
    ioDesiredSpeed = 0.0f;
    ioBehavior = hit.pedHit ? "YIELDING_PEDESTRIAN" : "FOLLOWING_LEAD";
    if (minProbFwd <= rssDist * kCloseGaussAebFrac &&
        ego.speed > kMinAebSpeedMs) {
      ioAeb = true;
    }
  } else {
    const float ramp =
      std::max(0.0f, minProbFwd - rssDist) * kStaticGaussRampGain;
    ioDesiredSpeed = std::min(ioDesiredSpeed, ramp);
  }
}
}  // namespace cybercab::v15
