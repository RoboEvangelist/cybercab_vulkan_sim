#include "planning/v15/AutoregressivePolicy.hpp"
#include "planning/RssCorridor.hpp"
#include "planning/v15/AutoregSafety.hpp"
#include "planning/v15/GaussianRolloutScorer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cybercab::v15 {
namespace {
constexpr float kOnRoadMaxLatM = 3.65f, kEvalMaxDistM = 24.0f, kVoxelMaxDistM = 18.0f, kVoxelMaxHeightM = 2.2f;
constexpr float kVoxelPedRadM = 1.95f, kVoxelVehRadM = 1.35f, kVoxelCostGain = 140.0f, kVoxelMaxCost = 280.0f;
constexpr float kTrackPedRadM = 2.10f, kTrackPedCostGain = 650.0f, kTrackVehCostGain = 320.0f;
constexpr float kBlindVoxMinX = -5.5f, kBlindVoxMaxX = 7.5f, kBlindVoxLatM = 1.35f;
constexpr int kBlindVoxMinCnt = 3, kCorridorSteps = 36;
constexpr float kStaticMaxVelMs = 0.35f, kStaticMinFwdM = 0.5f, kStaticMaxFwdM = 9.5f, kStaticLatM = 1.35f;
constexpr float kCandBlockedThresh = 40.0f, kSignalSouthZ = 36.0f, kSignalBackWinM = 36.0f, kSignalMarginM = 1.5f;
constexpr float kStopSignNorthZ = -36.0f, kStopSignMinX = 13.0f, kStopSignMaxX = 40.0f, kStopSignLineX = 14.5f;
constexpr float kMinKappaFloor = 0.015f, kSigmaCruiseGain = 0.32f, kSigmaBase = 0.22f, kCorridorMinM = 32.0f;
constexpr size_t kScratchReserveCap = 64u;
constexpr int kNumLaneCands = 2;
constexpr float kInvalidRoadLatM = 1e3f;
constexpr float kLane0NearMaxLatM = -0.4f;
constexpr float kLane0NearMinFwdM = -10.0f;
constexpr float kLane0NearMaxFwdM = 26.0f;
constexpr float kCorridorExtraLenM = 4.0f;
constexpr float kCorridorGroundYM = 0.08f;
constexpr float kMinSigmaSpeedScale = 0.65f;
} // namespace

float AutoregressivePolicy::clampToMiddleOfLane(float lat, const V15PolicyConfig& p) {
    const float sign = (lat >= 0.0f) ? 1.0f : -1.0f;
    return sign * clampf(std::abs(lat), p.minLaneAbsM, p.maxLaneAbsM);
}

float AutoregressivePolicy::computeTruckShyOffset(float baseLaneLat, const EgoState& ego,
                                                  const std::vector<Track3D>& tracks,
                                                  const V15PolicyConfig& p) {
    const float laneSign = (baseLaneLat >= 0.0f) ? 1.0f : -1.0f;
    float target = laneSign * p.laneCenterAbsM;
    const float sE = std::sin(ego.yaw), cE = std::cos(ego.yaw);
    for (const auto& tr : tracks) {
        if (tr.semantic != SemanticClass::Vehicle || tr.halfExtents.z < p.truckHalfLengthMinM) continue;
        const float dx = tr.worldPos.x - ego.pos.x, dz = tr.worldPos.z - ego.pos.z;
        const float sFwd = sE * dx + cE * dz, dLeft = cE * dx - sE * dz;
        const float truckRoadLat = ego.currentLateralOffset + dLeft;
        const float absLat = std::abs(truckRoadLat - baseLaneLat);
        if (sFwd > p.truckShyFwdMinM && sFwd < p.truckShyFwdMaxM &&
            absLat > p.adjLaneMinSepM && absLat < p.adjLaneMaxSepM) {
            const float shySign = (truckRoadLat > baseLaneLat) ? -1.0f : 1.0f;
            const float fadeSpan = std::max(1.0f, p.truckShyFwdMaxM - 8.0f);
            const float w = clampf(1.0f - std::max(0.0f, sFwd - 8.0f) / fadeSpan, 0.25f, 1.0f);
            target = laneSign * p.laneCenterAbsM + shySign * (p.maxTruckShyM * w);
            break;
        }
    }
    return clampToMiddleOfLane(target, p);
}

void AutoregressivePolicy::rolloutMiddleOfLane(const EgoState& ego, const ClosedCatmullRomCurve3& centerline,
                                               float loopU, float currentLat, float targetLat,
                                               int steps, float lookaheadDist,
                                               std::vector<Vec3>& outTraj, float& outMaxKappa,
                                               const V15PolicyConfig& p) {
    const int n = std::max(6, steps);
    const float totalLen = centerline.getLength(), cleanLat = clampToMiddleOfLane(targetLat, p);
    const float ds = lookaheadDist / static_cast<float>(n - 1);
    const float dtStep = ds / std::max(std::abs(ego.speed), p.rolloutMinSpeedMs);
    outTraj.resize(static_cast<size_t>(n));
    // Autoregressive lateral state rollout d_{k+1} = d_k + clamp(k_p*(d_tgt - d_k), +-v_lat_max) * dtStep
    float lat = currentLat;
    for (int s = 0; s < n; ++s) {
        const float distAlong = static_cast<float>(s) * ds;
        float u = std::fmod(loopU + distAlong / totalLen, 1.0f);
        if (u < 0.0f) u += 1.0f;
        const Vec3 cPt = centerline.getPointAt(u), tan = centerline.getTangentAt(u);
        const Vec3 left{tan.z, 0.0f, -tan.x};
        Vec3 wp = cPt + left * lat;
        wp.y = 0.12f;
        outTraj[static_cast<size_t>(s)] = wp;
        const float vLat = clampf((cleanLat - lat) * p.latGainPerS, -p.maxLatSpeedMs, p.maxLatSpeedMs);
        lat += vLat * dtStep;
    }
    outMaxKappa = 0.0f;
    const float halfChord = p.curveChordHalfM, duChord = halfChord / std::max(totalLen, 1.0f);
    for (float sM : {2.0f, 6.0f, 10.0f, 14.0f}) {
        float uc = std::fmod(loopU + sM / totalLen, 1.0f);
        if (uc < 0.0f) uc += 1.0f;
        float um = std::fmod(uc - duChord + 1.0f, 1.0f), up = std::fmod(uc + duChord, 1.0f);
        const Vec3 t0 = centerline.getTangentAt(um), t1 = centerline.getTangentAt(up);
        outMaxKappa = std::max(outMaxKappa, std::abs(wrapAngle(std::atan2(t1.x, t1.z) - std::atan2(t0.x, t0.z))) / (2.0f * halfChord));
    }
}

void AutoregressivePolicy::plan(const SimConfig& cfg, WorldEngine& world,
                                const PerceptionFrameOutput& perc, PlannerCommand& cmd,
                                const AutoregConfig& acfg) {
    (void)acfg;
    const V15PolicyConfig& p = cfg.v15Policy;
    const EgoState& ego = world.ego();
    const auto& centerline = world.centerline();
    const float totalLoopLen = centerline.getLength(), loopU = world.findClosestLoopU(ego.pos);
    const Vec3 curTan = centerline.getTangentAt(loopU);
    const float currentLat = (ego.pos - centerline.getPointAt(loopU)).dot({curTan.z, 0.0f, -curTan.x});
    const float latHeading = (std::sin(ego.yaw) * curTan.z) - (std::cos(ego.yaw) * curTan.x);
    const float lookaheadDist = std::max(p.minLookaheadM, std::max(0.0f, ego.speed) * p.lookaheadTimeSec + p.lookaheadBaseM);
    const int trajSteps = std::max(8, cfg.trajSteps);
    const bool sirenYield = (perc.v15.sirenDetected && perc.v15.sirenConfidence > p.sirenConfThresh);
    const float baseLats[2] = {-p.laneCenterAbsM, p.laneCenterAbsM};
    const int curLane = (currentLat >= 0.0f) ? 1 : 0;
    const float sE = std::sin(ego.yaw), cE = std::cos(ego.yaw);
    const float bumperFwd = ego.hz + ego.centerOffsetForward;
    const float egoObbCx = ego.pos.x + sE * ego.centerOffsetForward, egoObbCz = ego.pos.z + cE * ego.centerOffsetForward;
    const float frontBumperX = ego.pos.x + sE * bumperFwd, frontBumperZ = ego.pos.z + cE * bumperFwd;

    const bool redOrYellow = (ego.pos.z > kSignalSouthZ && ego.pos.x > WorldEngine::kStopLineX - kSignalBackWinM &&
                              ego.pos.x < WorldEngine::kStopLineX + kSignalMarginM &&
                              (world.signalPhase() == SignalPhase::Red || world.signalPhase() == SignalPhase::Yellow));
    const bool atStopSign = (ego.pos.z < kStopSignNorthZ && ego.pos.x > kStopSignMinX &&
                             ego.pos.x < kStopSignMaxX && ego.passedStopSignCooldown <= 0.0f);
    float stopHorizonM = kEvalMaxDistM;
    if (redOrYellow) stopHorizonM = std::min(stopHorizonM, std::max(4.5f, (WorldEngine::kStopLineX - frontBumperX) + 1.0f));
    if (atStopSign) stopHorizonM = std::min(stopHorizonM, std::max(4.5f, (frontBumperX - kStopSignLineX) + 1.0f));

  thread_local std::vector<float> trRoadLat;
  trRoadLat.reserve(kScratchReserveCap);
  trRoadLat.assign(perc.tracks.size(), kInvalidRoadLatM);
  bool lane0VehicleNearby = false;
  for (size_t i = 0; i < perc.tracks.size(); ++i) {
    const auto& tr = perc.tracks[i];
    if (tr.semantic != SemanticClass::Vehicle) continue;
    const float tu = world.findClosestLoopU(tr.worldPos);
    const Vec3 tt = centerline.getTangentAt(tu);
    trRoadLat[i] =
      (tr.worldPos - centerline.getPointAt(tu)).dot({tt.z, 0.0f, -tt.x});
    const float sFwd =
      sE * (tr.worldPos.x - ego.pos.x) + cE * (tr.worldPos.z - ego.pos.z);
    if (trRoadLat[i] < kLane0NearMaxLatM && trRoadLat[i] > -kOnRoadMaxLatM &&
        sFwd > kLane0NearMinFwdM && sFwd < kLane0NearMaxFwdM) {
      lane0VehicleNearby = true;
    }
  }

  cmd.candidateTrajectories.resize(kNumLaneCands);
  cmd.candidateBlocked.assign(kNumLaneCands, false);
  float candOffsets[kNumLaneCands]{};
  float candKappa[kNumLaneCands]{};
  float candLeadGap[kNumLaneCands]{kUnboundedGapM, kUnboundedGapM};
  float candSpeedScale[kNumLaneCands]{kUnitSpeedScale, kUnitSpeedScale};
  bool staticBlocked[kNumLaneCands]{false, false};
  int bestIdx = curLane;
  float bestScore = std::numeric_limits<float>::infinity();

  for (int c = 0; c < kNumLaneCands; ++c) {
    const float baseC =
      sirenYield ? p.sirenRightPullM
                 : computeTruckShyOffset(baseLats[c], ego, perc.tracks, p);
    const ComfortRolloutChoice choice =
      sirenYield
        ? ComfortRolloutChoice{p.sirenRightPullM, kUnitSpeedScale, 0.0f, 0.0f,
                               kUnboundedGapM}
        : selectComfortSubRollout(
            baseC, currentLat, (c == curLane) ? currentLat : baseLats[c], ego,
            centerline, loopU, lookaheadDist, stopHorizonM, perc.v15Gaussians,
            p);
    candOffsets[c] = choice.targetLatM;
    candSpeedScale[c] = choice.speedScale;
    candLeadGap[c] = std::min(candLeadGap[c], choice.minGaussianGapM);
    rolloutMiddleOfLane(ego, centerline, loopU, currentLat, candOffsets[c],
                        trajSteps, lookaheadDist, cmd.candidateTrajectories[c],
                        candKappa[c], p);
    float cost = choice.comfortJerkCost;
    float voxCost = 0.0f, obsCost = choice.gaussianCost;
        if (sirenYield) {
            cost = (c == 1) ? 1e4f : 0.0f;
        } else if (curLane == 0 && c == 1) {
            cost += p.rightLaneSwitchCost;
        } else if (curLane == 1) {
          const float returnCost =
            lane0VehicleNearby ? p.leftLaneReturnHoldCost : 0.0f;
          const float stayCost = lane0VehicleNearby ? 0.0f : p.leftLaneStayCost;
          cost += (c == 0) ? returnCost : stayCost;
        }
        if (!sirenYield && (redOrYellow || atStopSign) && c != curLane) cost += p.intersectionLaneLockCost;
        if (!sirenYield && std::abs(currentLat) < p.dividerBandM) {
            if (latHeading > p.headingCommitThresh && c == 0) cost += p.headingCommitCost;
            if (latHeading < -p.headingCommitThresh && c == 1) cost += p.headingCommitCost;
        }
        if (!sirenYield && c != curLane) {
            for (size_t i = 0; i < perc.tracks.size(); ++i) {
                const auto& tr = perc.tracks[i];
                if (tr.semantic != SemanticClass::Vehicle || std::abs(trRoadLat[i]) > kOnRoadMaxLatM) continue;
                const float sCenter = sE * (tr.worldPos.x - egoObbCx) + cE * (tr.worldPos.z - egoObbCz);
                const float trAlong = sE * tr.worldVel.x + cE * tr.worldVel.z;
                const float rearReq = ego.hz + tr.halfExtents.z + p.blindSpotRearGapM + std::max(0.0f, trAlong - ego.speed) * p.blindSpotClosingTimeS;
                const float fwdReq = ego.hz + tr.halfExtents.z + p.blindSpotFwdGapM + std::max(0.0f, ego.speed - trAlong) * p.blindSpotClosingTimeS;
                if (std::abs(trRoadLat[i] - candOffsets[c]) < (ego.hx + tr.halfExtents.x + p.blindSpotLatPadM) &&
                    sCenter > -rearReq && sCenter < fwdReq) {
                    obsCost += p.blindSpotBlockCost;
                }
            }
            int blindVox = 0;
            const float tgtLatRel = candOffsets[c] - currentLat;
            for (const auto& v : perc.activeVoxels) {
                if (v.sem != SemanticClass::Vehicle || v.wy > kVoxelMaxHeightM || v.confidence <= p.voxelMinConf) continue;
                const float sV = sE * (v.wx - egoObbCx) + cE * (v.wz - egoObbCz);
                const float lV = cE * (v.wx - ego.pos.x) - sE * (v.wz - ego.pos.z);
                if (sV > kBlindVoxMinX && sV < kBlindVoxMaxX && std::abs(lV - tgtLatRel) < kBlindVoxLatM) ++blindVox;
            }
            if (blindVox >= kBlindVoxMinCnt) obsCost += p.blindSpotBlockCost * 0.5f;
        }

        int voxHits = 0;
        for (int s = 1; s < trajSteps; ++s) {
            const float distAlong = (static_cast<float>(s) / static_cast<float>(trajSteps - 1)) * lookaheadDist;
            if (distAlong > stopHorizonM) break;
            const float tPred = distAlong / std::max(4.0f, std::abs(ego.speed));
            const Vec3& wp = cmd.candidateTrajectories[c][s];
            const Vec3 seg = wp - cmd.candidateTrajectories[c][s - 1];
            const float segL = std::max(1e-4f, std::hypot(seg.x, seg.z));
            const Vec3 wpTan{seg.x / segL, 0.0f, seg.z / segL};
            const Vec3 wpObb = wp + wpTan * ego.centerOffsetForward, wpFront = wp + wpTan * (ego.centerOffsetForward + ego.hz * 0.85f);
            if (distAlong <= kVoxelMaxDistM) {
                for (size_t k = 0; k < perc.activeVoxels.size(); k += 2) {
                    const auto& v = perc.activeVoxels[k];
                    if (v.wy > kVoxelMaxHeightM || v.confidence <= p.voxelMinConf || v.sem == SemanticClass::Overhang ||
                        v.sem == SemanticClass::SignalSign || v.sem == SemanticClass::Curb || v.sem == SemanticClass::Static) continue;
                    const float r = (v.sem == SemanticClass::Pedestrian) ? kVoxelPedRadM : kVoxelVehRadM;
                    const float vxP = v.wx + v.vx * std::min(1.5f, tPred), vzP = v.wz + v.vz * std::min(1.5f, tPred);
                    const float d = std::min(std::hypot(wpObb.x - vxP, wpObb.z - vzP), std::hypot(wpFront.x - vxP, wpFront.z - vzP));
                    if (d < r) { ++voxHits; voxCost += (r - d) * kVoxelCostGain; candLeadGap[c] = std::min(candLeadGap[c], distAlong); }
                }
            }
            for (size_t i = 0; i < perc.tracks.size(); ++i) {
                const auto& tr = perc.tracks[i];
                if (tr.semantic == SemanticClass::Static || tr.semantic == SemanticClass::SignalSign) continue;
                if (tr.semantic == SemanticClass::Vehicle && std::abs(trRoadLat[i]) > kOnRoadMaxLatM) continue;
                if (tr.semantic == SemanticClass::Vehicle && std::hypot(tr.worldVel.x, tr.worldVel.z) < kStaticMaxVelMs) {
                    const float sFwd = sE * (tr.worldPos.x - ego.pos.x) + cE * (tr.worldPos.z - ego.pos.z);
                    if (sFwd > kStaticMinFwdM && sFwd < kStaticMaxFwdM && std::abs(trRoadLat[i] - candOffsets[c]) < kStaticLatM) {
                        staticBlocked[c] = true;
                    }
                }
                const float tp = std::min(1.8f, tPred), px = tr.worldPos.x + tr.worldVel.x * tp, pz = tr.worldPos.z + tr.worldVel.z * tp;
                float d = std::min(std::hypot(wpObb.x - px, wpObb.z - pz), std::hypot(wpFront.x - px, wpFront.z - pz));
                if (tr.semantic == SemanticClass::Vehicle) {
                    const float hzOff = tr.halfExtents.z * p.obbHullSampleScale, sxT = std::sin(tr.yaw), czT = std::cos(tr.yaw);
                    for (const float sgn : {-1.0f, 1.0f}) {
                        const float hxP = px + sxT * hzOff * sgn, hzP = pz + czT * hzOff * sgn;
                        d = std::min({d, std::hypot(wpObb.x - hxP, wpObb.z - hzP), std::hypot(wpFront.x - hxP, wpFront.z - hzP)});
                    }
                }
                const float baseR = (tr.semantic == SemanticClass::Pedestrian) ? kTrackPedRadM : (ego.hx + tr.halfExtents.x + p.vehClearancePadM);
                const float r = baseR + clampf(tr.posSigmaM * 0.15f, 0.0f, 0.45f);
                if (d < r) {
                    obsCost += (r - d) * ((tr.semantic == SemanticClass::Pedestrian) ? kTrackPedCostGain : kTrackVehCostGain);
                    const float bumGap = std::max(0.0f, distAlong - ((tr.semantic == SemanticClass::Vehicle) ? tr.halfExtents.z : 0.0f));
                    candLeadGap[c] = std::min(candLeadGap[c], bumGap);
                }
            }
        }
        if (voxHits >= 2) obsCost += std::min(voxCost, kVoxelMaxCost);
        cost += obsCost;
        cmd.candidateBlocked[c] = (obsCost > kCandBlockedThresh);
        if (cost < bestScore) { bestScore = cost; bestIdx = c; }
    }

    cmd.bestCandIdx = bestIdx;
    cmd.optimalTrajectory = cmd.candidateTrajectories[bestIdx];
    const float excessSigma = std::max(0.0f, perc.forwardEpistemicSigma - kSigmaBase);
    const float rssDist = cfg.rssDBufferM + cfg.rssUncertaintyBufferGain * excessSigma +
                          std::max(0.0f, ego.speed) * cfg.rssTReactSec + (ego.speed * ego.speed) / (2.0f * cfg.rssABrakeMs2);
    cmd.rssSafeDist = rssDist;

  thread_local std::vector<Vec3> corrPts;
  thread_local std::vector<float> corrArc;
  corrPts.reserve(kScratchReserveCap);
  corrArc.reserve(kScratchReserveCap);
  const float corrLen = std::max(kCorridorMinM, rssDist + kCorridorExtraLenM);
  if (cfg.driveMode == "manual") {
    rss::buildRssCorridorPoints(ego, centerline, totalLoopLen, loopU,
                                lookaheadDist, currentLat, candOffsets[bestIdx],
                                cfg.wheelbaseM, true, corrLen, corrPts,
                                corrArc);
  } else {
    float bumpU = std::fmod(loopU + bumperFwd / totalLoopLen, 1.0f);
    if (bumpU < 0.0f) bumpU += 1.0f;
    const Vec3 bTan = centerline.getTangentAt(bumpU);
    const float bLat = (Vec3{frontBumperX, 0.0f, frontBumperZ} -
                        centerline.getPointAt(bumpU))
                         .dot({bTan.z, 0.0f, -bTan.x});
    float dummyK = 0.0f;
    rolloutMiddleOfLane(ego, centerline, bumpU, bLat, candOffsets[bestIdx],
                        kCorridorSteps, corrLen, corrPts, dummyK, p);
    corrPts[0] = {frontBumperX, kCorridorGroundYM, frontBumperZ};
    corrArc.resize(corrPts.size(), 0.0f);
    for (size_t i = 1; i < corrPts.size(); ++i) {
      corrPts[i].y = kCorridorGroundYM;
      corrArc[i] = corrArc[i - 1] + std::hypot(corrPts[i].x - corrPts[i - 1].x,
                                               corrPts[i].z - corrPts[i - 1].z);
    }
  }
  const float safeKappa = std::max(candKappa[bestIdx], kMinKappaFloor);
  const float curveCap =
    std::max(p.minCurveSpeedMs, std::sqrt(p.maxLatAccelMs2 / safeKappa));
  const float sigmaScale =
    clampf(1.0f - kSigmaCruiseGain * excessSigma, kMinSigmaSpeedScale, 1.0f);
  const float cruiseSpd =
    std::min(cfg.targetCruiseSpeedMs * sigmaScale, curveCap) *
    candSpeedScale[bestIdx];
    const bool activelyCrossing = (bestIdx != curLane) || (std::abs(std::abs(currentLat) - p.laneCenterAbsM) > p.crossingDevThreshM);
    const SafetyEvalOutput safe = evaluateAutoregSafety(world,
        {cfg, ego, perc, corrPts, corrArc, rssDist, candLeadGap[bestIdx], sirenYield, staticBlocked[0] && staticBlocked[1],
         &cmd.optimalTrajectory},
        activelyCrossing ? "LANE_CHANGE_OVERTAKE" : "CRUISING", cruiseSpd);

    cmd.gear = safe.gear;
    cmd.minObstacleDist = safe.minForwardObsDist;
    cmd.aebActive = cfg.aebEnabled && safe.aebTriggered;
    cmd.behaviorState = cmd.aebActive ? (safe.forwardRedundancyLost ? "SENSOR_SOILED_MRM" : "AEB_EMERGENCY_BRAKE")
                                      : (ego.collisionHoldTimer > 0.0f ? "COLLISION_CONTACT"
                                                                       : (cfg.driveMode == "manual" ? "MANUAL_DRIVE" : safe.behavior));
    cmd.targetSpeed = cmd.aebActive ? 0.0f : safe.desiredSpeed;
    rss::resamplePolyline(corrPts, corrArc, std::max(1.2f, std::min(rssDist, safe.minForwardObsDist + 0.5f)), 17, cmd.rssCarpetPoints);
    cmd.rssCarpetHalfWidth = ego.hx + 0.12f + clampf(excessSigma * 0.25f, 0.0f, 0.40f);
}

} // namespace cybercab::v15
