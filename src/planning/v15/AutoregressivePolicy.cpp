#include "planning/v15/AutoregressivePolicy.hpp"
#include "planning/RssCorridor.hpp"
#include "planning/v15/AutoregSafety.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cybercab::v15 {

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

    bool lane0VehicleNearby = false;
    for (const auto& tr : perc.tracks) {
        if (tr.semantic != SemanticClass::Vehicle) continue;
        const float tu = world.findClosestLoopU(tr.worldPos);
        const Vec3 tt = centerline.getTangentAt(tu);
        const float trLat = (tr.worldPos - centerline.getPointAt(tu)).dot({tt.z, 0.0f, -tt.x});
        const float sFwd = sE * (tr.worldPos.x - ego.pos.x) + cE * (tr.worldPos.z - ego.pos.z);
        if (trLat < -0.4f && trLat > -3.65f && sFwd > -10.0f && sFwd < 26.0f) lane0VehicleNearby = true;
    }

    cmd.candidateTrajectories.resize(2);
    cmd.candidateBlocked.assign(2, false);
    float candOffsets[2]{}, candKappa[2]{}, candLeadGap[2]{1e4f, 1e4f};
    bool staticBlocked[2]{false, false};
    int bestIdx = curLane;
    float bestScore = std::numeric_limits<float>::infinity();

    for (int c = 0; c < 2; ++c) {
        candOffsets[c] = sirenYield ? p.sirenRightPullM : computeTruckShyOffset(baseLats[c], ego, perc.tracks, p);
        rolloutMiddleOfLane(ego, centerline, loopU, currentLat, candOffsets[c], trajSteps, lookaheadDist,
                            cmd.candidateTrajectories[c], candKappa[c], p);
        float cost = 0.0f;
        if (sirenYield) {
            cost = (c == 1) ? 1e4f : 0.0f;
        } else if (curLane == 0 && c == 1) {
            cost = p.rightLaneSwitchCost;
        } else if (curLane == 1) {
            cost = (c == 0) ? (lane0VehicleNearby ? p.leftLaneReturnHoldCost : 0.0f) : (lane0VehicleNearby ? 0.0f : p.leftLaneStayCost);
        }
        if (!sirenYield && std::abs(currentLat) < p.dividerBandM) {
            if (latHeading > p.headingCommitThresh && c == 0) cost += p.headingCommitCost;
            if (latHeading < -p.headingCommitThresh && c == 1) cost += p.headingCommitCost;
        }
        int voxHits = 0;
        float voxCost = 0.0f, obsCost = 0.0f;
        for (int s = 1; s < trajSteps; ++s) {
            const float distAlong = (static_cast<float>(s) / static_cast<float>(trajSteps - 1)) * lookaheadDist;
            if (distAlong > 22.0f) break;
            const float tPred = distAlong / std::max(4.0f, std::abs(ego.speed));
            const Vec3& wp = cmd.candidateTrajectories[c][s];
            if (distAlong <= 18.0f) {
                for (size_t k = 0; k < perc.activeVoxels.size(); k += 2) {
                    const auto& v = perc.activeVoxels[k];
                    if (v.wy > 2.2f || v.confidence <= p.voxelMinConf || v.sem == SemanticClass::Overhang ||
                        v.sem == SemanticClass::SignalSign || v.sem == SemanticClass::Curb || v.sem == SemanticClass::Static) continue;
                    const float r = (v.sem == SemanticClass::Pedestrian) ? 1.95f : 1.35f;
                    const float d = std::hypot(wp.x - (v.wx + v.vx * std::min(1.5f, tPred)), wp.z - (v.wz + v.vz * std::min(1.5f, tPred)));
                    if (d < r) { ++voxHits; voxCost += (r - d) * 140.0f; candLeadGap[c] = std::min(candLeadGap[c], distAlong); }
                }
            }
            for (const auto& tr : perc.tracks) {
                if (tr.semantic == SemanticClass::Static || tr.semantic == SemanticClass::SignalSign) continue;
                if (tr.semantic == SemanticClass::Vehicle) {
                    const float tu = world.findClosestLoopU(tr.worldPos);
                    const Vec3 tt = centerline.getTangentAt(tu);
                    if (std::abs((tr.worldPos - centerline.getPointAt(tu)).dot({tt.z, 0.0f, -tt.x})) > 3.65f) continue;
                }
                if (tr.semantic == SemanticClass::Vehicle && std::hypot(tr.worldVel.x, tr.worldVel.z) < 0.35f) {
                    const float tu = world.findClosestLoopU(tr.worldPos);
                    const Vec3 tt = centerline.getTangentAt(tu);
                    const float trLat = (tr.worldPos - centerline.getPointAt(tu)).dot({tt.z, 0.0f, -tt.x});
                    const float sFwd = sE * (tr.worldPos.x - ego.pos.x) + cE * (tr.worldPos.z - ego.pos.z);
                    if (sFwd > 0.5f && sFwd < 9.5f && std::abs(trLat - candOffsets[c]) < 1.35f) staticBlocked[c] = true;
                }
                const float r = ((tr.semantic == SemanticClass::Pedestrian) ? 2.10f : 1.55f) + clampf(tr.posSigmaM * 0.15f, 0.0f, 0.45f);
                const float d = std::hypot(wp.x - (tr.worldPos.x + tr.worldVel.x * std::min(1.8f, tPred)),
                                           wp.z - (tr.worldPos.z + tr.worldVel.z * std::min(1.8f, tPred)));
                if (d < r) {
                    obsCost += (r - d) * ((tr.semantic == SemanticClass::Pedestrian) ? 650.0f : 320.0f);
                    candLeadGap[c] = std::min(candLeadGap[c], distAlong);
                }
            }
        }
        if (voxHits >= 2) obsCost += std::min(voxCost, 280.0f);
        cost += obsCost;
        cmd.candidateBlocked[c] = (obsCost > 40.0f);
        if (cost < bestScore) { bestScore = cost; bestIdx = c; }
    }

    cmd.bestCandIdx = bestIdx;
    cmd.optimalTrajectory = cmd.candidateTrajectories[bestIdx];
    const float excessSigma = std::max(0.0f, perc.forwardEpistemicSigma - 0.22f);
    const float rssDist = cfg.rssDBufferM + cfg.rssUncertaintyBufferGain * excessSigma +
                          std::max(0.0f, ego.speed) * cfg.rssTReactSec + (ego.speed * ego.speed) / (2.0f * cfg.rssABrakeMs2);
    cmd.rssSafeDist = rssDist;

    std::vector<Vec3> corrPts;
    std::vector<float> corrArc;
    rss::buildRssCorridorPoints(ego, centerline, totalLoopLen, loopU, lookaheadDist, currentLat,
                                candOffsets[bestIdx], cfg.wheelbaseM, cfg.driveMode == "manual",
                                std::max(32.0f, rssDist + 4.0f), corrPts, corrArc);
    const float curveCap = std::max(p.minCurveSpeedMs, std::sqrt(p.maxLatAccelMs2 / std::max(candKappa[bestIdx], 0.015f)));
    const float cruiseSpd = std::min(cfg.targetCruiseSpeedMs * clampf(1.0f - 0.32f * excessSigma, 0.65f, 1.0f), curveCap);
    const bool activelyCrossing = (bestIdx != curLane) || (std::abs(std::abs(currentLat) - p.laneCenterAbsM) > p.crossingDevThreshM);
    const SafetyEvalOutput safe = evaluateAutoregSafety(world,
        {cfg, ego, perc, corrPts, corrArc, rssDist, candLeadGap[bestIdx], sirenYield, staticBlocked[0] && staticBlocked[1]},
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
