#include "planning/v15/AutoregressivePolicy.hpp"
#include "planning/RssCorridor.hpp"
#include "planning/v15/AutoregSafety.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cybercab::v15 {
namespace {

float quinticLat(float d0, float d1, float tNorm) {
    const float t = clampf(tNorm, 0.0f, 1.0f);
    const float poly = t * t * t * (10.0f + t * (-15.0f + 6.0f * t));
    return d0 + (d1 - d0) * poly;
}

} // namespace

float AutoregressivePolicy::clampToMiddleOfLane(float lat) {
    const float sign = (lat >= 0.0f) ? 1.0f : -1.0f;
    return sign * clampf(std::abs(lat), AutoregConfig::kMinLaneAbsM, AutoregConfig::kMaxLaneAbsM);
}

float AutoregressivePolicy::computeTruckShyOffset(float baseLaneLat, const EgoState& ego,
                                                  const std::vector<Track3D>& tracks) {
    const float laneSign = (baseLaneLat >= 0.0f) ? 1.0f : -1.0f;
    float target = laneSign * AutoregConfig::kLaneCenterAbsM;
    const float sE = std::sin(ego.yaw), cE = std::cos(ego.yaw);
    for (const auto& tr : tracks) {
        if (tr.semantic != SemanticClass::Vehicle || tr.halfExtents.z < 3.2f) continue;
        const float dx = tr.worldPos.x - ego.pos.x, dz = tr.worldPos.z - ego.pos.z;
        const float sFwd = sE * dx + cE * dz, dLeft = cE * dx - sE * dz;
        const float latDiff = std::abs(dLeft - ego.currentLateralOffset);
        if (sFwd > -6.0f && sFwd < 26.0f && latDiff > 1.8f && latDiff < 5.5f) {
            const float shySign = (dLeft > baseLaneLat) ? -1.0f : 1.0f;
            const float w = clampf(1.0f - std::max(0.0f, sFwd - 8.0f) / 18.0f, 0.25f, 1.0f);
            target = laneSign * AutoregConfig::kLaneCenterAbsM + shySign * (AutoregConfig::kMaxTruckShyM * w);
            break;
        }
    }
    return clampToMiddleOfLane(target);
}

void AutoregressivePolicy::rolloutMiddleOfLane(const EgoState& ego, const ClosedCatmullRomCurve3& centerline,
                                               float loopU, float currentLat, float targetLat,
                                               int steps, float lookaheadDist,
                                               std::vector<Vec3>& outTraj, float& outMaxKappa) {
    (void)ego;
    const int n = std::max(6, steps);
    const float totalLen = centerline.getLength(), cleanLat = clampToMiddleOfLane(targetLat);
    outTraj.resize(static_cast<size_t>(n));
    for (int s = 0; s < n; ++s) {
        const float frac = static_cast<float>(s) / static_cast<float>(n - 1);
        float u = std::fmod(loopU + (frac * lookaheadDist) / totalLen, 1.0f);
        if (u < 0.0f) u += 1.0f;
        const Vec3 cPt = centerline.getPointAt(u), tan = centerline.getTangentAt(u);
        const Vec3 left{tan.z, 0.0f, -tan.x};
        Vec3 wp = cPt + left * quinticLat(currentLat, cleanLat, std::min(1.0f, frac / 0.55f));
        wp.y = 0.12f;
        outTraj[static_cast<size_t>(s)] = wp;
    }
    outMaxKappa = 0.0f;
    const int kLimit = std::min(n - 2, 10);
    for (int i = 1; i <= kLimit; ++i) {
        const Vec3 d0 = outTraj[i] - outTraj[i - 1], d1 = outTraj[i + 1] - outTraj[i];
        const float ds = std::hypot(d0.x, d0.z);
        if (ds >= 0.2f) {
            outMaxKappa = std::max(outMaxKappa, std::abs(wrapAngle(std::atan2(d1.x, d1.z) - std::atan2(d0.x, d0.z))) / ds);
        }
    }
}

void AutoregressivePolicy::plan(const SimConfig& cfg, WorldEngine& world,
                                const PerceptionFrameOutput& perc, PlannerCommand& cmd,
                                const AutoregConfig& acfg) {
    const EgoState& ego = world.ego();
    const auto& centerline = world.centerline();
    const float totalLoopLen = centerline.getLength(), loopU = world.findClosestLoopU(ego.pos);
    const Vec3 curTan = centerline.getTangentAt(loopU);
    const float currentLat = (ego.pos - centerline.getPointAt(loopU)).dot({curTan.z, 0.0f, -curTan.x});
    const float lookaheadDist = std::max(acfg.lookaheadMinM, std::max(0.0f, ego.speed) * 1.9f + 7.0f);
    const int trajSteps = std::max(8, cfg.trajSteps);
    const bool sirenYield = (perc.v15.sirenDetected && perc.v15.sirenConfidence > 0.35f);
    const float baseLats[2] = {-AutoregConfig::kLaneCenterAbsM, AutoregConfig::kLaneCenterAbsM};

    cmd.candidateTrajectories.resize(2);
    cmd.candidateBlocked.assign(2, false);
    float candOffsets[2]{}, candKappa[2]{}, candLeadGap[2]{1e4f, 1e4f};
    bool staticBlocked[2]{false, false};
    int bestIdx = 0;
    float bestScore = std::numeric_limits<float>::infinity();

    for (int c = 0; c < 2; ++c) {
        candOffsets[c] = sirenYield ? -2.05f : computeTruckShyOffset(baseLats[c], ego, perc.tracks);
        rolloutMiddleOfLane(ego, centerline, loopU, currentLat, candOffsets[c], trajSteps, lookaheadDist,
                            cmd.candidateTrajectories[c], candKappa[c]);
        float cost = (c == 1) ? (sirenYield ? 1e4f : 12.0f) : 0.0f;
        for (int s = 1; s < trajSteps; ++s) {
            const float distAlong = (static_cast<float>(s) / static_cast<float>(trajSteps - 1)) * lookaheadDist;
            const float tPred = distAlong / std::max(4.0f, std::abs(ego.speed));
            const Vec3& wp = cmd.candidateTrajectories[c][s];
            for (const auto& v : perc.activeVoxels) {
                if (v.wy > 2.2f || v.confidence <= 0.52f || v.sem == SemanticClass::Overhang ||
                    v.sem == SemanticClass::SignalSign || v.sem == SemanticClass::Curb || v.sem == SemanticClass::Static) continue;
                const float r = (v.sem == SemanticClass::Pedestrian) ? 2.1f : 1.55f;
                const float d = std::hypot(wp.x - (v.wx + v.vx * std::min(1.8f, tPred)), wp.z - (v.wz + v.vz * std::min(1.8f, tPred)));
                if (d < r) { cost += (r - d) * 400.0f; candLeadGap[c] = std::min(candLeadGap[c], distAlong); }
            }
            for (const auto& tr : perc.tracks) {
                if (tr.semantic == SemanticClass::Static || tr.semantic == SemanticClass::SignalSign) continue;
                const float r = ((tr.semantic == SemanticClass::Pedestrian) ? 2.15f : 1.75f) + clampf(tr.posSigmaM * 0.2f, 0.0f, 0.6f);
                const float d = std::hypot(wp.x - (tr.worldPos.x + tr.worldVel.x * std::min(2.0f, tPred)),
                                           wp.z - (tr.worldPos.z + tr.worldVel.z * std::min(2.0f, tPred)));
                if (d < r) {
                    cost += (r - d) * ((tr.semantic == SemanticClass::Pedestrian) ? 700.0f : 380.0f);
                    candLeadGap[c] = std::min(candLeadGap[c], distAlong);
                    if (tr.semantic == SemanticClass::Vehicle && distAlong < 9.5f && std::hypot(tr.worldVel.x, tr.worldVel.z) < 0.3f) {
                        staticBlocked[c] = true;
                    }
                }
            }
        }
        cmd.candidateBlocked[c] = (cost > 40.0f);
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
    const float cruiseSpd = std::min(cfg.targetCruiseSpeedMs * clampf(1.0f - 0.38f * excessSigma, 0.55f, 1.0f),
                                     std::sqrt(acfg.maxLatAccelMs2 / std::max(candKappa[bestIdx], 0.015f)));
    const SafetyEvalOutput safe = evaluateAutoregSafety(world,
        {cfg, ego, perc, corrPts, corrArc, rssDist, candLeadGap[bestIdx], sirenYield, staticBlocked[0] && staticBlocked[1]},
        (bestIdx == 1) ? "LANE_CHANGE_OVERTAKE" : "CRUISING", cruiseSpd);

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
