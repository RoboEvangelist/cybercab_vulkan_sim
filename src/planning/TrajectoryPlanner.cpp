#include "planning/TrajectoryPlanner.hpp"
#include "planning/RssCorridor.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cybercab {

TrajectoryPlanner::TrajectoryPlanner(const SimConfig& cfg) : cfg_(cfg) {
    corridorScratch_.reserve(40);
    corridorArcScratch_.reserve(40);
}

void TrajectoryPlanner::buildRssCorridor(const EgoState& ego, const ClosedCatmullRomCurve3& centerline,
                                         float loopLen, float loopU, float lookaheadDist,
                                         float currentLat, float targetLat, float maxLen) const {
    rss::buildRssCorridorPoints(ego, centerline, loopLen, loopU, lookaheadDist,
                                currentLat, targetLat, cfg_.wheelbaseM,
                                cfg_.driveMode == "manual", maxLen,
                                corridorScratch_, corridorArcScratch_);
}

float TrajectoryPlanner::quinticLateralOffset(float d0, float d1, float tNorm) {
    const float t = clampf(tNorm, 0.0f, 1.0f);
    const float poly = t * t * t * (10.0f + t * (-15.0f + 6.0f * t));
    return d0 + (d1 - d0) * poly;
}

void TrajectoryPlanner::plan(WorldEngine& world, const PerceptionFrameOutput& perc, PlannerCommand& cmd) const {
    const EgoState& ego = world.ego();
    const auto& centerline = world.centerline();
    const float totalLoopLen = centerline.getLength();
    const float loopU = world.findClosestLoopU(ego.pos);

    const Vec3 curCenter = centerline.getPointAt(loopU);
    const Vec3 curTan = centerline.getTangentAt(loopU);
    const Vec3 curLeft{curTan.z, 0.0f, -curTan.x};
    const float currentLat = (ego.pos - curCenter).dot(curLeft);

    const float candidateOffsets[9] = {-2.5f, -1.9f, -1.3f, -0.6f, 0.0f, 0.6f, 1.3f, 1.9f, 2.5f};
    const int numCand = std::min(9, cfg_.numPlannerCandidates);
    const int trajSteps = cfg_.trajSteps;
    const float lookaheadDist = std::max(18.0f, ego.speed * 2.5f + 8.0f);

    float bestScore = std::numeric_limits<float>::infinity();
    int bestCandIdx = 1;
    float bestLeadGap = std::numeric_limits<float>::infinity();
    float bestTargetOffset = -1.9f;

    cmd.candidateTrajectories.resize(static_cast<size_t>(numCand));
    cmd.candidateBlocked.assign(static_cast<size_t>(numCand), false);

    for (int c = 0; c < numCand; ++c) {
        const float targetLat = candidateOffsets[c];
        auto& pts = cmd.candidateTrajectories[c];
        pts.resize(static_cast<size_t>(trajSteps));
        float collisionCost = 0.0f;
        float minLeadDist = std::numeric_limits<float>::infinity();

        for (int s = 0; s < trajSteps; ++s) {
            const float frac = static_cast<float>(s) / static_cast<float>(trajSteps - 1);
            const float distAlong = frac * lookaheadDist;
            float u = std::fmod(loopU + distAlong / totalLoopLen, 1.0f);
            if (u < 0.0f) u += 1.0f;

            const Vec3 cPt = centerline.getPointAt(u);
            const Vec3 tan = centerline.getTangentAt(u);
            const Vec3 left{tan.z, 0.0f, -tan.x};
            const float lat = quinticLateralOffset(currentLat, targetLat, std::min(1.0f, frac * 1.35f));
            Vec3 wp = cPt + left * lat;
            wp.y = 0.12f;
            pts[s] = wp;

            if (s > 1) {
                const float tPred = distAlong / std::max(4.0f, ego.speed);
                for (size_t k = 0; k < perc.activeVoxels.size(); k += 2) {
                    const auto& v = perc.activeVoxels[k];
                    if (v.wy > 2.2f || v.sem == SemanticClass::Overhang ||
                        v.sem == SemanticClass::SignalSign || v.sem == SemanticClass::Curb) continue;
                    const float predVx = v.wx + v.vx * std::min(1.8f, tPred);
                    const float predVz = v.wz + v.vz * std::min(1.8f, tPred);
                    const float dObs = std::hypot(wp.x - predVx, wp.z - predVz);
                    const float safeRadius = ((v.sem == SemanticClass::Pedestrian) ? 2.1f : 1.55f)
                                           + clampf(v.epistemicSigma * 0.35f, 0.0f, 0.7f);
                    if (dObs < safeRadius) {
                        collisionCost += (safeRadius - dObs) * ((v.sem == SemanticClass::Pedestrian) ? 650.0f : 340.0f);
                        minLeadDist = std::min(minLeadDist, distAlong);
                    }
                }
                for (const auto& tr : perc.tracks) {
                    const float predTx = tr.worldPos.x + tr.worldVel.x * std::min(2.0f, tPred);
                    const float predTz = tr.worldPos.z + tr.worldVel.z * std::min(2.0f, tPred);
                    float dObs = std::hypot(wp.x - predTx, wp.z - predTz);
                    if (tr.semantic == SemanticClass::Vehicle) {
                        const float hzSample = std::min(tr.halfExtents.z, 2.9f) * 0.60f;
                        const float sx = std::sin(tr.yaw), cz = std::cos(tr.yaw);
                        for (const float sign : {-1.0f, 1.0f}) {
                            dObs = std::min(dObs, std::hypot(wp.x - (predTx + sx * hzSample * sign),
                                                             wp.z - (predTz + cz * hzSample * sign)));
                        }
                    }
                    const float safeRadius = ((tr.semantic == SemanticClass::Pedestrian) ? 2.15f : 1.75f)
                                           + clampf(tr.posSigmaM * 0.25f, 0.0f, 0.8f);
                    if (dObs < safeRadius) {
                        collisionCost += (safeRadius - dObs) * ((tr.semantic == SemanticClass::Pedestrian) ? 700.0f : 380.0f);
                        minLeadDist = std::min(minLeadDist, distAlong);
                    }
                }
            }
        }

        const float distToLane0 = std::abs(targetLat - (-1.9f));
        const float distToLane1 = std::abs(targetLat - 1.9f);
        const float laneCenterPenalty = std::min(distToLane0, distToLane1) * 14.0f + ((distToLane1 < 0.3f) ? 6.0f : 0.0f);
        const float lateralChangePenalty = std::abs(targetLat - currentLat) * 2.5f;
        const float totalCost = collisionCost + laneCenterPenalty + lateralChangePenalty;

        cmd.candidateBlocked[c] = (collisionCost > 40.0f);
        if (totalCost < bestScore) {
            bestScore = totalCost;
            bestCandIdx = c;
            bestLeadGap = minLeadDist;
            bestTargetOffset = targetLat;
        }
    }

    cmd.bestCandIdx = bestCandIdx;
    cmd.optimalTrajectory = cmd.candidateTrajectories[bestCandIdx];

    // Uncertainty-aware Responsibility-Sensitive Safety (RSS) stopping envelope
    const float excessSigma = std::max(0.0f, perc.forwardEpistemicSigma - 0.22f);
    const float uncertaintyBuffer = cfg_.rssUncertaintyBufferGain * excessSigma;
    const float rssDist = cfg_.rssDBufferM + uncertaintyBuffer
                        + ego.speed * cfg_.rssTReactSec
                        + (ego.speed * ego.speed) / (2.0f * cfg_.rssABrakeMs2);
    cmd.rssSafeDist = rssDist;

    float desiredSpeed = cfg_.targetCruiseSpeedMs * clampf(1.0f - 0.38f * excessSigma, 0.55f, 1.0f);
    std::string behavior = (std::abs(bestTargetOffset - 1.9f) < 0.6f) ? "LANE_CHANGE_OVERTAKE" : "CRUISING";
    float minForwardObsDist = 45.0f;
    bool aebTriggered = false;

    // FMEA Minimum Risk Maneuver (MRM): if ALL 3 forward cameras are simultaneously blinded by lens soiling,
    // fail safely to a controlled stop rather than cruising blindly across the +/-10 deg forward dead-zone.
    bool forwardRedundancyLost = true;
    for (int c = 0; c < 3; ++c) {
        if (!(perc.soilCameras[c].meanSoilProb > 0.60f && perc.soilCameras[c].tokenWeight < 0.40f)) {
            forwardRedundancyLost = false;
            break;
        }
    }
    if (forwardRedundancyLost) {
        desiredSpeed = 0.0f;
        behavior = "SENSOR_SOILED_MRM";
        if (ego.speed > 3.5f) {
            aebTriggered = true;
        }
    }

    const float sinEgo = std::sin(ego.yaw);
    const float cosEgo = std::cos(ego.yaw);
    const float bumperFwd = ego.hz + ego.centerOffsetForward;
    const float frontBumperX = ego.pos.x + sinEgo * bumperFwd;

    // Build curved RSS stopping-envelope corridor following the road / planned lane-change trajectory.
    buildRssCorridor(ego, centerline, totalLoopLen, loopU, lookaheadDist,
                     currentLat, bestTargetOffset, std::max(32.0f, rssDist + 4.0f));

    for (const auto& tr : perc.tracks) {
        if (tr.semantic == SemanticClass::SignalSign || tr.semantic == SemanticClass::Curb ||
            tr.semantic == SemanticClass::Static || tr.semantic == SemanticClass::Overhang) {
            continue;
        }
        float sTr = 0.0f, dLatTr = 0.0f, tX = sinEgo, tZ = cosEgo;
        rss::projectOntoPolyline(corridorScratch_, corridorArcScratch_,
                                 tr.worldPos.x, tr.worldPos.z, sTr, dLatTr, tX, tZ);

        const float sxTr = std::sin(tr.yaw), czTr = std::cos(tr.yaw);
        const float longProj = (tr.semantic == SemanticClass::Vehicle)
                             ? (std::abs(sxTr * tX + czTr * tZ) * tr.halfExtents.z +
                                std::abs(czTr * tX - sxTr * tZ) * tr.halfExtents.x)
                             : 0.0f;
        const float latProj = (tr.semantic == SemanticClass::Vehicle)
                            ? (std::abs(-sxTr * tZ + czTr * tX) * tr.halfExtents.z +
                               std::abs(czTr * tZ + sxTr * tX) * tr.halfExtents.x)
                            : 0.0f;
        const float fwdDist = sTr - longProj;
        const float latDist = std::max(0.0f, dLatTr - ((tr.semantic == SemanticClass::Vehicle)
                                                     ? clampf(latProj - 0.90f, 0.0f, 0.45f)
                                                     : 0.0f));

        const float trSpd = std::hypot(tr.worldVel.x, tr.worldVel.z);
        float sFut = sTr, latDistFut = latDist, ftX = tX, ftZ = tZ;
        if (trSpd > 0.25f) {
            rss::projectOntoPolyline(corridorScratch_, corridorArcScratch_,
                                     tr.worldPos.x + tr.worldVel.x * 1.50f,
                                     tr.worldPos.z + tr.worldVel.z * 1.50f,
                                     sFut, latDistFut, ftX, ftZ);
        }

        const float latCorridorNow = (tr.semantic == SemanticClass::Pedestrian)
                                   ? ((fwdDist < 12.0f) ? 1.72f : 1.55f)
                                   : 1.35f;
        const float latCorridorFut = (tr.semantic == SemanticClass::Pedestrian) ? 1.60f : 1.30f;
        const bool inNow = (latDist < latCorridorNow);
        const bool approachingCorridor = (latDistFut < latCorridorFut && latDistFut < latDist - 0.08f && trSpd > 0.25f);

        if (fwdDist > -0.6f && fwdDist < 30.0f && (inNow || approachingCorridor)) {
            minForwardObsDist = std::min(minForwardObsDist, std::max(0.0f, fwdDist));
            if (tr.semantic == SemanticClass::Pedestrian) {
                if (fwdDist <= rssDist + 1.5f) {
                    behavior = "YIELDING_PEDESTRIAN";
                    desiredSpeed = 0.0f;
                    if (fwdDist <= rssDist * 0.85f && ego.speed > 0.4f) {
                        aebTriggered = true;
                    }
                } else {
                    behavior = "YIELDING_PEDESTRIAN";
                    desiredSpeed = std::min(desiredSpeed, std::sqrt(2.0f * 2.5f * std::max(0.0f, fwdDist - rssDist)));
                }
            } else {
                const float vSpd = trSpd;
                if (fwdDist <= rssDist) {
                    if (bestLeadGap < 12.0f || latDist < 1.25f) {
                        desiredSpeed = 0.0f;
                        behavior = "FOLLOWING_LEAD";
                        if (fwdDist < rssDist * 0.72f && ego.speed > vSpd + 1.2f) {
                            aebTriggered = true;
                        }
                    }
                } else if (bestLeadGap < 18.0f) {
                    behavior = "FOLLOWING_LEAD";
                    desiredSpeed = std::min(desiredSpeed, vSpd + std::max(0.0f, (fwdDist - rssDist) * 0.45f));
                }
            }
        }
    }

    for (const auto& v : perc.activeVoxels) {
        if (v.wy > 2.1f || v.sem == SemanticClass::Overhang || v.sem == SemanticClass::Static ||
            v.sem == SemanticClass::SignalSign || v.sem == SemanticClass::Curb) {
            continue;
        }
        float fwd = 0.0f, lat = 0.0f, vtX = sinEgo, vtZ = cosEgo;
        rss::projectOntoPolyline(corridorScratch_, corridorArcScratch_, v.wx, v.wz, fwd, lat, vtX, vtZ);
        const float corr = (v.sem == SemanticClass::Pedestrian) ? 1.50f : 1.25f;
        if (fwd > 0.25f && fwd < rssDist + 4.0f && lat < corr && v.confidence > 0.52f) {
            minForwardObsDist = std::min(minForwardObsDist, std::max(0.0f, fwd));
            if (fwd <= rssDist * 0.88f && ego.speed > 0.5f) {
                aebTriggered = true;
                desiredSpeed = 0.0f;
                behavior = (v.sem == SemanticClass::Pedestrian) ? "YIELDING_PEDESTRIAN" : "FOLLOWING_LEAD";
            } else if (fwd <= rssDist + 1.5f) {
                desiredSpeed = std::min(desiredSpeed, 2.5f);
            }
        }
    }

    if (ego.pos.z > 36.0f && ego.pos.x < WorldEngine::kStopLineX + 1.5f && ego.pos.x > WorldEngine::kStopLineX - 36.0f) {
        const float distToStopBar = WorldEngine::kStopLineX - frontBumperX;
        if (world.signalPhase() == SignalPhase::Red || world.signalPhase() == SignalPhase::Yellow) {
            minForwardObsDist = std::min(minForwardObsDist, std::max(0.0f, distToStopBar));
            behavior = "RED_LIGHT_HOLD";
            if (distToStopBar <= 2.4f) {
                desiredSpeed = 0.0f;
            } else {
                desiredSpeed = std::min(desiredSpeed, std::sqrt(2.0f * 2.4f * std::max(0.0f, distToStopBar - 1.5f)));
            }
        }
    }

    EgoState& mutEgo = world.mutableEgo();
    if (mutEgo.passedStopSignCooldown > 0.0f) {
        mutEgo.passedStopSignCooldown -= 1.0f / cfg_.controlHz;
    }
    if (ego.pos.z < -36.0f && ego.pos.x > 13.0f && ego.pos.x < 40.0f && mutEgo.passedStopSignCooldown <= 0.0f) {
        const float distToSign = frontBumperX - 14.5f;
        minForwardObsDist = std::min(minForwardObsDist, std::max(0.0f, distToSign));
        if (distToSign <= 2.2f && ego.speed < 0.35f) {
            mutEgo.stopSignHoldTimer += 1.0f / cfg_.controlHz;
            behavior = "STOP_SIGN_HOLD";
            desiredSpeed = 0.0f;
            if (mutEgo.stopSignHoldTimer >= 1.2f) {
                mutEgo.stopSignHoldTimer = 0.0f;
                mutEgo.passedStopSignCooldown = 8.0f;
            }
        } else {
            behavior = "STOP_SIGN_HOLD";
            desiredSpeed = std::min(desiredSpeed, (distToSign < 1.5f) ? 0.0f : std::sqrt(2.0f * 2.2f * std::max(0.0f, distToSign - 1.0f)));
        }
    }

    cmd.minObstacleDist = minForwardObsDist;
    cmd.aebActive = cfg_.aebEnabled && aebTriggered;

    if (cmd.aebActive) {
        behavior = forwardRedundancyLost ? "SENSOR_SOILED_MRM" : "AEB_EMERGENCY_BRAKE";
        desiredSpeed = 0.0f;
    } else if (ego.collisionHoldTimer > 0.0f) {
        behavior = "COLLISION_CONTACT";
    } else if (cfg_.driveMode == "manual") {
        behavior = "MANUAL_DRIVE";
    }

    cmd.targetSpeed = desiredSpeed;
    cmd.behaviorState = behavior;
    const float envLen = std::max(1.2f, std::min(rssDist, minForwardObsDist + 0.5f));
    rss::resamplePolyline(corridorScratch_, corridorArcScratch_, envLen, 17, cmd.rssCarpetPoints);
    cmd.rssCarpetHalfWidth = ego.hx + 0.12f + clampf(excessSigma * 0.25f, 0.0f, 0.40f);
}

PlannerCommand TrajectoryPlanner::plan(WorldEngine& world, const PerceptionFrameOutput& perc) const {
    PlannerCommand cmd;
    plan(world, perc, cmd);
    return cmd;
}

} // namespace cybercab
