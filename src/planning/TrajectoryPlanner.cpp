#include "planning/TrajectoryPlanner.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cybercab {

TrajectoryPlanner::TrajectoryPlanner(const SimConfig& cfg) : cfg_(cfg) {}

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
    const Vec3 curLeft{-curTan.z, 0.0f, curTan.x};
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
            const Vec3 left{-tan.z, 0.0f, tan.x};
            const float lat = quinticLateralOffset(currentLat, targetLat, std::min(1.0f, frac * 1.35f));
            Vec3 wp = cPt + left * lat;
            wp.y = 0.12f;
            pts[s] = wp;

            if (s > 1) {
                const float tPred = distAlong / std::max(4.0f, ego.speed);
                for (size_t k = 0; k < perc.activeVoxels.size(); k += 2) {
                    const auto& v = perc.activeVoxels[k];
                    if (v.wy > 2.2f || v.sem == SemanticClass::Overhang) continue;
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
                    const float dObs = std::hypot(wp.x - predTx, wp.z - predTz);
                    const float safeRadius = ((tr.semantic == SemanticClass::Pedestrian) ? 2.15f : 1.85f)
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

    const float sinEgo = std::sin(ego.yaw);
    const float cosEgo = std::cos(ego.yaw);
    const float frontBumperX = ego.pos.x + sinEgo * 3.42f;
    const float frontBumperZ = ego.pos.z + cosEgo * 3.42f;

    for (const auto& tr : perc.tracks) {
        if (tr.semantic == SemanticClass::SignalSign || tr.semantic == SemanticClass::Curb ||
            tr.semantic == SemanticClass::Static || tr.semantic == SemanticClass::Overhang) {
            continue;
        }
        const float rx = tr.worldPos.x - frontBumperX;
        const float rz = tr.worldPos.z - frontBumperZ;
        const float fwdDist = (rx * sinEgo + rz * cosEgo) - ((tr.semantic == SemanticClass::Vehicle) ? tr.halfExtents.z : 0.0f);
        const float latDist = std::abs(rx * cosEgo - rz * sinEgo);

        const float rxFut = (tr.worldPos.x + tr.worldVel.x * 1.35f) - frontBumperX;
        const float rzFut = (tr.worldPos.z + tr.worldVel.z * 1.35f) - frontBumperZ;
        const float latDistFut = std::abs(rxFut * cosEgo - rzFut * sinEgo);
        const float trSpd = std::hypot(tr.worldVel.x, tr.worldVel.z);

        const float latCorridorNow = (tr.semantic == SemanticClass::Pedestrian) ? 1.55f : 1.35f;
        const float latCorridorFut = (tr.semantic == SemanticClass::Pedestrian) ? 1.45f : 1.30f;
        const bool inNow = (latDist < latCorridorNow);
        const bool approachingCorridor = (latDistFut < latCorridorFut && latDistFut < latDist - 0.12f && trSpd > 0.35f);

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
        const float fwd = v.egoX - 3.50f;
        const float lat = std::abs(v.egoY);
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
        behavior = "AEB_EMERGENCY_BRAKE";
        desiredSpeed = 0.0f;
    } else if (ego.collisionHoldTimer > 0.0f) {
        behavior = "COLLISION_CONTACT";
    } else if (cfg_.driveMode == "manual") {
        behavior = "MANUAL_DRIVE";
    }

    cmd.targetSpeed = desiredSpeed;
    cmd.behaviorState = behavior;
}

PlannerCommand TrajectoryPlanner::plan(WorldEngine& world, const PerceptionFrameOutput& perc) const {
    PlannerCommand cmd;
    plan(world, perc, cmd);
    return cmd;
}

} // namespace cybercab
