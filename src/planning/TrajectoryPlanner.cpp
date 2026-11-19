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

