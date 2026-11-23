#include "planning/v15/AutoregSafety.hpp"
#include "planning/RssCorridor.hpp"
#include <algorithm>
#include <cmath>

namespace cybercab::v15 {

SafetyEvalOutput evaluateAutoregSafety(WorldEngine& world, const SafetyEvalInput& in,
                                       std::string initialBehavior, float initialSpeed) {
    SafetyEvalOutput out{};
    out.desiredSpeed = initialSpeed;
    out.behavior = std::move(initialBehavior);
    out.forwardRedundancyLost = true;
    for (int c = 0; c < 3; ++c) {
        if (!(in.perc.soilCameras[c].meanSoilProb > 0.60f && in.perc.soilCameras[c].tokenWeight < 0.40f)) {
            out.forwardRedundancyLost = false;
            break;
        }
    }
    if (out.forwardRedundancyLost) {
        out.desiredSpeed = 0.0f;
        out.behavior = "SENSOR_SOILED_MRM";
        if (in.ego.speed > 3.5f) out.aebTriggered = true;
    }

    const float sinEgo = std::sin(in.ego.yaw), cosEgo = std::cos(in.ego.yaw);
    const float frontBumperX = in.ego.pos.x + sinEgo * (in.ego.hz + in.ego.centerOffsetForward);
    for (const auto& tr : in.perc.tracks) {
        if (tr.semantic == SemanticClass::SignalSign || tr.semantic == SemanticClass::Curb ||
            tr.semantic == SemanticClass::Static || tr.semantic == SemanticClass::Overhang) continue;
        float sTr = 0.0f, dLat = 0.0f, tX = sinEgo, tZ = cosEgo;
        rss::projectOntoPolyline(in.corrPts, in.corrArc, tr.worldPos.x, tr.worldPos.z, sTr, dLat, tX, tZ);
        const float corrW = (tr.semantic == SemanticClass::Pedestrian) ? 1.65f : 1.35f;
        if (sTr > -0.6f && sTr < 30.0f && dLat < corrW) {
            const float ext = (tr.semantic == SemanticClass::Vehicle) ? tr.halfExtents.z : 0.0f;
            const float fwdDist = std::max(0.0f, sTr - ext);
            out.minForwardObsDist = std::min(out.minForwardObsDist, fwdDist);
            const float trSpd = std::hypot(tr.worldVel.x, tr.worldVel.z);
            if (tr.semantic == SemanticClass::Pedestrian) {
                out.behavior = "YIELDING_PEDESTRIAN";
                out.desiredSpeed = (fwdDist <= in.rssDist + 1.5f) ? 0.0f
                    : std::min(out.desiredSpeed, std::sqrt(5.0f * (fwdDist - in.rssDist)));
                if (fwdDist <= in.rssDist * 0.85f && in.ego.speed > 0.4f) out.aebTriggered = true;
            } else if (fwdDist <= in.rssDist) {
                out.desiredSpeed = 0.0f;
                out.behavior = "FOLLOWING_LEAD";
                if (fwdDist < in.rssDist * 0.72f && in.ego.speed > trSpd + 1.2f) out.aebTriggered = true;
            } else if (in.bestLeadGap < 18.0f) {
                out.behavior = "FOLLOWING_LEAD";
                out.desiredSpeed = std::min(out.desiredSpeed, trSpd + std::max(0.0f, (fwdDist - in.rssDist) * 0.45f));
            }
        }
    }
    for (const auto& v : in.perc.activeVoxels) {
        if (v.wy > 2.1f || v.confidence <= 0.52f || v.sem == SemanticClass::Overhang ||
            v.sem == SemanticClass::Static || v.sem == SemanticClass::SignalSign || v.sem == SemanticClass::Curb) continue;
        float fwd = 0.0f, lat = 0.0f, vtX = sinEgo, vtZ = cosEgo;
        rss::projectOntoPolyline(in.corrPts, in.corrArc, v.wx, v.wz, fwd, lat, vtX, vtZ);
        if (fwd > 0.25f && fwd < in.rssDist + 4.0f && lat < ((v.sem == SemanticClass::Pedestrian) ? 1.50f : 1.25f)) {
            out.minForwardObsDist = std::min(out.minForwardObsDist, fwd);
            if (fwd <= in.rssDist * 0.88f && in.ego.speed > 0.5f) {
                out.aebTriggered = true; out.desiredSpeed = 0.0f;
                out.behavior = (v.sem == SemanticClass::Pedestrian) ? "YIELDING_PEDESTRIAN" : "FOLLOWING_LEAD";
            } else if (fwd <= in.rssDist + 1.5f) { out.desiredSpeed = std::min(out.desiredSpeed, 2.5f); }
        }
    }

    if (in.ego.pos.z > 36.0f && in.ego.pos.x < WorldEngine::kStopLineX + 1.5f &&
        in.ego.pos.x > WorldEngine::kStopLineX - 36.0f &&
        (world.signalPhase() == SignalPhase::Red || world.signalPhase() == SignalPhase::Yellow)) {
        const float distBar = WorldEngine::kStopLineX - frontBumperX;
        out.minForwardObsDist = std::min(out.minForwardObsDist, std::max(0.0f, distBar));
        out.behavior = "RED_LIGHT_HOLD";
        out.desiredSpeed = (distBar <= 2.4f) ? 0.0f
            : std::min(out.desiredSpeed, std::sqrt(4.8f * std::max(0.0f, distBar - 1.5f)));
    }
    EgoState& mutEgo = world.mutableEgo();
    if (mutEgo.passedStopSignCooldown > 0.0f) mutEgo.passedStopSignCooldown -= 1.0f / in.cfg.controlHz;
    if (in.ego.pos.z < -36.0f && in.ego.pos.x > 13.0f && in.ego.pos.x < 40.0f && mutEgo.passedStopSignCooldown <= 0.0f) {
        const float distSign = frontBumperX - 14.5f;
        out.minForwardObsDist = std::min(out.minForwardObsDist, std::max(0.0f, distSign));
        out.behavior = "STOP_SIGN_HOLD";
        if (distSign <= 2.2f && std::abs(in.ego.speed) < 0.35f) {
            mutEgo.stopSignHoldTimer += 1.0f / in.cfg.controlHz;
            out.desiredSpeed = 0.0f;
            if (mutEgo.stopSignHoldTimer >= 1.2f) { mutEgo.stopSignHoldTimer = 0.0f; mutEgo.passedStopSignCooldown = 8.0f; }
        } else {
            out.desiredSpeed = std::min(out.desiredSpeed, (distSign < 1.5f) ? 0.0f
                : std::sqrt(4.4f * std::max(0.0f, distSign - 1.0f)));
        }
    }

    if (in.sirenYield) {
        out.behavior = "YIELDING_SIREN";
        out.desiredSpeed = (in.perc.v15.sirenConfidence > 0.55f) ? 0.0f : std::min(out.desiredSpeed, 1.2f);
    } else if (!out.forwardRedundancyLost && out.behavior != "RED_LIGHT_HOLD" &&
               out.behavior != "STOP_SIGN_HOLD" && out.behavior != "YIELDING_PEDESTRIAN" &&
               (std::abs(in.ego.speed) < 0.35f || in.ego.gear < 0) && in.bothLanesStaticBlocked) {
        if (in.perc.v15.rearClearanceM > 6.0f) {
            out.gear = -1; out.desiredSpeed = -1.6f; out.behavior = "AUTOSHIFT_REVERSE"; out.aebTriggered = false;
        } else {
            out.gear = 1; out.desiredSpeed = 0.0f;
        }
    }
    return out;
}

} // namespace cybercab::v15
