#include "planning/v15/AutoregSafety.hpp"
#include "planning/RssCorridor.hpp"
#include "planning/v15/GaussianRolloutScorer.hpp"
#include "planning/v15/StaticVoxelGuard.hpp"
#include <algorithm>
#include <cmath>

namespace cybercab::v15 {
namespace {
constexpr float kSoilBlindProbThresh = 0.60f, kSoilBlindWeightMax = 0.40f, kMrmAebSpeedThreshMs = 3.5f;
constexpr float kOnRoadMaxLatM = 3.65f, kMaxSafetyRangeM = 30.0f, kDirectBumperRangeM = 3.2f;
constexpr float kVehFutHorizSec = 1.2f, kMinTrackMoveMs = 0.25f, kFutLatApproachM = 0.08f;
constexpr float kPedBrakeDecelGain = 5.0f, kPedStopMarginM = 1.5f, kPedAebFrac = 0.85f, kPedAebMinSpdMs = 0.4f;
constexpr float kAebMinScore = 0.38f, kAebCloseFwdM = 4.5f, kAebHoldMinSpdMs = 0.15f;
constexpr float kVoxelMaxHeightM = 2.1f, kVoxelCloseRssFrac = 0.88f, kVoxelWarnMarginM = 1.5f, kVoxelWarnCapMs = 3.5f;
constexpr float kPedVoxHighConf = 0.65f, kSignalSouthMinZ = 36.0f, kSignalWindowBackM = 36.0f;
constexpr float kSignalPassMarginM = 1.5f, kStopBarHoldDistM = 2.4f, kSignalDecelGain = 4.8f;
constexpr float kStopSignNorthMaxZ = -36.0f, kStopSignLineX = 14.5f, kStopSignMinX = 13.0f, kStopSignMaxX = 40.0f;
constexpr float kStopSignHoldDistM = 2.2f, kStopSignDwellSec = 1.2f, kStopSignCooldownSec = 8.0f;
constexpr float kStopSignDecelGain = 4.4f, kStopSignCreepDistM = 1.5f, kAutoshiftMaxStopSpdMs = 0.35f;

void minPlannedTrajDist(const std::vector<Vec3>* traj, float px, float pz, float& outLat, float& outS) {
    outLat = 1e4f; outS = 1e4f;
    if (!traj || traj->size() < 2) return;
    float arc = 0.0f;
    for (size_t i = 0; i + 1 < traj->size(); ++i) {
        const float ax = (*traj)[i].x, az = (*traj)[i].z, vx = (*traj)[i + 1].x - ax, vz = (*traj)[i + 1].z - az;
        const float segL = std::hypot(vx, vz), lenSq = segL * segL;
        const float t = (lenSq > 1e-6f) ? clampf(((px - ax) * vx + (pz - az) * vz) / lenSq, 0.0f, 1.0f) : 0.0f;
        const float d = std::hypot(px - (ax + t * vx), pz - (az + t * vz));
        if (d < outLat) { outLat = d; outS = arc + t * segL; }
        arc += segL;
    }
}
} // namespace

RotObbExtents projectVehicleExtents(const Track3D& tr, float dirX, float dirZ) {
    if (tr.semantic != SemanticClass::Vehicle) return {};
    const float sx = std::sin(tr.yaw), cz = std::cos(tr.yaw);
    return {
        std::abs(sx * dirX + cz * dirZ) * tr.halfExtents.z + std::abs(cz * dirX - sx * dirZ) * tr.halfExtents.x,
        std::abs(-sx * dirZ + cz * dirX) * tr.halfExtents.z + std::abs(cz * dirZ + sx * dirX) * tr.halfExtents.x
    };
}

SafetyEvalOutput evaluateAutoregSafety(WorldEngine& world, const SafetyEvalInput& in,
                                       std::string initialBehavior, float initialSpeed) {
    SafetyEvalOutput out{};
    out.desiredSpeed = initialSpeed;
    out.behavior = std::move(initialBehavior);
    out.forwardRedundancyLost = true;
    for (int c = 0; c < 3; ++c) {
        if (!(in.perc.soilCameras[c].meanSoilProb > kSoilBlindProbThresh &&
              in.perc.soilCameras[c].tokenWeight < kSoilBlindWeightMax)) {
            out.forwardRedundancyLost = false;
            break;
        }
    }
    if (out.forwardRedundancyLost) {
        out.desiredSpeed = 0.0f;
        out.behavior = "SENSOR_SOILED_MRM";
        if (in.ego.speed > kMrmAebSpeedThreshMs) out.aebTriggered = true;
    }

    const V15PolicyConfig& p = in.cfg.v15Policy;
    const float sinEgo = std::sin(in.ego.yaw), cosEgo = std::cos(in.ego.yaw);
    const float bumperFwd = in.ego.hz + in.ego.centerOffsetForward;
    const float frontBumperX = in.ego.pos.x + sinEgo * bumperFwd;
    const float frontBumperZ = in.ego.pos.z + cosEgo * bumperFwd;
    const auto& cl = world.centerline();
    for (const auto& tr : in.perc.tracks) {
        if (tr.semantic == SemanticClass::SignalSign || tr.semantic == SemanticClass::Curb ||
            tr.semantic == SemanticClass::Static || tr.semantic == SemanticClass::Overhang) continue;
        if (tr.semantic == SemanticClass::Vehicle) {
            const float tu = world.findClosestLoopU(tr.worldPos);
            const Vec3 tt = cl.getTangentAt(tu);
            if (std::abs((tr.worldPos - cl.getPointAt(tu)).dot({tt.z, 0.0f, -tt.x})) > kOnRoadMaxLatM) continue;
        }
        float sTr = 0.0f, dLat = 0.0f, tX = sinEgo, tZ = cosEgo, trajLat = 1e4f, trajS = 1e4f;
        rss::projectOntoPolyline(in.corrPts, in.corrArc, tr.worldPos.x, tr.worldPos.z, sTr, dLat, tX, tZ);
        minPlannedTrajDist(in.plannedTraj, tr.worldPos.x, tr.worldPos.z, trajLat, trajS);
        const RotObbExtents corrExt = projectVehicleExtents(tr, tX, tZ);
        const RotObbExtents dirExt = projectVehicleExtents(tr, sinEgo, cosEgo);
        const float dxB = tr.worldPos.x - frontBumperX, dzB = tr.worldPos.z - frontBumperZ;
        const float sDir = sinEgo * dxB + cosEgo * dzB, latDir = std::abs(cosEgo * dxB - sinEgo * dzB);
        const bool isPed = (tr.semantic == SemanticClass::Pedestrian);
        const float corrW = isPed ? p.trackCorridorPedM : (in.ego.hx + corrExt.latProj + p.vehHullSafetyPadM);
        const float dirW = isPed ? p.trackCorridorPedM : (in.ego.hx + dirExt.latProj + p.vehHullSafetyPadM * 0.65f);
        const float trSpd = std::hypot(tr.worldVel.x, tr.worldVel.z);
        bool approaching = false;
        if (trSpd > kMinTrackMoveMs) {
            float sFut = 0.0f, latFut = 0.0f, ftX = sinEgo, ftZ = cosEgo;
            rss::projectOntoPolyline(in.corrPts, in.corrArc, tr.worldPos.x + tr.worldVel.x * kVehFutHorizSec,
                                     tr.worldPos.z + tr.worldVel.z * kVehFutHorizSec, sFut, latFut, ftX, ftZ);
            approaching = (latFut < corrW * 0.92f && latFut < dLat - kFutLatApproachM &&
                           (sFut - corrExt.longProj) > -0.8f && sFut < kMaxSafetyRangeM);
        }
        const bool inCorr = (std::min(dLat, trajLat) < corrW) && ((sTr - corrExt.longProj) > -0.8f) && (sTr < kMaxSafetyRangeM);
        const bool inDirect = (latDir < dirW) && ((sDir - dirExt.longProj) > -0.6f) && (sDir < kDirectBumperRangeM);
        if (inCorr || inDirect || approaching) {
            const float rawCorr = inCorr ? (sTr - corrExt.longProj) : 1e4f;
            const float rawDir = inDirect ? (sDir - dirExt.longProj) : 1e4f;
            const float fwdDist = std::max(0.0f, std::min({rawCorr, rawDir, approaching ? (sTr - corrExt.longProj) : 1e4f}));
            out.minForwardObsDist = std::min(out.minForwardObsDist, fwdDist);
            if (isPed) {
                out.behavior = "YIELDING_PEDESTRIAN";
                out.desiredSpeed = (fwdDist <= in.rssDist + kPedStopMarginM) ? 0.0f
                    : std::min(out.desiredSpeed, std::sqrt(kPedBrakeDecelGain * (fwdDist - in.rssDist)));
                if (fwdDist <= in.rssDist * kPedAebFrac && in.ego.speed > kPedAebMinSpdMs) out.aebTriggered = true;
            } else if (fwdDist <= in.rssDist) {
                out.desiredSpeed = 0.0f;
                out.behavior = "FOLLOWING_LEAD";
                const bool relAeb = (fwdDist < in.rssDist * p.aebRssFrac && in.ego.speed > trSpd + p.aebRelSpeedMinMs &&
                                     (tr.score >= kAebMinScore || tr.hits >= 2 || fwdDist < kAebCloseFwdM));
                if (relAeb || (fwdDist < p.minHardAebDistM && in.ego.speed > kAebHoldMinSpdMs)) out.aebTriggered = true;
            } else if (fwdDist < in.rssDist + p.followRampSpanM || in.bestLeadGap < 20.0f) {
                out.behavior = "FOLLOWING_LEAD";
                out.desiredSpeed = std::min(out.desiredSpeed, trSpd + std::max(0.0f, fwdDist - in.rssDist) * p.followSpeedGain);
            }
        }
    }
    int closeVoxCnt = 0;
    float minVoxFwd = 45.0f;
    bool closePedVox = false;
    for (const auto& v : in.perc.activeVoxels) {
      if (v.wy > kVoxelMaxHeightM || v.confidence <= p.voxelMinConf ||
          v.sem == SemanticClass::Overhang ||
          v.sem == SemanticClass::SignalSign ||
          v.sem == SemanticClass::Curb) continue;
      if (v.sem == SemanticClass::Static &&
          !isOnRoadBlockingStaticVoxel(world, v)) {
        continue;
      }
        float fwd = 0.0f, lat = 0.0f, vtX = sinEgo, vtZ = cosEgo;
        rss::projectOntoPolyline(in.corrPts, in.corrArc, v.wx, v.wz, fwd, lat, vtX, vtZ);
        const float voxLatLim = (v.sem == SemanticClass::Pedestrian) ? p.voxelCorridorPedM : p.voxelCorridorVehM;
        if (fwd > 0.25f && fwd < in.rssDist + 3.0f && lat < voxLatLim) {
            bool adjTrackedVeh = false;
            if (v.sem == SemanticClass::Vehicle ||
                v.sem == SemanticClass::Static) {
                for (const auto& tr : in.perc.tracks) {
                    if (tr.semantic != SemanticClass::Vehicle) continue;
                    if (std::hypot(v.wx - tr.worldPos.x, v.wz - tr.worldPos.z) < p.adjVehVoxelSuppressM) {
                        float ts = 0.0f, tl = 0.0f, tx = sinEgo, tz = cosEgo;
                        rss::projectOntoPolyline(in.corrPts, in.corrArc, tr.worldPos.x, tr.worldPos.z, ts, tl, tx, tz);
                        const RotObbExtents te = projectVehicleExtents(tr, tx, tz);
                        if (tl > in.ego.hx + te.latProj + p.vehHullSafetyPadM) { adjTrackedVeh = true; break; }
                    }
                }
            }
            if (adjTrackedVeh) continue;
            if (fwd <= in.rssDist * kVoxelCloseRssFrac) {
                ++closeVoxCnt;
                minVoxFwd = std::min(minVoxFwd, fwd);
                if (v.sem == SemanticClass::Pedestrian && v.confidence > kPedVoxHighConf) closePedVox = true;
            } else if (fwd <= in.rssDist + kVoxelWarnMarginM) {
                out.minForwardObsDist = std::min(out.minForwardObsDist, fwd);
                out.desiredSpeed = std::min(out.desiredSpeed, kVoxelWarnCapMs);
            }
        }
    }
    if ((closeVoxCnt >= 2 || closePedVox) && in.ego.speed > 0.5f) {
        out.minForwardObsDist = std::min(out.minForwardObsDist, minVoxFwd);
        out.aebTriggered = true; out.desiredSpeed = 0.0f;
        out.behavior = closePedVox ? "YIELDING_PEDESTRIAN" : "FOLLOWING_LEAD";
    }
    evaluateCorridorGaussians(in.perc.v15Gaussians, in.corrPts, in.corrArc,
                              in.rssDist, in.ego, p, out.minForwardObsDist,
                              out.desiredSpeed, out.aebTriggered, out.behavior);

  const int egoLane = (in.ego.currentLateralOffset >= 0.0f) ? 1 : 0;
  const SignalPhase laneSig = world.signalPhaseForLane(egoLane);
  if (in.ego.pos.z > kSignalSouthMinZ &&
      in.ego.pos.x < WorldEngine::kStopLineX + kSignalPassMarginM &&
      in.ego.pos.x > WorldEngine::kStopLineX - kSignalWindowBackM &&
      (laneSig == SignalPhase::Red || laneSig == SignalPhase::Yellow)) {
        const float distBar = WorldEngine::kStopLineX - frontBumperX;
        out.minForwardObsDist = std::min(out.minForwardObsDist, std::max(0.0f, distBar));
        if (!out.aebTriggered && distBar <= out.minForwardObsDist + 0.5f) out.behavior = "RED_LIGHT_HOLD";
        out.desiredSpeed = (distBar <= kStopBarHoldDistM) ? 0.0f
            : std::min(out.desiredSpeed, std::sqrt(kSignalDecelGain * std::max(0.0f, distBar - kSignalPassMarginM)));
    }
    EgoState& mutEgo = world.mutableEgo();
    if (mutEgo.passedStopSignCooldown > 0.0f) mutEgo.passedStopSignCooldown -= 1.0f / in.cfg.controlHz;
    if (in.ego.pos.z < kStopSignNorthMaxZ && in.ego.pos.x > kStopSignMinX && in.ego.pos.x < kStopSignMaxX &&
        mutEgo.passedStopSignCooldown <= 0.0f) {
        const float distSign = frontBumperX - kStopSignLineX;
        out.minForwardObsDist = std::min(out.minForwardObsDist, std::max(0.0f, distSign));
        if (!out.aebTriggered && distSign <= out.minForwardObsDist + 0.5f) out.behavior = "STOP_SIGN_HOLD";
        if (distSign <= kStopSignHoldDistM && std::abs(in.ego.speed) < kAutoshiftMaxStopSpdMs) {
            mutEgo.stopSignHoldTimer += 1.0f / in.cfg.controlHz;
            out.desiredSpeed = 0.0f;
            if (mutEgo.stopSignHoldTimer >= kStopSignDwellSec) {
                mutEgo.stopSignHoldTimer = 0.0f;
                mutEgo.passedStopSignCooldown = kStopSignCooldownSec;
            }
        } else {
            out.desiredSpeed = std::min(out.desiredSpeed, (distSign < kStopSignCreepDistM) ? 0.0f
                : std::sqrt(kStopSignDecelGain * std::max(0.0f, distSign - 1.0f)));
        }
    }

    if (in.sirenYield) {
        out.behavior = "YIELDING_SIREN";
        out.desiredSpeed = (in.perc.v15.sirenConfidence > p.sirenStopConfThresh) ? 0.0f
            : std::min(out.desiredSpeed, p.sirenYieldSpeedMs);
    } else if (!out.forwardRedundancyLost && out.behavior != "RED_LIGHT_HOLD" &&
               out.behavior != "STOP_SIGN_HOLD" && out.behavior != "YIELDING_PEDESTRIAN" &&
               (std::abs(in.ego.speed) < kAutoshiftMaxStopSpdMs || in.ego.gear < 0) && in.bothLanesStaticBlocked) {
        if (in.perc.v15.rearClearanceM > p.autoshiftMinRearClearM) {
            out.gear = -1; out.desiredSpeed = p.autoshiftReverseSpeedMs;
            out.behavior = "AUTOSHIFT_REVERSE"; out.aebTriggered = false;
        } else {
            out.gear = 1; out.desiredSpeed = 0.0f;
        }
    }
    return out;
}

} // namespace cybercab::v15
