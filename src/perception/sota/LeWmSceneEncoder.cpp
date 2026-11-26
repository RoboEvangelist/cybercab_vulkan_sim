#include "perception/sota/LeWmSceneEncoder.hpp"

#include <algorithm>
#include <cmath>

namespace cybercab::sota {
namespace {

constexpr float kDefaultLatM = -1.95f;

float safeVal(float v, float fallback, uint32_t& mask) {
  if (!std::isfinite(v)) {
    mask |= kWmFaultNanInput;
    return fallback;
  }
  return v;
}

}  // namespace

bool encodeSceneLatent16(const SimConfig& cfg, const WorldEngine& world,
                         const PerceptionFrameOutput& perc,
                         const LeWmWeights& weights, LatentVec16& zOut,
                         uint32_t& faultMaskOut, LatentVec16* rawFeaturesOut) {
  faultMaskOut = kWmFaultNone;
  const EgoState& ego = world.ego();
  const auto& cl = world.centerline();
  const float uEgo = world.findClosestLoopU(ego.pos);
  const Vec3 tan = cl.getTangentAt(uEgo);
  const Vec3 cPt = cl.getPointAt(uEgo);
  const Vec3 leftVec{tan.z, 0.0f, -tan.x};
  const float ex = safeVal(ego.pos.x, 0.0f, faultMaskOut);
  const float ez = safeVal(ego.pos.z, 0.0f, faultMaskOut);
  const float eyaw = safeVal(ego.yaw, 0.0f, faultMaskOut);
  const float espd = safeVal(ego.speed, 0.0f, faultMaskOut);
  const float esteer = safeVal(ego.steerAngle, 0.0f, faultMaskOut);
  const float curLat =
    safeVal((Vec3{ex, 0.0f, ez} - cPt).dot(leftVec), kDefaultLatM, faultMaskOut);
  const int curLane = (curLat >= 0.0f) ? 1 : 0;
  const float sE = std::sin(eyaw), cE = std::cos(eyaw);
  const float latHead = sE * tan.z - cE * tan.x;
  const float maxCruise = std::max(1.0f, cfg.targetCruiseSpeedMs);
  const float bumperFwd = ego.hz + ego.centerOffsetForward;

  float sameGap = 40.0f, sameLeadSpd = maxCruise, sameStaticGap = 40.0f;
  float adjFwdGap = 40.0f, adjOncoming = 0.0f, adjBlind = 0.0f, vruThreat = 0.0f;
  for (const auto& tr : perc.tracks) {
    const float tx = safeVal(tr.worldPos.x, ex, faultMaskOut);
    const float tz = safeVal(tr.worldPos.z, ez, faultMaskOut);
    const float vx = safeVal(tr.worldVel.x, 0.0f, faultMaskOut);
    const float vz = safeVal(tr.worldVel.z, 0.0f, faultMaskOut);
    const float tu = world.findClosestLoopU({tx, 0.0f, tz});
    const Vec3 tt = cl.getTangentAt(tu);
    const float trLat = (Vec3{tx, 0.0f, tz} - cl.getPointAt(tu)).dot({tt.z, 0.0f, -tt.x});
    const float sFwd = sE * (tx - ex) + cE * (tz - ez) - bumperFwd;
    const float dLeft = cE * (tx - ex) - sE * (tz - ez);
    const float alongSpd = sE * vx + cE * vz;
    if (tr.semantic == SemanticClass::Pedestrian) {
      if (sFwd > -1.0f && sFwd < 28.0f && std::abs(dLeft) < 4.5f) {
        const float thr = clampf(1.0f - sFwd / 26.0f, 0.0f, 1.0f) *
                          clampf(1.0f - std::abs(dLeft) / 4.2f, 0.0f, 1.0f);
        vruThreat = std::max(vruThreat, thr);
      }
      continue;
    }
    if (tr.semantic != SemanticClass::Vehicle || std::abs(trLat) > 3.85f) continue;
    const int trLane = (trLat >= 0.0f) ? 1 : 0;
    const float netGap = std::max(0.0f, sFwd - tr.halfExtents.z);
    if (trLane == curLane) {
      if (sFwd > -1.0f && netGap < sameGap) {
        sameGap = netGap;
        sameLeadSpd = std::hypot(vx, vz);
      }
      if (sFwd > 0.5f && std::hypot(vx, vz) < 0.4f && netGap < sameStaticGap) {
        sameStaticGap = netGap;
      }
    } else {
      if (sFwd > -1.0f && netGap < adjFwdGap) adjFwdGap = netGap;
      if (sFwd > -8.0f && sFwd < 14.0f) {
        adjBlind = std::max(adjBlind, clampf(1.0f - std::abs(sFwd) / 14.0f, 0.25f, 1.0f));
      }
      if (sFwd > 0.0f && sFwd < 38.0f && alongSpd < -1.0f) {
        const float relClose = std::max(0.0f, espd - alongSpd);
        const float ttc = netGap / std::max(1.0f, relClose);
        adjOncoming = std::max(adjOncoming, clampf(1.0f - ttc / 4.2f, 0.15f, 1.3f));
      }
    }
  }
  for (const auto& v : perc.activeVoxels) {
    if (v.wy > 2.2f || v.confidence <= 0.45f || v.sem == SemanticClass::SignalSign ||
        v.sem == SemanticClass::Overhang || v.sem == SemanticClass::Curb) continue;
    const float wx = safeVal(v.wx, ex, faultMaskOut);
    const float wz = safeVal(v.wz, ez, faultMaskOut);
    const float sV = sE * (wx - ex) + cE * (wz - ez) - bumperFwd;
    const float lV = cE * (wx - ex) - sE * (wz - ez);
    if (sV > 0.5f && sV < 24.0f && std::abs(lV) < 1.35f) {
      sameStaticGap = std::min(sameStaticGap, sV);
    }
  }

  float sigRisk = 0.0f;
  const SignalPhase sig = world.signalPhaseForLane(curLane);
  if (ez > 36.0f && ex < WorldEngine::kStopLineX + 1.5f && ex > WorldEngine::kStopLineX - 36.0f &&
      (sig == SignalPhase::Red || sig == SignalPhase::Yellow)) {
    sigRisk = clampf(1.0f - std::max(0.0f, WorldEngine::kStopLineX - (ex + bumperFwd)) / 24.0f,
                     0.2f, 1.0f);
  }
  if (ez < -36.0f && ex > 13.0f && ex < 40.0f && ego.passedStopSignCooldown <= 0.0f) {
    sigRisk = std::max(sigRisk, clampf(1.0f - std::max(0.0f, (ex + bumperFwd) - 14.5f) / 22.0f,
                                       0.2f, 1.0f));
  }
  if (perc.v15.sirenDetected && perc.v15.sirenConfidence > cfg.v15Policy.sirenConfThresh) {
    sigRisk = std::max(sigRisk, clampf(perc.v15.sirenConfidence, 0.4f, 1.0f));
  }

  int soiledFwd = 0;
  for (int c = 0; c < 3; ++c) {
    if (perc.soilCameras[c].meanSoilProb > 0.55f) ++soiledFwd;
  }
  const float rawSig = safeVal(perc.forwardEpistemicSigma, 0.0f, faultMaskOut);

  LatentVec16 raw{};
  raw[0] = clampf(espd / maxCruise, 0.0f, 1.5f);
  raw[1] = clampf(curLat / 1.95f, -1.5f, 1.5f);
  raw[2] = clampf(latHead, -1.5f, 1.5f);
  raw[3] = clampf(esteer / std::max(0.1f, cfg.maxSteerRad), -1.5f, 1.5f);
  raw[4] = clampf(1.0f - sameGap / 32.0f, 0.0f, 1.0f);
  raw[5] = clampf((espd - sameLeadSpd) / 12.0f, -1.0f, 1.5f);
  raw[6] = clampf(1.0f - sameStaticGap / 24.0f, 0.0f, 1.0f);
  raw[7] = clampf(sameLeadSpd / maxCruise, 0.0f, 1.5f);
  raw[8] = clampf(1.0f - adjFwdGap / 32.0f, 0.0f, 1.0f);
  raw[9] = clampf(adjOncoming, 0.0f, 1.5f);
  raw[10] = clampf(adjBlind, 0.0f, 1.0f);
  raw[11] = clampf(1.0f - std::max(raw[8], std::max(raw[9], raw[10])), -0.5f, 1.0f);
  raw[12] = clampf(vruThreat, 0.0f, 1.5f);
  raw[13] = clampf(sigRisk, 0.0f, 1.0f);
  raw[14] = clampf(rawSig + static_cast<float>(soiledFwd) * 0.25f, 0.0f, 1.5f);
  raw[15] = clampf(static_cast<float>(perc.tracks.size()) * 0.10f +
                   (perc.gaussianFormer3D.active ? perc.gaussianFormer3D.meanOpacity : 0.20f),
                   0.0f, 1.5f);
  if (rawFeaturesOut) *rawFeaturesOut = raw;

  for (size_t i = 0; i < kLeWmLatentDim; ++i) {
    float acc = 0.0f;
    for (size_t j = 0; j < kLeWmLatentDim; ++j) acc += weights.projW[i][j] * raw[j];
    const float standardized = (acc - weights.sigregMean[i]) * weights.sigregInvStd[i];
    if (!std::isfinite(standardized)) {
      faultMaskOut |= kWmFaultNanInput;
      zOut = {};
      return false;
    }
    zOut[i] = standardized;
  }
  return (faultMaskOut & kWmFaultNanInput) == 0u;
}

}  // namespace cybercab::sota
