#include "perception/sota/HJepaHierarchy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>

namespace cybercab::sota {
namespace {

constexpr float kRightLaneM = -1.95f;
constexpr float kLeftLaneM = 1.95f;
constexpr float kRightStayPrior = 0.0f;
constexpr float kLeftStayPrior = 18.0f;
constexpr float kSwitchLeftBaseCost = 4.0f;
constexpr float kReturnRightBonus = -12.0f;

bool checkRedOrStopHold(const WorldEngine& world, const EgoState& ego) {
  const float bumperFwd = ego.hz + ego.centerOffsetForward;
  const int lane = (ego.currentLateralOffset >= 0.0f) ? 1 : 0;
  const SignalPhase sig = world.signalPhaseForLane(lane);
  if (ego.pos.z > 36.0f && ego.pos.x < WorldEngine::kStopLineX + 1.5f &&
      ego.pos.x > WorldEngine::kStopLineX - 36.0f &&
      (sig == SignalPhase::Red || sig == SignalPhase::Yellow)) {
    return (WorldEngine::kStopLineX - (ego.pos.x + bumperFwd)) < 26.0f;
  }
  return (ego.pos.z < -36.0f && ego.pos.x > 13.0f && ego.pos.x < 40.0f &&
          ego.passedStopSignCooldown <= 0.0f);
}

}  // namespace

void HJepaHierarchy::reset() {
  prevRecLatM_ = kRightLaneM;
  prevRecSpeedScale_ = 1.0f;
}

MacroVec8 HJepaHierarchy::projectMacro(const LatentVec16& z, const LeWmWeights& w) {
  MacroVec8 s{};
  for (size_t r = 0; r < kLeWmMacroDim; ++r) {
    float acc = 0.0f;
    for (size_t c = 0; c < kLeWmLatentDim; ++c) acc += w.macroW[r][c] * z[c];
    s[r] = acc;
  }
  return s;
}

MacroSubgoalResult HJepaHierarchy::planMacroSubgoal(const SimConfig& cfg,
                                                    const EgoState& ego,
                                                    const LatentVec16& z,
                                                    const LeWmLatentModel& model,
                                                    bool redOrStop,
                                                    bool sirenYield) {
  MacroSubgoalResult best{};
  best.valid = false;
  if (!std::isfinite(ego.speed) || !std::isfinite(ego.currentLateralOffset)) return best;
  const MacroVec8 s = projectMacro(z, model.weights());
  for (float v : s) {
    if (!std::isfinite(v)) return best;
  }

  const float maxCruise = std::max(1.0f, cfg.targetCruiseSpeedMs);
  const float curLat = clampf(z[1] * 1.95f, -2.5f, 2.5f);
  const int curLane = (curLat >= 0.0f) ? 1 : 0;
  const float sameBlock = clampf(s[2], 0.0f, 1.5f);
  const float adjThreat = clampf(s[3], 0.0f, 2.0f);
  const float vruSig = clampf(s[5], 0.0f, 2.0f);

  struct CandDef {
    MacroSubgoal code;
    float latM;
    float spdMs;
    float prior;
  };
  const std::array<CandDef, 5> defs{{
    {MacroSubgoal::KeepLane, (curLane == 1) ? kLeftLaneM : kRightLaneM,
     maxCruise * clampf(1.0f - 0.30f * std::max(0.0f, z[4]), 0.65f, 1.0f),
     (curLane == 1) ? kLeftStayPrior : kRightStayPrior},
    {MacroSubgoal::OvertakeLeft, kLeftLaneM, maxCruise,
     (curLane == 0) ? kSwitchLeftBaseCost : 12.0f},
    {MacroSubgoal::ReturnRight, kRightLaneM, maxCruise,
     (curLane == 1) ? kReturnRightBonus : 35.0f},
    {MacroSubgoal::YieldSlow,
     sirenYield ? cfg.v15Policy.sirenRightPullM
                : ((curLane == 1) ? kLeftLaneM : kRightLaneM),
     maxCruise * 0.58f, 16.0f},
    {MacroSubgoal::StopHold, (curLane == 1) ? kLeftLaneM : kRightLaneM, 0.0f, 28.0f},
  }};

  float bestCost = std::numeric_limits<float>::infinity();
  for (const auto& d : defs) {
    const LeWmAction act{
      clampf((d.spdMs - ego.speed) * 0.65f, -4.5f, 2.5f),
      clampf((d.latM - curLat) * 0.55f, -1.6f, 1.6f)};
    float rollE = 0.0f;
    if (!model.rolloutEnergy(z, act, cfg.wmSota.macroHorizonSteps,
                             cfg.wmSota.macroDtSec, rollE)) {
      return best;
    }
    float cost = d.prior + cfg.wmSota.energyCostWeight * 24.0f * rollE;
    if (d.code == MacroSubgoal::KeepLane && curLane == 0 && z[4] > 0.55f) {
      cost += 42.0f * (z[4] - 0.40f);  // tactical travel-time delay behind close blocked lead
    } else if (d.code == MacroSubgoal::OvertakeLeft) {
      if (z[9] > 0.15f || z[10] > 0.25f || adjThreat > 0.45f) {
        cost += cfg.wmSota.oncomingSafetyWeight * (0.4f + z[9] + z[10]);
      }
      if (sameBlock < 0.35f) cost += 22.0f;  // no reason to leave clear right lane
      if (redOrStop || sirenYield) cost += 250.0f;
    } else if (d.code == MacroSubgoal::ReturnRight) {
      if (z[4] > 0.40f || adjThreat > 0.45f) cost += 60.0f;
    } else if (d.code == MacroSubgoal::YieldSlow) {
      if (sirenYield || z[14] > 0.65f || vruSig > 0.45f ||
          (sameBlock > 0.50f && (z[9] > 0.20f || adjThreat > 0.40f))) {
        cost -= 26.0f;
      }
    } else if (d.code == MacroSubgoal::StopHold) {
      if (redOrStop || z[12] > 0.72f || (z[4] > 0.78f && z[5] > 0.20f)) {
        cost -= 42.0f;
      }
    }
    if (cost < bestCost) {
      bestCost = cost;
      best = {d.code, d.latM, d.spdMs, rollE, true};
    }
  }
  return best;
}

CemPlanResult HJepaHierarchy::optimizeActionCem(const SimConfig& cfg,
                                                const EgoState& ego,
                                                const LatentVec16& z,
                                                const MacroSubgoalResult& macro,
                                                const LeWmLatentModel& model,
                                                float prevRecLatM,
                                                float prevRecSpeedScale,
                                                uint32_t& faultMaskOut) {
  CemPlanResult out{};
  out.valid = false;
  if (!macro.valid || !validateWorldModelSotaParams(cfg.wmSota)) {
    faultMaskOut |= kWmFaultInvalidConfig;
    return out;
  }

  const float maxCruise = std::max(1.0f, cfg.targetCruiseSpeedMs);
  const float curLat = clampf(z[1] * 1.95f, -2.5f, 2.5f);
  float muLat = clampf(0.5f * (curLat + macro.targetLatM), -2.25f, 2.25f);
  float muSpd = clampf(macro.targetSpeedMs / maxCruise, 0.55f, 1.0f);
  float sigLat = 1.10f, sigSpd = 0.22f;

  struct EvalPt {
    float cost = 1e9f;
    float latM = -1.95f;
    float spdScale = 1.0f;
  };
  constexpr uint32_t kMaxCands = 16u;
  std::array<EvalPt, kMaxCands> pts{};
  const uint32_t nCand = std::min(kMaxCands, std::max(3u, cfg.wmSota.cemCandidates));
  const uint32_t nElites = std::min(nCand, std::max(1u, cfg.wmSota.cemElites));
  const uint32_t nIters = std::max(1u, cfg.wmSota.cemIterations);

  EvalPt globalBest{std::numeric_limits<float>::infinity(), macro.targetLatM, muSpd};
  for (uint32_t it = 0u; it < nIters; ++it) {
    const uint32_t side = (nCand >= 9u) ? 3u : nCand;
    for (uint32_t c = 0u; c < nCand; ++c) {
      const float uL = (static_cast<float>(c % side) / std::max(1.0f, static_cast<float>(side - 1u))) * 2.0f - 1.0f;
      const float uS = (static_cast<float>(c / side) / std::max(1.0f, static_cast<float>((nCand - 1u) / side))) * 2.0f - 1.0f;
      const float candLat = clampf(muLat + uL * sigLat, -2.25f, 2.25f);
      const float candScale = clampf(muSpd + uS * sigSpd, 0.55f, 1.0f);
      const float tgtVel = candScale * maxCruise;
      const LeWmAction act{
        clampf((tgtVel - ego.speed) * 0.8f, -5.0f, 3.0f),
        clampf((candLat - curLat) * 0.9f, -1.8f, 1.8f)};
      float rollE = 0.0f;
      if (!model.rolloutEnergy(z, act, cfg.wmSota.rolloutHorizonSteps,
                               cfg.wmSota.stepDtSec, rollE)) {
        faultMaskOut |= kWmFaultRolloutDivergence;
        return out;
      }
      const float cost =
        cfg.wmSota.energyCostWeight * 20.0f * rollE +
        cfg.wmSota.subgoalPriorWeight * std::abs(candLat - macro.targetLatM) +
        cfg.wmSota.comfortJerkWeight *
          (std::abs(candLat - prevRecLatM) * 0.5f + std::abs(candScale - prevRecSpeedScale) * 2.0f) +
        (candLat > 0.3f && z[9] > 0.15f ? cfg.wmSota.oncomingSafetyWeight : 0.0f);
      pts[c] = {cost, candLat, candScale};
    }
    std::partial_sort(pts.begin(), pts.begin() + nElites, pts.begin() + nCand,
                      [](const EvalPt& a, const EvalPt& b) { return a.cost < b.cost; });
    if (it == 0u) out.iter0BestCost = pts[0].cost;
    if (pts[0].cost < globalBest.cost) globalBest = pts[0];
    float sumL = 0.0f, sumS = 0.0f;
    for (uint32_t e = 0u; e < nElites; ++e) {
      sumL += pts[e].latM;
      sumS += pts[e].spdScale;
    }
    muLat = sumL / static_cast<float>(nElites);
    muSpd = sumS / static_cast<float>(nElites);
    sigLat *= 0.45f;
    sigSpd *= 0.45f;
  }
  out.finalBestCost = globalBest.cost;
  out.recommendedLatM = globalBest.latM;
  out.recommendedSpeedScale = globalBest.spdScale;
  out.valid = std::isfinite(out.finalBestCost) && std::isfinite(out.recommendedLatM) &&
              std::isfinite(out.recommendedSpeedScale);
  return out;
}

void HJepaHierarchy::runHierarchy(const SimConfig& cfg, const WorldEngine& world,
                                  const PerceptionFrameOutput& perc,
                                  const LeWmLatentModel& model,
                                  const LatentVec16& z,
                                  WorldModelSotaSummary& ioSummary) {
  const auto t0 = std::chrono::steady_clock::now();
  const EgoState& ego = world.ego();
  const bool redOrStop = checkRedOrStopHold(world, ego);
  const bool sirenYield =
    perc.v15.sirenDetected && perc.v15.sirenConfidence > cfg.v15Policy.sirenConfThresh;

  uint32_t fault = ioSummary.fmeaFaultMask;
  const MacroSubgoalResult macro =
    planMacroSubgoal(cfg, ego, z, model, redOrStop, sirenYield);
  if (!macro.valid) {
    fault |= kWmFaultRolloutDivergence;
  } else {
    const CemPlanResult cem = optimizeActionCem(
      cfg, ego, z, macro, model, prevRecLatM_, prevRecSpeedScale_, fault);
    if (cem.valid) {
      ioSummary.hierarchical = true;
      ioSummary.macroSubgoalCode = static_cast<int>(macro.code);
      ioSummary.macroSubgoalLatM = macro.targetLatM;
      ioSummary.macroSubgoalSpeedMs = macro.targetSpeedMs;
      ioSummary.bestRolloutEnergy = cem.finalBestCost;
      ioSummary.recommendedLatM = cem.recommendedLatM;
      ioSummary.recommendedSpeedScale = cem.recommendedSpeedScale;
    }
  }
  ioSummary.fmeaFaultMask = fault;
  if (cfg.wmSota.enableFmeaFallback &&
      (fault & (kWmFaultNanInput | kWmFaultLatentCollapse |
                kWmFaultRolloutDivergence | kWmFaultInvalidConfig)) != 0u) {
    ioSummary.fmeaFallbackTriggered = true;
    ioSummary.recommendedLatM = kRightLaneM;
    ioSummary.recommendedSpeedScale = 1.0f;
    ioSummary.surpriseSigmaDelta = 0.0f;
  } else {
    prevRecLatM_ = ioSummary.recommendedLatM;
    prevRecSpeedScale_ = ioSummary.recommendedSpeedScale;
  }
  const auto t1 = std::chrono::steady_clock::now();
  ioSummary.stageCpuMs +=
    std::chrono::duration<float, std::milli>(t1 - t0).count();
}

}  // namespace cybercab::sota
