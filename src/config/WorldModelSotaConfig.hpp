#pragma once
// Stage 2/3 LeWorldModel (SIGReg latent dynamics, arXiv:2603.19312) and H-JEPA parameters.
#include <cmath>
#include <cstdint>
#include <string>

namespace YAML {
class Node;
}

namespace cybercab {

struct WorldModelSotaParams {
  uint32_t latentDim = 16u;
  uint32_t rolloutHorizonSteps = 6u;
  float stepDtSec = 0.25f;
  uint32_t macroHorizonSteps = 6u;
  float macroDtSec = 0.50f;
  uint32_t cemIterations = 3u;
  uint32_t cemCandidates = 9u;
  uint32_t cemElites = 3u;
  float sigregWeight = 0.25f;
  float sigregTargetVar = 1.0f;
  float surpriseSigmaGain = 0.30f;
  float surpriseBaseline = 0.32f;
  float energyCostWeight = 0.85f;
  float subgoalPriorWeight = 26.0f;
  float oncomingSafetyWeight = 180.0f;
  float comfortJerkWeight = 12.0f;
  bool enableFmeaFallback = true;
  std::string weightsPath;
};

inline bool isValidWorldModelMode(const std::string& mode) {
  return mode == "legacy" || mode == "lewm" || mode == "hjepa";
}

inline bool validateWorldModelSotaParams(const WorldModelSotaParams& p) {
  auto finNonNeg = [](float v) { return std::isfinite(v) && v >= 0.0f; };
  auto finPos = [](float v) { return std::isfinite(v) && v > 0.0f; };
  return p.latentDim >= 4u && p.latentDim <= 16u &&
         p.rolloutHorizonSteps >= 1u && p.rolloutHorizonSteps <= 8u &&
         p.macroHorizonSteps >= 1u && p.macroHorizonSteps <= 8u &&
         finPos(p.stepDtSec) && p.stepDtSec <= 2.0f &&
         finPos(p.macroDtSec) && p.macroDtSec <= 4.0f &&
         p.cemIterations >= 1u && p.cemIterations <= 6u &&
         p.cemCandidates >= 3u && p.cemCandidates <= 16u &&
         p.cemElites >= 1u && p.cemElites <= p.cemCandidates &&
         finNonNeg(p.sigregWeight) && finPos(p.sigregTargetVar) &&
         finNonNeg(p.surpriseSigmaGain) && finNonNeg(p.surpriseBaseline) &&
         finNonNeg(p.energyCostWeight) && finNonNeg(p.subgoalPriorWeight) &&
         finNonNeg(p.oncomingSafetyWeight) && finNonNeg(p.comfortJerkWeight);
}

void loadWorldModelSotaYaml(const YAML::Node& node, WorldModelSotaParams& dst);

}  // namespace cybercab
