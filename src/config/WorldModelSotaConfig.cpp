#include "config/WorldModelSotaConfig.hpp"
#include <yaml-cpp/yaml.h>

namespace cybercab {
namespace {

template <typename T>
T readWm(const YAML::Node& node, const char* key, const T& fallback) {
  if (node && node[key]) {
    return node[key].as<T>();
  }
  return fallback;
}

}  // namespace

void loadWorldModelSotaYaml(const YAML::Node& node, WorldModelSotaParams& dst) {
  if (!node) return;
  dst.latentDim = readWm(node, "latent_dim", dst.latentDim);
  dst.rolloutHorizonSteps =
    readWm(node, "rollout_horizon_steps", dst.rolloutHorizonSteps);
  dst.stepDtSec = readWm(node, "step_dt_sec", dst.stepDtSec);
  dst.macroHorizonSteps =
    readWm(node, "macro_horizon_steps", dst.macroHorizonSteps);
  dst.macroDtSec = readWm(node, "macro_dt_sec", dst.macroDtSec);
  dst.cemIterations = readWm(node, "cem_iterations", dst.cemIterations);
  dst.cemCandidates = readWm(node, "cem_candidates", dst.cemCandidates);
  dst.cemElites = readWm(node, "cem_elites", dst.cemElites);
  dst.sigregWeight = readWm(node, "sigreg_weight", dst.sigregWeight);
  dst.sigregTargetVar = readWm(node, "sigreg_target_var", dst.sigregTargetVar);
  dst.surpriseSigmaGain =
    readWm(node, "surprise_sigma_gain", dst.surpriseSigmaGain);
  dst.surpriseBaseline =
    readWm(node, "surprise_baseline", dst.surpriseBaseline);
  dst.energyCostWeight =
    readWm(node, "energy_cost_weight", dst.energyCostWeight);
  dst.subgoalPriorWeight =
    readWm(node, "subgoal_prior_weight", dst.subgoalPriorWeight);
  dst.oncomingSafetyWeight =
    readWm(node, "oncoming_safety_weight", dst.oncomingSafetyWeight);
  dst.comfortJerkWeight =
    readWm(node, "comfort_jerk_weight", dst.comfortJerkWeight);
  dst.enableFmeaFallback =
    readWm(node, "enable_fmea_fallback", dst.enableFmeaFallback);
  dst.weightsPath = readWm<std::string>(node, "weights_path", dst.weightsPath);
}

}  // namespace cybercab
