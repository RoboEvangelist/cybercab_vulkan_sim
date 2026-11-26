#pragma once
// Stage-2 LeWorldModel (LeWM): SIGReg latent encoder, action-conditioned rollout & FMEA guards.
#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/sota/LeWmTypes.hpp"
#include "world/WorldEngine.hpp"
#include <array>
#include <cstdint>

namespace cybercab::sota {

class LeWmLatentModel {
public:
  void init(const SimConfig& cfg);
  void reset();

  bool encodeScene(const SimConfig& cfg, const WorldEngine& world,
                   const PerceptionFrameOutput& perc, LatentVec16& zOut,
                   uint32_t& faultMaskOut,
                   LatentVec16* rawFeaturesOut = nullptr) const;

  SigRegMetrics updateSigReg(const LatentVec16& z, float targetVar = 1.0f);

  bool predictNext(const LatentVec16& z, const LeWmAction& act, float dtSec,
                   LatentVec16& zNextOut) const;

  bool rolloutEnergy(const LatentVec16& z0, const LeWmAction& act,
                     uint32_t steps, float dtSec, float& energyOut,
                     LatentVec16* terminalZOut = nullptr) const;

  void stepFrame(const SimConfig& cfg, const WorldEngine& world,
                 const PerceptionFrameOutput& perc, float dtSec,
                 WorldModelSotaSummary& outSummary);

  const LeWmWeights& weights() const { return weights_; }
  const LatentVec16& lastLatent() const { return lastLatent_; }
  bool weightsLoadedFromFile() const { return weightsLoadedFromFile_; }
  uint32_t initFaultMask() const { return initFaultMask_; }

private:
  LeWmWeights weights_{};
  bool weightsLoadedFromFile_ = false;
  uint32_t initFaultMask_ = kWmFaultNone;

  std::array<LatentVec16, kLeWmSigRegWindow> latentRing_{};
  std::array<LatentVec16, kLeWmSigRegWindow> rawRing_{};
  uint32_t ringHead_ = 0u;
  uint32_t ringCount_ = 0u;

  bool hasPrevPred_ = false;
  LatentVec16 prevPredZ_{};
  LatentVec16 lastLatent_{};
  float prevRecLatM_ = -1.95f;
  float prevRecSpeedScale_ = 1.0f;

  static float latentNormSq(const LatentVec16& v);
  static float stateRiskEnergy(const LatentVec16& z, const LeWmAction& act);
  float computeRawRingVariance() const;
};

}  // namespace cybercab::sota
