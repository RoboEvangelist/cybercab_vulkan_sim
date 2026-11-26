#pragma once
// Stage-2 AD-E2E-JEPA compressed scene feature extraction & SIGReg latent projection.
#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/sota/LeWmTypes.hpp"
#include "world/WorldEngine.hpp"
#include <cstdint>

namespace cybercab::sota {

bool encodeSceneLatent16(const SimConfig& cfg,
                         const WorldEngine& world,
                         const PerceptionFrameOutput& perc,
                         const LeWmWeights& weights,
                         LatentVec16& zOut,
                         uint32_t& faultMaskOut,
                         LatentVec16* rawFeaturesOut = nullptr);

}  // namespace cybercab::sota
