#pragma once
// FSD v15 Module 4: Self-supervised next-frame latent predictor & sparse/KV/FP16 compute telemetry.
#include "config/V15PolicyConfig.hpp"
#include "perception/v15/PatchTokenizer.hpp"
#include "perception/v15/StaticSpatialMemory.hpp"
#include "world/WorldTypes.hpp"
#include <vector>

namespace cybercab::v15 {

struct V15ComputeTelemetry {
    float tokenSparsityRatio = 0.0f;
    float staticSkipRatio = 0.0f;
    float kvCacheReuseRatio = 0.0f;
    float effectiveSpeedupX = 1.0f;
};

struct LatentEvalResult {
    float latentNovelty = 0.0f;
    float surpriseSigmaDelta = 0.0f;
    V15ComputeTelemetry telemetry{};
};

class LatentPredictor {
public:
    void reset();
    LatentEvalResult evaluate(const PatchTokenizer& patches, const StaticSpatialMemory& staticMem,
                              const V15PolicyConfig& policy = {});

    static LatentEvalResult evaluateTokens(const PatchTokenGpu* prevSlot, const PatchTokenGpu* currSlot,
                                           const EgoState& prevEgo, const EgoState& currEgo,
                                           uint32_t validSlots, float staticSkipRatio,
                                           const V15PolicyConfig& policy = {});
    const LatentEvalResult& lastResult() const { return last_; }

private:
    LatentEvalResult last_{};
    mutable std::vector<PatchTokenGpu> currScratch_;
    mutable std::vector<PatchTokenGpu> prevScratch_;
};

} // namespace cybercab::v15
