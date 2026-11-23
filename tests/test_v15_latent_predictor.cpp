#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/LatentPredictor.hpp"

#include <algorithm>
#include <vector>

namespace cybercab {
namespace {
using namespace v15;

using V15LatentPredTest = testutil::SharedGpuMetricsTest;

TEST(V15LatentPredUnitTest, GuardsTelemetryAndSuddenCorridorNoveltyJump) {
    LatentPredictor pred;
    PatchTokenizer emptyTok;
    StaticSpatialMemory emptyMem;
    EXPECT_FLOAT_EQ(pred.evaluate(emptyTok, emptyMem).latentNovelty, 0.0f);
    EXPECT_FLOAT_EQ(LatentPredictor::evaluateTokens(nullptr, nullptr, EgoState{}, EgoState{}, 0u, 0.0f).surpriseSigmaDelta, 0.0f);

    std::vector<PatchTokenGpu> prev(PatchTokenizer::kTokensPerFrame);
    for (auto& t : prev) {
        t.app0 = {0.20f, 0.22f, 0.21f, 1.0f};
        t.app1 = {0.21f, 0.05f, 0.01f, 0.01f};
    }
    auto currCalm = prev;
    const EgoState e0{};
    const LatentEvalResult calm = LatentPredictor::evaluateTokens(prev.data(), currCalm.data(), e0, e0, 4u, 0.70f);
    EXPECT_NEAR(calm.telemetry.kvCacheReuseRatio, 0.75f, 1e-4f);
    EXPECT_NEAR(calm.telemetry.staticSkipRatio, 0.70f, 1e-4f);
    EXPECT_GT(calm.telemetry.effectiveSpeedupX, 2.0f);
    EXPECT_LT(calm.latentNovelty, 0.05f);
    EXPECT_LT(calm.surpriseSigmaDelta, 0.05f);

    // Inject unmodeled sudden hazard into central corridor of forward views 1 and 8.
    auto currHazard = prev;
    for (int v : {1, 8}) {
        const size_t base = static_cast<size_t>(v) * PatchTokenizer::kGridW * PatchTokenizer::kGridH;
        for (int gy = 10; gy <= 15; ++gy) {
            for (int gx = 12; gx <= 17; ++gx) {
                auto& t = currHazard[base + static_cast<size_t>(gy) * PatchTokenizer::kGridW + gx];
                t.app0 = {0.95f, 0.45f, 0.10f, 1.0f};
                t.app1 = {0.75f, 0.55f, 0.35f, -0.35f};
            }
        }
    }
    const LatentEvalResult spike = LatentPredictor::evaluateTokens(prev.data(), currHazard.data(), e0, e0, 4u, 0.70f);
    EXPECT_GT(spike.latentNovelty, 0.45f);
    EXPECT_GT(spike.surpriseSigmaDelta, 0.20f);
    pred.reset();
    EXPECT_FLOAT_EQ(pred.lastResult().latentNovelty, 0.0f);
}

TEST_F(V15LatentPredTest, PipelineComputesTelemetryAndDetectsJaywalkerSurprise) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline perc(cfg);
    perc.init(s_ctx, rig);

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    for (int f = 0; f < 5; ++f) {
        rig.renderAllCameras(world, cfg, 0.05f * f, nullptr);
        perc.runFrame(s_ctx, rig, world, dt, true, out, nullptr);
        world.stepAgents(dt);
    }
    EXPECT_TRUE(out.v15.active);
    EXPECT_NEAR(out.v15.kvCacheReuseRatio, 0.75f, 1e-4f);
    EXPECT_GT(out.v15.effectiveSpeedupX, 2.0f);
    EXPECT_GT(out.v15.rearClearanceM, 5.0f);
    EXPECT_LT(out.v15.surpriseSigmaDelta, 0.12f);
    const float nomNovelty = out.v15.latentNovelty;
    const float nomSigma = out.forwardEpistemicSigma;

    world.triggerJaywalkerCutIn();
    float peakNovelty = 0.0f, peakSigma = 0.0f;
    for (int f = 0; f < 6; ++f) {
        rig.renderAllCameras(world, cfg, 0.25f + 0.05f * f, nullptr);
        perc.runFrame(s_ctx, rig, world, dt, true, out, nullptr);
        peakNovelty = std::max(peakNovelty, out.v15.latentNovelty);
        peakSigma = std::max(peakSigma, out.forwardEpistemicSigma);
        world.stepAgents(dt);
    }
    EXPECT_GE(peakNovelty, nomNovelty);
    EXPECT_GE(peakSigma, nomSigma);

    RecordProperty("m4_nom_novelty", testutil::fmtProp(nomNovelty));
    RecordProperty("m4_jaywalk_novelty", testutil::fmtProp(peakNovelty));
    RecordProperty("m4_kv_reuse", testutil::fmtProp(out.v15.kvCacheReuseRatio));
    RecordProperty("m4_speedup_x", testutil::fmtProp(out.v15.effectiveSpeedupX));
    perc.destroy();
    rig.destroy();
}

} // namespace
} // namespace cybercab
