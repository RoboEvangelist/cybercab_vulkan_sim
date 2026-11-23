#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/GaussianWorldModel.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cybercab {
namespace {
using namespace v15;

using V15GaussianWorldTest = testutil::SharedGpuMetricsTest;

TEST_F(V15GaussianWorldTest, GuardsEwaFiniteDiffRolloutAndGaussianFormerSuperposition) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    GaussianWorldModel gw;
    EXPECT_FLOAT_EQ(gw.queryOccupancyProb({0.0f, 1.0f, 0.0f}), 0.0f);
    std::vector<uint8_t> rgba(16, 1u);
    gw.downloadRealityViewRGBA(rgba);
    EXPECT_TRUE(rgba.empty());
    VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
    gw.dispatchRealitySplat(cmd, cfg, EgoState{});
    s_ctx.endSingleTimeCommands(cmd);

    const Gaussian3DGpu g1{{3.2f, 1.4f, 14.5f, 0.75f}, {0.95f, 0.72f, 2.20f, 1.0f}, {1.6f, -0.8f, 0.42f, 1.0f}, {0.9f, 0.2f, 0.2f, 0.9f}};
    const Mat4 camFromWorld = Mat4::translation({-0.3f, 1.1f, 0.5f}) * Mat4::rotY(-0.18f);
    const CameraIntrinsics intr = cfg.cameras[0].intrinsics(GaussianWorldModel::kRealityW, GaussianWorldModel::kRealityH);

    Vec2 uv{};
    float cov[3]{}, depth = 0.0f;
    GaussianWorldModel::projectEwa2D(g1, camFromWorld, intr, uv, cov, depth);
    const float det = cov[0] * cov[2] - cov[1] * cov[1];
    EXPECT_GT(det, 0.0f);

    auto projPt = [&](const Vec3& pw) -> Vec2 {
        const Vec3 pc = camFromWorld.transformPoint(pw);
        return {intr.cx + intr.fx * pc.x / pc.z, intr.cy + intr.fy * pc.y / pc.z};
    };
    const float sy = std::sin(g1.velYaw.z), cy = std::cos(g1.velYaw.z);
    const Vec3 axes[3] = {{cy * g1.scaleSem.x, 0.0f, -sy * g1.scaleSem.x}, {0.0f, g1.scaleSem.y, 0.0f}, {sy * g1.scaleSem.z, 0.0f, cy * g1.scaleSem.z}};
    const Vec3 ctr{g1.meanOpacity.x, g1.meanOpacity.y, g1.meanOpacity.z};
    constexpr float eps = 1e-3f;
    float fdCov[3]{0.3f, 0.0f, 0.3f};
    for (int k = 0; k < 3; ++k) {
        const Vec2 duv = (projPt(ctr + axes[k] * eps) - projPt(ctr - axes[k] * eps)) * (0.5f / eps);
        fdCov[0] += duv.x * duv.x;
        fdCov[1] += duv.x * duv.y;
        fdCov[2] += duv.y * duv.y;
    }
    double maxRel = 0.0;
    for (int i = 0; i < 3; ++i) {
        maxRel = std::max(maxRel, static_cast<double>(std::abs(cov[i] - fdCov[i]) / std::max(1.0f, std::abs(fdCov[i]))));
    }
    EXPECT_LT(maxRel, 1e-3);

    gw.setGaussiansForTest({g1});
    const float pSingle = gw.queryOccupancyProb(ctr, 0.0f);
    EXPECT_NEAR(pSingle, 0.75f, 1e-4f);
    gw.setGaussiansForTest({g1, g1});
    const float pSuper = gw.queryOccupancyProb(ctr, 0.0f);
    EXPECT_NEAR(pSuper, 1.0f - 0.25f * 0.25f, 1e-4f);
    EXPECT_GT(pSuper, pSingle);
    EXPECT_LE(pSuper, 1.0f);

    std::vector<Gaussian3DGpu> pred;
    gw.rollout4D(1.5f, pred);
    EXPECT_NEAR(pred[0].meanOpacity.x, g1.meanOpacity.x + 1.5f * g1.velYaw.x, 1e-4f);
    EXPECT_NEAR(pred[0].meanOpacity.z, g1.meanOpacity.z + 1.5f * g1.velYaw.y, 1e-4f);
    EXPECT_NEAR(pred[0].scaleSem.x, g1.scaleSem.x * 1.225f, 1e-4f);
    EXPECT_NEAR(gw.queryOccupancyProb({pred[0].meanOpacity.x, pred[0].meanOpacity.y, pred[0].meanOpacity.z}, 1.5f), pSuper, 1e-3f);
    RecordProperty("m6_ewa_fd_rel_err", testutil::fmtProp(maxRel));
}

TEST_F(V15GaussianWorldTest, GpuRealitySplatDepthSortsOccludesAndRunsInV15Pipeline) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    GaussianWorldModel gw;
    gw.initSplatGpu(s_ctx);
    // Insert far blue BEFORE near red to verify front-to-back depth sorting + occlusion.
    gw.setGaussiansForTest({
        {{0.0f, 0.0f, 15.0f, 0.98f}, {1.6f, 1.6f, 0.5f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.05f, 0.10f, 0.98f, 1.0f}},
        {{0.0f, 0.0f,  6.0f, 0.98f}, {1.2f, 1.2f, 0.5f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.98f, 0.05f, 0.05f, 1.0f}},
    });
    const CameraIntrinsics intr = cfg.cameras[0].intrinsics(GaussianWorldModel::kRealityW, GaussianWorldModel::kRealityH);
    VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
    gw.dispatchRealitySplat(cmd, Mat4::identity(), intr);
    s_ctx.endSingleTimeCommands(cmd);

    std::vector<uint8_t> rgba;
    gw.downloadRealityViewRGBA(rgba);
    ASSERT_EQ(rgba.size(), static_cast<size_t>(GaussianWorldModel::kRealityW * GaussianWorldModel::kRealityH * 4));
    const size_t ctrIdx = (static_cast<size_t>(GaussianWorldModel::kRealityH / 2) * GaussianWorldModel::kRealityW +
                           (GaussianWorldModel::kRealityW / 2)) * 4;
    EXPECT_GT(rgba[ctrIdx + 0], 210u);
    EXPECT_LT(rgba[ctrIdx + 2], 40u);
    gw.destroy();

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline perc(cfg);
    perc.init(s_ctx, rig);
    GpuProfiler profiler;
    profiler.init(s_ctx, 64);
    PerceptionFrameOutput out;
    const float dt = 1.0f / cfg.controlHz;
    for (int f = 0; f < 4; ++f) {
        rig.renderAllCameras(world, cfg, 0.05f * f, &profiler);
        perc.runFrame(s_ctx, rig, world, dt, true, out, &profiler);
        world.stepAgents(dt);
    }
    EXPECT_EQ(out.gpuStageMs.count("6_v15_gaussian_splat"), 1u);
    const auto& gwPipe = perc.v15()->gaussianWorld();
    EXPECT_FALSE(gwPipe.gaussians().empty());
    gwPipe.downloadRealityViewRGBA(rgba);
    uint8_t minR = 255u, maxR = 0u;
    for (size_t i = 0; i < rgba.size(); i += 4) {
        minR = std::min(minR, rgba[i]);
        maxR = std::max(maxR, rgba[i]);
    }
    EXPECT_GT(static_cast<int>(maxR) - static_cast<int>(minR), 10);
    RecordProperty("m6_gaussians", static_cast<int>(gwPipe.gaussians().size()));
    RecordProperty("m6_splat_ms", testutil::fmtProp(out.gpuStageMs.at("6_v15_gaussian_splat")));
    profiler.destroy();
    perc.destroy();
    rig.destroy();
}

} // namespace
} // namespace cybercab
