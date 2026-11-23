#include "MetricsEvalHelpers.hpp"
#include "core/GpuProfiler.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <chrono>
#include <map>
#include <unordered_map>

namespace cybercab {
namespace {

using WorldModelAndLatencyTest = testutil::SharedGpuMetricsTest;

TEST_F(WorldModelAndLatencyTest, WorldModel4DRolloutAccuracyAndUncertaintyCalibration) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext& ctx = s_ctx;

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);

    const float dt = 1.0f / cfg.controlHz;
    struct Snap {
        double t = 0.0;
        std::unordered_map<std::string, OBB> map;
    };
    struct RollPred {
        double tIssue = 0.0;
        std::string gtId;
        float horizon = 0.0f;
        float px = 0.0f;
        float pz = 0.0f;
        float sigma = 0.0f;
    };
    std::vector<Snap> snaps;
    std::vector<RollPred> preds;
    PerceptionFrameOutput out;
    PlannerCommand cmd;

    for (int s = 0; s < 110; ++s) {
        const float simT = static_cast<float>(s) * dt;
        rig.renderAllCameras(world, cfg, simT, nullptr);
        perception.runFrame(ctx, rig, world, dt, true, out, nullptr);
        planner.plan(world, out, cmd);

        const auto dyn = world.dynamicGroundTruthOBBs();
        Snap sn;
        sn.t = simT;
        for (const auto& g : dyn) sn.map[g.id] = g;
        snaps.push_back(std::move(sn));

        if (s >= 4) {
            std::unordered_map<int, std::string> trkToGt;
            for (const auto& tr : out.tracks) {
                if (tr.state != TrackState::Tracked || tr.hits < 3) continue;
                float bd = (tr.semantic == SemanticClass::Pedestrian) ? 3.5f : 5.5f;
                std::string gid;
                for (const auto& g : dyn) {
                    if (g.semantic != tr.semantic) continue;
                    const float d = std::hypot(tr.worldPos.x - g.cx, tr.worldPos.z - g.cz);
                    if (d < bd) { bd = d; gid = g.id; }
                }
                if (!gid.empty()) trkToGt[tr.trackId] = gid;
            }
            for (const auto& gh : out.worldGhosts) {
                const int tid = static_cast<int>(std::round(gh.yawMeta.w));
                if (trkToGt.count(tid)) {
                    preds.push_back({simT, trkToGt[tid], gh.yawMeta.y,
                                     gh.worldPosScale.x, gh.worldPosScale.z, gh.yawMeta.z});
                }
            }
        }

        world.stepAgents(dt);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
    }

    struct HorStats {
        double errSum = 0.0;
        double sigSum = 0.0;
        int n = 0;
        int cov25 = 0;
    };
    std::map<int, HorStats> byHorMs;
    for (const auto& pr : preds) {
        const double tTgt = pr.tIssue + pr.horizon;
        for (const auto& sn : snaps) {
            if (std::abs(sn.t - tTgt) < 0.018 && sn.map.count(pr.gtId)) {
                const auto& gt = sn.map.at(pr.gtId);
                const float err = std::hypot(pr.px - gt.cx, pr.pz - gt.cz);
                auto& hs = byHorMs[static_cast<int>(std::round(pr.horizon * 1000.0f))];
                hs.errSum += err;
                hs.sigSum += pr.sigma;
                hs.n++;
                if (err <= 2.5f * pr.sigma) hs.cov25++;
                break;
            }
        }
    }

    ASSERT_GT(byHorMs[500].n, 20);
    ASSERT_GT(byHorMs[2000].n, 15);
    const double mae05 = byHorMs[500].errSum / byHorMs[500].n;
    const double mae10 = byHorMs[1000].errSum / byHorMs[1000].n;
    const double mae20 = byHorMs[2000].errSum / byHorMs[2000].n;
    const double sig05 = byHorMs[500].sigSum / byHorMs[500].n;
    const double sig20 = byHorMs[2000].sigSum / byHorMs[2000].n;
    const double cov20 = static_cast<double>(byHorMs[2000].cov25) / byHorMs[2000].n;

    RecordProperty("wm_mae_0_5s", testutil::fmtProp(mae05));
    RecordProperty("wm_mae_1_0s", testutil::fmtProp(mae10));
    RecordProperty("wm_mae_2_0s", testutil::fmtProp(mae20));
    RecordProperty("wm_cov25_2_0s", testutil::fmtProp(cov20));

    EXPECT_LT(mae05, 3.2);
    EXPECT_LT(mae10, 4.2);
    EXPECT_GT(sig20, sig05 + 0.8);
    EXPECT_GE(cov20, 0.65);

    perception.destroy();
    rig.destroy();
}

TEST_F(WorldModelAndLatencyTest, PipelineBottleneckAndFrameLatencySla) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext& ctx = s_ctx;

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    GpuProfiler profiler;
    profiler.init(ctx, 64);

    auto measureRes = [&](int wPx, int hPx, const char* tag,
                          double& totalWallMs, double& occReadMs, double& gpuSumMs) {
        if (rig.camWidth() != wPx || rig.camHeight() != hPx) {
            rig.resize(wPx, hPx);
            perception.onCameraResized(rig);
        }
        world.reset();
        perception.reset();
        const float dt = 1.0f / cfg.controlHz;
        PerceptionFrameOutput out;
        PlannerCommand cmd;

        double wallSum = 0.0, occSum = 0.0, gSum = 0.0;
        constexpr int kWarm = 5;
        constexpr int kRun = 25;
        for (int s = 0; s < kWarm + kRun; ++s) {
            const auto t0 = std::chrono::steady_clock::now();
            rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt, &profiler);
            perception.runFrame(ctx, rig, world, dt, true, out, &profiler);
            const auto tOcc0 = std::chrono::steady_clock::now();
            perception.occupancyFlow().readbackVoxels(cfg, world.ego(), out.activeVoxels, out.shadowVoxels);
            const auto tOcc1 = std::chrono::steady_clock::now();
            planner.plan(world, out, cmd);
            world.stepAgents(dt);
            world.stepEgoAndPhysics(dt, cmd, ManualInput{});
            const auto t1 = std::chrono::steady_clock::now();

            if (s >= kWarm) {
                wallSum += std::chrono::duration<double, std::milli>(t1 - t0).count();
                occSum += std::chrono::duration<double, std::milli>(tOcc1 - tOcc0).count();
                for (const auto& kv : out.gpuStageMs) gSum += kv.second;
            }
        }
        totalWallMs = wallSum / kRun;
        occReadMs = occSum / kRun;
        gpuSumMs = gSum / kRun;
        RecordProperty(std::string(tag) + "_total_wall_ms", testutil::fmtProp(totalWallMs));
        RecordProperty(std::string(tag) + "_occ_read_ms", testutil::fmtProp(occReadMs));
        RecordProperty(std::string(tag) + "_gpu_sum_ms", testutil::fmtProp(gpuSumMs));
    };

    double halfWall = 0.0, halfOcc = 0.0, halfGpu = 0.0;
    double fullWall = 0.0, fullOcc = 0.0, fullGpu = 0.0;
    measureRes(cfg.camWidth, cfg.camHeight, "half", halfWall, halfOcc, halfGpu);
    measureRes(cfg.fullCamWidth, cfg.fullCamHeight, "full", fullWall, fullOcc, fullGpu);

    EXPECT_LT(halfOcc, 0.25);
    EXPECT_LT(fullOcc, 0.25);
    EXPECT_LT(halfWall, 10.0);
    EXPECT_LT(fullWall, 16.0);

    profiler.destroy();
    perception.destroy();
    rig.destroy();
}

} // namespace
} // namespace cybercab
