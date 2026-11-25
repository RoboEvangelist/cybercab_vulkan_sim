#include "MetricsEvalHelpers.hpp"
#include "core/GpuProfiler.hpp"
#include "core/TraceLogger.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <gtest/gtest.h>
#include <json/json.h>
#include <malloc.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace cybercab::sota {
namespace {

using GaussianFormer3DAbMissionsNfrTest = testutil::SharedGpuMetricsTest;

constexpr float kMinNomDistM = 30.0f;
constexpr float kMaxLatDevM = 0.35f;
constexpr float kMaxComfortAcc = 12.0f;

struct AbMissionMetrics {
    std::string mode;
    std::string spatial;
    bool passed = false;
    bool collision = false;
    bool seenSiren = false;
    float distM = 0.0f;
    float maxLatDevM = 0.0f;
    float maxAbsAcc = 0.0f;
    float meanAbsJerk = 0.0f;
    float minDynClearM = 99.0f;
    float minSpeedMs = 99.0f;
    float meanRssSafeM = 0.0f;
    int firstBrakeStep = -1;
    int aebFrames = 0;
};

struct AbLatencyMetrics {
    std::string res;
    std::string spatial;
    double meanWallMs = 0.0;
    double p50WallMs = 0.0;
    double p95WallMs = 0.0;
    double maxWallMs = 0.0;
    double meanGpuMs = 0.0;
    double meanGf3dCpuMs = 0.0;
};

SimConfig makeNfrCfg(const std::string& pipeVer, const std::string& spatial, const std::string& res = "half") {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = pipeVer;
    cfg.spatialRepMode = spatial;
    cfg.initialResMode = res;
    const CamResPreset p = cfg.resolveInitialRes();
    cfg.camWidth = p.width;
    cfg.camHeight = p.height;
    return cfg;
}

float minDynamicObbClearanceM(const WorldEngine& w, std::vector<OBB>& gtScratch) {
    const OBB ego = w.getEgoWorldOBB();
    w.collectDynamicGroundTruthOBBs(gtScratch);
    float best = 99.0f;
    for (const auto& b : gtScratch) {
        const float gx = std::max(0.0f, std::abs(ego.cx - b.cx) - (ego.hx + b.hx));
        const float gz = std::max(0.0f, std::abs(ego.cz - b.cz) - (ego.hz + b.hz));
        best = std::min(best, std::hypot(gx, gz));
    }
    return best;
}

AbMissionMetrics runMissionCase(VulkanContext& ctx,
                                const std::string& pipeVer,
                                const std::string& spatial,
                                const std::string& mode,
                                int steps) {
    SimConfig cfg = makeNfrCfg(pipeVer, spatial);
    if (mode == "night_glare_jaywalk") {
        cfg.lightingMode = "night_glare";
        cfg.photonRccbMosaic = true;
    }
    WorldEngine w(cfg);
    if (mode == "jaywalk" || mode == "night_glare_jaywalk") w.triggerJaywalkerCutIn();
    else if (mode == "occluded") w.triggerOccludedDartOut();
    else if (mode == "hardbrake") w.triggerLeadHardBrake();
    else if (mode == "siren") w.triggerEmergencySiren();

    CameraRigRenderer rig; rig.init(ctx, cfg, w);
    PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    PerceptionFrameOutput out{}; PlannerCommand cmd{};
    std::vector<OBB> gtScratch;
    const float dt = 1.0f / cfg.controlHz;

    AbMissionMetrics r{};
    r.mode = pipeVer + ":" + mode;
    r.spatial = spatial;
    Vec3 prevPos = w.ego().pos;
    float prevAccel = w.ego().accel;
    double jerkSum = 0.0, rssSum = 0.0;

    for (int s = 0; s < steps; ++s) {
        rig.renderAllCameras(w, cfg, static_cast<float>(s) * dt);
        pipe.runFrame(ctx, rig, w, dt, true, out);
        planner.plan(w, out, cmd);
        w.stepAgents(dt);
        w.stepEgoAndPhysics(dt, cmd, {});

        if (w.lastContact().occurred) r.collision = true;
        if (cmd.behaviorState == "YIELDING_SIREN") r.seenSiren = true;
        if (cmd.aebActive) ++r.aebFrames;
        if (r.firstBrakeStep < 0 && (cmd.aebActive || cmd.targetSpeed < w.ego().speed - 0.5f)) {
            r.firstBrakeStep = s;
        }
        const EgoState& e = w.ego();
        r.distM += std::hypot(e.pos.x - prevPos.x, e.pos.z - prevPos.z);
        prevPos = e.pos;
        r.maxAbsAcc = std::max(r.maxAbsAcc, std::abs(e.accel));
        if (s > 0) jerkSum += std::abs(e.accel - prevAccel) / dt;
        prevAccel = e.accel;
        r.minSpeedMs = std::min(r.minSpeedMs, e.speed);
        r.minDynClearM = std::min(r.minDynClearM, minDynamicObbClearanceM(w, gtScratch));
        rssSum += cmd.rssSafeDist;
        if (cmd.behaviorState == "CRUISING" && cmd.gear == 1) {
            const float latDev = std::abs(std::abs(e.currentLateralOffset) - cfg.v15Policy.laneCenterAbsM);
            r.maxLatDevM = std::max(r.maxLatDevM, latDev);
        }
    }
    r.meanAbsJerk = (steps > 1) ? static_cast<float>(jerkSum / (steps - 1)) : 0.0f;
    r.meanRssSafeM = static_cast<float>(rssSum / steps);
    const bool nomOk = (r.distM > kMinNomDistM) && (pipeVer != "v15" || r.maxLatDevM < kMaxLatDevM);
    r.passed = !r.collision && (r.maxAbsAcc <= kMaxComfortAcc) &&
               (mode != "nominal" || nomOk) && (mode != "siren" || r.seenSiren);
    pipe.destroy(); rig.destroy();
    return r;
}

TEST_F(GaussianFormer3DAbMissionsNfrTest, ABSixMissionSafetyAndDecisionComparison) {
    struct Spec { const char* pipe; const char* mode; int steps; };
    const Spec specs[] = {
        {"v15", "nominal", 180},
        {"v15", "jaywalk", 60},
        {"v15", "occluded", 60},
        {"v15", "hardbrake", 60},
        {"v15", "siren", 60},
        {"v15", "night_glare_jaywalk", 60},
        {"v14", "nominal", 180},
        {"v14", "jaywalk", 60},
    };
    int legPass = 0, gfPass = 0, tot = 0, legCol = 0, gfCol = 0;
    for (const auto& sp : specs) {
        const AbMissionMetrics l = runMissionCase(s_ctx, sp.pipe, "legacy", sp.mode, sp.steps);
        const AbMissionMetrics g = runMissionCase(s_ctx, sp.pipe, "gaussianformer3d", sp.mode, sp.steps);
        ++tot;
        if (l.passed) ++legPass;
        if (g.passed) ++gfPass;
        if (l.collision) ++legCol;
        if (g.collision) ++gfCol;

        std::printf("[AB_MISSIONS] %-22s | pass=%d/%d col=%d/%d dist=%.1f/%.1f maxAcc=%.2f/%.2f jerk=%.2f/%.2f minClr=%.2f/%.2f brkStep=%d/%d aeb=%d/%d minV=%.2f/%.2f rss=%.2f/%.2f\n",
                    l.mode.c_str(), l.passed, g.passed, l.collision, g.collision,
                    l.distM, g.distM, l.maxAbsAcc, g.maxAbsAcc, l.meanAbsJerk, g.meanAbsJerk,
                    l.minDynClearM, g.minDynClearM, l.firstBrakeStep, g.firstBrakeStep,
                    l.aebFrames, g.aebFrames, l.minSpeedMs, g.minSpeedMs, l.meanRssSafeM, g.meanRssSafeM);

        EXPECT_TRUE(l.passed) << l.mode << " legacy";
        EXPECT_TRUE(g.passed) << g.mode << " gf3d";
    }
    RecordProperty("legacy_missions_passed", std::to_string(legPass) + "/" + std::to_string(tot));
    RecordProperty("gf3d_missions_passed", std::to_string(gfPass) + "/" + std::to_string(tot));
    EXPECT_EQ(legPass, tot);
    EXPECT_EQ(gfPass, tot);
    EXPECT_EQ(legCol, 0);
    EXPECT_EQ(gfCol, 0);
}

AbLatencyMetrics benchLatency(VulkanContext& ctx, const std::string& res, const std::string& spatial) {
    const SimConfig cfg = makeNfrCfg("v15", spatial, res);
    WorldEngine w(cfg);
    CameraRigRenderer rig; rig.init(ctx, cfg, w);
    PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
    GpuProfiler prof; prof.init(ctx, 64);
    PerceptionFrameOutput out{};
    const float dt = 1.0f / cfg.controlHz;

    constexpr int kWarm = 4, kMeas = 20;
    std::vector<double> wallSamples;
    wallSamples.reserve(kMeas);
    double gpuSum = 0.0, gfStageSum = 0.0;

    std::vector<ActiveVoxel> rawVoxels;
    for (int i = 0; i < kWarm + kMeas; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        rig.renderAllCameras(w, cfg, static_cast<float>(i) * dt, &prof);
        pipe.runFrame(ctx, rig, w, dt, true, out, &prof);
        const auto t1 = std::chrono::steady_clock::now();
        if (i == kWarm) {
            pipe.occupancyFlow().readbackVoxels(cfg, w.ego(), rawVoxels, out.shadowVoxels);
        }
        if (i >= kWarm) {
            wallSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            gpuSum += out.eval.totalPipelineMs;
        }
        w.stepAgents(dt);
    }

    if (pipe.gaussianFormer3D()) {
        auto* mutablePipe = const_cast<GaussianFormer3DPipeline*>(pipe.gaussianFormer3D());
        const auto g0 = std::chrono::steady_clock::now();
        for (int k = 0; k < kMeas; ++k) {
            out.activeVoxels = rawVoxels;
            mutablePipe->runStage(cfg, w.ego(), &rig, &pipe.v15()->staticMemory(), out);
        }
        const auto g1 = std::chrono::steady_clock::now();
        gfStageSum = std::chrono::duration<double, std::milli>(g1 - g0).count() / kMeas;
    }

    std::sort(wallSamples.begin(), wallSamples.end());
    AbLatencyMetrics m{};
    m.res = res;
    m.spatial = spatial;
    m.meanWallMs = std::accumulate(wallSamples.begin(), wallSamples.end(), 0.0) / wallSamples.size();
    m.p50WallMs = wallSamples[wallSamples.size() / 2];
    m.p95WallMs = wallSamples[std::min(wallSamples.size() - 1, (wallSamples.size() * 95) / 100)];
    m.maxWallMs = wallSamples.back();
    m.meanGpuMs = gpuSum / kMeas;
    m.meanGf3dCpuMs = gfStageSum;

    prof.destroy(); pipe.destroy(); rig.destroy();
    return m;
}

TEST_F(GaussianFormer3DAbMissionsNfrTest, ABNonFunctionalLatencyDeterminismAndTraceAudit) {
    const char* presets[] = {"half", "full", "cybercab_binned"};
    for (const char* res : presets) {
        const AbLatencyMetrics l = benchLatency(s_ctx, res, "legacy");
        const AbLatencyMetrics g = benchLatency(s_ctx, res, "gaussianformer3d");
        std::printf("[AB_LATENCY] %-16s | wallMean=%.2f/%.2fms p50=%.2f/%.2fms p95=%.2f/%.2fms max=%.2f/%.2fms | gpu=%.2f/%.2fms | gf3dCpu=%.3fms\n",
                    res, l.meanWallMs, g.meanWallMs, l.p50WallMs, g.p50WallMs, l.p95WallMs, g.p95WallMs,
                    l.maxWallMs, g.maxWallMs, l.meanGpuMs, g.meanGpuMs, g.meanGf3dCpuMs);
        EXPECT_LT(g.p95WallMs, 100.0) << res;
    }

    // Heap allocation & capacity stability over steady-state Stage 1 calls.
    const SimConfig cfg = makeNfrCfg("v15", "gaussianformer3d", "half");
    EgoState ego{};
    GaussianFormer3DPipeline stage1;
    stage1.init(cfg);
    PerceptionFrameOutput out{};
    out.activeVoxels.reserve(static_cast<size_t>(cfg.occTotalVoxels()));
    out.reserveV15Scratch(1024u, 320u * 180u * 4u);
    auto seed = [&]() {
        out.tracks.resize(1u);
        out.tracks[0].state = TrackState::Tracked;
        out.tracks[0].semantic = SemanticClass::Vehicle;
        out.tracks[0].worldPos = {12.0f, 0.7f, 0.0f};
        out.tracks[0].halfExtents = {1.0f, 0.72f, 2.1f};
        out.tracks[0].score = 0.9f;
        out.activeVoxels.assign(2u, ActiveVoxel{12.0f, 0.0f, 0.75f, 12.0f, 0.0f, SemanticClass::Vehicle, 0.85f, 4.0f, 0.0f, 0.2f});
    };
    auto getHeapUsedBytes = []() -> long {
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
        return static_cast<long>(::mallinfo2().uordblks);
#else
        return static_cast<long>(::mallinfo().uordblks);
#endif
    };
    for (int w = 0; w < 3; ++w) { seed(); stage1.runStage(cfg, ego, nullptr, nullptr, out); }
    const long m0Bytes = getHeapUsedBytes();
    const size_t cPrim = stage1.primitives().capacity(), cV15 = stage1.v15Gaussians().capacity();
    for (int i = 0; i < 50; ++i) {
        seed(); stage1.runStage(cfg, ego, nullptr, nullptr, out);
        EXPECT_EQ(stage1.primitives().capacity(), cPrim);
        EXPECT_EQ(stage1.v15Gaussians().capacity(), cV15);
    }
    const long heapDelta = getHeapUsedBytes() - m0Bytes;
    std::printf("[AB_NFR] stage1_heap_delta_bytes=%ld prim_mem_bytes=%zu\n",
                heapDelta, sizeof(Gaussian3DPrimitive) * stage1.config().maxGaussians);
    EXPECT_EQ(heapDelta, 0L);

    // Determinism & paired trace audit (12 steps).
    auto runTraceRun = [&](const std::string& spatial, std::vector<std::string>& jsons,
                           std::vector<PlannerCommand>& cmds, std::vector<float>& steers, Vec3& finalPos) {
        const SimConfig c = makeNfrCfg("v15", spatial, "half");
        WorldEngine w(c);
        CameraRigRenderer rig; rig.init(s_ctx, c, w);
        PerceptionPipeline pipe(c); pipe.init(s_ctx, rig);
        TrajectoryPlanner planner(c);
        PerceptionFrameOutput po{}; PlannerCommand pc{};
        const float dt = 1.0f / c.controlHz;
        jsons.clear(); cmds.clear(); steers.clear();
        for (int s = 0; s < 12; ++s) {
            rig.renderAllCameras(w, c, static_cast<float>(s) * dt);
            pipe.runFrame(s_ctx, rig, w, dt, true, po);
            planner.plan(w, po, pc);
            jsons.push_back(TraceLogger::formatFrameJson(s, s * dt, c, w.ego(), pc, po, w.lastContact()));
            cmds.push_back(pc);
            steers.push_back(w.ego().steerAngle);
            w.stepAgents(dt);
            w.stepEgoAndPhysics(dt, pc, {});
        }
        finalPos = w.ego().pos;
        pipe.destroy(); rig.destroy();
    };

    std::vector<std::string> gfRunA, gfRunB, legRun;
    std::vector<PlannerCommand> gfCmdA, gfCmdB, legCmd;
    std::vector<float> gfSteerA, gfSteerB, legSteer;
    Vec3 gfPosA{}, gfPosB{}, legPos{};
    runTraceRun("gaussianformer3d", gfRunA, gfCmdA, gfSteerA, gfPosA);
    runTraceRun("gaussianformer3d", gfRunB, gfCmdB, gfSteerB, gfPosB);
    runTraceRun("legacy", legRun, legCmd, legSteer, legPos);

    EXPECT_FLOAT_EQ(gfPosA.x, gfPosB.x);
    EXPECT_FLOAT_EQ(gfPosA.z, gfPosB.z);
    int behDiff = 0;
    float maxSpdDiff = 0.0f, maxSteerDiff = 0.0f, maxRssDiff = 0.0f;
    for (size_t s = 0; s < gfRunA.size(); ++s) {
        EXPECT_EQ(gfRunA[s], gfRunB[s]);
        if (gfCmdA[s].behaviorState != legCmd[s].behaviorState) ++behDiff;
        maxSpdDiff = std::max(maxSpdDiff, std::abs(gfCmdA[s].targetSpeed - legCmd[s].targetSpeed));
        maxSteerDiff = std::max(maxSteerDiff, std::abs(gfSteerA[s] - legSteer[s]));
        maxRssDiff = std::max(maxRssDiff, std::abs(gfCmdA[s].rssSafeDist - legCmd[s].rssSafeDist));

        Json::CharReaderBuilder rb;
        Json::Value jG, jL;
        std::string errs;
        std::istringstream isG(gfRunA[s]), isL(legRun[s]);
        ASSERT_TRUE(Json::parseFromStream(rb, isG, &jG, &errs)) << errs;
        ASSERT_TRUE(Json::parseFromStream(rb, isL, &jL, &errs)) << errs;
        EXPECT_TRUE(jG.isMember("gaussianformer3d"));
        EXPECT_FALSE(jL.isMember("gaussianformer3d"));
    }
    std::printf("[AB_TRACE] det_exact=1 behavior_diff_frames=%d/12 max_spd_delta=%.4fm/s max_steer_delta=%.4frad max_rss_delta=%.4fm\n",
                behDiff, maxSpdDiff, maxSteerDiff, maxRssDiff);
}

} // namespace
} // namespace cybercab::sota
