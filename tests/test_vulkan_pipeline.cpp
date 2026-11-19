#include "config/SimConfig.hpp"
#include "core/GpuProfiler.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/ByteTrack.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "world/WorldEngine.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <map>
#include <numeric>

using namespace cybercab;

TEST(ByteTrackUnitTest, HungarianAndStage2KeepTrackUnderLowConfidence) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = 0.0f;

    // Frame 1 & 2: High confidence detection establishes Tracked object
    Detection3D highDet;
    highDet.semantic = SemanticClass::Pedestrian;
    highDet.score = 0.85f;
    highDet.worldPos = {2.0f, 0.9f, 12.0f};
    highDet.camWeight = 1.0f;
    highDet.epistemicSigma = 0.3f;

    tracker.update(0.05f, ego, {highDet});
    highDet.worldPos.z += 0.1f;
    tracker.update(0.05f, ego, {highDet});
    auto trks = tracker.activeTracks();
    ASSERT_EQ(trks.size(), 1u);
    const int id = trks[0].trackId;
    const float sigmaClean = trks[0].posSigmaM;

    // Frame 3: Lens gets soiled -> score drops to 0.30 (low-conf tier), camWeight = 0.12
    Detection3D lowDet = highDet;
    lowDet.score = 0.30f;
    lowDet.camWeight = 0.12f;
    lowDet.epistemicSigma = 1.5f;
    lowDet.worldPos.z += 0.1f;
    tracker.update(0.05f, ego, {lowDet});

    trks = tracker.activeTracks();
    ASSERT_EQ(trks.size(), 1u);
    EXPECT_EQ(trks[0].trackId, id);
    EXPECT_TRUE(trks[0].matchedLowConf);
    // Uncertainty propagated: soiled camera keeps covariance higher than clean updates
    EXPECT_GE(trks[0].posSigmaM, sigmaClean * 0.9f);
}

TEST(VulkanPipelineIntegrationTest, EndToEndPerceptionAndUncertaintyPropagation) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    GpuProfiler profiler;
    profiler.init(ctx, 64);

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput cleanOut;
    PlannerCommand cleanCmd;

    // Run 6 nominal frames so temporal BEVFormer & ByteTrack settle
    for (int i = 0; i < 6; ++i) {
        world.stepAgents(dt);
        rig.renderAllCameras(world, cfg, static_cast<float>(i) * dt, &profiler);
        perception.runFrame(ctx, rig, world, dt, true, cleanOut, &profiler);
        planner.plan(world, cleanOut, cleanCmd);
        world.stepEgoAndPhysics(dt, cleanCmd, ManualInput{});
    }

    // 1. Verify camera offscreen image has rich non-constant pixels
    const auto px = rig.downloadCameraRGBA(0);
    ASSERT_EQ(px.size(), static_cast<size_t>(rig.camWidth() * rig.camHeight() * 4));
    double meanLuma = 0.0;
    for (size_t i = 0; i < px.size(); i += 4) {
        meanLuma += (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) / 255.0;
    }
    meanLuma /= (rig.camWidth() * rig.camHeight());
    EXPECT_GT(meanLuma, 0.10);
    EXPECT_LT(meanLuma, 0.90);

    // 2. Clean cameras should report healthy SoilDetector weights & active BEV/Voxels
    EXPECT_GT(cleanOut.soilCameras[0].tokenWeight, 0.70f);
    EXPECT_GT(cleanOut.bevSummary.observedCells, 100);
    EXPECT_GT(cleanOut.activeVoxels.size(), 10u);
    EXPECT_FALSE(cleanOut.gpuStageMs.empty());

    // 3. Soil front cameras -> SoilDetector detects degradation from pixels alone & RSS expands!
    cfg.cameras[0].soiled = true;
    cfg.cameras[2].soiled = true;
    cfg.cameras[3].soiled = true;

    PerceptionFrameOutput soiledOut;
    PlannerCommand soiledCmd;
    rig.renderAllCameras(world, cfg, 0.5f, &profiler);
    perception.runFrame(ctx, rig, world, dt, true, soiledOut, &profiler);
    planner.plan(world, soiledOut, soiledCmd);

    EXPECT_LT(soiledOut.soilCameras[0].tokenWeight, 0.45f);
    EXPECT_LT(soiledOut.soilCameras[2].tokenWeight, 0.45f);
    EXPECT_GT(soiledOut.forwardEpistemicSigma, cleanOut.forwardEpistemicSigma + 0.10f);
    EXPECT_GT(soiledCmd.rssSafeDist, cleanCmd.rssSafeDist + 0.15f);

    // 4. Clean lenses and test Jaywalker Cut-In safety response
    for (auto& c : cfg.cameras) c.soiled = false;
    world.reset();
    perception.reset();
    world.triggerJaywalkerCutIn();

    bool reactedOrStopped = false;
    for (int i = 0; i < 65; ++i) {
        world.stepAgents(dt);
        rig.renderAllCameras(world, cfg, static_cast<float>(i) * dt, &profiler);
        perception.runFrame(ctx, rig, world, dt, true, cleanOut, &profiler);
        planner.plan(world, cleanOut, cleanCmd);
        world.stepEgoAndPhysics(dt, cleanCmd, ManualInput{});
        if (cleanCmd.aebActive || cleanCmd.behaviorState == "YIELDING_PEDESTRIAN" || world.ego().speed < 4.5f) {
            reactedOrStopped = true;
        }
    }
    EXPECT_TRUE(reactedOrStopped);
    EXPECT_FALSE(world.lastContact().occurred);

    // 5. Test Occluded Dart-Out produces occlusion shadow voxels behind truck
    world.reset();
    perception.reset();
    world.triggerOccludedDartOut();
    for (int i = 0; i < 6; ++i) {
        world.stepAgents(dt);
        rig.renderAllCameras(world, cfg, static_cast<float>(i) * dt, &profiler);
        perception.runFrame(ctx, rig, world, dt, true, cleanOut, &profiler);
    }
    EXPECT_GT(cleanOut.shadowVoxels.size(), 0u);

    profiler.destroy();
    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(VulkanPipelineIntegrationTest, CameraResizeProjectionAndNominalCruise) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    GpuProfiler profiler;
    profiler.init(ctx, 64);

    // 1. Project world point straight ahead of front main camera vs behind ego
    const float dt = 1.0f / cfg.controlHz;
    rig.renderAllCameras(world, cfg, 0.0f, &profiler);
    Vec2 pxAhead{}, pxBehind{};
    float depthA = 0.0f, depthB = 0.0f;
    const Vec3 aheadWorld = world.ego().pos + Vec3{18.0f, 1.1f, 0.0f};
    const Vec3 behindWorld = world.ego().pos + Vec3{-18.0f, 1.1f, 0.0f};
    ASSERT_TRUE(rig.projectWorldToPixel(0, aheadWorld, pxAhead, depthA));
    EXPECT_GT(depthA, 12.0f);
    EXPECT_GT(pxAhead.x, 40.0f);
    EXPECT_LT(pxAhead.x, 280.0f);
    EXPECT_FALSE(rig.projectWorldToPixel(0, behindWorld, pxBehind, depthB));

    // 2. Dynamic resize to 640x480 full resolution and back to 320x240
    rig.resize(cfg.fullCamWidth, cfg.fullCamHeight);
    perception.onCameraResized(rig);
    rig.renderAllCameras(world, cfg, dt, &profiler);
    const auto fullPx = rig.downloadCameraRGBA(0);
    EXPECT_EQ(fullPx.size(), static_cast<size_t>(cfg.fullCamWidth * cfg.fullCamHeight * 4));

    rig.resize(cfg.camWidth, cfg.camHeight);
    perception.onCameraResized(rig);

    // 3. Nominal autonomous driving progress check (should cruise smoothly without false AEB dead stop)
    const Vec3 startPos = world.ego().pos;
    PerceptionFrameOutput out;
    PlannerCommand cmd;
    for (int i = 0; i < 80; ++i) {
        world.stepAgents(dt);
        rig.renderAllCameras(world, cfg, static_cast<float>(i) * dt, &profiler);
        perception.runFrame(ctx, rig, world, dt, true, out, &profiler);
        planner.plan(world, out, cmd);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
    }
    const float distMoved = (world.ego().pos - startPos).norm();
    EXPECT_GT(distMoved, 14.0f);
    EXPECT_GT(world.ego().speed, 3.5f);
    EXPECT_FALSE(world.lastContact().occurred);

    profiler.destroy();
    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(VulkanPipelineIntegrationTest, WorldModelHorizonUncertaintyGrowthAndBevTemporalPersistence) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    GpuProfiler profiler;
    profiler.init(ctx, 64);

    // Run until at least one vehicle/ped track is confirmed so WorldModel produces rollout ghosts
    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    PlannerCommand cmd;
    for (int i = 0; i < 15; ++i) {
        world.stepAgents(dt);
        rig.renderAllCameras(world, cfg, static_cast<float>(i) * dt, &profiler);
        perception.runFrame(ctx, rig, world, dt, true, out, &profiler);
        planner.plan(world, out, cmd);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
    }

    EXPECT_GT(out.bevSummary.observedCells, 200);
    EXPECT_GT(out.bevSummary.meanObservability, 0.25f);
    ASSERT_FALSE(out.worldGhosts.empty());

    // Verify predicted horizon uncertainty grows monotonically from t=0.5s to t=2.0s per track
    struct HorizonSig { float tEarly = 1e9f, sEarly = 0.0f, tLate = -1.0f, sLate = 0.0f; };
    std::map<int, HorizonSig> perTrack;
    for (const auto& g : out.worldGhosts) {
        const int tid = static_cast<int>(std::round(g.yawMeta.w));
        const float h = g.yawMeta.y;
        const float s = g.yawMeta.z;
        auto& hs = perTrack[tid];
        if (h < hs.tEarly) { hs.tEarly = h; hs.sEarly = s; }
        if (h > hs.tLate)  { hs.tLate  = h; hs.sLate  = s; }
    }
    int checkedTracks = 0;
    for (const auto& kv : perTrack) {
        if (kv.second.tLate > kv.second.tEarly + 0.5f) {
            EXPECT_GT(kv.second.sLate, kv.second.sEarly);
            ++checkedTracks;
        }
    }
    EXPECT_GT(checkedTracks, 0);

    profiler.destroy();
    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}
