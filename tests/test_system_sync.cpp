#include <gtest/gtest.h>

#include "config/SimConfig.hpp"
#include "core/Math.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/BEVFormer.hpp"
#include "perception/ByteTrack.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/SimulationSync.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "TestPerceptionHelpers.hpp"
#include "world/RoadCurve.hpp"
#include "world/WorldEngine.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace cybercab {
namespace {

TEST(SystemSyncTest, CoordinateConventionsAndPlannerLaneParity) {
    const SimConfig cfg = SimConfig::defaultConfig();

    // 1. Verify BEVFormer ISO-Ego <-> World matrix matches WorldEngine vehicle matrix & ByteTrack TrackPrior projection across headings.
    for (const float yaw : {0.0f, kPi * 0.5f, -kPi * 0.5f, kPi, 0.73f}) {
        EgoState ego{};
        ego.pos = {12.5f, 0.0f, -8.25f};
        ego.yaw = yaw;

        const Mat4 bevWorldFromIso = BEVFormer::worldFromIsoEgo(ego);
        const Vec3 isoTestPt{14.0f, 3.5f, 1.2f}; // +14m fwd, +3.5m left, +1.2m up
        const Vec3 bevW = bevWorldFromIso.transformPoint(isoTestPt);

        Mat4 worldFromVeh = Mat4::translation(ego.pos) * Mat4::rotY(ego.yaw);
        const Vec3 vehW = worldFromVeh.transformPoint(isoEgoToVehicleLocal(isoTestPt));
        EXPECT_NEAR(bevW.x, vehW.x, 1e-4f);
        EXPECT_NEAR(bevW.y, vehW.y, 1e-4f);
        EXPECT_NEAR(bevW.z, vehW.z, 1e-4f);

        // Round-trip via ByteTracker GPU export projection
        ByteTracker tracker(cfg);
        Detection3D d{};
        d.semantic = SemanticClass::Vehicle;
        d.score = 0.95f;
        d.worldPos = bevW;
        tracker.update(0.03f, ego, {d});
        tracker.update(0.03f, ego, {d});
        std::vector<TrackPriorGpu> priors;
        tracker.exportTrackPriorsGpu(ego, priors);
        ASSERT_GT(priors[0].yawValid.w, 0.5f);
        EXPECT_NEAR(priors[0].posVel.x, isoTestPt.x, 0.25f) << "yaw=" << yaw;
        EXPECT_NEAR(priors[0].posVel.y, isoTestPt.y, 0.25f) << "yaw=" << yaw;
    }

    // 2. Verify WorldEngine::resetEgo() starts on Lane 0 (-1.9m outer lane at z = 45.9)
    // and TrajectoryPlanner keeps empty-road cruising centered in Lane 0 (NOT crossing to Lane 1 at z = 42.1!).
    WorldEngine world(cfg);
    TrajectoryPlanner planner(cfg);
    const EgoState& startEgo = world.ego();
    EXPECT_NEAR(startEgo.pos.z, WorldEngine::kLoopHalfZ + 1.9f, 1e-3f);

    const PerceptionFrameOutput emptyPerc{};
    PlannerCommand cruiseCmd;
    planner.plan(world, emptyPerc, cruiseCmd);
    EXPECT_EQ(cruiseCmd.behaviorState, "CRUISING");
    ASSERT_FALSE(cruiseCmd.optimalTrajectory.empty());
    EXPECT_NEAR(cruiseCmd.optimalTrajectory.front().z, WorldEngine::kLoopHalfZ + 1.9f, 0.25f);
    EXPECT_NEAR(cruiseCmd.optimalTrajectory.back().z, WorldEngine::kLoopHalfZ + 1.9f, 0.35f)
        << "Empty-road CRUISING drifted into inner lane z=" << cruiseCmd.optimalTrajectory.back().z;

    // 3. When Lane 0 ahead is blocked outside emergency RSS distance, planner MUST overtake left onto Lane 1 (z = 42.1f) with LANE_CHANGE_OVERTAKE.
    PerceptionFrameOutput blockedPerc;
    Track3D lead;
    lead.trackId = 1;
    lead.semantic = SemanticClass::Vehicle;
    lead.state = TrackState::Tracked;
    lead.score = 0.95f;
    lead.worldPos = {startEgo.pos.x + 22.0f, 0.72f, startEgo.pos.z};
    lead.halfExtents = {0.96f, 0.72f, 2.2f};
    blockedPerc.tracks.push_back(lead);

    PlannerCommand overtakeCmd;
    planner.plan(world, blockedPerc, overtakeCmd);
    EXPECT_EQ(overtakeCmd.behaviorState, "LANE_CHANGE_OVERTAKE");
    EXPECT_NEAR(overtakeCmd.optimalTrajectory.back().z, WorldEngine::kLoopHalfZ - 1.9f, 0.45f)
        << "Overtake failed to select inner passing lane z=" << (WorldEngine::kLoopHalfZ - 1.9f);
}

TEST(SystemSyncTest, VulkanValidationAndMultiFrameResizeBarriersClean) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(true, nullptr);

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    PlannerCommand cmd;

    for (int step = 0; step < 5; ++step) {
        world.stepAgents(dt);
        rig.renderAllCameras(world, cfg, static_cast<float>(step) * dt, nullptr);
        perception.runFrame(ctx, rig, world, dt, true, out, nullptr);
        planner.plan(world, out, cmd);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
    }

    // Exercise dynamic camera resolution change + descriptor rebinding under validation
    rig.resize(cfg.fullCamWidth, cfg.fullCamHeight);
    perception.onCameraResized(rig);
    EXPECT_EQ(rig.camWidth(), cfg.fullCamWidth);
    EXPECT_EQ(rig.camHeight(), cfg.fullCamHeight);

    for (int step = 5; step < 10; ++step) {
        world.stepAgents(dt);
        rig.renderAllCameras(world, cfg, static_cast<float>(step) * dt, nullptr);
        perception.runFrame(ctx, rig, world, dt, true, out, nullptr);
        planner.plan(world, out, cmd);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
    }

    if (ctx.validationEnabled()) {
        EXPECT_EQ(ctx.validationIssueCount(), 0u)
            << "Vulkan validation layers reported warnings/errors during multi-frame render/compute/resize";
    }
    EXPECT_FALSE(std::isnan(out.forwardEpistemicSigma));
    EXPECT_GT(out.bevSummary.observedCells, 100);

    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(SystemSyncTest, ColdStartFrameZeroHasZeroBogusMotionAndMatchesSteadyEmptyScene) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    testutil::IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    testutil::fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    const size_t totalCells = static_cast<size_t>(8 * cfg.featWidth * cfg.featHeight);
    std::vector<Vec4> s4(totalCells * 4u);
    h.detr.featS4SSBO().downloadVector(s4);

    float maxMotFrame0 = 0.0f;
    for (size_t i = 0; i < totalCells; ++i) {
        maxMotFrame0 = std::max(maxMotFrame0, s4[i * 4u + 2u].z); // f2.z == mMot
    }
    EXPECT_LT(maxMotFrame0, 0.01f)
        << "Frame 0 cold-start produced bogus optical flow mMot=" << maxMotFrame0 << " against zero-initialized prevCamTensor";

    // Frame 1 with identical image + re-check static motion, then draw moving target on Frame 2.
    testutil::drawCamRect(h.synthPixelsScratch, 0, w, hPx, 70, 125, 110, 165, {0.88f, 0.11f, 0.28f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);
    h.runDetr(h.world->ego(), out2D, out3D);

    testutil::fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);
    testutil::drawCamRect(h.synthPixelsScratch, 0, w, hPx, 150, 125, 190, 165, {0.88f, 0.11f, 0.28f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);
    h.runDetr(h.world->ego(), out2D, out3D);
    h.detr.featS4SSBO().downloadVector(s4);

    float maxMovedCellMot = 0.0f;
    float maxStaticSkyMot = 0.0f;
    for (int fy = 0; fy < cfg.featHeight; ++fy) {
        for (int fx = 0; fx < cfg.featWidth; ++fx) {
            const size_t idx = (static_cast<size_t>(fy) * cfg.featWidth + fx) * 4u + 2u; // cam 0
            if (fy < 6) {
                maxStaticSkyMot = std::max(maxStaticSkyMot, s4[idx].z);
            } else {
                maxMovedCellMot = std::max(maxMovedCellMot, s4[idx].z);
            }
        }
    }
    EXPECT_GT(maxMovedCellMot, 0.20f) << "True motion across frames did not appear in mMot";
    EXPECT_LT(maxStaticSkyMot, 0.01f);

    // Rebinding/resizing camera tensor must reset prev-frame gate so post-resize Frame 0 also has zero bogus motion.
    testutil::fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);
    h.uploadSyntheticPixels(h.synthPixelsScratch);
    h.detr.updateCameraTensorBinding(h.synthCamTensorSSBO, w, hPx);
    h.runDetr(h.world->ego(), out2D, out3D);
    h.detr.featS4SSBO().downloadVector(s4);

    float maxMotPostReset = 0.0f;
    for (size_t i = 0; i < totalCells; ++i) {
        maxMotPostReset = std::max(maxMotPostReset, s4[i * 4u + 2u].z);
    }
    EXPECT_LT(maxMotPostReset, 0.01f)
        << "Post-resize/rebind Frame 0 leaked bogus motion mMot=" << maxMotPostReset;
}

TEST(SystemSyncTest, VariableDtSynchronizesAgentsEgoAndKalmanVelocity) {
    const SimConfig cfg = SimConfig::defaultConfig();
    const float dtControl = 1.0f / cfg.controlHz;

    WorldEngine refWorld(cfg);
    WorldEngine varWorld(cfg);
    PlannerCommand cmd{};
    cmd.targetSpeed = 8.5f;

    // Step reference world for 54 uniform control ticks (1.500s total).
    ByteTracker refTracker(cfg);
    EgoState stationaryEgo{};
    stationaryEgo.pos = {0.0f, 0.0f, 0.0f};
    stationaryEgo.yaw = 0.0f;
    const Vec3 trueVel{8.0f, 0.0f, -3.0f};
    const Vec3 p0{5.0f, 0.72f, 20.0f};

    for (int i = 0; i < 54; ++i) {
        const float t = static_cast<float>(i) * dtControl;
        refTracker.update(dtControl, stationaryEgo,
                          {testutil::makeDet3D(SemanticClass::Vehicle, p0 + trueVel * t, 0.95f)});
        refWorld.stepAgents(dtControl);
        refWorld.stepEgoAndPhysics(dtControl, cmd, ManualInput{});
    }

    // Step variable-FPS world using SimulationSyncClock across 3x cycles of sub-frame & dropped multi-step intervals
    // (54 * dtControl = 1.500s total), verifying exact ego+agent sync and ByteTrack velocity invariance.
    SimulationSyncClock clock;
    ByteTracker tracker(cfg);

    const std::vector<float> cycleDts = {
        0.010f, 0.010f, 0.050f, 0.005f, 0.045f, 0.048f,
        0.014f, 0.014f, 0.052f, 0.028f, 0.050f, 0.040f,
        0.015f, 0.049f, 0.036f, 0.034f
    }; // One cycle = 0.500s (= 18 * dtControl at 36 Hz)

    float agentClockAtPerception = 0.0f;
    for (int cycle = 0; cycle < 3; ++cycle) {
        for (const float fdt : cycleDts) {
            clock.stepFrame(fdt, dtControl, 0.06f,
                [&](float simTime, float percDt) {
                    const float worldTime = static_cast<float>(clock.stepIndex) * dtControl;
                    EXPECT_NEAR(simTime, worldTime, 1e-4f);
                    agentClockAtPerception = simTime;

                    const Detection3D det = testutil::makeDet3D(
                        SemanticClass::Vehicle, p0 + trueVel * simTime, 0.95f);
                    tracker.update(percDt, stationaryEgo, {det});
                },
                [&](int, float, float tickDt) {
                    varWorld.stepAgents(tickDt);
                    varWorld.stepEgoAndPhysics(tickDt, cmd, ManualInput{});
                });
        }
    }

    ASSERT_EQ(clock.stepIndex, 54);
    EXPECT_NEAR(varWorld.ego().pos.x, refWorld.ego().pos.x, 2e-3f);
    EXPECT_NEAR(varWorld.ego().pos.z, refWorld.ego().pos.z, 2e-3f);

    const auto refGt = refWorld.dynamicGroundTruthOBBs();
    const auto varGt = varWorld.dynamicGroundTruthOBBs();
    ASSERT_EQ(refGt.size(), varGt.size());
    for (size_t i = 0; i < refGt.size(); ++i) {
        EXPECT_NEAR(varGt[i].cx, refGt[i].cx, 2e-3f) << refGt[i].id;
        EXPECT_NEAR(varGt[i].cz, refGt[i].cz, 2e-3f) << refGt[i].id;
    }

    const float tailDt = 1.50f - agentClockAtPerception;
    if (tailDt > 1e-4f) {
        const Detection3D finalDet = testutil::makeDet3D(
            SemanticClass::Vehicle, p0 + trueVel * 1.50f, 0.95f);
        tracker.update(tailDt, stationaryEgo, {finalDet});
    }
    const auto tracks = tracker.activeTracks();
    const auto refTracks = refTracker.activeTracks();
    ASSERT_EQ(tracks.size(), 1u);
    ASSERT_EQ(refTracks.size(), 1u);
    EXPECT_NEAR(tracks[0].worldVel.x, trueVel.x, 0.35f);
    EXPECT_NEAR(tracks[0].worldVel.z, trueVel.z, 0.35f);
    EXPECT_NEAR(tracks[0].worldVel.x, refTracks[0].worldVel.x, 0.35f);
    EXPECT_NEAR(tracks[0].worldVel.z, refTracks[0].worldVel.z, 0.35f);
}

TEST(SystemSyncTest, CameraTensorDataQualityAndSprungProjectionParity) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    // Inject non-zero chassis pitch/roll (hard braking + cornering) to test sprung extrinsics sync.
    world.mutableEgo().pitch = -0.055f;
    world.mutableEgo().roll = 0.035f;

    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    rig.renderAllCameras(world, cfg, 0.25f, nullptr);

    const int camW = rig.camWidth();
    const int camH = rig.camHeight();
    const size_t perCamPx = static_cast<size_t>(camW) * camH;
    std::vector<Vec4> tensor(8u * perCamPx);
    rig.cameraTensorSSBO().downloadVector(tensor);

    for (int c = 0; c < 8; ++c) {
        double lumSum = 0.0;
        double lumSqSum = 0.0;
        for (size_t i = 0; i < perCamPx; ++i) {
            const Vec4& px = tensor[static_cast<size_t>(c) * perCamPx + i];
            ASSERT_FALSE(std::isnan(px.x) || std::isnan(px.y) || std::isnan(px.z) || std::isnan(px.w));
            ASSERT_FALSE(std::isinf(px.x) || std::isinf(px.y) || std::isinf(px.z) || std::isinf(px.w));
            EXPECT_GE(px.x, 0.0f); EXPECT_LE(px.x, 1.0f);
            EXPECT_GE(px.y, 0.0f); EXPECT_LE(px.y, 1.0f);
            EXPECT_GE(px.z, 0.0f); EXPECT_LE(px.z, 1.0f);
            EXPECT_FLOAT_EQ(px.w, 1.0f);
            const double lum = 0.299 * px.x + 0.587 * px.y + 0.114 * px.z;
            lumSum += lum;
            lumSqSum += lum * lum;
        }
        const double meanLum = lumSum / static_cast<double>(perCamPx);
        const double varLum = std::max(0.0, lumSqSum / static_cast<double>(perCamPx) - meanLum * meanLum);
        EXPECT_GT(std::sqrt(varLum), 0.03) << "Camera " << c << " appears uniform/dead";
    }

    // Verify 3D ground-point -> pixel (with Brown-Conrady distortion & sprung chassis) -> undistorted 3D ray inverse
    // recovers true world points within < 0.05m across Front Main (0), Front Wide (2), Pillar R (4), Rear (7).
    const EgoState& ego = world.ego();
    const Mat4 worldFromIso = BEVFormer::worldFromIsoEgo(ego);
    struct GroundProbe { int cam; Vec3 isoPt; };
    const GroundProbe probes[] = {
        {0, {18.0f,  0.0f, 0.0f}},
        {2, { 9.0f,  5.5f, 0.0f}},
        {4, { 6.0f, -8.5f, 0.0f}},
        {7, {-10.0f, 0.0f, 0.0f}},
    };
    for (const auto& pr : probes) {
        const Vec3 ptW = worldFromIso.transformPoint(pr.isoPt);
        Vec2 px{};
        float depth = 0.0f;
        ASSERT_TRUE(rig.projectWorldToPixel(pr.cam, ptW, px, depth)) << "cam=" << pr.cam;
        EXPECT_GT(px.x, 5.0f);
        EXPECT_LT(px.x, static_cast<float>(camW) - 5.0f);
        EXPECT_GT(px.y, 5.0f);
        EXPECT_LT(px.y, static_cast<float>(camH) - 5.0f);

        const auto& cm = rig.cameraMatrices(pr.cam);
        const auto& spec = cfg.cameras.at(pr.cam);
        const Vec2 xu = undistortNormalized((px.x - cm.intr.cx) / cm.intr.fx,
                                            (px.y - cm.intr.cy) / cm.intr.fy,
                                            spec.distortion);
        const Vec3 camOrigW = cm.worldFromCam.transformPoint({0.0f, 0.0f, 0.0f});
        const Vec3 rayDirW = cm.worldFromCam.transformVector({xu.x, xu.y, 1.0f}).normalized();
        ASSERT_LT(rayDirW.y, -1e-3f);
        const float tGround = -camOrigW.y / rayDirW.y;
        const Vec3 liftedW = camOrigW + rayDirW * tGround;
        EXPECT_NEAR(liftedW.x, ptW.x, 0.05f) << "cam=" << pr.cam;
        EXPECT_NEAR(liftedW.z, ptW.z, 0.05f) << "cam=" << pr.cam;
    }

    // Verify per-camera physical ISP soil flag propagates accurately through SoilDetector.
    cfg.cameras[2].soiled = true;
    SoilDetector soil;
    soil.init(ctx, cfg, rig.cameraTensorSSBO());
    rig.renderAllCameras(world, cfg, 0.25f, nullptr);
    {
        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        soil.recordDispatch(cmd, cfg, camW, camH);
        ctx.endSingleTimeCommands(cmd);
    }
    std::array<SoilCameraSummary, 8> soilCams{};
    std::vector<SoilPatchStats> soilPatches;
    float fwdSigma = 0.0f;
    soil.readbackResults(soilCams, soilPatches, fwdSigma);
    EXPECT_GT(soilCams[2].meanSoilProb, 0.55f);
    EXPECT_LT(soilCams[2].tokenWeight, 0.55f);
    EXPECT_GT(soilCams[0].tokenWeight, 0.90f);

    soil.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(SystemSyncTest, RenderedPixelAndBevProjectionMatchWorldGroundTruthUnderPitch) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    testutil::IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindRealRigTensor();

    const int camW = h.rig.camWidth();
    const int camH = h.rig.camHeight();
    std::vector<Vec4> tensor(static_cast<size_t>(8 * camW * camH));

    // Check known visible painted actors: PED_05 (emerald green {0.063, 0.725, 0.506}) in CAM_PILLAR_RIGHT (4) & CAM_FRONT_WIDE (2),
    // and PED_08 (crimson {0.957, 0.247, 0.369}) in CAM_FRONT_WIDE (2) / FRONT_MAIN (0).
    const auto& peds = h.world->pedestrians();
    int ped5Idx = -1;
    for (size_t i = 0; i < peds.size(); ++i) {
        if (peds[i].id == "PED_05") ped5Idx = static_cast<int>(i);
    }
    ASSERT_GE(ped5Idx, 0);
    const Vec3 chestW{peds[ped5Idx].x, 1.15f, peds[ped5Idx].z};

    auto sampleBestGreenMatch = [&](int cam, const Vec2& centerPx) {
        float bestDist = 10.0f;
        const int cx = static_cast<int>(std::round(centerPx.x));
        const int cy = static_cast<int>(std::round(centerPx.y));
        const size_t camOff = static_cast<size_t>(cam) * camW * camH;
        for (int dy = -4; dy <= 4; ++dy) {
            for (int dx = -4; dx <= 4; ++dx) {
                const int px = std::clamp(cx + dx, 0, camW - 1);
                const int py = std::clamp(cy + dy, 0, camH - 1);
                const Vec4& c = tensor[camOff + static_cast<size_t>(py) * camW + px];
                // Shaded emerald green shirt (g significantly dominates r and b)
                if (c.y > c.x * 1.6f && c.y > c.z * 1.1f && c.y > 0.20f) {
                    bestDist = std::min(bestDist, std::hypot(static_cast<float>(dx), static_cast<float>(dy)));
                }
            }
        }
        return bestDist;
    };

    // Pass 1: Level chassis (pitch = 0.0)
    h.world->mutableEgo().pitch = 0.0f;
    h.rig.renderAllCameras(*h.world, cfg, 0.0f, nullptr);
    h.rig.cameraTensorSSBO().downloadVector(tensor);

    Vec2 pxLevelCam2{}, pxLevelCam4{};
    float d2 = 0.0f, d4 = 0.0f;
    ASSERT_TRUE(h.rig.projectWorldToPixel(2, chestW, pxLevelCam2, d2));
    ASSERT_TRUE(h.rig.projectWorldToPixel(4, chestW, pxLevelCam4, d4));
    EXPECT_LT(sampleBestGreenMatch(2, pxLevelCam2), 3.2f)
        << "Rendered PED_05 green shirt missed projected pixel on distorted CAM_FRONT_WIDE (k1=-0.245) at ("
        << pxLevelCam2.x << "," << pxLevelCam2.y << ")";
    EXPECT_LT(sampleBestGreenMatch(4, pxLevelCam4), 3.2f)
        << "Rendered PED_05 green shirt missed projected pixel on CAM_PILLAR_RIGHT (-55 deg) at ("
        << pxLevelCam4.x << "," << pxLevelCam4.y << ")";

    // Pass 2: Hard-braking nose dive (pitch = +0.060 rad): horizon & objects must shift UP in image (smaller py),
    // and both rendered ISP pixels and projectWorldToPixel must track identically.
    h.world->mutableEgo().pitch = 0.060f;
    h.rig.renderAllCameras(*h.world, cfg, 0.03f, nullptr);
    h.rig.cameraTensorSSBO().downloadVector(tensor);

    Vec2 pxDiveCam4{};
    float d4Dive = 0.0f;
    ASSERT_TRUE(h.rig.projectWorldToPixel(4, chestW, pxDiveCam4, d4Dive));
    EXPECT_LT(pxDiveCam4.y, pxLevelCam4.y - 3.0f);
    EXPECT_LT(sampleBestGreenMatch(4, pxDiveCam4), 3.2f)
        << "Rendered pixel vs projectWorldToPixel diverged under sprung pitch=+0.06 rad";

    // Also run full RT-DETR + BEVFormer on Pass 2: 3D detection of PED_05 must stay accurate even under braking dive.
    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);
    float bestPed5Err = 999.0f;
    for (const auto& d : out3D) {
        if (d.semantic == SemanticClass::Pedestrian) {
            bestPed5Err = std::min(bestPed5Err, std::hypot(d.worldPos.x - chestW.x, d.worldPos.z - chestW.z));
        }
    }
    EXPECT_LT(bestPed5Err, 2.5f) << "3D detection error for PED_05 under braking pitch=" << bestPed5Err;
}

TEST(SystemSyncTest, ClosedLoopNominal60StepsMaintainsCleanTracksAndZeroCollisions) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);

    const float dtControl = 1.0f / cfg.controlHz;
    SimulationSyncClock clock;
    PerceptionFrameOutput out;
    PlannerCommand cmd;

    int worstOrphanTracks = 0;
    std::ostringstream trackLog;

    for (int step = 0; step < 60; ++step) {
        clock.stepFrame(dtControl, dtControl, dtControl * 2.0f,
            [&](float simTime, float percDt) {
                rig.renderAllCameras(world, cfg, simTime, nullptr);
                perception.runFrame(ctx, rig, world, percDt, cfg.occEnableShadows, out, nullptr);
            },
            [&](int tickIdx, float, float tickDt) {
                planner.plan(world, out, cmd);
                world.stepAgents(tickDt);
                world.stepEgoAndPhysics(tickDt, cmd, ManualInput{});
                EXPECT_FALSE(world.lastContact().occurred) << "Unexpected SAT collision at tick " << tickIdx;
            });

        if (step == 20 || step == 40 || step == 59) {
            const auto gtDyn = world.dynamicGroundTruthOBBs();
            int orphanAtStep = 0;
            for (const auto& tr : out.tracks) {
                float minDynDist = 999.0f;
                for (const auto& g : gtDyn) {
                    const float dx = std::max(0.0f, std::abs(tr.worldPos.x - g.cx) - g.hx);
                    const float dz = std::max(0.0f, std::abs(tr.worldPos.z - g.cz) - g.hz);
                    minDynDist = std::min(minDynDist, std::hypot(dx, dz));
                }
                if (minDynDist > 4.5f) {
                    ++orphanAtStep;
                    trackLog << "\n  [step " << step << " ORPHAN TRK] id=" << tr.trackId
                             << " sem=" << static_cast<int>(tr.semantic) << " sc=" << tr.score
                             << " w=(" << tr.worldPos.x << "," << tr.worldPos.z << ") minDynDist=" << minDynDist;
                }
            }
            if (orphanAtStep > 1) {
                const int cw = rig.camWidth();
                const int ch = rig.camHeight();
                std::vector<Vec4> tensor(static_cast<size_t>(8 * cw * ch));
                rig.cameraTensorSSBO().downloadVector(tensor);
                for (const auto& d : out.detections3D) {
                    if (d.semantic == SemanticClass::SignalSign) continue;
                    const int px = std::clamp(static_cast<int>(d.det2d.cxNorm * cw), 0, cw - 1);
                    const int py = std::clamp(static_cast<int>(d.det2d.cyNorm * ch), 0, ch - 1);
                    const Vec4& rgb = tensor[(static_cast<size_t>(d.camIndex) * ch + py) * cw + px];
                    trackLog << "\n    [step " << step << " D3D] cam=" << d.camIndex
                             << " sem=" << static_cast<int>(d.semantic) << " sc=" << d.score
                             << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm << "," << d.det2d.wNorm << "," << d.det2d.hNorm
                             << ") rgb=(" << rgb.x << "," << rgb.y << "," << rgb.z << ") w=(" << d.worldPos.x << "," << d.worldPos.z << ")";
                }
            }
            worstOrphanTracks = std::max(worstOrphanTracks, orphanAtStep);
        }
    }

    perception.destroy();
    rig.destroy();
    ctx.cleanup();

    EXPECT_LE(worstOrphanTracks, 1)
        << "Long-horizon closed-loop nominal run accumulated " << worstOrphanTracks
        << " orphan dynamic tracks (>4.5m from any real dynamic GT):" << trackLog.str();
}

TEST(SystemSyncTest, ClosedLoopJaywalkScenarioStopsBeforeCollisionAndTriggersYieldOrAeb) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    world.triggerJaywalkerCutIn();
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);

    const float dtControl = 1.0f / cfg.controlHz;
    SimulationSyncClock clock;
    PerceptionFrameOutput out;
    PlannerCommand cmd;

    bool sawBrakeReaction = false;
    float minSpeedReached = 999.0f;

    for (int step = 0; step < 65; ++step) {
        clock.stepFrame(dtControl, dtControl, dtControl * 2.0f,
            [&](float simTime, float percDt) {
                rig.renderAllCameras(world, cfg, simTime, nullptr);
                perception.runFrame(ctx, rig, world, percDt, cfg.occEnableShadows, out, nullptr);
            },
            [&](int tickIdx, float, float tickDt) {
                planner.plan(world, out, cmd);
                if (cmd.aebActive || cmd.behaviorState == "AEB_EMERGENCY_BRAKE" ||
                    cmd.behaviorState == "YIELDING_PEDESTRIAN") {
                    sawBrakeReaction = true;
                }
                world.stepAgents(tickDt);
                world.stepEgoAndPhysics(tickDt, cmd, ManualInput{});
                minSpeedReached = std::min(minSpeedReached, world.ego().speed);
                EXPECT_FALSE(world.lastContact().occurred)
                    << "Collision with " << world.lastContact().otherId << " at jaywalk tick " << tickIdx;
            });

        for (const auto& g : out.worldGhosts) {
            EXPECT_FALSE(std::isnan(g.worldPosScale.x) || std::isnan(g.worldPosScale.z));
        }
    }

    perception.destroy();
    rig.destroy();
    ctx.cleanup();

    EXPECT_TRUE(sawBrakeReaction) << "Planner never triggered YIELDING_PEDESTRIAN or AEB during jaywalker cut-in";
    EXPECT_LT(minSpeedReached, 1.0f) << "Ego failed to brake safely for jaywalker (minSpeed=" << minSpeedReached << ")";
}

TEST(SystemSyncTest, ClosedLoopStoppedTruckScenarioDetectsTruckAndPreventsCollision) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    world.triggerOccludedDartOut(); // Places TRUCK_01 stationary 13.5m ahead in Lane 0
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);

    const float dtControl = 1.0f / cfg.controlHz;
    SimulationSyncClock clock;
    PerceptionFrameOutput out;
    PlannerCommand cmd;

    int truckDet3dHits = 0;
    int truckTrackHits = 0;
    int truckVoxelHits = 0;
    bool collisionOccurred = false;
    std::ostringstream diag;

    for (int step = 0; step < 70; ++step) {
        clock.stepFrame(dtControl, dtControl, dtControl * 2.0f,
            [&](float simTime, float percDt) {
                rig.renderAllCameras(world, cfg, simTime, nullptr);
                perception.runFrame(ctx, rig, world, percDt, cfg.occEnableShadows, out, nullptr);
            },
            [&](int tickIdx, float, float tickDt) {
                const auto& truck = world.vehicles().at(0);
                float bestDetDist = 999.0f;
                for (const auto& d : out.detections3D) {
                    const float dist = std::hypot(d.worldPos.x - truck.pos.x, d.worldPos.z - truck.pos.z);
                    if (d.semantic == SemanticClass::Vehicle && dist < 5.5f) {
                        bestDetDist = std::min(bestDetDist, dist);
                    }
                }
                if (bestDetDist < 5.5f) ++truckDet3dHits;

                float bestTrkDist = 999.0f;
                for (const auto& t : out.tracks) {
                    const float dist = std::hypot(t.worldPos.x - truck.pos.x, t.worldPos.z - truck.pos.z);
                    if (t.semantic == SemanticClass::Vehicle && dist < 5.5f) {
                        bestTrkDist = std::min(bestTrkDist, dist);
                    }
                }
                if (bestTrkDist < 5.5f) ++truckTrackHits;

                int voxOnTruck = 0;
                for (const auto& v : out.activeVoxels) {
                    const float dx = std::abs(v.wx - truck.pos.x);
                    const float dz = std::abs(v.wz - truck.pos.z);
                    if (dx <= truck.halfExtents.z + 1.0f && dz <= truck.halfExtents.x + 1.0f) {
                        ++voxOnTruck;
                    }
                }
                if (voxOnTruck > 0) ++truckVoxelHits;

                planner.plan(world, out, cmd);
                if (tickIdx <= 3 || tickIdx % 10 == 0) {
                    diag << "step=" << tickIdx << " ego=(" << world.ego().pos.x << "," << world.ego().pos.z
                         << ") v=" << world.ego().speed << " truck=(" << truck.pos.x << "," << truck.pos.z
                         << ") beh=" << cmd.behaviorState << " minObs=" << cmd.minObstacleDist
                         << " det2D=" << out.detections2D.size() << " det3D=" << out.detections3D.size()
                         << "(bestTruckD=" << bestDetDist << ") trk=" << out.tracks.size()
                         << "(bestTrkD=" << bestTrkDist << ") voxOnTruck=" << voxOnTruck << "\n";
                    if (tickIdx == 0 || tickIdx == 30 || tickIdx == 50) {
                        for (const auto& d2 : out.detections2D) {
                            if (d2.camIndex <= 2) {
                                diag << "   D2D cam=" << d2.camIndex << " sem=" << static_cast<int>(d2.semantic)
                                     << " sc=" << d2.score << " c=(" << d2.cxNorm << "," << d2.cyNorm
                                     << ") wh=(" << d2.wNorm << "," << d2.hNorm << ")\n";
                            }
                        }
                        for (const auto& d3 : out.detections3D) {
                            diag << "   D3D cam=" << d3.camIndex << " sem=" << static_cast<int>(d3.semantic)
                                 << " sc=" << d3.score << " pos=(" << d3.worldPos.x << "," << d3.worldPos.y
                                 << "," << d3.worldPos.z << ")\n";
                        }
                    }
                }
                world.stepAgents(tickDt);
                world.stepEgoAndPhysics(tickDt, cmd, ManualInput{});
                if (world.lastContact().occurred && world.lastContact().otherId == "TRUCK_01") {
                    collisionOccurred = true;
                }
            });
    }

    perception.destroy();
    rig.destroy();
    ctx.cleanup();

    EXPECT_FALSE(collisionOccurred) << "Cybercab collided with TRUCK_01!\n" << diag.str();
    EXPECT_GT(truckDet3dHits, 10) << "TRUCK_01 rarely detected in 3D!\n" << diag.str();
    EXPECT_GT(truckTrackHits, 10) << "TRUCK_01 rarely tracked by ByteTrack!\n" << diag.str();
    EXPECT_GT(truckVoxelHits, 5) << "TRUCK_01 rarely represented in activeVoxels!\n" << diag.str();
}

} // namespace
} // namespace cybercab
