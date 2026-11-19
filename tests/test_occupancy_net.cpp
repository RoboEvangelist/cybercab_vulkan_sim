#include "config/SimConfig.hpp"
#include "core/Math.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "TestPerceptionHelpers.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

using namespace cybercab;
using namespace cybercab::testutil;

TEST(OccupancyNetExtendedTest, CleanEmptyRoadProducesZeroFalseDynamicVoxels) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch, true, false);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<ActiveVoxel> occVox, shadowVox;
    const std::vector<TrackPriorGpu> noTracks;
    const float dt = 1.0f / cfg.controlHz;
    for (int i = 0; i < 5; ++i) {
        h.runBevAndOcc(h.world->ego(), dt, noTracks, false, occVox, shadowVox);
    }

    int roadGhostCount = 0;
    for (const auto& v : occVox) {
        if ((v.sem == SemanticClass::Vehicle || v.sem == SemanticClass::Pedestrian) &&
            v.egoX > 3.5f && v.egoX < 35.0f && std::abs(v.egoY) < 6.0f) {
            ++roadGhostCount;
        }
    }
    EXPECT_EQ(roadGhostCount, 0)
        << "Clean empty road produced " << roadGhostCount << " false Vehicle/Pedestrian voxels";
}

TEST(OccupancyNetExtendedTest, EgoMotionWarpsDecayingVoxelsInsteadOfDraggingGhostWake) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = kPi * 0.5f; // Facing +X_world

    std::vector<ActiveVoxel> occVox, shadowVox;
    std::vector<TrackPriorGpu> priors;
    priors.push_back(makeTrackPrior(ego, {22.0f, 0.72f, 0.0f}, {0.0f, 0.0f, 0.0f}, ego.yaw, 0.96f, 2.18f));

    // Step 0: stamp stationary obstacle at egoX = 22.0m (X_world = 22.0m).
    h.runBevAndOcc(ego, 0.05f, priors, false, occVox, shadowVox);
    ASSERT_FALSE(occVox.empty());

    // Clear track priors & move ego forward +6.0m over 6 ticks (total 0.30s < decay time ~0.76s).
    // Stationary obstacle at world X=22 must shift backward in ego frame to egoX ~ 16.0m, NOT remain at egoX >= 20m!
    priors.clear();
    for (int step = 0; step < 6; ++step) {
        ego.pos.x += 1.0f;
        h.runBevAndOcc(ego, 0.05f, priors, false, occVox, shadowVox);
    }

    int stuckAheadGhosts = 0;
    int properlyShiftedVox = 0;
    for (const auto& v : occVox) {
        if (v.sem != SemanticClass::Vehicle) continue;
        if (v.egoX >= 19.8f) ++stuckAheadGhosts;
        if (v.egoX >= 13.0f && v.egoX <= 18.8f && std::abs(v.egoY) < 1.8f) ++properlyShiftedVox;
    }

    EXPECT_EQ(stuckAheadGhosts, 0)
        << "05_occupancy_flow.comp failed to warp voxel grid with ego motion: "
        << stuckAheadGhosts << " voxels stayed stuck at old egoX >= 19.8m";
    EXPECT_GT(properlyShiftedVox, 0)
        << "Expected stationary obstacle voxels to shift backward to egoX ~ 16m as ego advanced +6m";
}

TEST(OccupancyNetExtendedTest, MovingEgoPassingTrackedCarDoesNotLeaveTrailingWake) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = kPi * 0.5f;
    const Vec3 parkedCarW{25.0f, 0.72f, -3.5f};

    std::vector<ActiveVoxel> occVox, shadowVox;
    std::vector<TrackPriorGpu> priors(1);
    const float dt = 0.04f;

    // Ego passes parked car from X=0 to X=15 over 10 ticks (1.5 m/tick).
    for (int step = 0; step < 10; ++step) {
        ego.pos.x += 1.5f;
        priors[0] = makeTrackPrior(ego, parkedCarW, {0.0f, 0.0f, 0.0f}, ego.yaw, 0.96f, 2.18f);
        h.runBevAndOcc(ego, dt, priors, false, occVox, shadowVox);
    }

    // Current egoX of parked car is 25 - 15 = 10.0m (halfLength 2.18 -> rear edge at 12.2m, front at 7.8m).
    int wakeGhostCount = 0;
    for (const auto& v : occVox) {
        if (v.sem == SemanticClass::Vehicle && v.egoX > 14.5f && v.confidence > 0.25f) {
            ++wakeGhostCount;
        }
    }
    EXPECT_EQ(wakeGhostCount, 0)
        << "Unwarped occupancy persistence dragged " << wakeGhostCount << " high-conf ghost wake voxels behind car";
}

TEST(OccupancyNetExtendedTest, TrackPriorOBBRotation90DegAlignsVoxelFootprintAndFlow) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = kPi * 0.5f; // +X_world forward
    std::vector<ActiveVoxel> occVox, shadowVox;
    std::vector<TrackPriorGpu> priors(1);

    // Case 1: cross-traffic car at yaw = 0 (+Z_world = -Y_iso), so halfLength=2.2 aligns along egoY!
    priors[0] = makeTrackPrior(ego, {15.0f, 0.72f, 0.0f}, {0.0f, 0.0f, 6.5f}, 0.0f, 0.96f, 2.20f);
    h.runBevAndOcc(ego, 0.03f, priors, false, occVox, shadowVox);

    float minEgoX = 1e9f, maxEgoX = -1e9f, minEgoY = 1e9f, maxEgoY = -1e9f;
    int cnt = 0;
    for (const auto& v : occVox) {
        if (v.sem != SemanticClass::Vehicle) continue;
        minEgoX = std::min(minEgoX, v.egoX);
        maxEgoX = std::max(maxEgoX, v.egoX);
        minEgoY = std::min(minEgoY, v.egoY);
        maxEgoY = std::max(maxEgoY, v.egoY);
        EXPECT_NEAR(v.vz, 6.5f, 0.1f);
        EXPECT_NEAR(v.vx, 0.0f, 0.1f);
        ++cnt;
    }
    ASSERT_GT(cnt, 4) << "occVox.size()=" << occVox.size();
    const float spanX = maxEgoX - minEgoX;
    const float spanY = maxEgoY - minEgoY;
    EXPECT_GT(spanY, spanX + 1.0f)
        << "Cross-traffic 90-deg car should span longer in egoY than egoX (spanX=" << spanX << ", spanY=" << spanY << ")";
}

TEST(OccupancyNetExtendedTest, EgoFootprintMaskExcludesSelfOccupiedVoxels) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    const EgoState& ego = h.world->ego();
    std::vector<ActiveVoxel> occVox, shadowVox;
    std::vector<TrackPriorGpu> priors(1);
    // Inject a false track right inside ego's own chassis bounds (egoX=1.2, egoY=0.0).
    priors[0] = makeTrackPrior(ego,
                               ego.pos + Vec3{std::sin(ego.yaw) * 1.2f, 0.72f, std::cos(ego.yaw) * 1.2f},
                               {0.0f, 0.0f, 0.0f}, ego.yaw, 0.96f, 2.18f);
    h.runBevAndOcc(ego, 0.03f, priors, false, occVox, shadowVox);

    int selfVox = 0;
    for (const auto& v : occVox) {
        if (v.egoX > -1.35f && v.egoX < 3.75f && std::abs(v.egoY) < 1.25f) {
            ++selfVox;
        }
    }
    EXPECT_EQ(selfVox, 0);
}

TEST(OccupancyNetExtendedTest, VoxelWorldTransformMatchesTrackPoseAtArbitraryHeading) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    EgoState ego{};
    ego.pos = {12.5f, 0.0f, -18.0f};
    ego.yaw = -kPi * 0.35f;

    const float fwdX = std::sin(ego.yaw), fwdZ = std::cos(ego.yaw);
    const float leftX = std::cos(ego.yaw), leftZ = -std::sin(ego.yaw);
    const Vec3 obsW{ego.pos.x + 15.0f * fwdX - 4.0f * leftX, 0.9f, ego.pos.z + 15.0f * fwdZ - 4.0f * leftZ};

    std::vector<ActiveVoxel> occVox, shadowVox;
    std::vector<TrackPriorGpu> priors(1);
    priors[0] = makeTrackPrior(ego, obsW, {0.0f, 0.0f, 0.0f}, ego.yaw, 0.45f, 0.45f, SemanticClass::Pedestrian);
    h.runBevAndOcc(ego, 0.03f, priors, false, occVox, shadowVox);

    ASSERT_FALSE(occVox.empty());
    float bestWorldDist = std::numeric_limits<float>::infinity();
    for (const auto& v : occVox) {
        if (v.sem == SemanticClass::Pedestrian) {
            bestWorldDist = std::min(bestWorldDist, std::hypot(v.wx - obsW.x, v.wz - obsW.z));
        }
    }
    EXPECT_LT(bestWorldDist, 0.85f);
}

TEST(OccupancyNetExtendedTest, BevFormerTemporalWarpDecaysMemoryAndGrowsEpistemicVar) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    // Frame 0: render real scene with rich cameras so BEVFormer builds high spatial observability.
    h.bindRealRigTensor();
    h.rig.renderAllCameras(*h.world, cfg, 0.0f, nullptr);
    std::vector<ActiveVoxel> occVox, shadowVox;
    h.runBevAndOcc(h.world->ego(), 1.0f / 36.0f, {}, false, occVox, shadowVox);
    const BevSummary s0 = h.bev.readbackSummary();
    ASSERT_GT(s0.meanObservability, 0.30f);

    // Download currBevMetaSSBO across steps when cameras become blind/zeroed out.
    std::vector<Vec4> meta0(static_cast<size_t>(cfg.bevNx * cfg.bevNy));
    h.bev.currBevMetaSSBO().downloadVector(meta0);

    // Feed blank black soiled tensor (minimum camera patch weight & zero features) to inspect temporal decay.
    h.bindSyntheticTensor();
    std::fill(h.synthPixelsScratch.begin(), h.synthPixelsScratch.end(), Vec4{0.0f, 0.0f, 0.0f, 1.0f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    EgoState ego = h.world->ego();
    for (int i = 0; i < 6; ++i) {
        ego.pos.x += 0.3f;
        h.runBevAndOcc(ego, 0.10f, {}, false, occVox, shadowVox);
    }
    const BevSummary sLater = h.bev.readbackSummary();
    EXPECT_GT(sLater.meanEpistemicVar, s0.meanEpistemicVar)
        << "Epistemic variance must increase as camera features degrade / age";
}

TEST(OccupancyNetExtendedTest, OcclusionShadowRaycastGeometryAndDisableClearance) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = kPi * 0.5f;
    std::vector<ActiveVoxel> occVox, shadowVox;
    std::vector<TrackPriorGpu> priors(1);
    priors[0] = makeTrackPrior(ego, {14.0f, 0.72f, 0.0f}, {0.0f, 0.0f, 0.0f}, ego.yaw, 1.1f, 2.2f);

    h.runBevAndOcc(ego, 0.03f, priors, true, occVox, shadowVox);
    ASSERT_FALSE(shadowVox.empty()) << "occVox.size()=" << occVox.size();
    for (const auto& sv : shadowVox) {
        EXPECT_GT(sv.egoX, 14.0f) << "Occlusion shadow leaked in front of occluder at egoX=" << sv.egoX;
        EXPECT_LT(sv.egoX, 27.0f) << "Occlusion shadow exceeded 9.5m cast limit at egoX=" << sv.egoX;
        EXPECT_LT(std::abs(sv.egoY), 4.5f);
    }

    // Disabling shadows must immediately clear all SEM_OCC_SHADOW voxels.
    h.runBevAndOcc(ego, 0.03f, priors, false, occVox, shadowVox);
    std::vector<Vec4> rawStates(cfg.occNx() * cfg.occNy() * cfg.occNz());
    h.occ.voxelStateSSBO().downloadVector(rawStates);
    float maxConf = 0.0f;
    for (const auto& s : rawStates) maxConf = std::max(maxConf, s.y);
    std::vector<TrackPriorGpu> uploaded(cfg.btMaxTracks);
    h.occ.trackPriorsSSBO().downloadVector(uploaded);
    EXPECT_TRUE(shadowVox.empty());
}

TEST(OccupancyNetExtendedTest, FullSceneVoxelAuditCountsTrueVsOrphanGhostVoxels) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    PlannerCommand cmd;
    std::ostringstream sampleLog;
    for (int i = 0; i < 12; ++i) {
        world.stepAgents(dt);
        rig.renderAllCameras(world, cfg, static_cast<float>(i) * dt, nullptr);
        perception.runFrame(ctx, rig, world, dt, true, out, nullptr);
        planner.plan(world, out, cmd);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
        for (const auto& d : out.detections3D) {
            if (d.semantic == SemanticClass::Vehicle) {
                sampleLog << "\n  [STEP " << i << " D3D] cam=" << d.camIndex << " sem=1 sc=" << d.score
                          << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm << "," << d.det2d.wNorm << "," << d.det2d.hNorm
                          << ") w=(" << d.worldPos.x << "," << d.worldPos.z << ")";
            }
        }
    }

    const auto gtAll = world.allGroundTruthOBBs();
    int dynVoxels = 0;
    int orphanDynVoxels = 0;
    int orphanVeh = 0;
    int orphanPed = 0;
    for (const auto& g : world.dynamicGroundTruthOBBs()) {
        sampleLog << "\n  [GT] " << g.id << " sem=" << static_cast<int>(g.semantic)
                  << " w=(" << g.cx << "," << g.cz << ") h=(" << g.hx << "," << g.hz << ")";
    }

    for (const auto& v : out.activeVoxels) {
        if (v.sem != SemanticClass::Vehicle && v.sem != SemanticClass::Pedestrian) continue;
        ++dynVoxels;
        float minDist = std::numeric_limits<float>::infinity();
        for (const auto& g : gtAll) {
            const float dx = std::max(0.0f, std::abs(v.wx - g.cx) - g.hx);
            const float dz = std::max(0.0f, std::abs(v.wz - g.cz) - g.hz);
            minDist = std::min(minDist, std::hypot(dx, dz));
        }
        if (minDist > 3.5f) {
            ++orphanDynVoxels;
            if (v.sem == SemanticClass::Vehicle) ++orphanVeh;
            else ++orphanPed;
            if (orphanDynVoxels <= 8) {
                sampleLog << "\n  [ORPHAN] sem=" << static_cast<int>(v.sem)
                          << " ego=(" << v.egoX << "," << v.egoY << ")"
                          << " w=(" << v.wx << "," << v.wy << "," << v.wz << ")"
                          << " conf=" << v.confidence << " flow=(" << v.vx << "," << v.vz << ") dist=" << minDist;
            }
        }
    }

    const float orphanRatio = (dynVoxels > 0)
        ? (static_cast<float>(orphanDynVoxels) / static_cast<float>(dynVoxels))
        : 0.0f;
    RecordProperty("dyn_voxels", dynVoxels);
    RecordProperty("orphan_ghost_voxels", orphanDynVoxels);
    RecordProperty("orphan_ghost_voxel_pct", static_cast<int>(orphanRatio * 100.0f));

    perception.destroy();
    rig.destroy();
    ctx.cleanup();

    EXPECT_GT(dynVoxels, 0);
    for (const auto& t : out.tracks) {
        float tMinGt = std::numeric_limits<float>::infinity();
        for (const auto& g : gtAll) {
            const float dx = std::max(0.0f, std::abs(t.worldPos.x - g.cx) - g.hx);
            const float dz = std::max(0.0f, std::abs(t.worldPos.z - g.cz) - g.hz);
            tMinGt = std::min(tMinGt, std::hypot(dx, dz));
        }
        sampleLog << "\n  [TRK] id=" << t.trackId << " sem=" << static_cast<int>(t.semantic)
                  << " st=" << static_cast<int>(t.state) << " hits=" << t.hits
                  << " lost=" << t.timeSinceUpdateSec << " score=" << t.score
                  << " w=(" << t.worldPos.x << "," << t.worldPos.z << ") ego=(" << t.egoPos.x << "," << t.egoPos.y
                  << ") minGt=" << tMinGt;
    }
    for (const auto& d : out.detections3D) {
        if (d.semantic == SemanticClass::SignalSign) continue;
        sampleLog << "\n  [D3D] cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
                  << " score=" << d.score << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm << ","
                  << d.det2d.wNorm << "," << d.det2d.hNorm << ") w=(" << d.worldPos.x << "," << d.worldPos.z << ")";
    }
    EXPECT_LT(orphanRatio, 0.15f)
        << "Full scene emitted " << orphanDynVoxels << " / " << dynVoxels
        << " (veh=" << orphanVeh << ", ped=" << orphanPed << ", trk=" << out.tracks.size() << ")"
        << " (" << (orphanRatio * 100.0f) << "%) dynamic voxels >3.5m away from ANY GT:" << sampleLog.str();
}

TEST(OccupancyNetExtendedTest, EgoYawRotationWarpsDecayingVoxelsAtWorldCoordinate) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = 0.0f;

    const Vec3 worldTarget{10.0f, 0.72f, 14.0f};
    std::vector<TrackPriorGpu> priors = {
        makeTrackPrior(ego, worldTarget, {0.0f, 0.0f, 0.0f}, 0.0f, 1.0f, 2.2f, SemanticClass::Vehicle, 0.95f, 1)
    };
    std::vector<ActiveVoxel> occ0, occ1, shad0, shad1;
    h.runBevAndOcc(ego, 0.03f, priors, false, occ0, shad0);
    ASSERT_FALSE(occ0.empty());

    // Step ego forward AND turn sharply by +45 deg with priors removed:
    // decaying voxels must warp back onto worldTarget (10.0, 14.0), NOT rotate in ego frame.
    ego.pos = {2.0f, 0.0f, 3.0f};
    ego.yaw = kPi * 0.25f;
    h.runBevAndOcc(ego, 0.05f, {}, false, occ1, shad1);

    int warpedVehVoxels = 0;
    float worstErr = 0.0f;
    for (const auto& v : occ1) {
        if (v.sem != SemanticClass::Vehicle) continue;
        ++warpedVehVoxels;
        const float dx = std::max(0.0f, std::abs(v.wx - worldTarget.x) - 1.25f);
        const float dz = std::max(0.0f, std::abs(v.wz - worldTarget.z) - 2.45f);
        worstErr = std::max(worstErr, std::hypot(dx, dz));
    }
    EXPECT_GT(warpedVehVoxels, 4);
    EXPECT_LT(worstErr, 1.5f) << "Ego yaw rotation warped decaying occupancy voxels away from stationary world object";
}
