#include <gtest/gtest.h>

#include "TestPerceptionHelpers.hpp"
#include "perception/ByteTrack.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace cybercab {
namespace {

TEST(FMEAPerceptionTest, SoilDetectorSoiledFrontCamDegradesWeightsInflatesVarAndMaintainsSafety) {
    // FMEA (Sensor Contamination / Soiling):
    // Validates ISP mud/haze input injection, SoilDetector patch/camera telemetry, parameter bounds,
    // RSS uncertainty buffer inflation, and fail-safe obstacle stopping under soiled front optics.
    const SimConfig cleanCfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine cleanWorld(cleanCfg);
    CameraRigRenderer cleanRig;
    cleanRig.init(ctx, cleanCfg, cleanWorld);
    PerceptionPipeline cleanPerc(cleanCfg);
    cleanPerc.init(ctx, cleanRig);

    PerceptionFrameOutput cleanOut;
    cleanRig.renderAllCameras(cleanWorld, cleanCfg, 0.0f, nullptr);
    cleanPerc.runFrame(ctx, cleanRig, cleanWorld, 1.0f / 36.0f, cleanCfg.occEnableShadows, cleanOut, nullptr);

    EXPECT_EQ(cleanOut.soilPatches.size(), static_cast<size_t>(8 * cleanCfg.soilPatchCols * cleanCfg.soilPatchRows));
    for (int c = 0; c < 8; ++c) {
        const auto& sc = cleanOut.soilCameras[c];
        EXPECT_TRUE(std::isfinite(sc.meanSoilProb) && std::isfinite(sc.tokenWeight) && std::isfinite(sc.meanEpistemicVar));
        EXPECT_LT(sc.meanSoilProb, 0.32f) << "Clean camera " << c << " reported elevated soilProb";
        EXPECT_GT(sc.tokenWeight, 0.80f) << "Clean camera " << c << " degraded tokenWeight unexpectedly";
        EXPECT_GE(sc.meanEpistemicVar, cleanCfg.soilBaseEpistemicVar - 1e-4f);
    }

    // 1. Single/dual front lens soiling isolation (CAM 0 + CAM 2 soiled, CAM 1 clean)
    SimConfig soilCfg = cleanCfg;
    soilCfg.cameras[0].soiled = true;
    soilCfg.cameras[2].soiled = true;
    soilCfg.soilMinTokenWeight = 0.18f;
    soilCfg.soilVarGain = 1.15f;

    CameraRigRenderer soilRig;
    soilRig.init(ctx, soilCfg, cleanWorld);
    PerceptionPipeline soilPerc(soilCfg);
    soilPerc.init(ctx, soilRig);

    PerceptionFrameOutput soilOut;
    soilRig.renderAllCameras(cleanWorld, soilCfg, 0.0f, nullptr);
    soilPerc.runFrame(ctx, soilRig, cleanWorld, 1.0f / 36.0f, soilCfg.occEnableShadows, soilOut, nullptr);

    std::ostringstream soilDiag;
    for (int c = 0; c < 8; ++c) {
        const auto& s = soilOut.soilCameras[c];
        soilDiag << "\n  cam=" << c << " soiled=" << soilCfg.cameras[c].soiled
                 << " p=" << s.meanSoilProb << " w=" << s.tokenWeight
                 << " var=" << s.meanEpistemicVar << " edge=" << s.meanEdge;
    }

    for (const int soiledIdx : {0, 2}) {
        const auto& s = soilOut.soilCameras[soiledIdx];
        const auto& base = cleanOut.soilCameras[soiledIdx];
        EXPECT_GT(s.meanSoilProb, 0.65f) << "Soiled cam " << soiledIdx << " undetected:" << soilDiag.str();
        EXPECT_LT(s.tokenWeight, 0.45f) << "Soiled cam " << soiledIdx << " failed to attenuate weight:" << soilDiag.str();
        EXPECT_GE(s.tokenWeight, soilCfg.soilMinTokenWeight - 1e-3f) << "Weight fell below config min floor";
        EXPECT_GT(s.meanEpistemicVar, base.meanEpistemicVar + 0.35f)
            << "Soiled cam " << soiledIdx << " did not inflate epistemic variance:" << soilDiag.str();
        EXPECT_LT(s.meanEdge, base.meanEdge) << "Mud/haze should suppress sharp edge gradient energy";
    }
    EXPECT_LT(soilOut.soilCameras[1].meanSoilProb, 0.32f) << "Clean CAM 1 contaminated by soiled CAM 0/2:" << soilDiag.str();
    EXPECT_GT(soilOut.soilCameras[1].tokenWeight, 0.80f);
    EXPECT_GT(soilOut.forwardEpistemicSigma, cleanOut.forwardEpistemicSigma + 0.30f) << soilDiag.str();

    for (const auto& p : soilOut.soilPatches) {
        EXPECT_TRUE(std::isfinite(p.soilProb) && std::isfinite(p.patchConfidence) && std::isfinite(p.epistemicVar));
        EXPECT_GE(p.soilProb, 0.0f);
        EXPECT_LE(p.soilProb, 1.0f);
        EXPECT_GE(p.patchConfidence, soilCfg.soilMinTokenWeight - 1e-4f);
        EXPECT_LE(p.patchConfidence, 1.0001f);
        EXPECT_GE(p.epistemicVar, soilCfg.soilBaseEpistemicVar - 1e-4f);
    }

    // 2. Planner RSS safety buffer expansion under high forward epistemic uncertainty
    TrajectoryPlanner cleanPlanner(cleanCfg);
    TrajectoryPlanner soilPlanner(soilCfg);
    PlannerCommand cmdClean, cmdSoiled;
    cleanPlanner.plan(cleanWorld, cleanOut, cmdClean);
    soilPlanner.plan(cleanWorld, soilOut, cmdSoiled);

    const float expectedExcess = std::max(0.0f, soilOut.forwardEpistemicSigma - 0.22f);
    EXPECT_GT(expectedExcess, 0.20f);
    EXPECT_NE(cmdSoiled.behaviorState, "SENSOR_SOILED_MRM")
        << "Single clean CAM 1 forward path should prevent premature MRM lockout";
    EXPECT_GT(cmdSoiled.rssSafeDist, cmdClean.rssSafeDist + 0.35f)
        << "RSS envelope did not expand under elevated forward epistemic uncertainty (clean="
        << cmdClean.rssSafeDist << ", soiled=" << cmdSoiled.rssSafeDist << ")";

    // 3. Closed-loop fail-safe obstacle stopping even with ALL 3 front cameras soiled
    SimConfig failSafeCfg = cleanCfg;
    failSafeCfg.cameras[0].soiled = true;
    failSafeCfg.cameras[1].soiled = true;
    failSafeCfg.cameras[2].soiled = true;
    WorldEngine stopWorld(failSafeCfg);
    for (auto& p : stopWorld.mutablePedestrians()) {
        p.x = -300.0f; p.z = -300.0f;
        p.ax = p.x; p.az = p.z;
        p.bx = p.x; p.bz = p.z;
        p.targetX = p.x; p.targetZ = p.z;
    }
    const EgoState& e0 = stopWorld.ego();
    const float sinE = std::sin(e0.yaw), cosE = std::cos(e0.yaw);
    auto& leadTruck = stopWorld.mutableVehicles().at(0);
    leadTruck.pos = {e0.pos.x + sinE * 16.5f, 0.0f, e0.pos.z + cosE * 16.5f};
    leadTruck.yaw = e0.yaw;
    leadTruck.speed = 0.0f; leadTruck.vx = 0.0f; leadTruck.vz = 0.0f;

    CameraRigRenderer stopRig;
    stopRig.init(ctx, failSafeCfg, stopWorld);
    PerceptionPipeline stopPerc(failSafeCfg);
    stopPerc.init(ctx, stopRig);
    TrajectoryPlanner stopPlanner(failSafeCfg);
    PerceptionFrameOutput stopOut;
    PlannerCommand stopCmd;
    const float dt = 1.0f / failSafeCfg.controlHz;
    bool slowedOrStopped = false;
    bool observedMrmStop = false;
    std::ostringstream stopDiag;

    for (int step = 0; step < 55; ++step) {
        stopRig.renderAllCameras(stopWorld, failSafeCfg, static_cast<float>(step) * dt, nullptr);
        stopPerc.runFrame(ctx, stopRig, stopWorld, dt, failSafeCfg.occEnableShadows, stopOut, nullptr);
        stopPlanner.plan(stopWorld, stopOut, stopCmd);
        stopWorld.stepEgoAndPhysics(dt, stopCmd, ManualInput{});
        if (stopWorld.ego().speed < 3.0f || stopCmd.aebActive) slowedOrStopped = true;
        if (stopCmd.behaviorState == "SENSOR_SOILED_MRM" && stopCmd.targetSpeed == 0.0f) observedMrmStop = true;
        if (step % 5 == 0 || stopWorld.lastContact().occurred) {
            float minFwdVox = 99.0f, maxVoxConf = 0.0f;
            for (const auto& v : stopOut.activeVoxels) {
                if (v.wy <= 2.1f && v.sem == SemanticClass::Vehicle && std::abs(v.egoY) < 1.3f && v.egoX > 2.0f) {
                    minFwdVox = std::min(minFwdVox, v.egoX - 3.5f);
                    maxVoxConf = std::max(maxVoxConf, v.confidence);
                }
            }
            stopDiag << "\n  [s=" << step << "] spd=" << stopWorld.ego().speed
                     << " tgtSpd=" << stopCmd.targetSpeed << " rss=" << stopCmd.rssSafeDist
                     << " minObs=" << stopCmd.minObstacleDist << " aeb=" << stopCmd.aebActive
                     << " st=" << stopCmd.behaviorState << " dets=" << stopOut.detections3D.size()
                     << " trks=" << stopOut.tracks.size() << " minFwdVox=" << minFwdVox
                     << " maxVoxC=" << maxVoxConf;
        }
        EXPECT_FALSE(stopWorld.lastContact().occurred)
            << "Soiled front-camera fail-safe crashed into " << stopWorld.lastContact().otherId
            << " at step=" << step << " fwdSig=" << stopOut.forwardEpistemicSigma
            << " vox=" << stopOut.activeVoxels.size() << " dets=" << stopOut.detections3D.size()
            << stopDiag.str();
    }
    EXPECT_TRUE(observedMrmStop) << "Expected SENSOR_SOILED_MRM when all 3 forward cameras are soiled";
    EXPECT_TRUE(slowedOrStopped)
        << "Cybercab failed to decelerate toward stopped TRUCK_01 when front cameras were soiled";

    stopPerc.destroy();
    stopRig.destroy();
    soilPerc.destroy();
    soilRig.destroy();
    cleanPerc.destroy();
    cleanRig.destroy();
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, OcclusionShadowRaycastAndWorldModelRolloutCoherence) {
    // 1. Validate WorldModel + Shadow input configuration parameters
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    ASSERT_TRUE(cfg.occEnableShadows);
    ASSERT_EQ(cfg.wmHorizonsSec.size(), 4u);
    EXPECT_FLOAT_EQ(cfg.wmUncertaintyGrowthRate, 0.55f);

    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    world.mutableEgo().speed = 0.0f;
    for (auto& p : world.mutablePedestrians()) {
        p.x = -350.0f; p.z = -350.0f;
        p.ax = p.x; p.az = p.z;
        p.bx = p.x; p.bz = p.z;
        p.targetX = p.x; p.targetZ = p.z;
    }
    for (size_t i = 0; i < world.mutableVehicles().size(); ++i) {
        auto& v = world.mutableVehicles()[i];
        v.speed = 0.0f;
        v.vx = 0.0f;
        v.vz = 0.0f;
        v.pos = {-400.0f - static_cast<float>(i) * 20.0f, 0.0f, -400.0f};
    }

    // Place stopped TRUCK_01 14m ahead of ego in lane 0
    const EgoState ego0 = world.ego();
    const float sinE = std::sin(ego0.yaw);
    const float cosE = std::cos(ego0.yaw);
    {
        auto& truck = world.mutableVehicles().at(0);
        truck.pos = {ego0.pos.x + sinE * 14.0f, 0.0f, ego0.pos.z + cosE * 14.0f};
        truck.yaw = ego0.yaw;
        truck.speed = 0.0f;
        truck.hardBrakeTimer = 10.0f;
    }

    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);

    // 2. Occlusion shadow raycast geometry, uncertainty, & enable/disable toggle FMEA check
    PerceptionFrameOutput out;
    for (int s = 0; s < 4; ++s) {
        rig.renderAllCameras(world, cfg, static_cast<float>(s) * 0.05f, nullptr);
        perception.runFrame(ctx, rig, world, 0.05f, true, out, nullptr);
    }

    EXPECT_GT(out.shadowVoxels.size(), 6u)
        << "Stopped TRUCK_01 at 14m ahead should cast BEV occlusion shadows when enableShadows=true";
    for (const auto& sv : out.shadowVoxels) {
        EXPECT_TRUE(std::isfinite(sv.egoX) && std::isfinite(sv.egoY) && std::isfinite(sv.epistemicSigma));
        EXPECT_EQ(sv.sem, SemanticClass::OccShadow);
        EXPECT_GE(sv.epistemicSigma, 0.80f) << "Occlusion shadow must inflate epistemic uncertainty >= 0.85";
        EXPECT_GE(sv.confidence, 0.50f);
        EXPECT_LE(sv.confidence, 0.65f);
        // Shadow rays cast behind TRUCK_01 (~11.1..16.9) through occluded cone within sensor range [10.5, occMaxX]
        EXPECT_GE(sv.egoX, 10.5f) << "Shadow leaked ahead of occluder TRUCK_01 front face at egoX=" << sv.egoX;
        EXPECT_LE(sv.egoX, cfg.occMaxX) << "Shadow outside BEV occlusion grid at egoX=" << sv.egoX;
        EXPECT_LE(std::abs(sv.egoY), 4.5f) << "Shadow raycast spread outside truck angular shadow cone";
    }

    // Disabling shadows must immediately clear all OccShadow voxels
    rig.renderAllCameras(world, cfg, 0.25f, nullptr);
    perception.runFrame(ctx, rig, world, 0.05f, false, out, nullptr);
    EXPECT_EQ(out.shadowVoxels.size(), 0u)
        << "Disabling occEnableShadows failed to clear stale shadow voxels";

    // 3. WorldModel 4D rollout coherence: moving SEDAN_01 vs stopped TRUCK_01
    {
        auto& sedan = world.mutableVehicles().at(1);
        const float loopLen = world.centerline().getLength();
        sedan.lane = 1;
        sedan.u = std::fmod(ego0.loopProgressU + 8.0f / loopLen, 1.0f);
        sedan.speed = 6.5f;
        sedan.hardBrakeTimer = 0.0f;
        const Vec3 sPt = world.laneCurve(1).getPointAt(sedan.u);
        const Vec3 sTan = world.laneCurve(1).getTangentAt(sedan.u);
        sedan.pos = sPt;
        sedan.yaw = std::atan2(sTan.x, sTan.z);
        sedan.vx = sTan.x * sedan.speed;
        sedan.vz = sTan.z * sedan.speed;
    }

    const float dt = 0.05f;
    std::ostringstream wmDiag;
    for (int s = 0; s < 10; ++s) {
        world.stepAgents(dt);
        // Re-pin stopped TRUCK_01 speed = 0 so it remains strictly stationary
        world.mutableVehicles().at(0).speed = 0.0f;
        world.mutableVehicles().at(0).vx = 0.0f;
        world.mutableVehicles().at(0).vz = 0.0f;
        rig.renderAllCameras(world, cfg, 0.30f + static_cast<float>(s) * dt, nullptr);
        perception.runFrame(ctx, rig, world, dt, true, out, nullptr);
        const auto& sd = world.vehicles().at(1);
        for (const auto& d : out.detections3D) {
            if (d.semantic == SemanticClass::Vehicle && std::hypot(d.worldPos.x - sd.pos.x, d.worldPos.z - sd.pos.z) < 5.0f) {
                wmDiag << "\n  [s" << s << " D3D] cam=" << d.camIndex << " x=" << d.worldPos.x
                       << " gtX=" << sd.pos.x << " sig=" << d.epistemicSigma << " sc=" << d.score;
            }
        }
        for (const auto& tr : out.tracks) {
            if (tr.semantic == SemanticClass::Vehicle && std::hypot(tr.worldPos.x - sd.pos.x, tr.worldPos.z - sd.pos.z) < 5.0f) {
                wmDiag << "\n  [s" << s << " TRK] id=" << tr.trackId << " hits=" << tr.hits
                       << " x=" << tr.worldPos.x << " spd=" << std::hypot(tr.worldVel.x, tr.worldVel.z);
            }
        }
    }

    const auto& truck = world.vehicles().at(0);
    const auto& sedan = world.vehicles().at(1);
    const Track3D* truckTrk = nullptr;
    const Track3D* sedanTrk = nullptr;
    for (const auto& tr : out.tracks) {
        if (tr.semantic != SemanticClass::Vehicle) continue;
        if (std::hypot(tr.worldPos.x - truck.pos.x, tr.worldPos.z - truck.pos.z) < 4.0f) truckTrk = &tr;
        if (std::hypot(tr.worldPos.x - sedan.pos.x, tr.worldPos.z - sedan.pos.z) < 4.5f) sedanTrk = &tr;
    }
    ASSERT_NE(truckTrk, nullptr) << "Expected active track on stationary TRUCK_01";
    ASSERT_NE(sedanTrk, nullptr) << "Expected active track on moving SEDAN_01";

    const float truckSpd = std::hypot(truckTrk->worldVel.x, truckTrk->worldVel.z);
    const float sedanSpd = std::hypot(sedanTrk->worldVel.x, sedanTrk->worldVel.z);
    EXPECT_LT(truckSpd, 0.45f) << "Stationary TRUCK_01 Kalman velocity drifted to " << truckSpd << " m/s";
    EXPECT_GT(sedanSpd, 2.0f) << "Moving SEDAN_01 Kalman speed failed to converge (" << sedanSpd
                               << " m/s, gtSpeed=" << sedan.speed << ", trkPos=("
                               << sedanTrk->worldPos.x << "," << sedanTrk->worldPos.z
                               << "), gtPos=(" << sedan.pos.x << "," << sedan.pos.z << "))" << wmDiag.str();

    int stoppedTruckGhosts = 0;
    std::array<std::vector<const WorldGhost*>, 4> byHorizon;
    for (const auto& g : out.worldGhosts) {
        EXPECT_TRUE(std::isfinite(g.worldPosScale.x) && std::isfinite(g.worldPosScale.z) &&
                    std::isfinite(g.yawMeta.y) && std::isfinite(g.yawMeta.z));
        const int gTid = static_cast<int>(std::round(g.yawMeta.w));
        if (gTid == truckTrk->trackId && truckSpd < 0.25f) {
            ++stoppedTruckGhosts;
        }
        if (gTid == sedanTrk->trackId) {
            for (size_t hi = 0; hi < cfg.wmHorizonsSec.size(); ++hi) {
                if (std::abs(g.yawMeta.y - cfg.wmHorizonsSec[hi]) < 0.05f) {
                    byHorizon[hi].push_back(&g);
                }
            }
        }
    }
    EXPECT_EQ(stoppedTruckGhosts, 0)
        << "WorldModel emitted future ghosts for strictly stopped TRUCK_01 (speed < 0.25)";

    float prevSigma = 0.0f;
    for (size_t hi = 0; hi < 4; ++hi) {
        ASSERT_EQ(byHorizon[hi].size(), 4u)
            << "Expected 2x2 ghost corner cluster (4) at horizon t=" << cfg.wmHorizonsSec[hi];
        float meanX = 0.0f, meanZ = 0.0f, meanSigma = 0.0f;
        for (const auto* g : byHorizon[hi]) {
            meanX += g->worldPosScale.x;
            meanZ += g->worldPosScale.z;
            meanSigma += g->yawMeta.z;
        }
        meanX *= 0.25f;
        meanZ *= 0.25f;
        meanSigma *= 0.25f;

        const float tFut = cfg.wmHorizonsSec[hi];
        const float expX = sedanTrk->worldPos.x + sedanTrk->worldVel.x * tFut;
        const float expZ = sedanTrk->worldPos.z + sedanTrk->worldVel.z * tFut;
        const float expSigma = sedanTrk->posSigmaM + cfg.wmUncertaintyGrowthRate * tFut;
        EXPECT_NEAR(meanX, expX, 0.15f) << "WorldModel X rollout mismatch at t=" << tFut;
        EXPECT_NEAR(meanZ, expZ, 0.15f) << "WorldModel Z rollout mismatch at t=" << tFut;
        EXPECT_NEAR(meanSigma, expSigma, 0.05f) << "WorldModel uncertainty growth mismatch at t=" << tFut;
        if (hi > 0) {
            EXPECT_GT(meanSigma, prevSigma + 0.15f) << "Epistemic sigma did not grow monotonically at t=" << tFut;
        }
        prevSigma = meanSigma;
    }

    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, ParameterBoundaryValidationAndInvalidConfigRejection) {
    // FMEA (Configuration & Numerical Boundary Validation):
    // Tests missing/malformed YAML rejection, partial override merging, ByteTrack saturation & dt edge cases,
    // Planner RSS uncertainty monotonic scaling under extreme sigma, and clean repeated Vulkan init/destroy cycles.
    const SimConfig defCfg = SimConfig::defaultConfig();
    EXPECT_EQ(defCfg.cameras.size(), 8u);
    EXPECT_GT(defCfg.occTotalVoxels(), 0);

    // 1. Missing YAML config file must throw cleanly without corrupting defaults
    EXPECT_THROW(SimConfig::loadFromYaml("/tmp/cybercab_definitely_missing_config_99999.yaml"), std::exception);

    // 2. Malformed YAML with invalid camera count (!= 8) must throw std::runtime_error
    const std::filesystem::path badCamPath = std::filesystem::temp_directory_path() / "cybercab_bad_cams_test.yaml";
    {
        std::ofstream ofs(badCamPath);
        ofs << "cameras:\n"
            << "  - id: CAM_0\n    hfov_deg: 50.0\n"
            << "  - id: CAM_1\n    hfov_deg: 28.0\n";
    }
    EXPECT_THROW(SimConfig::loadFromYaml(badCamPath.string()), std::runtime_error);
    std::filesystem::remove(badCamPath);

    // 3. Partial YAML config must preserve valid 8-camera defaults while merging overrides
    const std::filesystem::path partialPath = std::filesystem::temp_directory_path() / "cybercab_partial_test.yaml";
    {
        std::ofstream ofs(partialPath);
        ofs << "simulation:\n  control_hz: 40.0\n"
            << "perception:\n  soil_detector:\n    min_token_weight: 0.12\n";
    }
    const SimConfig partialCfg = SimConfig::loadFromYaml(partialPath.string());
    std::filesystem::remove(partialPath);
    EXPECT_FLOAT_EQ(partialCfg.controlHz, 40.0f);
    EXPECT_FLOAT_EQ(partialCfg.soilMinTokenWeight, 0.12f);
    EXPECT_EQ(partialCfg.cameras.size(), 8u);

    // 3b. Invalid YAML parameter bounds (negative control_hz, inverted conf thresholds, zero min_token_weight)
    const std::filesystem::path badBoundsPath = std::filesystem::temp_directory_path() / "cybercab_bad_bounds_test.yaml";
    {
        std::ofstream ofs(badBoundsPath);
        ofs << "simulation:\n  control_hz: -10.0\n";
    }
    EXPECT_THROW(SimConfig::loadFromYaml(badBoundsPath.string()), std::runtime_error);
    {
        std::ofstream ofs(badBoundsPath);
        ofs << "perception:\n  rtdetr:\n    high_conf_thresh: 0.20\n    low_conf_thresh: 0.55\n";
    }
    EXPECT_THROW(SimConfig::loadFromYaml(badBoundsPath.string()), std::runtime_error);
    {
        std::ofstream ofs(badBoundsPath);
        ofs << "perception:\n  soil_detector:\n    min_token_weight: 0.0\n";
    }
    EXPECT_THROW(SimConfig::loadFromYaml(badBoundsPath.string()), std::runtime_error);
    std::filesystem::remove(badBoundsPath);

    // 4. ByteTrack capacity cap (> btMaxTracks), zero/negative/large dt spikes, & NaN immunity
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    ByteTracker tracker(cfg);
    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = 0.0f;

    std::vector<Detection3D> saturatedDets;
    saturatedDets.reserve(90);
    for (int i = 0; i < 90; ++i) {
        saturatedDets.push_back(testutil::makeDet3D(
            SemanticClass::Vehicle,
            {static_cast<float>(i % 10) * 10.0f - 45.0f, 0.72f, 8.0f + static_cast<float>(i / 10) * 8.0f},
            0.88f, 1.0f, 0.25f));
    }
    // Feed two frames to confirm promoted Tracked entries respect btMaxTracks cap
    tracker.update(0.05f, ego, saturatedDets);
    tracker.update(0.05f, ego, saturatedDets);
    EXPECT_LE(static_cast<int>(tracker.activeTracks().size()), cfg.btMaxTracks);
    EXPECT_LE(static_cast<int>(tracker.exportTrackPriorsGpu(ego).size()), cfg.btMaxTracks);

    // Zero dt, negative dt fault, and large dt spike must never produce NaN/Inf or negative posSigmaM
    tracker.update(0.0f, ego, saturatedDets);
    tracker.update(-0.05f, ego, saturatedDets);
    tracker.update(0.25f, ego, {});
    for (const auto& tr : tracker.activeTracks()) {
        EXPECT_TRUE(std::isfinite(tr.worldPos.x) && std::isfinite(tr.worldPos.z));
        EXPECT_TRUE(std::isfinite(tr.worldVel.x) && std::isfinite(tr.worldVel.z));
        EXPECT_TRUE(std::isfinite(tr.posSigmaM) && tr.posSigmaM > 0.0f);
    }

    // Drained empty observations past btMaxTimeLostSec must cleanly expire tracks
    for (int k = 0; k < 40; ++k) {
        tracker.update(0.05f, ego, {});
    }
    EXPECT_EQ(tracker.activeTracks().size(), 0u)
        << "Expired tracks failed to prune after > 1.5s without observations";

    // 5. Planner boundary resilience & Vulkan init/reset/destroy lifecycle cleanliness (2 full cycles)
    VulkanContext ctx;
    ctx.init(false, nullptr);
    for (int cycle = 0; cycle < 2; ++cycle) {
        WorldEngine world(cfg);
        world.mutablePedestrians().clear();
        for (size_t i = 0; i < world.mutableVehicles().size(); ++i) {
            world.mutableVehicles()[i].pos = {-450.0f - static_cast<float>(i) * 20.0f, 0.0f, -450.0f};
            world.mutableVehicles()[i].speed = 0.0f;
            world.mutableVehicles()[i].vx = 0.0f;
            world.mutableVehicles()[i].vz = 0.0f;
        }
        CameraRigRenderer rig;
        rig.init(ctx, cfg, world);
        PerceptionPipeline perc(cfg);
        perc.init(ctx, rig);
        TrajectoryPlanner planner(cfg);
        PerceptionFrameOutput out;
        PlannerCommand cmd;

        rig.renderAllCameras(world, cfg, 0.0f, nullptr);
        perc.runFrame(ctx, rig, world, 0.05f, cfg.occEnableShadows, out, nullptr);
        out.forwardEpistemicSigma = 2.5f;
        planner.plan(world, out, cmd);

        EXPECT_TRUE(std::isfinite(cmd.targetSpeed) && std::isfinite(cmd.minObstacleDist) && std::isfinite(cmd.rssSafeDist));
        EXPECT_GE(cmd.rssSafeDist, cfg.rssDBufferM + cfg.rssUncertaintyBufferGain * 2.5f - 1e-3f);
        EXPECT_LE(cmd.targetSpeed, cfg.targetCruiseSpeedMs + 1e-4f);
        EXPECT_FALSE(cmd.optimalTrajectory.empty());
        for (const auto& pt : cmd.optimalTrajectory) {
            EXPECT_TRUE(std::isfinite(pt.x) && std::isfinite(pt.z));
        }

        perc.reset();
        perc.destroy();
        rig.destroy();
    }
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, RSSCarpetFollowsRoadCurveAndPreventsCurveRearEndCollisions) {
    // FMEA (RSS Corridor Frame, Lateral Coordinate Sign Consistency & Curve Safety):
    // 1. Verify WorldEngine::currentLateralOffset sign matches TrajectoryPlanner (-1.9m outer lane 0, +1.9m inner lane 1)
    //    so triggerLeadHardBrake() spawns the hazard in ego's occupied lane after physics steps.
    // 2. Verify RSS carpet attaches flush at front bumper, tracks road curvature around 90-deg turns instead of projecting
    //    straight off-road, scales width/length with epistemic uncertainty, detects stopped curve lead sedans without
    //    false-triggering on adjacent-lane traffic, and maintains < 1ms planning latency.
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    const float dt = 1.0f / cfg.controlHz;

    // 1. Coordinate sign consistency for lane 0 & lane 1 after physics updates
    for (int initLane : {0, 1}) {
        WorldEngine world(cfg);
        TrajectoryPlanner planner(cfg);
        auto& ego = world.mutableEgo();
        const Vec3 pt = world.laneCurve(initLane).getPointAt(0.03f);
        const Vec3 tan = world.laneCurve(initLane).getTangentAt(0.03f);
        ego.pos = pt;
        ego.yaw = std::atan2(tan.x, tan.z);
        ego.currentLateralOffset = (initLane == 0) ? -1.9f : 1.9f;
        PerceptionFrameOutput emptyPerc;
        PlannerCommand cmd;
        for (int s = 0; s < 10; ++s) {
            planner.plan(world, emptyPerc, cmd);
            world.stepAgents(dt);
            world.stepEgoAndPhysics(dt, cmd, ManualInput{});
        }
        const float expectedSign = (initLane == 0) ? -1.0f : 1.0f;
        EXPECT_GT(world.ego().currentLateralOffset * expectedSign, 1.0f)
            << "currentLateralOffset flipped sign for lane " << initLane
            << " (got " << world.ego().currentLateralOffset << ")";
        world.triggerLeadHardBrake();
        EXPECT_EQ(world.vehicles().at(1).lane, initLane)
            << "triggerLeadHardBrake spawned SEDAN_01 in wrong lane for ego lane " << initLane;
    }

    // 2. RSS Carpet geometry on a sharp 90-deg road curve (u = 0.25, R = 14m)
    WorldEngine curveWorld(cfg);
    TrajectoryPlanner curvePlanner(cfg);
    const float loopLen = curveWorld.centerline().getLength();
    const float curveU = 0.25f;
    {
        auto& ego = curveWorld.mutableEgo();
        ego.loopProgressU = curveU;
        const Vec3 pt = curveWorld.laneCurve(0).getPointAt(curveU);
        const Vec3 tan = curveWorld.laneCurve(0).getTangentAt(curveU);
        ego.pos = pt;
        ego.yaw = std::atan2(tan.x, tan.z);
        ego.speed = 10.5f;
        ego.currentLateralOffset = -1.9f;
    }

    PerceptionFrameOutput cleanPerc;
    cleanPerc.forwardEpistemicSigma = 0.20f;
    PlannerCommand cleanCmd;
    const auto t0 = std::chrono::steady_clock::now();
    curvePlanner.plan(curveWorld, cleanPerc, cleanCmd);
    const double planMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(planMs, 5.0) << "TrajectoryPlanner latency budget exceeded: " << planMs << "ms";

    ASSERT_GE(cleanCmd.rssCarpetPoints.size(), 8u) << "RSS carpet points not populated";
    const float bumperFwd = curveWorld.ego().hz + curveWorld.ego().centerOffsetForward;
    const Vec3 bumperPos{
        curveWorld.ego().pos.x + std::sin(curveWorld.ego().yaw) * bumperFwd,
        0.0f,
        curveWorld.ego().pos.z + std::cos(curveWorld.ego().yaw) * bumperFwd
    };
    const float startErr = std::hypot(cleanCmd.rssCarpetPoints.front().x - bumperPos.x,
                                      cleanCmd.rssCarpetPoints.front().z - bumperPos.z);
    EXPECT_LT(startErr, 0.20f) << "RSS carpet must anchor flush to ego front bumper";

    float carpetArcLen = 0.0f;
    for (size_t i = 0; i < cleanCmd.rssCarpetPoints.size(); ++i) {
        const auto& p = cleanCmd.rssCarpetPoints[i];
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
        if (i > 0) {
            carpetArcLen += std::hypot(p.x - cleanCmd.rssCarpetPoints[i - 1].x,
                                       p.z - cleanCmd.rssCarpetPoints[i - 1].z);
        }
    }
    EXPECT_NEAR(carpetArcLen, cleanCmd.rssSafeDist, 0.25f)
        << "Empty-road RSS carpet arc length should match rssSafeDist";

    // Measure distance from RSS carpet tip to curved lane 0 centerline around the turn
    const float tipU = std::fmod(curveU + (bumperFwd + cleanCmd.rssSafeDist) / loopLen, 1.0f);
    const Vec3 expectedTipCenter = curveWorld.laneCurve(0).getPointAt(tipU);
    const float tipDeviationM = std::hypot(cleanCmd.rssCarpetPoints.back().x - expectedTipCenter.x,
                                           cleanCmd.rssCarpetPoints.back().z - expectedTipCenter.z);
    EXPECT_LT(tipDeviationM, 1.25f)
        << "RSS carpet tip left the curved road! deviation=" << tipDeviationM << "m";

    // Epistemic uncertainty widens and lengthens RSS carpet monotonically
    PerceptionFrameOutput noisyPerc = cleanPerc;
    noisyPerc.forwardEpistemicSigma = 0.85f;
    PlannerCommand noisyCmd;
    curvePlanner.plan(curveWorld, noisyPerc, noisyCmd);
    EXPECT_GT(noisyCmd.rssSafeDist, cleanCmd.rssSafeDist + 0.8f);
    EXPECT_GT(noisyCmd.rssCarpetHalfWidth, cleanCmd.rssCarpetHalfWidth);

    // 3. Curve obstacle discrimination: adjacent-lane sedan does NOT false-trigger AEB;
    //    in-lane stopped sedan at 12m arc distance on the turn DOES trigger AEB and clips carpet!
    auto makeCarTrack = [](int id, const Vec3& pt, const Vec3& tan, int hits) {
        Track3D tr;
        tr.trackId = id;
        tr.semantic = SemanticClass::Vehicle;
        tr.state = TrackState::Tracked;
        tr.worldPos = {pt.x, 0.69f, pt.z};
        tr.halfExtents = {0.96f, 0.69f, 2.18f};
        tr.yaw = std::atan2(tan.x, tan.z);
        tr.score = 0.92f;
        tr.hits = hits;
        return tr;
    };
    const float obsU = std::fmod(curveU + 12.5f / loopLen, 1.0f);
    {
        PerceptionFrameOutput adjPerc = cleanPerc;
        const Vec3 adjPt = curveWorld.laneCurve(1).getPointAt(obsU);
        const Vec3 adjTan = curveWorld.laneCurve(1).getTangentAt(obsU);
        adjPerc.tracks.push_back(makeCarTrack(101, adjPt, adjTan, 2));
        PlannerCommand adjCmd;
        curvePlanner.plan(curveWorld, adjPerc, adjCmd);
        EXPECT_FALSE(adjCmd.aebActive) << "Adjacent-lane sedan on curve falsely triggered AEB";
    }
    {
        PerceptionFrameOutput leadPerc = cleanPerc;
        const Vec3 leadPt = curveWorld.laneCurve(0).getPointAt(obsU);
        const Vec3 leadTan = curveWorld.laneCurve(0).getTangentAt(obsU);
        leadPerc.tracks.push_back(makeCarTrack(102, leadPt, leadTan, 3));
        const float blkU = std::fmod(curveU + 10.5f / loopLen, 1.0f);
        leadPerc.tracks.push_back(makeCarTrack(103, curveWorld.laneCurve(1).getPointAt(blkU),
                                               curveWorld.laneCurve(1).getTangentAt(blkU), 3));
        PlannerCommand leadCmd;
        curvePlanner.plan(curveWorld, leadPerc, leadCmd);
        EXPECT_TRUE(leadCmd.aebActive)
            << "Stopped sedan 12.5m ahead on curve failed to trigger curved RSS AEB (minObs="
            << leadCmd.minObstacleDist << "m, behavior=" << leadCmd.behaviorState << ")";
        EXPECT_LT(leadCmd.minObstacleDist, 10.0f);
        float clippedLen = 0.0f;
        for (size_t i = 1; i < leadCmd.rssCarpetPoints.size(); ++i) {
            clippedLen += std::hypot(leadCmd.rssCarpetPoints[i].x - leadCmd.rssCarpetPoints[i - 1].x,
                                     leadCmd.rssCarpetPoints[i].z - leadCmd.rssCarpetPoints[i - 1].z);
        }
        EXPECT_LT(clippedLen, leadCmd.rssSafeDist - 2.0f)
            << "Blocked RSS carpet failed to clip to obstacle distance";
    }
    {
        // Active occupancy voxel alone on the curve also triggers curved RSS corridor stopping
        PerceptionFrameOutput voxPerc = cleanPerc;
        const Vec3 voxPt = curveWorld.laneCurve(0).getPointAt(obsU);
        const Vec3 blkPt = curveWorld.laneCurve(1).getPointAt(std::fmod(curveU + 10.5f / loopLen, 1.0f));
        for (const Vec3& p : {voxPt, blkPt}) {
            ActiveVoxel v;
            v.wx = p.x;
            v.wy = 0.75f;
            v.wz = p.z;
            v.sem = SemanticClass::Vehicle;
            v.confidence = 0.90f;
            voxPerc.activeVoxels.push_back(v);
        }
        PlannerCommand voxCmd;
        curvePlanner.plan(curveWorld, voxPerc, voxCmd);
        EXPECT_LT(voxCmd.minObstacleDist, 10.0f)
            << "Curved RSS corridor missed obstacle voxel on turn";
    }
    {
        // Manual steering mode bends RSS carpet along instantaneous bicycle-model yaw arc
        SimConfig manCfg = cfg;
        manCfg.driveMode = "manual";
        WorldEngine manWorld(manCfg);
        manWorld.mutableEgo().steerAngle = 0.28f;
        manWorld.mutableEgo().speed = 10.5f;
        TrajectoryPlanner manPlanner(manCfg);
        PlannerCommand manCmd;
        manPlanner.plan(manWorld, cleanPerc, manCmd);
        ASSERT_GE(manCmd.rssCarpetPoints.size(), 8u);
        const auto& tip = manCmd.rssCarpetPoints.back();
        const float sinE = std::sin(manWorld.ego().yaw), cosE = std::cos(manWorld.ego().yaw);
        const float latDev = std::abs((tip.x - manWorld.ego().pos.x) * cosE - (tip.z - manWorld.ego().pos.z) * sinE);
        EXPECT_GT(latDev, 2.5f) << "Manual steering failed to curve the RSS carpet";
    }
}

TEST(FMEAPerceptionTest, SmallSedanAndSuvBackboneDecoderConsistency) {
    // FMEA (Backbone FPN + Deformable RT-DETR Small-Car Consistency):
    // 1. Negative baseline: empty-road + buildings produces zero false Vehicle/Pedestrian detections.
    // 2. Close range (7.5m): Cam0 does not clip silver SUV or red sedan bottoms at v=0.72/0.78.
    // 3. Curve approach (15 frames at u=0.25, 22m -> 18m): all small cars (SEDAN_01..03, SUV_01) maintain
    //    consistent 2D tire-grounded boxes (no >45% single-frame height collapse in Cam3 and zero high-score fragment steals).
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perc(cfg);
    perc.init(ctx, rig);
    const float dt = 1.0f / cfg.controlHz;

    auto isolateCar = [&](size_t vehIdx, float egoU, float aheadM, float egoSpd) {
        world.reset();
        perc.reset();
        const float loopLen = world.centerline().getLength();
        auto& ego = world.mutableEgo();
        ego.loopProgressU = egoU;
        const Vec3 ePt = world.laneCurve(0).getPointAt(egoU);
        const Vec3 eTan = world.laneCurve(0).getTangentAt(egoU);
        ego.pos = ePt;
        ego.yaw = std::atan2(eTan.x, eTan.z);
        ego.speed = egoSpd;
        for (auto& v : world.mutableVehicles()) {
            v.lane = 1;
            v.u = std::fmod(egoU + 0.55f, 1.0f);
            v.pos = {-420.0f, 0.0f, -420.0f};
            v.baseSpeed = 0.0f;
            v.speed = 0.0f;
            v.vx = 0.0f;
            v.vz = 0.0f;
            v.hardBrakeTimer = 8.0f;
        }
        for (auto& p : world.mutablePedestrians()) {
            p.x = -420.0f; p.z = -420.0f; p.ax = -420.0f; p.az = -420.0f;
            p.bx = -420.0f; p.bz = -420.0f; p.targetX = -420.0f; p.targetZ = -420.0f;
        }
        if (vehIdx < world.mutableVehicles().size()) {
            auto& v = world.mutableVehicles()[vehIdx];
            v.lane = 0;
            v.u = std::fmod(egoU + aheadM / loopLen, 1.0f);
            v.pos = world.laneCurve(0).getPointAt(v.u);
            const Vec3 vTan = world.laneCurve(0).getTangentAt(v.u);
            v.yaw = std::atan2(vTan.x, vTan.z);
        }
    };

    // 1. Empty road + buildings guard (straight and curve)
    PerceptionFrameOutput out;
    for (float uCheck : {0.03f, 0.25f}) {
        isolateCar(99, uCheck, 0.0f, 0.0f);
        rig.renderAllCameras(world, cfg, 0.0f, nullptr);
        perc.runFrame(ctx, rig, world, dt, false, out, nullptr);
        int falseVeh = 0, falsePed = 0;
        for (const auto& d3 : out.detections3D) {
            if (d3.semantic == SemanticClass::Vehicle) ++falseVeh;
            if (d3.semantic == SemanticClass::Pedestrian) ++falsePed;
        }
        EXPECT_EQ(falseVeh, 0) << "Empty road/building scene produced false Vehicle at u=" << uCheck;
        EXPECT_EQ(falsePed, 0) << "Empty road/building scene produced false Pedestrian at u=" << uCheck;
    }

    // 2. Close range (7.5m) Cam0 bottom box coverage for silver SUV_01 and red SEDAN_02
    for (size_t vi : {size_t(2), size_t(3)}) {
        isolateCar(vi, 0.03f, 7.5f, 0.0f);
        rig.renderAllCameras(world, cfg, 0.0f, nullptr);
        perc.runFrame(ctx, rig, world, dt, false, out, nullptr);
        const auto& gt = world.vehicles()[vi];
        Vec2 pxRear{};
        float dRear = 0.0f;
        const Vec3 rearW{gt.pos.x - std::sin(gt.yaw) * gt.halfExtents.z, 0.0f, gt.pos.z - std::cos(gt.yaw) * gt.halfExtents.z};
        ASSERT_TRUE(rig.projectWorldToPixel(0, rearW, pxRear, dRear));
        const float gtRearV = pxRear.y / static_cast<float>(rig.camHeight());

        float bestCam0VBot = 0.0f;
        float bestCam0Score = 0.0f;
        for (const auto& d2 : out.detections2D) {
            if (d2.camIndex == 0 && d2.semantic == SemanticClass::Vehicle && d2.score > bestCam0Score) {
                bestCam0Score = d2.score;
                bestCam0VBot = d2.cyNorm + d2.hNorm * 0.5f;
            }
        }
        EXPECT_GT(bestCam0Score, cfg.rtdetrHighConfThresh)
            << gt.id << " at 7.5m in Cam0 failed high-conf detection (score=" << bestCam0Score << ")";
        EXPECT_GT(bestCam0VBot, std::min(0.86f, gtRearV - 0.07f))
            << gt.id << " Cam0 box truncated above ground tires (vBot=" << bestCam0VBot << " vs gtRearV=" << gtRearV << ")";
    }

    // 3. 15-step curve approach consistency in Cam3 (u=0.25, 22m -> 18m): check continuous Cam3 detection & no >45% h collapse
    TrajectoryPlanner planner(cfg);
    PlannerCommand cmd;
    const float loopLen = world.centerline().getLength();
    for (size_t vi : {size_t(1), size_t(2), size_t(3), size_t(5)}) {
        isolateCar(vi, 0.25f, 22.0f, 10.5f);
        // Keep adjacent lane 1 occupied behind ego so ego stays in Lane 0 without blocking Cam3 ahead.
        auto& blk = world.mutableVehicles()[(vi + 1) % world.vehicles().size()];
        blk.lane = 1;
        blk.u = std::fmod(0.25f - 2.0f / loopLen + 1.0f, 1.0f);
        blk.pos = world.laneCurve(1).getPointAt(blk.u);
        blk.yaw = std::atan2(world.laneCurve(1).getTangentAt(blk.u).x,
                             world.laneCurve(1).getTangentAt(blk.u).z);
        float simT = 0.0f;
        float prevCam3H = -1.0f;
        float prevCam3VBot = -1.0f;
        int cam3DetFrames = 0;
        for (int s = 0; s < 15; ++s) {
            rig.renderAllCameras(world, cfg, simT, nullptr);
            perc.runFrame(ctx, rig, world, dt, false, out, nullptr);
            planner.plan(world, out, cmd);
            const auto& gt = world.vehicles()[vi];
            Vec2 gtPx{};
            float gtD = 0.0f;
            const bool gtInCam3 = rig.projectWorldToPixel(3, {gt.pos.x, gt.halfExtents.y, gt.pos.z}, gtPx, gtD);
            const float gtU = gtPx.x / static_cast<float>(rig.camWidth());
            const float gtV = gtPx.y / static_cast<float>(rig.camHeight());

            float matchH = 0.0f, cam3VBot = 0.0f;
            float bestUvDist = 0.18f;
            for (const auto& d2 : out.detections2D) {
                if (d2.camIndex == 3 && d2.semantic == SemanticClass::Vehicle && d2.score >= cfg.rtdetrHighConfThresh) {
                    const float uvDist = gtInCam3 ? std::hypot(d2.cxNorm - gtU, d2.cyNorm - gtV) : 0.0f;
                    if (uvDist < bestUvDist) {
                        bestUvDist = uvDist;
                        matchH = d2.hNorm;
                        cam3VBot = d2.cyNorm + d2.hNorm * 0.5f;
                    }
                }
            }
            if (matchH > 0.0f) {
                ++cam3DetFrames;
                if (prevCam3H > 0.0f) {
                    EXPECT_GT(matchH, prevCam3H * 0.62f)
                        << gt.id << " Cam3 height collapsed at step " << s
                        << " (" << prevCam3H << " -> " << matchH << ")";
                    EXPECT_LT(std::abs(cam3VBot - prevCam3VBot), 0.025f)
                        << gt.id << " Cam3 vBot jumped at step " << s
                        << " (" << prevCam3VBot << " -> " << cam3VBot << ")";
                }
                prevCam3H = matchH;
                prevCam3VBot = cam3VBot;
            }
            world.stepAgents(dt);
            world.stepEgoAndPhysics(dt, cmd, ManualInput{});
            simT += dt;
        }
        EXPECT_GE(cam3DetFrames, 14)
            << world.vehicles()[vi].id << " dropped high-conf Cam3 boxes across curve approach ("
            << cam3DetFrames << "/15)";
    }

    perc.destroy();
    rig.destroy();
    ctx.cleanup();
}

// FMEA #83: Small sedans and SUVs must never false-trigger isTallTruck (2.68m height prior / +1.65m offset)
// across close/mid ranges or braking pitch, true 3.1m trucks must still lift as tall trucks, and border-clipped
// 28-deg telephoto (Cam1) detections must yield to unclipped wide/main cameras in 3D NMS.
TEST(FMEAPerceptionTest, SmallSedanAndSuv3DLiftAndTelephotoBorderNMS) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perc(cfg);
    perc.init(ctx, rig);
    const float dt = 1.0f / cfg.controlHz;
    const float loopLen = world.centerline().getLength();

    auto placeIsolated = [&](size_t vehIdx, float egoU, float aheadM, int vehLane, float egoPitch, float egoSpd) {
        world.reset();
        perc.reset();
        auto& ego = world.mutableEgo();
        ego.loopProgressU = egoU;
        ego.pos = world.laneCurve(0).getPointAt(egoU);
        const Vec3 eTan = world.laneCurve(0).getTangentAt(egoU);
        ego.yaw = std::atan2(eTan.x, eTan.z);
        ego.pitch = egoPitch;
        ego.speed = egoSpd;
        for (auto& v : world.mutableVehicles()) {
            v.lane = 1;
            v.u = std::fmod(egoU + 0.55f, 1.0f);
            v.pos = {-420.0f, 0.0f, -420.0f};
            v.baseSpeed = 0.0f;
            v.speed = 0.0f;
            v.vx = 0.0f;
            v.vz = 0.0f;
            v.hardBrakeTimer = 8.0f;
        }
        for (auto& p : world.mutablePedestrians()) {
            p.x = -420.0f; p.z = -420.0f; p.ax = -420.0f; p.az = -420.0f;
            p.bx = -420.0f; p.bz = -420.0f; p.targetX = -420.0f; p.targetZ = -420.0f;
        }
        auto& v = world.mutableVehicles().at(vehIdx);
        v.lane = vehLane;
        v.u = std::fmod(egoU + aheadM / loopLen, 1.0f);
        v.pos = world.laneCurve(vehLane).getPointAt(v.u);
        const Vec3 vTan = world.laneCurve(vehLane).getTangentAt(v.u);
        v.yaw = std::atan2(vTan.x, vTan.z);
    };

    PerceptionFrameOutput out;

    // 1. SUV_01 (1.58m) & SEDAN_01 (1.38m) across close/mid range and braking pitch: zero false isTallTruck
    for (size_t vi : {size_t(1), size_t(2)}) {
        for (auto [aheadM, pitch] : std::vector<std::pair<float, float>>{
                 {8.5f, 0.0f}, {15.0f, 0.0f}, {20.0f, 0.045f}, {28.0f, 0.0f}}) {
            placeIsolated(vi, 0.03f, aheadM, 0, pitch, 0.0f);
            rig.renderAllCameras(world, cfg, 0.0f, nullptr);
            perc.runFrame(ctx, rig, world, dt, false, out, nullptr);

            const auto& gt = world.vehicles()[vi];
            const float gtRange = std::hypot(gt.pos.x - world.ego().pos.x, gt.pos.z - world.ego().pos.z);
            float bestDist = 1e6f;
            float bestRangeErr = 1e6f;
            bool anyTallMis = false;
            for (const auto& d3 : out.detections3D) {
                if (d3.semantic != SemanticClass::Vehicle) continue;
                const float dist = std::hypot(d3.worldPos.x - gt.pos.x, d3.worldPos.z - gt.pos.z);
                if (dist < 6.0f && d3.halfExtents.y > 1.20f) anyTallMis = true;
                if (dist < bestDist) {
                    bestDist = dist;
                    const float detR = std::hypot(d3.worldPos.x - world.ego().pos.x, d3.worldPos.z - world.ego().pos.z);
                    bestRangeErr = std::abs(detR - gtRange);
                }
            }
            EXPECT_FALSE(anyTallMis)
                << gt.id << " misclassified as TallTruck at dist=" << aheadM << "m pitch=" << pitch;
            EXPECT_LT(bestDist, 3.8f)
                << gt.id << " missing accurate 3D lift at dist=" << aheadM << "m pitch=" << pitch
                << " (bestDist=" << bestDist << ")";
            EXPECT_LT(bestRangeErr, 3.5f)
                << gt.id << " 3D range error too large at dist=" << aheadM << "m pitch=" << pitch;
        }
    }

    // 2. Real TRUCK_01 (3.10m) at 10m and 25m must still classify as TallTruck
    for (float aheadM : {10.0f, 25.0f}) {
        placeIsolated(0, 0.03f, aheadM, 0, 0.0f, 0.0f);
        rig.renderAllCameras(world, cfg, 0.0f, nullptr);
        perc.runFrame(ctx, rig, world, dt, false, out, nullptr);
        const auto& gt = world.vehicles()[0];
        bool foundTall = false;
        float bestDist = 1e6f;
        for (const auto& d3 : out.detections3D) {
            if (d3.semantic != SemanticClass::Vehicle) continue;
            const float dist = std::hypot(d3.worldPos.x - gt.pos.x, d3.worldPos.z - gt.pos.z);
            bestDist = std::min(bestDist, dist);
            if (dist < 5.0f && d3.halfExtents.y > 1.20f) foundTall = true;
        }
        EXPECT_TRUE(foundTall) << "TRUCK_01 at " << aheadM << "m failed isTallTruck preservation";
        EXPECT_LT(bestDist, 4.0f) << "TRUCK_01 at " << aheadM << "m 3D lift error too large";
    }

    // 3. Closed-loop curve chase on SEDAN_03 across telephoto Cam1 border -> Cam2/3 handoff:
    //    border-clipped Cam1 must never override an unclipped wide/main 2D detection in fused 3D NMS.
    TrajectoryPlanner planner(cfg);
    PlannerCommand cmd;
    placeIsolated(5, 0.25f, 16.0f, 0, 0.0f, 10.0f);
    // Occupy adjacent Lane 1 behind ego so planner stays in Lane 0 without blocking forward cameras
    auto& blk = world.mutableVehicles()[1];
    blk.lane = 1;
    blk.u = std::fmod(0.25f - 2.0f / loopLen + 1.0f, 1.0f);
    blk.pos = world.laneCurve(1).getPointAt(blk.u);
    blk.yaw = std::atan2(world.laneCurve(1).getTangentAt(blk.u).x, world.laneCurve(1).getTangentAt(blk.u).z);

    float simT = 0.0f;
    for (int s = 0; s < 30; ++s) {
        rig.renderAllCameras(world, cfg, simT, nullptr);
        perc.runFrame(ctx, rig, world, dt, false, out, nullptr);
        planner.plan(world, out, cmd);
        const auto& gt = world.vehicles()[5];
        const float gtR = std::hypot(gt.pos.x - world.ego().pos.x, gt.pos.z - world.ego().pos.z);

        bool hasUnclippedNonTele = false;
        for (const auto& d2 : out.detections2D) {
            if (d2.semantic != SemanticClass::Vehicle || d2.score < cfg.rtdetrHighConfThresh) continue;
            const bool clipX = (d2.cxNorm - d2.wNorm * 0.5f) < 0.025f || (d2.cxNorm + d2.wNorm * 0.5f) > 0.975f;
            if (cfg.cameras.at(d2.camIndex).hfovDeg >= 40.0f && !clipX) {
                hasUnclippedNonTele = true;
            }
        }
        float bestRangeErr = 1e6f;
        const Detection3D* nearestD3 = nullptr;
        float minDist = 1e6f;
        for (const auto& d3 : out.detections3D) {
            if (d3.semantic != SemanticClass::Vehicle) continue;
            const float dist = std::hypot(d3.worldPos.x - gt.pos.x, d3.worldPos.z - gt.pos.z);
            if (dist < minDist) {
                minDist = dist;
                nearestD3 = &d3;
                const float detR = std::hypot(d3.worldPos.x - world.ego().pos.x, d3.worldPos.z - world.ego().pos.z);
                bestRangeErr = std::abs(detR - gtR);
            }
        }
        ASSERT_NE(nearestD3, nullptr) << "SEDAN_03 lost 3D detection on curve at step=" << s;
        const bool d3ClippedX = (nearestD3->det2d.cxNorm - nearestD3->det2d.wNorm * 0.5f) < 0.025f ||
                                (nearestD3->det2d.cxNorm + nearestD3->det2d.wNorm * 0.5f) > 0.975f;
        if (hasUnclippedNonTele && cfg.cameras.at(nearestD3->camIndex).hfovDeg < 40.0f && d3ClippedX) {
            ADD_FAILURE() << "Border-clipped telephoto Cam" << nearestD3->camIndex
                          << " overrode unclipped wide/main camera at step=" << s
                          << " rangeErr=" << bestRangeErr;
        }
        EXPECT_LT(bestRangeErr, 4.5f) << "SEDAN_03 3D range error spiked at step=" << s;

        world.stepAgents(dt);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
        EXPECT_FALSE(world.lastContact().occurred)
            << "Closed-loop curve chase collided with " << world.lastContact().otherId << " at step=" << s;
        simT += dt;
    }

    perc.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, ByteTrackSedanConsistencyNoDuplicatesOrVelocityExplosion) {
    // FMEA (Tracker Handoff / Monocular Jitter):
    // 1) Camera-handoff radial jumps must preserve active track ID and damp velocity spikes.
    // 2) Close-range collinear partial/cross-class detections must not spawn duplicate tracks.
    // 3) Closed-loop sedan chases (CRV_SEDAN01_22M, STR_SEDAN02_22M) must maintain 1 stable track.
    const SimConfig cfg = SimConfig::defaultConfig();
    const float dt = 1.0f / cfg.controlHz;
    {
        ByteTracker tracker(cfg);
        EgoState ego{};
        ego.pos = {0.0f, 0.0f, 0.0f};
        ego.yaw = 0.0f;
        std::vector<Track3D> active;
        for (int f = 0; f < 15; ++f) {
            tracker.update(dt, ego, {testutil::makeDet3D(SemanticClass::Vehicle, {0.0f, 0.72f, 16.0f}, 0.90f)});
        }
        tracker.collectActiveTracks(active);
        ASSERT_EQ(active.size(), 1u);
        const uint32_t baseId = active[0].trackId;

        // FMEA-BT-01: 4.4m radial camera-handoff depth jump along the same viewing ray.
        tracker.update(dt, ego, {testutil::makeDet3D(SemanticClass::Vehicle, {0.05f, 0.72f, 20.4f}, 0.85f)});
        tracker.collectActiveTracks(active);
        ASSERT_EQ(active.size(), 1u);
        EXPECT_EQ(active[0].trackId, baseId) << "Radial handoff jump spawned or dropped track ID";
        EXPECT_EQ(active[0].state, TrackState::Tracked) << "Radial handoff dropped Tracked state to Lost";
        EXPECT_LT(std::hypot(active[0].worldVel.x, active[0].worldVel.z), 2.5f)
            << "Single-frame radial jump spiked stationary target velocity";

        // FMEA-BT-02: +/-1.4m multi-frame monocular box quantization oscillation.
        for (int f = 0; f < 12; ++f) {
            const float z = 16.0f + (((f / 2) % 2 == 0) ? 1.4f : -1.4f);
            tracker.update(dt, ego, {testutil::makeDet3D(SemanticClass::Vehicle, {0.0f, 0.72f, z}, 0.88f)});
        }
        tracker.collectActiveTracks(active);
        ASSERT_EQ(active.size(), 1u);
        EXPECT_EQ(active[0].trackId, baseId);
        EXPECT_LT(std::hypot(active[0].worldVel.x, active[0].worldVel.z), 2.5f)
            << "Monocular radial quantization oscillation exploded stationary velocity";
    }
    {
        ByteTracker tracker(cfg);
        EgoState ego{};
        ego.pos = {0.0f, 0.0f, 0.0f};
        ego.yaw = 0.0f;
        std::vector<Track3D> active;
        for (int f = 0; f < 10; ++f) {
            tracker.update(dt, ego, {testutil::makeDet3D(SemanticClass::Vehicle, {0.0f, 0.72f, 10.3f}, 0.92f)});
        }
        // FMEA-BT-03: Close-range (<=12m) collinear same-class + cross-class telephoto partial detections.
        for (int f = 0; f < 5; ++f) {
            tracker.update(dt, ego, {
                testutil::makeDet3D(SemanticClass::Vehicle, {0.0f, 0.72f, 10.3f}, 0.80f, 1.0f, 0.25f, 0),
                testutil::makeDet3D(SemanticClass::Vehicle, {1.40f, 0.72f, 17.2f}, 0.80f, 1.0f, 0.25f, 1),
                testutil::makeDet3D(SemanticClass::Pedestrian, {0.38f, 0.90f, 4.95f}, 0.76f, 1.0f, 0.25f, 1)
            });
        }
        tracker.collectActiveTracks(active);
        EXPECT_EQ(active.size(), 1u) << "Close-range collinear partial detections spawned duplicate tracks";

        // Genuine adjacent-lane vehicle (+3.6m lateral) must still spawn its own distinct track.
        for (int f = 0; f < 3; ++f) {
            tracker.update(dt, ego, {
                testutil::makeDet3D(SemanticClass::Vehicle, {0.0f, 0.72f, 10.3f}, 0.90f, 1.0f, 0.25f, 0),
                testutil::makeDet3D(SemanticClass::Vehicle, {3.6f, 0.72f, 10.5f}, 0.90f, 1.0f, 0.25f, 0)
            });
        }
        tracker.collectActiveTracks(active);
        EXPECT_EQ(active.size(), 2u) << "Adjacent-lane vehicle was erroneously suppressed";
    }

    VulkanContext ctx;
    ctx.init(false, nullptr);
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perc(cfg);
    perc.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    const float loopLen = world.centerline().getLength();

    struct ChaseCase { const char* name; size_t vehIdx; float egoU; } cases[] = {
        {"CRV_SEDAN01_22M", 1, 0.25f},
        {"STR_SEDAN02_22M", 3, 0.03f},
    };
    for (const auto& c : cases) {
        world.reset();
        perc.reset();
        auto& ego = world.mutableEgo();
        ego.loopProgressU = c.egoU;
        ego.pos = world.laneCurve(0).getPointAt(c.egoU);
        const Vec3 eTan = world.laneCurve(0).getTangentAt(c.egoU);
        ego.yaw = std::atan2(eTan.x, eTan.z);
        ego.speed = 10.5f;

        auto& vehs = world.mutableVehicles();
        for (size_t i = 0; i < vehs.size(); ++i) {
            vehs[i].lane = 1;
            vehs[i].u = std::fmod(c.egoU + 0.45f + 0.05f * float(i), 1.0f);
            vehs[i].pos = world.laneCurve(1).getPointAt(vehs[i].u);
            const Vec3 t = world.laneCurve(1).getTangentAt(vehs[i].u);
            vehs[i].yaw = std::atan2(t.x, t.z);
            vehs[i].speed = 0.0f;
            vehs[i].hardBrakeTimer = 8.0f;
        }
        for (auto& p : world.mutablePedestrians()) { p.x = p.z = p.targetX = p.targetZ = 0.0f; }
        auto& v = vehs[c.vehIdx];
        v.lane = 0;
        v.u = std::fmod(c.egoU + 22.0f / loopLen, 1.0f);
        v.pos = world.laneCurve(0).getPointAt(v.u);
        const Vec3 vTan = world.laneCurve(0).getTangentAt(v.u);
        v.yaw = std::atan2(vTan.x, vTan.z);
        v.speed = v.vx = v.vz = 0.0f;
        v.hardBrakeTimer = 8.0f;

        PerceptionFrameOutput out;
        PlannerCommand cmd;
        float simT = 0.0f, maxSpd = 0.0f;
        int missTrk = 0, multiTrk = 0;
        for (int s = 0; s < 90; ++s) {
            rig.renderAllCameras(world, cfg, simT, nullptr);
            perc.runFrame(ctx, rig, world, dt, true, out, nullptr);
            planner.plan(world, out, cmd);
            const auto& gt = world.vehicles()[c.vehIdx];
            int nearVehTrk = 0, nearAllTrk = 0;
            for (const auto& tr : out.tracks) {
                const float err = std::hypot(tr.worldPos.x - gt.pos.x, tr.worldPos.z - gt.pos.z);
                if (err < 8.0f) {
                    nearAllTrk++;
                    if (tr.semantic == SemanticClass::Vehicle) nearVehTrk++;
                    if (s >= 15) maxSpd = std::max(maxSpd, std::hypot(tr.worldVel.x, tr.worldVel.z));
                }
            }
            if (s >= 2 && nearVehTrk == 0) missTrk++;
            if (nearAllTrk > 1) multiTrk++;
            world.stepAgents(dt);
            world.stepEgoAndPhysics(dt, cmd, ManualInput{});
            ASSERT_FALSE(world.lastContact().occurred) << c.name << " collided at s=" << s;
            simT += dt;
        }
        EXPECT_EQ(missTrk, 0) << c.name << " dropped vehicle track";
        EXPECT_LE(multiTrk, 2) << c.name << " spawned duplicate tracks near sedan";
        EXPECT_LT(maxSpd, 3.0f) << c.name << " stationary sedan track speed exploded";
    }

    perc.destroy();
    rig.destroy();
    ctx.cleanup();
}

} // namespace
} // namespace cybercab
