#include "MetricsEvalHelpers.hpp"
#include "perception/v15/AudioTokenizer.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "planning/v15/AutoregSafety.hpp"
#include "planning/v15/AutoregressivePolicy.hpp"
#include "world/CybercabMesh.hpp"

#include <algorithm>
#include <cmath>

namespace cybercab {
namespace {

TEST(V15AutoregPlannerTest, ClosedLoopStraightAndCurvesStayStrictlyInMiddleOfLane) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    WorldEngine world(cfg);
    TrajectoryPlanner planner(cfg);

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput clearPerc{};
    PlannerCommand cmd;
    double maxDevFromLaneCenter = 0.0;
    float minLat = 10.0f, maxLat = -10.0f;

    for (int step = 0; step < 520; ++step) {
        planner.plan(world, clearPerc, cmd);
        EXPECT_EQ(cmd.gear, 1);
        EXPECT_EQ(cmd.candidateTrajectories.size(), 2u);
        world.stepEgoAndPhysics(dt, cmd, {});
        const float lat = world.ego().currentLateralOffset;
        minLat = std::min(minLat, lat);
        maxLat = std::max(maxLat, lat);
        maxDevFromLaneCenter = std::max(maxDevFromLaneCenter, static_cast<double>(std::abs(lat - (-1.95f))));
        ASSERT_GE(lat, -2.35f) << "Step " << step << " drifted toward outer curb limit (-4.23 m)";
        ASSERT_LE(lat, -1.55f) << "Step " << step << " drifted toward center divider (0.0 m)";
    }
    RecordProperty("m7_curve_min_lat_m", testutil::fmtProp(minLat));
    RecordProperty("m7_curve_max_lat_m", testutil::fmtProp(maxLat));
    RecordProperty("m7_max_center_dev_m", testutil::fmtProp(maxDevFromLaneCenter));
}

TEST(V15AutoregPlannerTest, TruckShyAwaySirenYieldAutoshiftReverseAndV14Parity) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    WorldEngine world(cfg);
    TrajectoryPlanner planner(cfg);
    const EgoState ego0 = world.ego();
    const float sE = std::sin(ego0.yaw), cE = std::cos(ego0.yaw);
    const Vec3 egoFwd{sE, 0.0f, cE}, egoLeft{cE, 0.0f, -sE};

    EXPECT_LE(v15::AutoregressivePolicy::clampToMiddleOfLane(-0.2f), -1.65f);
    EXPECT_GE(v15::AutoregressivePolicy::clampToMiddleOfLane(0.2f), 1.65f);

    Track3D adjTruck{};
    adjTruck.semantic = SemanticClass::Vehicle;
    adjTruck.worldPos = ego0.pos + egoFwd * 12.0f + egoLeft * 3.85f;
    adjTruck.halfExtents = {1.25f, 1.50f, 4.40f};
    const float shyLat = v15::AutoregressivePolicy::computeTruckShyOffset(-1.95f, ego0, {adjTruck});
    EXPECT_LT(shyLat, -2.03f);
    EXPECT_GE(shyLat, -2.25f);

    // Blind-corner emergency siren yield.
    world.triggerEmergencySiren();
    const v15::AudioToken aud = v15::AudioTokenizer::stepFromWorld(world, 0);
    ASSERT_TRUE(aud.sirenDetected);
    PerceptionFrameOutput sirenPerc{};
    sirenPerc.v15.active = true;
    sirenPerc.v15.sirenDetected = aud.sirenDetected;
    sirenPerc.v15.sirenConfidence = std::max(aud.confidence, 0.65f);
    PlannerCommand cmd;
    planner.plan(world, sirenPerc, cmd);
    EXPECT_EQ(cmd.behaviorState, "YIELDING_SIREN");
    EXPECT_LE(cmd.targetSpeed, 1.2f);
    EXPECT_EQ(cmd.bestCandIdx, 0);

    // Dead-end stopped vehicles blocking both lanes -> Autoshift Reverse when rear is clear.
    world.reset();
    world.mutableEgo().speed = 0.0f;
    PerceptionFrameOutput deadEndPerc{};
    deadEndPerc.v15.active = true;
    deadEndPerc.v15.rearClearanceM = 18.0f;
    for (float latOff : {-1.95f, 1.95f}) {
        Track3D blocker{};
        blocker.semantic = SemanticClass::Vehicle;
        blocker.worldPos = world.ego().pos + egoFwd * 6.0f + egoLeft * (latOff - world.ego().currentLateralOffset);
        blocker.halfExtents = {1.0f, 0.8f, 2.2f};
        blocker.posSigmaM = 0.2f;
        deadEndPerc.tracks.push_back(blocker);
    }
    planner.plan(world, deadEndPerc, cmd);
    EXPECT_EQ(cmd.gear, -1);
    EXPECT_LT(cmd.targetSpeed, -0.5f);
    EXPECT_EQ(cmd.behaviorState, "AUTOSHIFT_REVERSE");

    const Vec3 startPos = world.ego().pos;
    const float dt = 1.0f / cfg.controlHz;
    for (int i = 0; i < 18; ++i) world.stepEgoAndPhysics(dt, cmd, {});
    EXPECT_LT(world.ego().speed, -0.4f);
    EXPECT_LT((world.ego().pos - startPos).dot(egoFwd), -0.05f);

    // Inhibits reverse when rear is blocked.
    deadEndPerc.v15.rearClearanceM = 2.0f;
    planner.plan(world, deadEndPerc, cmd);
    EXPECT_EQ(cmd.gear, 1);
    EXPECT_FLOAT_EQ(cmd.targetSpeed, 0.0f);

    // v14 equivalence guard (C-EQ): 9 quintic candidates preserved under pipelineVersion = "v14".
    SimConfig v14Cfg = cfg;
    v14Cfg.pipelineVersion = "v14";
    TrajectoryPlanner v14Planner(v14Cfg);
    PlannerCommand v14Cmd;
    v14Planner.plan(world, PerceptionFrameOutput{}, v14Cmd);
    EXPECT_EQ(v14Cmd.candidateTrajectories.size(), 9u);
    EXPECT_EQ(v14Cmd.gear, 1);
    RecordProperty("m7_truck_shy_lat_m", testutil::fmtProp(shyLat));
}

TEST(V15AutoregPlannerTest, RolloutBlindSpotTruckHullAndSafetyHoldProtectAgainstCollisions) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    WorldEngine world(cfg);
    TrajectoryPlanner planner(cfg);
    const auto& cl = world.centerline();

    // 1. Autoregressive lateral rollout crosses >1.0 m within first half of lookahead (no quintic Zeno lag).
    std::vector<Vec3> traj;
    float maxK = 0.0f;
    v15::AutoregressivePolicy::rolloutMiddleOfLane(world.ego(), cl, world.ego().loopProgressU,
                                                    -1.95f, 1.95f, 24, 24.0f, traj, maxK, cfg.v15Policy);
    ASSERT_EQ(traj.size(), 24u);
    const float midU = world.findClosestLoopU(traj[10]);
    const Vec3 midT = cl.getTangentAt(midU);
    const float midLat = (traj[10] - cl.getPointAt(midU)).dot({midT.z, 0.0f, -midT.x});
    EXPECT_GT(midLat, -0.95f);
    const float endU = world.findClosestLoopU(traj.back());
    const Vec3 endT = cl.getTangentAt(endU);
    const float endLat = (traj.back() - cl.getPointAt(endU)).dot({endT.z, 0.0f, -endT.x});
    EXPECT_GE(endLat, cfg.v15Policy.minLaneAbsM);
    EXPECT_LE(endLat, cfg.v15Policy.maxLaneAbsM);

    // 2. Blind-spot side-by-side car, fast closing car, and 8.8 m truck hull block unsafe merge into Lane 1.
    const EgoState ego0 = world.ego();
    const Vec3 fwd{std::sin(ego0.yaw), 0.0f, std::cos(ego0.yaw)}, left{std::cos(ego0.yaw), 0.0f, -std::sin(ego0.yaw)};
    auto makeVeh = [&](float sF, float roadLat, float spd, float hz) {
        Track3D t{};
        t.semantic = SemanticClass::Vehicle;
        t.worldPos = ego0.pos + fwd * sF + left * (roadLat - ego0.currentLateralOffset);
        t.worldVel = fwd * spd;
        t.halfExtents = {1.0f, 0.75f, hz};
        t.yaw = ego0.yaw;
        t.score = 0.95f;
        t.hits = 4;
        return t;
    };
    PlannerCommand cmd;
    PerceptionFrameOutput blindPerc{};
    blindPerc.tracks = {makeVeh(11.0f, -1.95f, 2.0f, 2.2f), makeVeh(0.0f, 1.95f, 8.5f, 2.2f)};
    planner.plan(world, blindPerc, cmd);
    EXPECT_TRUE(cmd.candidateBlocked[1]);
    EXPECT_EQ(cmd.bestCandIdx, 0);

    blindPerc.tracks = {makeVeh(11.0f, -1.95f, 2.0f, 2.2f), makeVeh(-9.0f, 1.95f, ego0.speed + 6.0f, 2.2f)};
    planner.plan(world, blindPerc, cmd);
    EXPECT_TRUE(cmd.candidateBlocked[1]);
    EXPECT_EQ(cmd.bestCandIdx, 0);

    blindPerc.tracks = {makeVeh(11.0f, -1.95f, 2.0f, 2.2f), makeVeh(6.5f, 1.95f, 4.0f, 4.4f)};
    planner.plan(world, blindPerc, cmd);
    EXPECT_TRUE(cmd.candidateBlocked[1]);
    EXPECT_EQ(cmd.bestCandIdx, 0);

    // 3. Red-light intersection lock prevents swerving into adjacent crosswalk when lead queues at stop line.
    world.setForcedRedSignal(true);
    world.mutableEgo().pos = {-21.0f, 0.0f, WorldEngine::kLoopHalfZ + 1.90f};
    world.mutableEgo().yaw = kPi * 0.5f;
    world.mutableEgo().speed = 6.5f;
    world.mutableEgo().currentLateralOffset = -1.90f;
    PerceptionFrameOutput redPerc{};
    Track3D redQueue{};
    redQueue.semantic = SemanticClass::Vehicle;
    redQueue.worldPos = {-13.5f, 0.0f, WorldEngine::kLoopHalfZ + 1.90f};
    redQueue.halfExtents = {0.96f, 0.70f, 2.18f};
    redQueue.yaw = kPi * 0.5f;
    redQueue.score = 0.95f;
    Track3D cwPed{};
    cwPed.semantic = SemanticClass::Pedestrian;
    cwPed.worldPos = {-7.6f, 0.0f, WorldEngine::kLoopHalfZ - 1.90f};
    cwPed.score = 0.95f;
    redPerc.tracks = {redQueue, cwPed};
    for (int s = 0; s < 35; ++s) {
        planner.plan(world, redPerc, cmd);
        EXPECT_EQ(cmd.bestCandIdx, 0);
        world.stepEgoAndPhysics(1.0f / cfg.controlHz, cmd, {});
        ASSERT_FALSE(world.lastContact().occurred);
    }
    EXPECT_LT(world.ego().speed, 2.5f);

    // 4. AutoregSafety rotated OBB lateral overlap (1.75 m > old 1.22 m) + close-range (< 1.85 m) AEB hold.
    std::vector<Vec3> corrPts(16);
    std::vector<float> corrArc(16);
    const EgoState& e = world.ego();
    const float bFwd = e.hz + e.centerOffsetForward;
    for (int i = 0; i < 16; ++i) {
        corrArc[i] = static_cast<float>(i) * 1.5f;
        corrPts[i] = {e.pos.x + bFwd + corrArc[i], 0.08f, e.pos.z};
    }
    EgoState slowEgo = e;
    slowEgo.speed = 0.40f;
    Track3D angled{};
    angled.semantic = SemanticClass::Vehicle;
    angled.yaw = kPi * 0.5f + 0.35f;
    angled.halfExtents = {0.96f, 0.70f, 2.20f};
    angled.worldPos = {corrPts[0].x + 2.4f, 0.0f, corrPts[0].z - 1.75f};
    angled.score = 0.95f;
    angled.hits = 3;
    PerceptionFrameOutput obbPerc{};
    obbPerc.tracks = {angled};
    const v15::SafetyEvalOutput safe = v15::evaluateAutoregSafety(
        world, {cfg, slowEgo, obbPerc, corrPts, corrArc, 4.0f, 2.0f, false, false, nullptr}, "CRUISING", 8.0f);
    EXPECT_EQ(safe.behavior, "FOLLOWING_LEAD");
    EXPECT_FLOAT_EQ(safe.desiredSpeed, 0.0f);
    EXPECT_TRUE(safe.aebTriggered);
}

TEST(V15AutoregPlannerTest, CybercabMeshEmitsValidUnitNormalsAndGearBrakeStates) {
    auto evalMesh = [](int gear, bool braking, bool wire) -> double {
        EgoState e{};
        e.gear = gear;
        e.aebActive = braking;
        std::vector<SceneVertex> v;
        std::vector<uint32_t> idx;
        appendCybercabVehicleMesh(v, idx, e, Mat4::identity(), Mat4::identity(), wire);
        EXPECT_FALSE(v.empty());
        EXPECT_FALSE(idx.empty());
        EXPECT_EQ(idx.size() % 3u, 0u);
        for (const uint32_t id : idx) EXPECT_LT(id, v.size());
        double emSum = 0.0;
        for (const auto& vx : v) {
            const float nLen = std::sqrt(vx.nx * vx.nx + vx.ny * vx.ny + vx.nz * vx.nz);
            EXPECT_NEAR(nLen, 1.0f, 1e-3f);
            emSum += vx.emissive;
        }
        return emSum;
    };
    const double emCruise = evalMesh(1, false, false);
    const double emBrake = evalMesh(1, true, false);
    const double emRev = evalMesh(-1, false, false);
    const double emWire = evalMesh(1, false, true);
    EXPECT_GT(emBrake, emCruise);
    EXPECT_GT(emRev, emCruise);
    EXPECT_GT(emWire, 1.0);
}

} // namespace
} // namespace cybercab
