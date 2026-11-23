#include "MetricsEvalHelpers.hpp"
#include "perception/v15/AudioTokenizer.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "planning/v15/AutoregressivePolicy.hpp"

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

} // namespace
} // namespace cybercab
