#include "config/SimConfig.hpp"
#include "world/WorldEngine.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace cybercab;

TEST(PhysicsSatTest, SatDetectsAndSeparatesOverlappingOBBs) {
    const OBB a{"A", 0.0f, 0.7f, 0.0f, 1.0f, 0.7f, 2.0f, 0.2f, 0, 0, SemanticClass::Vehicle, ""};
    const OBB b{"B", 1.4f, 0.7f, 0.3f, 1.0f, 0.7f, 2.0f, -0.1f, 0, 0, SemanticClass::Vehicle, ""};
    const SatHit hit = WorldEngine::checkSAT_OBB(a, b);
    EXPECT_TRUE(hit.hit);
    EXPECT_GT(hit.overlap, 0.0f);

    // Pushing A along MTV normal resolves overlap
    OBB aMoved = a;
    aMoved.cx += hit.normal.x * (hit.overlap + 0.02f);
    aMoved.cz += hit.normal.y * (hit.overlap + 0.02f);
    const SatHit after = WorldEngine::checkSAT_OBB(aMoved, b);
    EXPECT_FALSE(after.hit);
}

TEST(PhysicsSatTest, PedestrianCapsuleZeroInterpenetrationAgainstEgoOBB) {
    const SimConfig cfg = SimConfig::defaultConfig();
    WorldEngine world(cfg);
    const OBB egoObb = world.getEgoWorldOBB();

    Pedestrian ped;
    ped.id = "TEST_PED";
    ped.x = egoObb.cx + 0.3f;
    ped.z = egoObb.cz + 0.4f;
    ped.vx = -1.0f;
    ped.vz = 0.0f;
    ped.radius = 0.42f;

    const bool hit = WorldEngine::resolvePedestrianVsVehicleOBB(ped, egoObb, &world.mutableEgo());
    EXPECT_TRUE(hit);

    // Verify pedestrian is now strictly outside vehicle OBB skin
    const bool hitAfter = WorldEngine::resolvePedestrianVsVehicleOBB(ped, egoObb, nullptr);
    EXPECT_FALSE(hitAfter);
}

TEST(PhysicsSatTest, SuspensionSpringDamperStabilizesAfterBrakingJolt) {
    const SimConfig cfg = SimConfig::defaultConfig();
    WorldEngine world(cfg);
    PlannerCommand cmd;
    cmd.targetSpeed = 0.0f;
    cmd.aebActive = true;
    cmd.rssSafeDist = 10.0f;
    cmd.minObstacleDist = 5.0f;
    cmd.behaviorState = "AEB_EMERGENCY_BRAKE";

    const float dt = 1.0f / 36.0f;
    for (int i = 0; i < 45; ++i) {
        world.stepAgents(dt);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
    }
    EXPECT_NEAR(world.ego().speed, 0.0f, 0.1f);
    EXPECT_TRUE(std::isfinite(world.ego().pitch));
    EXPECT_TRUE(std::isfinite(world.ego().roll));
}

TEST(PhysicsSatTest, SatReturnsMissForSeparatedRotatedOBBs) {
    const OBB a{"A", 0.0f, 0.7f, 0.0f, 1.0f, 0.7f, 2.2f, 0.65f, 0, 0, SemanticClass::Vehicle, ""};
    const OBB b{"B", 4.6f, 0.7f, 1.5f, 1.0f, 0.7f, 2.2f, -0.45f, 0, 0, SemanticClass::Vehicle, ""};
    const SatHit hit = WorldEngine::checkSAT_OBB(a, b);
    EXPECT_FALSE(hit.hit);
}

TEST(PhysicsSatTest, PedestrianToPedestrianSeparationAndHelbingSignalHold) {
    const SimConfig cfg = SimConfig::defaultConfig();
    WorldEngine world(cfg);
    auto& peds = world.mutablePedestrians();
    ASSERT_GE(peds.size(), 2u);

    // Place two pedestrians overlapping right next to each other -> stepPhysics must push them apart
    peds[0].x = -7.6f; peds[0].z = 49.0f;
    peds[1].x = -7.7f; peds[1].z = 49.1f;
    const float initialDist = std::hypot(peds[0].x - peds[1].x, peds[0].z - peds[1].z);
    const float dt = 1.0f / 36.0f;
    world.stepAgents(dt);
    const float afterDist = std::hypot(peds[0].x - peds[1].x, peds[0].z - peds[1].z);
    EXPECT_GT(afterDist, initialDist);

    // Reset: signal-obeying pedestrian (PED_01) at crosswalk curb should wait when traffic light is GREEN
    world.reset();
    const float zStart = world.pedestrians()[0].z;
    for (int i = 0; i < 20; ++i) {
        world.stepAgents(dt);
    }
    EXPECT_NEAR(world.pedestrians()[0].z, zStart, 0.65f);
}

TEST(PhysicsSatTest, IdmRedLightHoldAndStaticBarrierContactFlash) {
    const SimConfig cfg = SimConfig::defaultConfig();
    WorldEngine world(cfg);
    world.setForcedRedSignal(true);
    const float dt = 1.0f / 36.0f;

    // Put TRUCK_01 right upstream of red stop line and verify IDM brakes cleanly
    auto& truck = world.mutableVehicles().at(0);
    truck.u = std::fmod(world.findClosestLoopU({-18.0f, 0.0f, 45.9f}), 1.0f);
    if (truck.u < 0.0f) truck.u += 1.0f;
    const float initSpd = truck.speed;
    for (int i = 0; i < 45; ++i) {
        world.stepAgents(dt);
    }
    EXPECT_LT(truck.speed, initSpd);

    // Verify driving ego into western barrier triggers SAT resolution & contactFlashAlpha
    world.reset();
    world.mutableEgo().pos = {-WorldEngine::kLoopHalfX - 5.1f, 0.0f, 0.0f};
    world.mutableEgo().yaw = -kPi * 0.5f;
    world.mutableEgo().speed = 5.0f;
    PlannerCommand manCmd;
    manCmd.targetSpeed = 5.0f;
    ManualInput man{};
    man.forward = true;
    world.stepEgoAndPhysics(dt, manCmd, man);
    EXPECT_TRUE(world.lastContact().occurred);
    EXPECT_GT(world.contactFlashAlpha(), 0.5f);
}

TEST(PhysicsSatTest, CybercabAdvancesAroundLoopInCruiseMode) {
    const SimConfig cfg = SimConfig::defaultConfig();
    WorldEngine world(cfg);
    const Vec3 startPos = world.ego().pos;

    PlannerCommand cmd;
    cmd.targetSpeed = 9.5f;
    cmd.behaviorState = "CRUISING";
    cmd.optimalTrajectory.resize(14);

    const float dt = 1.0f / 36.0f;
    for (int step = 0; step < 90; ++step) {
        const float u = world.findClosestLoopU(world.ego().pos);
        for (int s = 0; s < 14; ++s) {
            const float uS = std::fmod(u + (static_cast<float>(s) * 2.0f) / world.centerline().getLength(), 1.0f);
            cmd.optimalTrajectory[s] = world.laneCurve(0).getPointAt(uS);
        }
        world.stepAgents(dt);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
    }

    const float distMoved = (world.ego().pos - startPos).norm();
    EXPECT_GT(distMoved, 20.0f);
    EXPECT_GT(world.ego().speed, 8.0f);
    EXPECT_GT(world.ego().wheelSpin, 20.0f);
}
