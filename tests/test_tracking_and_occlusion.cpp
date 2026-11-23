#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <unordered_map>

namespace cybercab {
namespace {

using TrackingAndOcclusionTest = testutil::SharedGpuMetricsTest;

TEST_F(TrackingAndOcclusionTest, ClearMotIdSwitchingAndClassConsistencyAcrossScenarios) {
    const SimConfig baseCfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    const float dt = 1.0f / baseCfg.controlHz;
    VulkanContext& ctx = s_ctx;

    struct MotSummary {
        double mota = 0.0;
        double motp = 0.0;
        int idsw = 0;
        int frag = 0;
        int dupGt = 0;
        int clsSwaps = 0;
        bool collision = false;
    };

    auto runMotScenario = [&](const char* name, int steps, bool soiledFront) -> MotSummary {
        SimConfig cfg = baseCfg;
        if (soiledFront) {
            cfg.cameras[0].soiled = cfg.cameras[2].soiled = cfg.cameras[3].soiled = true;
        }
        WorldEngine world(cfg);
        if (std::string(name) == "jaywalk") world.triggerJaywalkerCutIn();
        else if (std::string(name) == "hardbrake") world.triggerLeadHardBrake();
        CameraRigRenderer rig;
        rig.init(ctx, cfg, world);
        PerceptionPipeline perception(cfg);
        perception.init(ctx, rig);
        TrajectoryPlanner planner(cfg);

        std::unordered_map<std::string, int> lastTrackId;
        std::unordered_map<std::string, SemanticClass> lastTrackSem;
        std::unordered_map<std::string, bool> wasTracked;

        MotSummary res{};
        int gtObs = 0, tp = 0, fp = 0, fn = 0;
        double distSum = 0.0;
        PerceptionFrameOutput out;
        PlannerCommand cmd;

        for (int s = 0; s < steps; ++s) {
            rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt, nullptr);
            perception.runFrame(ctx, rig, world, dt, true, out, nullptr);
            planner.plan(world, out, cmd);
            if (world.lastContact().occurred) res.collision = true;

            if (s >= 2) {
                const EgoState& ego = world.ego();
                const auto dyn = world.dynamicGroundTruthOBBs();
                std::vector<OBB> gtInRange;
                for (const auto& g : dyn) {
                    if (std::hypot(g.cx - ego.pos.x, g.cz - ego.pos.z) <= 35.0f) gtInRange.push_back(g);
                }
                gtObs += static_cast<int>(gtInRange.size());

                std::vector<uint8_t> trUsed(out.tracks.size(), 0);
                std::unordered_map<std::string, int> gtMatchCount;

                for (const auto& g : gtInRange) {
                    int bestTi = -1;
                    float bestD = (g.semantic == SemanticClass::Pedestrian) ? 3.5f : 5.5f;
                    for (size_t ti = 0; ti < out.tracks.size(); ++ti) {
                        if (trUsed[ti]) continue;
                        const float d = std::hypot(out.tracks[ti].worldPos.x - g.cx, out.tracks[ti].worldPos.z - g.cz);
                        if (d < bestD) { bestD = d; bestTi = static_cast<int>(ti); }
                    }
                    if (bestTi >= 0) {
                        trUsed[bestTi] = 1;
                        ++tp;
                        distSum += bestD;
                        const auto& tr = out.tracks[bestTi];
                        gtMatchCount[g.id]++;
                        if (lastTrackId.count(g.id) && lastTrackId[g.id] != tr.trackId) ++res.idsw;
                        if (lastTrackSem.count(g.id) && lastTrackSem[g.id] != tr.semantic) ++res.clsSwaps;
                        if (wasTracked[g.id] && tr.state == TrackState::Lost) ++res.frag;
                        lastTrackId[g.id] = tr.trackId;
                        lastTrackSem[g.id] = tr.semantic;
                        wasTracked[g.id] = (tr.state == TrackState::Tracked);
                    } else {
                        ++fn;
                    }
                }
                for (const auto& kv : gtMatchCount) {
                    if (kv.second > 1) res.dupGt += kv.second - 1;
                }
                for (size_t ti = 0; ti < out.tracks.size(); ++ti) {
                    if (!trUsed[ti] && std::hypot(out.tracks[ti].egoPos.x, out.tracks[ti].egoPos.y) <= 35.0f) {
                        float minGtD = 999.0f;
                        for (const auto& g : dyn) {
                            minGtD = std::min(minGtD, std::hypot(out.tracks[ti].worldPos.x - g.cx,
                                                                 out.tracks[ti].worldPos.z - g.cz));
                        }
                        if (minGtD > 5.5f) ++fp;
                    }
                }
            }
            world.stepAgents(dt);
            world.stepEgoAndPhysics(dt, cmd, ManualInput{});
        }

        res.mota = (gtObs > 0) ? (1.0 - static_cast<double>(fn + fp + res.idsw) / gtObs) : 0.0;
        res.motp = (tp > 0) ? (distSum / tp) : 0.0;
        perception.destroy();
        rig.destroy();
        return res;
    };

    const MotSummary nom = runMotScenario("nominal", 90, false);
    const MotSummary jay = runMotScenario("jaywalk", 60, false);
    const MotSummary hb = runMotScenario("hardbrake", 65, false);
    const MotSummary soil = runMotScenario("soil_front", 65, true);

    RecordProperty("mota_nominal", testutil::fmtProp(nom.mota));
    RecordProperty("mota_jaywalk", testutil::fmtProp(jay.mota));
    RecordProperty("mota_hardbrake", testutil::fmtProp(hb.mota));
    RecordProperty("mota_soil_front", testutil::fmtProp(soil.mota));

    for (const auto& m : {nom, jay, hb, soil}) {
        EXPECT_FALSE(m.collision);
        EXPECT_EQ(m.idsw, 0);
        EXPECT_EQ(m.dupGt, 0);
        EXPECT_EQ(m.clsSwaps, 0);
        EXPECT_LE(m.motp, 1.05);
    }
    EXPECT_GE(nom.mota, 0.56);
    EXPECT_GE(jay.mota, 0.56);
    EXPECT_GE(hb.mota, 0.65);
    EXPECT_GE(soil.mota, 0.40);
}

TEST_F(TrackingAndOcclusionTest, OcclusionShadowDetectionAndEmergenceTrackingLatency) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext& ctx = s_ctx;

    WorldEngine world(cfg);
    world.triggerOccludedDartOut();
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    PlannerCommand cmd;

    int shadowBehind = 0, shadowInFront = 0;
    float minEgoSpd = 999.0f;
    for (int s = 0; s < 90; ++s) {
        rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt, nullptr);
        perception.runFrame(ctx, rig, world, dt, true, out, nullptr);
        planner.plan(world, out, cmd);
        EXPECT_FALSE(world.lastContact().occurred);
        minEgoSpd = std::min(minEgoSpd, world.ego().speed);

        const auto& trk = world.vehicles().at(0);
        const float trkDist = std::hypot(trk.pos.x - world.ego().pos.x, trk.pos.z - world.ego().pos.z);
        for (const auto& sv : out.shadowVoxels) {
            const float svDist = std::hypot(sv.wx - world.ego().pos.x, sv.wz - world.ego().pos.z);
            if (svDist > trkDist - trk.halfExtents.z * 0.5f) ++shadowBehind;
            else ++shadowInFront;
        }
        world.stepAgents(dt);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
    }
    EXPECT_GE(shadowBehind, 400);
    EXPECT_LT(shadowInFront, std::max(50, shadowBehind / 15));
    EXPECT_LT(minEgoSpd, 2.2f);

    rig.renderAllCameras(world, cfg, 2.0f, nullptr);
    perception.runFrame(ctx, rig, world, dt, false, out, nullptr);
    EXPECT_EQ(out.shadowVoxels.size(), 0u);

    world.reset();
    perception.reset();
    for (size_t i = 1; i < world.mutableVehicles().size(); ++i) {
        auto& v = world.mutableVehicles()[i];
        v.pos = {-350.0f - static_cast<float>(i) * 20.0f, 0.0f, -350.0f};
        v.speed = v.vx = v.vz = 0.0f;
    }
    for (auto& p : world.mutablePedestrians()) {
        p.x = -320.0f; p.z = -320.0f;
        p.ax = p.bx = p.targetX = p.x;
        p.az = p.bz = p.targetZ = p.z;
    }
    const EgoState& ego0 = world.ego();
    auto& truck = world.mutableVehicles().at(0);
    truck.pos = {ego0.pos.x + 14.5f, 0.0f, ego0.pos.z};
    truck.yaw = kPi * 0.5f;
    truck.speed = truck.vx = truck.vz = 0.0f;

    auto& ped = world.mutablePedestrians().at(5);
    ped.shirtColor = {0.976f, 0.451f, 0.086f};
    ped.jaywalking = true;
    ped.x = ego0.pos.x + 18.2f;
    ped.z = ego0.pos.z - 0.5f;
    ped.ax = ped.x; ped.az = ped.z;
    ped.bx = ped.x; ped.bz = ego0.pos.z - 5.5f;
    ped.targetX = ped.bx; ped.targetZ = ped.bz;
    ped.obeySignal = false;
    ped.baseSpeed = 2.6f;

    int emergeStep = -1, detStep = -1, trkStep = -1;
    for (int s = 0; s < 75; ++s) {
        world.stepAgents(dt);
        if (s > 0 && world.ego().speed > 2.5f) world.mutableEgo().speed = 2.5f;
        rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt, nullptr);
        perception.runFrame(ctx, rig, world, dt, true, out, nullptr);

        if (emergeStep < 0 && std::abs(ped.z - truck.pos.z) > truck.halfExtents.x + 0.55f) emergeStep = s;
        if (emergeStep >= 0) {
            if (detStep < 0) {
                for (const auto& d : out.detections3D) {
                    if (d.semantic == SemanticClass::Pedestrian &&
                        std::hypot(d.worldPos.x - ped.x, d.worldPos.z - ped.z) < 3.8f) { detStep = s; break; }
                }
            }
            if (trkStep < 0) {
                for (const auto& tr : out.tracks) {
                    if (tr.semantic == SemanticClass::Pedestrian &&
                        std::hypot(tr.worldPos.x - ped.x, tr.worldPos.z - ped.z) < 3.8f) { trkStep = s; break; }
                }
            }
        }
        if (detStep >= 0 && trkStep >= 0) break;
    }

    ASSERT_GE(emergeStep, 0);
    ASSERT_GE(detStep, emergeStep);
    ASSERT_GE(trkStep, emergeStep);
    EXPECT_LE(detStep - emergeStep, 8);
    EXPECT_LE(trkStep - emergeStep, 12);

    perception.destroy();
    rig.destroy();
}

} // namespace
} // namespace cybercab
