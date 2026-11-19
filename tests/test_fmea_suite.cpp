#include <gtest/gtest.h>

#include "TestPerceptionHelpers.hpp"
#include "core/GpuProfiler.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/SimulationSync.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <vector>

namespace cybercab {
namespace {

Vec3 sampleCamRgb(const CameraRigRenderer& rig, int camIdx, float cxNorm, float cyNorm) {
    const auto rgba = rig.downloadCameraRGBA(camIdx);
    const int px = clampf(cxNorm * static_cast<float>(rig.camWidth()), 0.0f, static_cast<float>(rig.camWidth() - 1));
    const int py = clampf(cyNorm * static_cast<float>(rig.camHeight()), 0.0f, static_cast<float>(rig.camHeight() - 1));
    const size_t idx = (static_cast<size_t>(py) * rig.camWidth() + px) * 4u;
    return {rgba[idx] / 255.0f, rgba[idx + 1] / 255.0f, rgba[idx + 2] / 255.0f};
}

TEST(FMEAPerceptionTest, GhostPedestrianZeroToleranceWithoutPedestriansAndNominalAudit) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    for (auto& p : world.mutablePedestrians()) {
        p.x = -300.0f; p.z = -300.0f;
        p.ax = p.x; p.az = p.z;
        p.bx = p.x; p.bz = p.z;
        p.targetX = p.x; p.targetZ = p.z;
    }

    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);

    PerceptionFrameOutput out;
    std::ostringstream ghostLog;
    int staticGhostPeds = 0;

    const float yaws[3] = {kPi * 0.5f, kPi * 0.5f + 0.18f, kPi * 0.5f - 0.18f};
    for (int f = 0; f < 3; ++f) {
        world.mutableEgo().yaw = yaws[f];
        perception.reset();
        rig.renderAllCameras(world, cfg, 0.0f, nullptr);
        perception.runFrame(ctx, rig, world, 1.0f / 36.0f, true, out, nullptr);

        for (const auto& d : out.detections3D) {
            if (d.semantic == SemanticClass::Pedestrian) {
                ++staticGhostPeds;
                const Vec3 rgb = sampleCamRgb(rig, d.camIndex, d.det2d.cxNorm, d.det2d.cyNorm);
                ghostLog << "  [STATIC GHOST PED] yawIdx=" << f << " cam=" << d.camIndex
                         << " sc=" << d.score << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm
                         << "," << d.det2d.wNorm << "," << d.det2d.hNorm << ")"
                         << " rgb=(" << rgb.x << "," << rgb.y << "," << rgb.z << ")"
                         << " w=(" << d.worldPos.x << "," << d.worldPos.z << ")\n";
            }
        }
    }

    world.reset();
    perception.reset();
    TrajectoryPlanner planner(cfg);
    SimulationSyncClock clock;
    const float dt = 1.0f / cfg.controlHz;
    PlannerCommand cmd;
    int nominalGhostPeds = 0;
    std::ostringstream nomLog;

    for (int step = 0; step < 60; ++step) {
        clock.stepFrame(dt, dt, dt * 2.0f,
            [&](float simTime, float percDt) {
                rig.renderAllCameras(world, cfg, simTime, nullptr);
                perception.runFrame(ctx, rig, world, percDt, cfg.occEnableShadows, out, nullptr);
            },
            [&](int tickIdx, float, float tickDt) {
                if (out.eval.ghostPedDetections > 0) {
                    nominalGhostPeds += out.eval.ghostPedDetections;
                    for (const auto& d : out.detections3D) {
                        if (d.semantic != SemanticClass::Pedestrian || d.rangeM > 42.0f) continue;
                        float minPed = 999.0f;
                        int minPedIdx = -1;
                        for (size_t pi = 0; pi < world.pedestrians().size(); ++pi) {
                            const auto& p = world.pedestrians()[pi];
                            const float dp = std::hypot(d.worldPos.x - p.x, d.worldPos.z - p.z);
                            if (dp < minPed) { minPed = dp; minPedIdx = static_cast<int>(pi); }
                        }
                        float minVeh = 999.0f;
                        std::string minVehLbl;
                        for (const auto& v : world.vehicles()) {
                            const float dv = std::hypot(d.worldPos.x - v.pos.x, d.worldPos.z - v.pos.z);
                            if (dv < minVeh) { minVeh = dv; minVehLbl = v.id; }
                        }
                        if (minPed >= 3.5f) {
                            const Vec3 rgb = sampleCamRgb(rig, d.camIndex, d.det2d.cxNorm, d.det2d.cyNorm);
                            const auto& gp = world.pedestrians()[minPedIdx];
                            const float gtRange = std::hypot(gp.x - world.ego().pos.x, gp.z - world.ego().pos.z);
                            std::string uvMatch = "none";
                            float bestUv = 999.0f;
                            for (const auto& obb : world.allGroundTruthOBBs()) {
                                Vec2 px{}; float depth = 0.0f;
                                if (rig.projectWorldToPixel(d.camIndex, {obb.cx, obb.cy, obb.cz}, px, depth)) {
                                    const float uvDist = std::hypot(px.x / rig.camWidth() - d.det2d.cxNorm,
                                                                    px.y / rig.camHeight() - d.det2d.cyNorm);
                                    if (uvDist < bestUv) {
                                        bestUv = uvDist;
                                        uvMatch = obb.id + "(duv=" + std::to_string(uvDist).substr(0, 5) + ",depth=" + std::to_string(depth).substr(0, 4) + ")";
                                    }
                                }
                            }
                            nomLog << "  [NOM s=" << tickIdx << " GHOST PED] cam=" << d.camIndex
                                   << " sc=" << d.score << " r=" << d.rangeM
                                   << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm
                                   << "," << d.det2d.wNorm << "," << d.det2d.hNorm << ")"
                                   << " rgb=(" << rgb.x << "," << rgb.y << "," << rgb.z << ")"
                                   << " w=(" << d.worldPos.x << "," << d.worldPos.z << ")"
                                   << " uvHit=" << uvMatch
                                   << " ped=" << gp.id << "(err=" << minPed << ",gtR=" << gtRange << ")\n";
                        }
                    }
                }
                planner.plan(world, out, cmd);
                world.stepAgents(tickDt);
                world.stepEgoAndPhysics(tickDt, cmd, ManualInput{});
            });
    }

    perception.destroy();
    rig.destroy();
    ctx.cleanup();

    EXPECT_EQ(staticGhostPeds, 0) << "Static environment generated ghost pedestrians:\n" << ghostLog.str();
    EXPECT_EQ(nominalGhostPeds, 0) << "Nominal closed-loop generated ghost pedestrians:\n" << nomLog.str();
}

TEST(FMEAPerceptionTest, TruckFullVolumeVehicleVoxelsAndClassAcrossRanges) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    // Isolate TRUCK_01 without nearby pedestrians so any Pedestrian class or voxel inside TRUCK_01 is an isolated defect
    for (auto& p : world.mutablePedestrians()) {
        p.x = -300.0f; p.z = -300.0f;
        p.ax = p.x; p.az = p.z;
        p.bx = p.x; p.bz = p.z;
        p.targetX = p.x; p.targetZ = p.z;
    }

    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);

    PerceptionFrameOutput out;
    const float ranges[] = {22.0f, 12.5f, 8.0f, 18.0f, 20.0f};
    const float yawOffsets[] = {0.0f, 0.0f, 0.35f, 2.35f, kPi};

    for (size_t c = 0; c < 5; ++c) {
        const EgoState& ego = world.ego();
        const float sinE = std::sin(ego.yaw);
        const float cosE = std::cos(ego.yaw);
        auto& truck = world.mutableVehicles().at(0);
        truck.pos = {ego.pos.x + sinE * ranges[c], 0.0f, ego.pos.z + cosE * ranges[c]};
        truck.yaw = ego.yaw + yawOffsets[c];
        truck.speed = 0.0f;
        truck.vx = 0.0f;
        truck.vz = 0.0f;
        perception.reset();

        // Check single-frame visual occupancy before temporal/track smoothing as well as settled 3-frame output
        rig.renderAllCameras(world, cfg, 0.0f, nullptr);
        perception.runFrame(ctx, rig, world, 0.05f, cfg.occEnableShadows, out, nullptr);
        EXPECT_EQ(out.eval.truckPedVoxels, 0)
            << "Frame-0 visual occupancy generated Pedestrian voxels (" << out.eval.truckPedVoxels
            << ") on TRUCK_01 at range=" << ranges[c] << " yawOff=" << yawOffsets[c];
        EXPECT_GE(out.eval.truckVehVoxels, 6)
            << "Frame-0 visual occupancy generated too few Vehicle voxels (" << out.eval.truckVehVoxels
            << ") on TRUCK_01 at range=" << ranges[c] << " yawOff=" << yawOffsets[c];

        for (int settle = 1; settle < 3; ++settle) {
            rig.renderAllCameras(world, cfg, static_cast<float>(settle) * 0.05f, nullptr);
            perception.runFrame(ctx, rig, world, 0.05f, cfg.occEnableShadows, out, nullptr);
        }

        const float cosY = std::cos(truck.yaw), sinY = std::sin(truck.yaw);
        float minLocX = 99.0f, maxLocX = -99.0f, maxWy = 0.0f;
        for (const auto& v : out.activeVoxels) {
            const float dx = v.wx - truck.pos.x;
            const float dz = v.wz - truck.pos.z;
            const float locX = dx * cosY - dz * sinY;
            const float locZ = dx * sinY + dz * cosY;
            if (v.sem == SemanticClass::Vehicle &&
                std::abs(locX) <= truck.halfExtents.x + 0.6f &&
                std::abs(locZ) <= truck.halfExtents.z + 0.6f) {
                minLocX = std::min(minLocX, locX);
                maxLocX = std::max(maxLocX, locX);
                maxWy = std::max(maxWy, v.wy);
            }
        }
        const float latSpan = (maxLocX >= minLocX) ? (maxLocX - minLocX) : 0.0f;

        bool hasVehDet = false;
        float bestVehDist = 999.0f;
        std::ostringstream detDiag;
        for (const auto& d : out.detections3D) {
            const float dist = std::hypot(d.worldPos.x - truck.pos.x, d.worldPos.z - truck.pos.z);
            if (d.semantic == SemanticClass::Vehicle) {
                bestVehDist = std::min(bestVehDist, dist);
                if (dist < 5.0f) hasVehDet = true;
            }
            detDiag << "\n  [3D] cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
                    << " dist=" << dist << " r=" << d.rangeM << " sc=" << d.score
                    << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm
                    << "," << d.det2d.wNorm << "," << d.det2d.hNorm << ") he=("
                    << d.halfExtents.x << "," << d.halfExtents.y << "," << d.halfExtents.z << ")";
        }
        for (const auto& d2 : out.detections2D) {
            if (d2.camIndex <= 2 && d2.semantic != SemanticClass::SignalSign) {
                detDiag << "\n  [2D] cam=" << d2.camIndex << " sem=" << static_cast<int>(d2.semantic)
                        << " sc=" << d2.score << " box=(" << d2.cxNorm << "," << d2.cyNorm
                        << "," << d2.wNorm << "," << d2.hNorm << ")";
            }
        }

        EXPECT_EQ(out.eval.misclassifiedTrucks, 0)
            << "Truck misclassified as Pedestrian at range=" << ranges[c] << detDiag.str();
        EXPECT_TRUE(hasVehDet)
            << "Missing Vehicle 3D box on TRUCK_01 at range=" << ranges[c] << " bestDist=" << bestVehDist << detDiag.str();
        EXPECT_EQ(out.eval.truckPedVoxels, 0)
            << "TRUCK_01 produced Pedestrian voxels (" << out.eval.truckPedVoxels << ") at range=" << ranges[c];
        EXPECT_GE(out.eval.truckVehVoxels, 12)
            << "TRUCK_01 produced partial Vehicle voxels (" << out.eval.truckVehVoxels << ") at range=" << ranges[c];
        EXPECT_GE(latSpan, 1.5f)
            << "TRUCK_01 Vehicle voxel lateral span too narrow (" << latSpan << "m) at range=" << ranges[c] << detDiag.str();
        EXPECT_GE(maxWy, 2.2f)
            << "TRUCK_01 upper cargo wall missing from voxels (maxWy=" << maxWy << "m) at range=" << ranges[c] << detDiag.str();
    }

    // Closed-loop occluded stopped-truck audit: verify 0 misclassified trucks, 0 truck ped voxels, full volume
    world.reset();
    world.triggerOccludedDartOut();
    perception.reset();
    TrajectoryPlanner planner(cfg);
    SimulationSyncClock clock;
    PlannerCommand cmd;
    const float dtControl = 1.0f / cfg.controlHz;
    int worstMisTruck = 0, worstTruckPedVox = 0, minTruckVehVox = 999;
    std::ostringstream pedVoxDiag;

    for (int step = 0; step < 30; ++step) {
        clock.stepFrame(dtControl, dtControl, dtControl * 2.0f,
            [&](float simTime, float percDt) {
                rig.renderAllCameras(world, cfg, simTime, nullptr);
                perception.runFrame(ctx, rig, world, percDt, cfg.occEnableShadows, out, nullptr);
                if (step >= 2) {
                    worstMisTruck = std::max(worstMisTruck, out.eval.misclassifiedTrucks);
                    if (out.eval.truckPedVoxels > worstTruckPedVox) {
                        worstTruckPedVox = out.eval.truckPedVoxels;
                        pedVoxDiag.str("");
                        pedVoxDiag << "\nWorst step=" << step << " ego=(" << world.ego().pos.x << "," << world.ego().pos.z
                                   << ",yaw=" << world.ego().yaw << ",spd=" << world.ego().speed << ")";
                        const auto& tr = world.vehicles().at(0);
                        pedVoxDiag << " truck=(" << tr.pos.x << "," << tr.pos.z << ",yaw=" << tr.yaw << ")";
                        const float cosY = std::cos(tr.yaw), sinY = std::sin(tr.yaw);
                        for (const auto& v : out.activeVoxels) {
                            const float dx = v.wx - tr.pos.x, dz = v.wz - tr.pos.z;
                            const float lx = dx * cosY - dz * sinY, lz = dx * sinY + dz * cosY;
                            if (v.sem == SemanticClass::Pedestrian &&
                                std::abs(lx) <= tr.halfExtents.x + 0.6f &&
                                std::abs(lz) <= tr.halfExtents.z + 0.6f &&
                                v.wy <= tr.halfExtents.y * 2.0f + 0.6f) {
                                pedVoxDiag << "\n  [PVOX] loc=(" << lx << "," << v.wy << "," << lz << ") conf=" << v.confidence;
                            }
                        }
                        for (const auto& tk : out.tracks) {
                            const float dx = tk.worldPos.x - tr.pos.x, dz = tk.worldPos.z - tr.pos.z;
                            pedVoxDiag << "\n  [TRK] id=" << tk.trackId << " sem=" << static_cast<int>(tk.semantic)
                                       << " dTruck=(" << dx << "," << dz << ") sc=" << tk.score;
                        }
                    }
                    minTruckVehVox = std::min(minTruckVehVox, out.eval.truckVehVoxels);
                }
            },
            [&](int tickIdx, float, float tickDt) {
                planner.plan(world, out, cmd);
                world.stepAgents(tickDt);
                world.stepEgoAndPhysics(tickDt, cmd, ManualInput{});
                EXPECT_FALSE(world.lastContact().occurred)
                    << "Collision with " << world.lastContact().otherId << " at tick " << tickIdx;
            });
    }

    EXPECT_EQ(worstMisTruck, 0) << "Occluded truck misclassified as Pedestrian during closed-loop braking";
    EXPECT_EQ(worstTruckPedVox, 0) << "Occluded truck generated Pedestrian-class voxels (" << worstTruckPedVox << ")" << pedVoxDiag.str();
    EXPECT_GE(minTruckVehVox, 12) << "Occluded truck Vehicle voxel count dropped to " << minTruckVehVox;

    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, AllVehicleColorsAndAnglesReceiveBoundingBoxes) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    for (auto& p : world.mutablePedestrians()) {
        p.x = -300.0f; p.z = -300.0f;
        p.ax = p.x; p.az = p.z;
        p.bx = p.x; p.bz = p.z;
        p.targetX = p.x; p.targetZ = p.z;
    }

    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);

    PerceptionFrameOutput out;
    const float yawOffsets[] = {0.0f, kPi * 0.25f, kPi};
    const char* yawLabels[] = {"rear", "oblique45", "front"};
    const EgoState& ego = world.ego();
    const float sinE = std::sin(ego.yaw);
    const float cosE = std::cos(ego.yaw);
    const size_t nVeh = world.vehicles().size();

    for (size_t vi = 0; vi < nVeh; ++vi) {
        for (size_t ai = 0; ai < 3; ++ai) {
            for (size_t k = 0; k < nVeh; ++k) {
                auto& vk = world.mutableVehicles()[k];
                vk.speed = 0.0f;
                vk.vx = 0.0f;
                vk.vz = 0.0f;
                if (k == vi) {
                    vk.pos = {ego.pos.x + sinE * 18.0f, 0.0f, ego.pos.z + cosE * 18.0f};
                    vk.yaw = ego.yaw + yawOffsets[ai];
                } else {
                    vk.pos = {-400.0f - static_cast<float>(k) * 20.0f, 0.0f, -400.0f};
                }
            }
            perception.reset();
            rig.renderAllCameras(world, cfg, 0.0f, nullptr);
            perception.runFrame(ctx, rig, world, 0.05f, cfg.occEnableShadows, out, nullptr);

            const auto& tv = world.vehicles()[vi];
            bool has2D = false;
            bool has3D = false;
            float bestDist = 999.0f;
            float bestScore = 0.0f;
            std::ostringstream vehDiag;
            for (const auto& d2 : out.detections2D) {
                if ((d2.camIndex == 0 || d2.camIndex == 1 || d2.camIndex == 2) &&
                    d2.semantic == SemanticClass::Vehicle && d2.score >= cfg.rtdetrLowConfThresh) {
                    has2D = true;
                }
                if (d2.camIndex <= 2) {
                    vehDiag << "\n  [2D] cam=" << d2.camIndex << " sem=" << static_cast<int>(d2.semantic)
                            << " sc=" << d2.score << " box=(" << d2.cxNorm << "," << d2.cyNorm
                            << "," << d2.wNorm << "," << d2.hNorm << ")";
                }
            }
            for (const auto& d3 : out.detections3D) {
                const float dist = std::hypot(d3.worldPos.x - tv.pos.x, d3.worldPos.z - tv.pos.z);
                if (d3.semantic == SemanticClass::Vehicle) {
                    bestDist = std::min(bestDist, dist);
                    bestScore = std::max(bestScore, d3.score);
                    if (dist <= 4.5f) has3D = true;
                }
                if (d3.camIndex <= 2) {
                    vehDiag << "\n  [3D] cam=" << d3.camIndex << " sem=" << static_cast<int>(d3.semantic)
                            << " sc=" << d3.score << " r=" << d3.rangeM << " dist=" << dist
                            << " he=(" << d3.halfExtents.x << "," << d3.halfExtents.y << "," << d3.halfExtents.z << ")";
                }
            }

            Vec2 projPx{};
            float projDepth = 0.0f;
            rig.projectWorldToPixel(0, {tv.pos.x, tv.halfExtents.y, tv.pos.z}, projPx, projDepth);
            const Vec3 ctrRgb = sampleCamRgb(rig, 0, projPx.x / static_cast<float>(rig.camWidth()),
                                             projPx.y / static_cast<float>(rig.camHeight()));

            EXPECT_TRUE(has2D)
                << "Vehicle " << tv.id << " (" << yawLabels[ai] << ") missing 2D Vehicle bounding box; centerRGB=("
                << ctrRgb.x << "," << ctrRgb.y << "," << ctrRgb.z << ")" << vehDiag.str();
            EXPECT_TRUE(has3D)
                << "Vehicle " << tv.id << " (" << yawLabels[ai] << ") missing 3D Vehicle bounding box within 4.5m; bestDist="
                << bestDist << " maxVehScore=" << bestScore << " centerRGB=("
                << ctrRgb.x << "," << ctrRgb.y << "," << ctrRgb.z << ")" << vehDiag.str();
        }
    }

    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, TrafficSignsLightsAndPolesPopulateOccupancyVoxels) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    for (auto& p : world.mutablePedestrians()) {
        p.x = -300.0f; p.z = -300.0f;
        p.ax = p.x; p.az = p.z;
        p.bx = p.x; p.bz = p.z;
        p.targetX = p.x; p.targetZ = p.z;
    }
    for (size_t k = 0; k < world.vehicles().size(); ++k) {
        auto& vk = world.mutableVehicles()[k];
        vk.pos = {-400.0f - static_cast<float>(k) * 20.0f, 0.0f, -400.0f};
        vk.speed = 0.0f;
        vk.vx = 0.0f;
        vk.vz = 0.0f;
    }

    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);

    struct SignCheck {
        const char* name;
        Vec3 egoPos;
        float egoYaw;
        Vec2 anchorXZ;
        float minHighWy;
    };
    const SignCheck checks[] = {
        {"SIGN_SPEED35", {-48.0f, 0.0f, WorldEngine::kLoopHalfZ}, kPi * 0.5f, {-34.0f, WorldEngine::kLoopHalfZ + 5.6f}, 2.0f},
        {"SIGN_STOP",    { 26.5f, 0.0f, -WorldEngine::kLoopHalfZ}, -kPi * 0.5f, {12.5f, -WorldEngine::kLoopHalfZ - 5.6f}, 2.0f},
        {"SIGNAL_MAST",  {-10.0f, 0.0f, WorldEngine::kLoopHalfZ}, kPi * 0.5f, {6.5f, WorldEngine::kLoopHalfZ - 6.2f}, 2.0f},
    };

    PerceptionFrameOutput out;
    for (const auto& sc : checks) {
        world.mutableEgo().pos = sc.egoPos;
        world.mutableEgo().yaw = sc.egoYaw;
        world.mutableEgo().speed = 0.0f;
        world.mutableEgo().pitch = 0.0f;
        world.mutableEgo().roll = 0.0f;
        perception.reset();

        for (int f = 0; f < 3; ++f) {
            rig.renderAllCameras(world, cfg, static_cast<float>(f) * 0.05f, nullptr);
            perception.runFrame(ctx, rig, world, 0.05f, cfg.occEnableShadows, out, nullptr);
        }

        int anchorVox = 0;
        int lowPoleVox = 0;
        int highSignVox = 0;
        std::ostringstream voxDiag;
        for (const auto& v : out.activeVoxels) {
            const float d = std::hypot(v.wx - sc.anchorXZ.x, v.wz - sc.anchorXZ.y);
            if (d <= 2.2f) {
                voxDiag << "\n  [VOX] sem=" << static_cast<int>(v.sem) << " wy=" << v.wy
                        << " conf=" << v.confidence << " d=" << d;
                if (v.sem == SemanticClass::SignalSign || v.sem == SemanticClass::Static) {
                    ++anchorVox;
                    if (v.wy <= 1.6f) ++lowPoleVox;
                    if (v.wy >= sc.minHighWy) ++highSignVox;
                }
            }
        }
        int sig2d = 0;
        for (const auto& d2 : out.detections2D) {
            if (d2.semantic == SemanticClass::SignalSign && d2.camIndex <= 4) {
                ++sig2d;
                voxDiag << "\n  [2D] cam=" << d2.camIndex << " sc=" << d2.score
                        << " box=(" << d2.cxNorm << "," << d2.cyNorm << "," << d2.wNorm << "," << d2.hNorm << ")";
            }
        }

        EXPECT_GE(out.eval.signPoleVoxels, 2)
            << sc.name << ": eval.signPoleVoxels=" << out.eval.signPoleVoxels
            << " sig2d=" << sig2d << voxDiag.str();
        EXPECT_GE(anchorVox, 2)
            << sc.name << ": expected >=2 SignalSign/Static voxels within 2.2m, got " << anchorVox << voxDiag.str();
        EXPECT_GE(lowPoleVox, 1)
            << sc.name << ": missing lower pole voxel (wy<=1.6m) near anchor" << voxDiag.str();
        EXPECT_GE(highSignVox, 1)
            << sc.name << ": missing upper sign/mast voxel (wy>=" << sc.minHighWy << "m) near anchor" << voxDiag.str();
    }

    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, CameraPoseSkewAndFrameDelayDegradesFusionSlas) {
    // 1. Validate SimulationSyncClock time-step & perception dt synchronization under variable display jitter
    SimulationSyncClock clock;
    const float dtControl = 1.0f / 36.0f;
    int percCalls = 0;
    int physTicks = 0;
    float accumPercDt = 0.0f;
    const float jitteredDt[] = {0.012f, 0.044f, 0.018f, 0.055f, 0.010f, 0.039f, 0.022f, 0.050f};
    for (float rawDt : jitteredDt) {
        clock.stepFrame(rawDt, dtControl, dtControl * 3.0f,
            [&](float, float percDt) {
                ++percCalls;
                accumPercDt += percDt;
            },
            [&](int, float, float tickDt) {
                ++physTicks;
                EXPECT_NEAR(tickDt, dtControl, 1e-6f);
            });
    }
    const float totalRawDt = 0.012f + 0.044f + 0.018f + 0.055f + 0.010f + 0.039f + 0.022f + 0.050f;
    EXPECT_EQ(physTicks, static_cast<int>(std::floor((totalRawDt + 1e-6f) / dtControl)));
    EXPECT_NEAR(clock.simTimeSec + clock.accumSec, totalRawDt, 1e-5f);
    EXPECT_GT(percCalls, 0);
    EXPECT_NEAR(accumPercDt, static_cast<float>(physTicks) * dtControl, dtControl * 1.5f);

    // 2. Compare synchronized 0-delay camera/pose vs 3-frame (~83ms) stale camera/pose skew in GPU perception
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    for (auto& p : world.mutablePedestrians()) {
        p.x = -300.0f; p.z = -300.0f;
        p.ax = p.x; p.az = p.z;
        p.bx = p.x; p.bz = p.z;
        p.targetX = p.x; p.targetZ = p.z;
    }

    auto setupTruckScenario = [&]() {
        auto& ego = world.mutableEgo();
        ego.pos = {-24.0f, 0.0f, 45.9f};
        ego.yaw = kPi * 0.5f;
        ego.speed = 10.5f;
        ego.pitch = 0.0f;
        ego.roll = 0.0f;
        for (size_t k = 0; k < world.vehicles().size(); ++k) {
            auto& vk = world.mutableVehicles()[k];
            if (k == 0) {
                vk.pos = {-2.0f, 0.0f, 45.9f}; // 22m ahead on straight
                vk.yaw = kPi * 0.5f;
                vk.speed = 12.0f;
                vk.vx = 12.0f;
                vk.vz = 0.0f;
            } else {
                vk.pos = {-400.0f - static_cast<float>(k) * 20.0f, 0.0f, -400.0f};
                vk.speed = 0.0f;
                vk.vx = 0.0f;
                vk.vz = 0.0f;
            }
        }
    };

    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);

    auto measureTruckErr = [&](const PerceptionFrameOutput& out) {
        const Vec3 gtPos = world.vehicles()[0].pos;
        float bestErr = 999.0f;
        for (const auto& d : out.detections3D) {
            if (d.semantic == SemanticClass::Vehicle) {
                bestErr = std::min(bestErr, std::hypot(d.worldPos.x - gtPos.x, d.worldPos.z - gtPos.z));
            }
        }
        return bestErr;
    };

    auto measureSignedVoxLatErr = [&](const PerceptionFrameOutput& out) {
        const Vec3 gtPos = world.vehicles()[0].pos;
        double sumZ = 0.0;
        int n = 0;
        for (const auto& v : out.activeVoxels) {
            if (v.sem == SemanticClass::Vehicle && std::hypot(v.wx - gtPos.x, v.wz - gtPos.z) < 7.5f) {
                sumZ += v.wz;
                ++n;
            }
        }
        return (n > 0) ? static_cast<float>(std::abs((sumZ / n) - gtPos.z)) : 999.0f;
    };

    // Synchronized baseline (camera rendered at exact evaluation Ego & world pose)
    setupTruckScenario();
    perception.reset();
    PerceptionFrameOutput outSync;
    for (int f = 0; f < 3; ++f) {
        rig.renderAllCameras(world, cfg, static_cast<float>(f) * dtControl, nullptr);
        perception.runFrame(ctx, rig, world, dtControl, cfg.occEnableShadows, outSync, nullptr);
    }
    const float syncErr = measureTruckErr(outSync);
    const float syncVoxLat = measureSignedVoxLatErr(outSync);
    EXPECT_LT(syncErr, 2.0f) << "Synchronized camera-pose pipeline exceeded 2.0m error SLA";
    EXPECT_LT(syncVoxLat, 0.55f);
    EXPECT_EQ(outSync.eval.misclassifiedTrucks, 0);
    EXPECT_GE(outSync.eval.truckVehVoxels, 12);

    // Delayed/skewed pass: camera buffer captured ~111ms (4 ticks) earlier while oncoming/braking truck closed 1.56m & Ego yawed
    setupTruckScenario();
    perception.reset();
    PerceptionFrameOutput outSkew;
    for (int f = 0; f < 2; ++f) {
        rig.renderAllCameras(world, cfg, static_cast<float>(f) * dtControl, nullptr);
        perception.runFrame(ctx, rig, world, dtControl, cfg.occEnableShadows, outSkew, nullptr);
    }
    // Render cameras at t_shutter, then advance world traffic & ego pose over 4 ticks (~111ms) before fusion
    rig.renderAllCameras(world, cfg, 2.0f * dtControl, nullptr);
    world.mutableVehicles()[0].pos.x -= 14.0f * (4.0f * dtControl);
    world.mutableEgo().pos.x += 0.90f;
    world.mutableEgo().yaw += 0.085f;
    world.mutableEgo().pitch = 0.045f;
    perception.runFrame(ctx, rig, world, dtControl, cfg.occEnableShadows, outSkew, nullptr);

    const float skewErr = measureTruckErr(outSkew);
    EXPECT_GT(skewErr, syncErr + 1.0f)
        << "Expected ~111ms camera-to-world delay to measurably degrade 3D localization (syncErr="
        << syncErr << ", skewErr=" << skewErr << ")";

    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, ConfigValidationPipelineLatencyBudgetAndVulkanClean) {
    // 1. Validate YAML configuration FMEA bounds & architectural consistency
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    EXPECT_GT(cfg.controlHz, 0.0f);
    EXPECT_LE(cfg.controlHz, 240.0f);
    EXPECT_GT(cfg.camWidth, 0);
    EXPECT_GT(cfg.camHeight, 0);
    EXPECT_GE(cfg.fullCamWidth, cfg.camWidth);
    EXPECT_GE(cfg.fullCamHeight, cfg.camHeight);
    EXPECT_EQ(cfg.camWidth % cfg.featWidth, 0);
    EXPECT_EQ(cfg.camHeight % cfg.featHeight, 0);
    ASSERT_EQ(cfg.cameras.size(), 8u);
    for (size_t c = 0; c < cfg.cameras.size(); ++c) {
        const auto& cam = cfg.cameras[c];
        EXPECT_FALSE(cam.id.empty());
        EXPECT_GT(cam.hfovDeg, 10.0f);
        EXPECT_LT(cam.hfovDeg, 170.0f);
        EXPECT_GT(cam.maxRangeM, 10.0f);
        const Mat4 t = cam.extrinsicEgoFromCamISO();
        const Mat4 inv = cam.extrinsicCamFromEgoISO();
        const Vec4 idCheck = t * (inv * Vec4{1.2f, -0.5f, 3.4f, 1.0f});
        EXPECT_NEAR(idCheck.x, 1.2f, 1e-4f) << "cam " << c;
        EXPECT_NEAR(idCheck.y, -0.5f, 1e-4f) << "cam " << c;
        EXPECT_NEAR(idCheck.z, 3.4f, 1e-4f) << "cam " << c;
    }
    ASSERT_EQ(cfg.bevZAnchors.size(), 4u);
    for (size_t i = 1; i < cfg.bevZAnchors.size(); ++i) {
        EXPECT_GT(cfg.bevZAnchors[i], cfg.bevZAnchors[i - 1]);
    }
    EXPECT_EQ(cfg.occNx(), cfg.bevNx);
    EXPECT_EQ(cfg.occNy(), cfg.bevNy);
    EXPECT_GT(cfg.occNz(), 0);
    EXPECT_LT(cfg.rtdetrLowConfThresh, cfg.rtdetrHighConfThresh);
    EXPECT_LE(cfg.btHighMatchDistM, cfg.btLowMatchDistM);
    EXPECT_GT(cfg.soilMinTokenWeight, 0.0f);
    EXPECT_LT(cfg.soilMinTokenWeight, 1.0f);
    EXPECT_GT(cfg.rssTReactSec, 0.0f);
    EXPECT_GT(cfg.rssABrakeMs2, 0.0f);
    EXPECT_GE(cfg.rssABrakeAebMs2, cfg.rssABrakeMs2);
    EXPECT_GT(cfg.rssDBufferM, 0.0f);

    // 2. Non-functional latency SLA & Vulkan validation cleanliness across scenarios
    VulkanContext ctx;
    ctx.init(true, nullptr);

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    GpuProfiler profiler;
    profiler.init(ctx, 64);

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    PlannerCommand cmd;
    double maxPostWarmupWallMs = 0.0;
    double maxGpuPipeMs = 0.0;

    for (int scen = 0; scen < 3; ++scen) {
        world.reset();
        if (scen == 1) world.triggerOccludedDartOut();
        else if (scen == 2) world.triggerJaywalkerCutIn();
        perception.reset();

        for (int step = 0; step < 4; ++step) {
            const auto t0 = std::chrono::steady_clock::now();
            world.stepAgents(dt);
            rig.renderAllCameras(world, cfg, static_cast<float>(step) * dt, &profiler);
            perception.runFrame(ctx, rig, world, dt, cfg.occEnableShadows, out, &profiler);
            planner.plan(world, out, cmd);
            world.stepEgoAndPhysics(dt, cmd, ManualInput{});
            const auto t1 = std::chrono::steady_clock::now();
            const double wallMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            EXPECT_FALSE(std::isnan(out.forwardEpistemicSigma));
            EXPECT_FALSE(std::isnan(out.bevSummary.meanEpistemicVar));
            EXPECT_FALSE(out.gpuStageMs.empty());
            EXPECT_TRUE(std::isfinite(out.eval.totalPipelineMs));
            if (!(scen == 0 && step < 2)) {
                maxPostWarmupWallMs = std::max(maxPostWarmupWallMs, wallMs);
                maxGpuPipeMs = std::max(maxGpuPipeMs, out.eval.totalPipelineMs);
            }
        }
    }

    EXPECT_LT(maxPostWarmupWallMs, 250.0)
        << "Post-warmup frame wall latency exceeded SLA budget: " << maxPostWarmupWallMs << " ms";
    EXPECT_LT(maxGpuPipeMs, 200.0)
        << "Post-warmup GPU pipeline latency exceeded SLA budget: " << maxGpuPipeMs << " ms";
    if (ctx.validationEnabled()) {
        EXPECT_EQ(ctx.validationIssueCount(), 0u)
            << "Vulkan validation layers reported warnings/errors during multi-scenario FMEA sweep";
    }

    profiler.destroy();
    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, NoSpuriousSignalSignVoxelFloodInNominalAndBuildingScenes) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);

    const Vec2 signAnchors[] = {
        {6.5f, WorldEngine::kLoopHalfZ - 6.2f},
        {12.5f, -WorldEngine::kLoopHalfZ - 5.6f},
        {-34.0f, WorldEngine::kLoopHalfZ + 5.6f},
        {WorldEngine::kLoopHalfX - 6.0f, 0.0f},
        {WorldEngine::kLoopHalfX + 6.0f, 0.0f},
    };

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    PlannerCommand cmd;

    int worstSpuriousSigVox = 0;
    int worstStep = -1;
    std::string worstDiag;

    for (int step = 0; step < 45; ++step) {
        world.stepAgents(dt);
        rig.renderAllCameras(world, cfg, static_cast<float>(step) * dt, nullptr);
        perception.runFrame(ctx, rig, world, dt, cfg.occEnableShadows, out, nullptr);
        planner.plan(world, out, cmd);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});

        int trueSigVox = 0;
        int spuriousSigVox = 0;
        std::array<int, 4> spuriousOctantCount{}; // (+x,+z), (-x,+z), (+x,-z), (-x,-z) ego
        for (const auto& v : out.activeVoxels) {
            if (v.sem != SemanticClass::SignalSign) continue;
            bool nearAnchor = false;
            for (const auto& sa : signAnchors) {
                if (std::hypot(v.wx - sa.x, v.wz - sa.y) <= 2.5f) {
                    nearAnchor = true;
                    break;
                }
            }
            if (nearAnchor) {
                ++trueSigVox;
            } else {
                ++spuriousSigVox;
                const int qx = (v.egoX >= 0.0f) ? 0 : 1;
                const int qy = (v.egoY >= 0.0f) ? 0 : 1;
                spuriousOctantCount[qx * 2 + qy]++;
            }
        }

        if (spuriousSigVox > worstSpuriousSigVox) {
            worstSpuriousSigVox = spuriousSigVox;
            worstStep = step;

            std::vector<Vec4> s4(static_cast<size_t>(8 * cfg.featWidth * cfg.featHeight * 4));
            perception.backboneRtdetr().featS4SSBO().downloadVector(s4);
            std::vector<Vec4> pixels(static_cast<size_t>(8 * rig.camWidth() * rig.camHeight()));
            rig.cameraTensorSSBO().downloadVector(pixels);

            std::ostringstream oss;
            std::ostringstream spSamples;
            int spPrinted = 0;
            for (const auto& v : out.activeVoxels) {
                if (v.sem != SemanticClass::SignalSign) continue;
                bool nearAnchor = false;
                float minA = 999.0f;
                for (const auto& sa : signAnchors) {
                    const float da = std::hypot(v.wx - sa.x, v.wz - sa.y);
                    minA = std::min(minA, da);
                    if (da <= 2.5f) nearAnchor = true;
                }
                if (!nearAnchor && std::abs(v.wy - 1.5f) < 0.3f && spPrinted < 12) {
                    spSamples << "\n    [SPUR] w=(" << v.wx << "," << v.wz << ") ego=(" << v.egoX << "," << v.egoY << ") minAnchorD=" << minA << " conf=" << v.confidence;
                    ++spPrinted;
                }
            }
            oss << "step=" << step << " totalVox=" << out.activeVoxels.size()
                << " trueSigVox=" << trueSigVox << " spuriousSigVox=" << spuriousSigVox
                << " octant(FL=" << spuriousOctantCount[0] << ",FR=" << spuriousOctantCount[1]
                << ",RL=" << spuriousOctantCount[2] << ",RR=" << spuriousOctantCount[3] << ")"
                << " ego=(" << world.ego().pos.x << "," << world.ego().pos.z << ",yaw=" << world.ego().yaw
                << ",pitch=" << world.ego().pitch << ")" << spSamples.str();
            for (int c = 0; c < 8; ++c) {
                int hotPoleCells = 0;
                int hotSigCells = 0;
                float bestC2w = 0.0f;
                int bestFx = 0, bestFy = 0;
                std::ostringstream sampleCells;
                int dumped = 0;
                for (int fy = 0; fy < cfg.featHeight; ++fy) {
                    for (int fx = 0; fx < cfg.featWidth; ++fx) {
                        const size_t idx = (static_cast<size_t>(c * cfg.featHeight + fy) * cfg.featWidth + fx) * 4u;
                        const float sig = s4[idx + 1u].z;
                        const float pole = s4[idx + 2u].w;
                        if (pole > 0.08f) {
                            ++hotPoleCells;
                            if (dumped < 3 && fy >= 3 && fy <= 24) {
                                const int spx = std::clamp(fx * 8 + 4, 0, rig.camWidth() - 1);
                                const int spy = std::clamp(fy * 8 + 4, 0, rig.camHeight() - 1);
                                const Vec4& sp = pixels[(static_cast<size_t>(c) * rig.camHeight() + spy) * rig.camWidth() + spx];
                                const Vec4& spL = pixels[(static_cast<size_t>(c) * rig.camHeight() + spy) * rig.camWidth() + std::max(0, spx - 4)];
                                const Vec4& spR = pixels[(static_cast<size_t>(c) * rig.camHeight() + spy) * rig.camWidth() + std::min(rig.camWidth() - 1, spx + 4)];
                                sampleCells << " [(" << fx << "," << fy << ") w=" << pole
                                            << " rgb=(" << sp.x << "," << sp.y << "," << sp.z << ")"
                                            << " L=(" << spL.x << "," << spL.y << "," << spL.z << ")"
                                            << " R=(" << spR.x << "," << spR.y << "," << spR.z << ")]";
                                ++dumped;
                            }
                        }
                        if (sig > 0.08f) ++hotSigCells;
                        if (pole > bestC2w) {
                            bestC2w = pole;
                            bestFx = fx;
                            bestFy = fy;
                        }
                    }
                }
                const int px = std::clamp(bestFx * (rig.camWidth() / cfg.featWidth) + 4, 0, rig.camWidth() - 1);
                const int py = std::clamp(bestFy * (rig.camHeight() / cfg.featHeight) + 4, 0, rig.camHeight() - 1);
                const Vec4& rgb = pixels[(static_cast<size_t>(c) * rig.camHeight() + py) * rig.camWidth() + px];
                oss << "\n  [CAM " << c << "] hotPoleCells=" << hotPoleCells
                    << " hotSigCells=" << hotSigCells
                    << " maxC2w=" << bestC2w << "@(" << bestFx << "," << bestFy << ")"
                    << " px=(" << px << "," << py << ") v=" << (static_cast<float>(py) / rig.camHeight())
                    << " rgb=(" << rgb.x << "," << rgb.y << "," << rgb.z << ")"
                    << sampleCells.str();
            }
            worstDiag = oss.str();
        }
    }

    EXPECT_LE(worstSpuriousSigVox, 20)
        << "Detected yellow/gold SignalSign voxel flood (>2.5m from any real sign/pole)! Worst @ "
        << worstStep << ":\n" << worstDiag;

    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

TEST(FMEAPerceptionTest, SingleTruckUnderBrakingPitchAndObliquityNeverSplitsBoxes) {
    // FMEA (Split Truck / Multiple Overlapping Boxes on FRONT NARROW & 3D NMS):
    // Reproduces V:14m + V:16m split detections on TRUCK_01 under curve yaw obliquity and AEB braking pitch (+2.3 deg).
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);

    WorldEngine world(cfg);
    // Move pedestrians and non-truck vehicles clear so every near-TRUCK_01 detection belongs strictly to TRUCK_01.
    for (auto& p : world.mutablePedestrians()) {
        p.x = -300.0f; p.z = -300.0f;
        p.ax = p.x; p.az = p.z;
        p.bx = p.x; p.bz = p.z;
        p.targetX = p.x; p.targetZ = p.z;
    }
    for (size_t i = 1; i < world.mutableVehicles().size(); ++i) {
        auto& v = world.mutableVehicles()[i];
        v.pos = {-260.0f - static_cast<float>(i) * 15.0f, 0.0f, -260.0f};
        v.speed = 0.0f; v.vx = 0.0f; v.vz = 0.0f;
    }

    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);

    PerceptionFrameOutput out;

    struct TruckCase {
        float rangeM;
        float latOffM;
        float yawOffRad;
        float egoPitchRad;
    };
    const TruckCase cases[] = {
        {14.5f, 0.0f, 0.00f, 0.040f},
        {15.5f, 0.8f, 0.28f, 0.040f},
        {14.0f, -0.7f, -0.30f, 0.040f},
        {16.5f, 1.1f, 0.45f, 0.035f},
        {12.5f, 0.4f, 0.22f, 0.042f},
        {18.0f, 0.0f, 0.18f, 0.025f},
    };

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const auto& tc = cases[ci];
        world.mutableEgo().pitch = tc.egoPitchRad;
        const EgoState& ego = world.ego();
        const float sinE = std::sin(ego.yaw), cosE = std::cos(ego.yaw);
        auto& truck = world.mutableVehicles().at(0);
        truck.pos = {ego.pos.x + sinE * tc.rangeM + cosE * tc.latOffM,
                     0.0f,
                     ego.pos.z + cosE * tc.rangeM - sinE * tc.latOffM};
        truck.yaw = ego.yaw + tc.yawOffRad;
        truck.speed = 0.0f; truck.vx = 0.0f; truck.vz = 0.0f;

        perception.reset();
        for (int s = 0; s < 3; ++s) {
            rig.renderAllCameras(world, cfg, static_cast<float>(s) * (1.0f / 36.0f), nullptr);
            perception.runFrame(ctx, rig, world, 1.0f / 36.0f, cfg.occEnableShadows, out, nullptr);
        }

        int truck3DCount = 0;
        int truckVeh3DCount = 0;
        int cam13DCount = 0;
        float bestDist3D = 999.0f;
        std::ostringstream diag;
        for (const auto& d : out.detections3D) {
            const float dist = std::hypot(d.worldPos.x - truck.pos.x, d.worldPos.z - truck.pos.z);
            if (dist < 7.5f || d.camIndex == 1) {
                diag << "\n  [3D] cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
                     << " dist=" << dist << " r=" << d.rangeM << " sc=" << d.score
                     << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm
                     << "," << d.det2d.wNorm << "," << d.det2d.hNorm << ")";
            }
            if (dist < 7.5f && d.semantic != SemanticClass::SignalSign) {
                ++truck3DCount;
                if (d.semantic == SemanticClass::Vehicle) {
                    ++truckVeh3DCount;
                    bestDist3D = std::min(bestDist3D, dist);
                    if (d.camIndex == 1) ++cam13DCount;
                }
            }
        }
        int cam02D = 0, cam12D = 0;
        for (const auto& d2 : out.detections2D) {
            if (d2.semantic != SemanticClass::SignalSign && (d2.camIndex == 0 || d2.camIndex == 1)) {
                diag << "\n  [2D] cam=" << d2.camIndex << " sem=" << static_cast<int>(d2.semantic)
                     << " sc=" << d2.score << " box=(" << d2.cxNorm << "," << d2.cyNorm
                     << "," << d2.wNorm << "," << d2.hNorm << ")";
                if (d2.camIndex == 0) ++cam02D;
                if (d2.camIndex == 1) ++cam12D;
            }
        }
        int truckTracks = 0;
        for (const auto& tr : out.tracks) {
            const float dist = std::hypot(tr.worldPos.x - truck.pos.x, tr.worldPos.z - truck.pos.z);
            if (dist < 7.5f && tr.semantic != SemanticClass::SignalSign) {
                ++truckTracks;
                diag << "\n  [TRK] id=" << tr.trackId << " sem=" << static_cast<int>(tr.semantic)
                     << " dist=" << dist << " hits=" << tr.hits;
            }
        }

        EXPECT_LE(truck3DCount, 1)
            << "TRUCK_01 split into " << truck3DCount << " 3D detections at case=" << ci
            << " (r=" << tc.rangeM << ", yaw=" << tc.yawOffRad << ", pitch=" << tc.egoPitchRad << ")" << diag.str();
        EXPECT_EQ(truckVeh3DCount, 1)
            << "Expected single fused Vehicle 3D box on TRUCK_01 at case=" << ci << " bestDist=" << bestDist3D << diag.str();
        EXPECT_LE(cam13DCount, 1)
            << "CAM 1 (FRONT NARROW) emitted multiple 3D boxes on TRUCK_01 at case=" << ci << diag.str();
        EXPECT_LE(cam02D, 1)
            << "CAM 0 (FRONT MAIN) split TRUCK_01 into " << cam02D << " 2D boxes at case=" << ci << diag.str();
        EXPECT_LE(cam12D, 1)
            << "CAM 1 (FRONT NARROW) split TRUCK_01 into " << cam12D << " 2D boxes at case=" << ci << diag.str();
        EXPECT_EQ(truckTracks, 1)
            << "ByteTrack should hold exactly 1 track for TRUCK_01 at case=" << ci << diag.str();
    }

    // Closed-loop curved AEB approach (reproduces Braking Pitch + oblique corner geometry around TRUCK_01).
    WorldEngine hbWorld(cfg);
    for (auto& p : hbWorld.mutablePedestrians()) {
        p.x = -300.0f; p.z = -300.0f;
        p.ax = p.x; p.az = p.z;
        p.bx = p.x; p.bz = p.z;
        p.targetX = p.x; p.targetZ = p.z;
    }
    for (size_t i = 1; i < hbWorld.mutableVehicles().size(); ++i) {
        auto& v = hbWorld.mutableVehicles()[i];
        v.pos = {-260.0f - static_cast<float>(i) * 15.0f, 0.0f, -260.0f};
        v.speed = 0.0f; v.vx = 0.0f; v.vz = 0.0f;
    }
    const float loopLen = hbWorld.centerline().getLength();
    {
        auto& ego = hbWorld.mutableEgo();
        ego.loopProgressU = 0.215f;
        ego.currentLateralOffset = -1.9f;
        ego.pos = hbWorld.laneCurve(0).getPointAt(ego.loopProgressU);
        const Vec3 eTan = hbWorld.laneCurve(0).getTangentAt(ego.loopProgressU);
        ego.yaw = std::atan2(eTan.x, eTan.z);
        ego.speed = 9.0f;

        auto& truck = hbWorld.mutableVehicles().at(0);
        truck.lane = 0;
        truck.u = std::fmod(ego.loopProgressU + 15.5f / loopLen, 1.0f);
        truck.speed = 0.0f;
        truck.hardBrakeTimer = 6.0f;
        truck.pos = hbWorld.laneCurve(0).getPointAt(truck.u);
        const Vec3 tTan = hbWorld.laneCurve(0).getTangentAt(truck.u);
        truck.yaw = std::atan2(tTan.x, tTan.z);
        truck.vx = 0.0f; truck.vz = 0.0f;
    }

    CameraRigRenderer hbRig;
    hbRig.init(ctx, cfg, hbWorld);
    PerceptionPipeline hbPerc(cfg);
    hbPerc.init(ctx, hbRig);
    TrajectoryPlanner planner(cfg);
    PerceptionFrameOutput hbOut;
    PlannerCommand cmd;
    const float dt = 1.0f / cfg.controlHz;
    bool observedBrakingPitch = false;

    for (int step = 0; step < 50; ++step) {
        hbWorld.stepAgents(dt);
        hbRig.renderAllCameras(hbWorld, cfg, static_cast<float>(step) * dt, nullptr);
        hbPerc.runFrame(ctx, hbRig, hbWorld, dt, cfg.occEnableShadows, hbOut, nullptr);
        planner.plan(hbWorld, hbOut, cmd);

        if (hbWorld.ego().pitch > 0.01f) observedBrakingPitch = true;

        const auto& truck = hbWorld.vehicles().at(0);
        int nearTruck3D = 0;
        int nearCam13D = 0;
        int cam12D = 0;
        std::ostringstream hbDiag;
        for (const auto& d : hbOut.detections3D) {
            const float dist = std::hypot(d.worldPos.x - truck.pos.x, d.worldPos.z - truck.pos.z);
            if (dist < 7.0f && d.semantic != SemanticClass::SignalSign) {
                ++nearTruck3D;
                if (d.camIndex == 1) ++nearCam13D;
                hbDiag << "\n  [HB 3D] cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
                       << " r=" << d.rangeM << " dist=" << dist << " sc=" << d.score
                       << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm
                       << "," << d.det2d.wNorm << "," << d.det2d.hNorm << ")";
            }
        }
        for (const auto& d2 : hbOut.detections2D) {
            if (d2.camIndex == 1 && d2.semantic != SemanticClass::SignalSign) {
                ++cam12D;
                hbDiag << "\n  [HB CAM1 2D] sem=" << static_cast<int>(d2.semantic) << " sc=" << d2.score
                       << " box=(" << d2.cxNorm << "," << d2.cyNorm << "," << d2.wNorm << "," << d2.hNorm << ")";
            }
        }
        EXPECT_LE(nearTruck3D, 1)
            << "Closed-loop curve AEB split TRUCK_01 at step=" << step
            << " pitch=" << hbWorld.ego().pitch << " state=" << cmd.behaviorState << hbDiag.str();
        EXPECT_LE(nearCam13D, 1)
            << "CAM 1 emitted multiple 3D boxes in curve AEB at step=" << step << hbDiag.str();
        if (cam12D > 1) {
            const auto rgba = hbRig.downloadCameraRGBA(1);
            for (const auto& d2 : hbOut.detections2D) {
                if (d2.camIndex == 1 && d2.semantic != SemanticClass::SignalSign) {
                    const int px = std::clamp(static_cast<int>(d2.cxNorm * 320.0f), 0, 319);
                    const int py = std::clamp(static_cast<int>(d2.cyNorm * 240.0f), 0, 239);
                    const size_t o = (static_cast<size_t>(py) * 320u + static_cast<size_t>(px)) * 4u;
                    const int pxL = std::clamp(px - 8, 0, 319);
                    const int pxR = std::clamp(px + 8, 0, 319);
                    const int pyU = std::clamp(py - 18, 0, 239);
                    const int pyD = std::clamp(py + 12, 0, 239);
                    const size_t oL = (static_cast<size_t>(py) * 320u + static_cast<size_t>(pxL)) * 4u;
                    const size_t oR = (static_cast<size_t>(py) * 320u + static_cast<size_t>(pxR)) * 4u;
                    const size_t oU = (static_cast<size_t>(pyU) * 320u + static_cast<size_t>(px)) * 4u;
                    const size_t oD = (static_cast<size_t>(pyD) * 320u + static_cast<size_t>(px)) * 4u;
                    hbDiag << "\n    rgb@(" << px << "," << py << ")=("
                           << rgba[o] / 255.0f << "," << rgba[o + 1] / 255.0f << "," << rgba[o + 2] / 255.0f
                           << ") L=(" << rgba[oL] / 255.0f << "," << rgba[oL + 1] / 255.0f << "," << rgba[oL + 2] / 255.0f
                           << ") R=(" << rgba[oR] / 255.0f << "," << rgba[oR + 1] / 255.0f << "," << rgba[oR + 2] / 255.0f
                           << ") U=(" << rgba[oU] / 255.0f << "," << rgba[oU + 1] / 255.0f << "," << rgba[oU + 2] / 255.0f
                           << ") D=(" << rgba[oD] / 255.0f << "," << rgba[oD + 1] / 255.0f << "," << rgba[oD + 2] / 255.0f << ")";
                }
            }
        }
        EXPECT_LE(cam12D, 1)
            << "CAM 1 emitted multiple 2D boxes in curve AEB at step=" << step << hbDiag.str();

        hbWorld.stepEgoAndPhysics(dt, cmd, ManualInput{});
        EXPECT_FALSE(hbWorld.lastContact().occurred)
            << "Curve AEB collision against " << hbWorld.lastContact().otherId << " at step=" << step;
    }
    EXPECT_TRUE(observedBrakingPitch) << "Expected closed-loop AEB to induce positive suspension braking pitch";

    hbPerc.destroy();
    hbRig.destroy();
    perception.destroy();
    rig.destroy();
    ctx.cleanup();
}

} // namespace
} // namespace cybercab
