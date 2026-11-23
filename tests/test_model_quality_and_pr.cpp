#include "MetricsEvalHelpers.hpp"
#include "core/GpuProfiler.hpp"
#include "core/TraceLogger.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <json/json.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace cybercab {
namespace {

using ModelQualityAndPrTest = testutil::SharedGpuMetricsTest;

TEST_F(ModelQualityAndPrTest, DataQualityContractsAcrossPipelineAndTrace) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext& ctx = s_ctx;

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);
    GpuProfiler profiler;
    profiler.init(ctx, 64);

    const std::filesystem::path tracePath =
        std::filesystem::temp_directory_path() / "cybercab_data_quality_contract.jsonl";
    TraceLogger logger;
    ASSERT_TRUE(logger.open(tracePath.string()));

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    PlannerCommand cmd;

    for (int step = 0; step < 6; ++step) {
        rig.renderAllCameras(world, cfg, static_cast<float>(step) * dt, &profiler);
        perception.runFrame(ctx, rig, world, dt, true, out, &profiler);
        planner.plan(world, out, cmd);
        logger.logFrame(step, static_cast<float>(step) * dt, cfg, world.ego(), cmd, out, world.lastContact());

        ASSERT_EQ(out.soilCameras.size(), 8u);
        for (int c = 0; c < 8; ++c) {
            const auto& sc = out.soilCameras[c];
            EXPECT_TRUE(std::isfinite(sc.meanSoilProb));
            EXPECT_GE(sc.meanSoilProb, 0.0f);
            EXPECT_LE(sc.meanSoilProb, 1.0f);
            EXPECT_GE(sc.tokenWeight, cfg.soilMinTokenWeight - 1e-4f);
            EXPECT_LE(sc.tokenWeight, 1.0f + 1e-4f);
            EXPECT_TRUE(std::isfinite(sc.meanEpistemicVar));
            EXPECT_GT(sc.meanEpistemicVar, 0.0f);
        }
        for (const auto& sp : out.soilPatches) {
            EXPECT_TRUE(std::isfinite(sp.soilProb) && sp.soilProb >= 0.0f && sp.soilProb <= 1.0f);
            EXPECT_TRUE(std::isfinite(sp.patchConfidence) && sp.patchConfidence >= 0.0f && sp.patchConfidence <= 1.0f);
            EXPECT_TRUE(std::isfinite(sp.epistemicVar) && sp.epistemicVar > 0.0f);
        }

        std::vector<Vec4> s4(static_cast<size_t>(8 * cfg.featWidth * cfg.featHeight * 4));
        perception.backboneRtdetr().featS4SSBO().downloadVector(s4);
        for (size_t i = 0; i < s4.size(); i += 4) {
            const Vec4& c1 = s4[i + 1];
            const Vec4& c2 = s4[i + 2];
            EXPECT_TRUE(std::isfinite(c1.x) && std::isfinite(c1.y) && std::isfinite(c1.z) && std::isfinite(c1.w));
            EXPECT_GE(c1.x, 0.0f);
            EXPECT_GE(c1.y, 0.0f);
            EXPECT_GE(c1.z, 0.0f);
            EXPECT_GE(c1.w, 0.0f);
            EXPECT_LE(c1.w, 1.05f);
            EXPECT_TRUE(std::isfinite(c2.x) && std::isfinite(c2.y) && std::isfinite(c2.z));
        }

        for (const auto& d2 : out.detections2D) {
            EXPECT_GE(d2.camIndex, 0);
            EXPECT_LT(d2.camIndex, 8);
            EXPECT_GE(d2.cxNorm, 0.0f);
            EXPECT_LE(d2.cxNorm, 1.0f);
            EXPECT_GE(d2.cyNorm, 0.0f);
            EXPECT_LE(d2.cyNorm, 1.0f);
            EXPECT_GT(d2.wNorm, 0.0f);
            EXPECT_GT(d2.hNorm, 0.0f);
            EXPECT_GE(d2.score, cfg.rtdetrLowConfThresh - 1e-4f);
            EXPECT_LE(d2.score, 1.0f + 1e-4f);
        }
        for (const auto& d3 : out.detections3D) {
            EXPECT_TRUE(std::isfinite(d3.worldPos.x) && std::isfinite(d3.worldPos.z));
            EXPECT_GT(d3.halfExtents.x, 0.0f);
            EXPECT_GT(d3.halfExtents.z, 0.0f);
            EXPECT_GE(d3.rangeM, 0.0f);
            EXPECT_LE(d3.rangeM, cfg.rtdetrMaxRangeM + 2.0f);
            EXPECT_TRUE(std::isfinite(d3.epistemicSigma) && d3.epistemicSigma > 0.0f);
        }

        EXPECT_GT(out.bevSummary.observedCells, 0);
        EXPECT_TRUE(std::isfinite(out.bevSummary.meanObservability));
        for (const auto& v : out.activeVoxels) {
            EXPECT_TRUE(std::isfinite(v.wx) && std::isfinite(v.wy) && std::isfinite(v.wz));
            EXPECT_GE(v.confidence, 0.08f);
            EXPECT_LE(v.confidence, 1.0f + 1e-4f);
            EXPECT_NE(v.sem, SemanticClass::Empty);
            const bool insideEgoSkin = (v.egoX > -1.3f && v.egoX < 1.3f && std::abs(v.egoY) < 0.7f && v.wy < 1.3f);
            EXPECT_FALSE(insideEgoSkin) << "Active dynamic voxel intruded into ego footprint!";
        }

        for (const auto& gh : out.worldGhosts) {
            EXPECT_TRUE(std::isfinite(gh.worldPosScale.x) && std::isfinite(gh.worldPosScale.z));
            EXPECT_GE(gh.yawMeta.y, 0.45f);
            EXPECT_LE(gh.yawMeta.y, 2.05f);
            EXPECT_TRUE(std::isfinite(gh.yawMeta.z) && gh.yawMeta.z > 0.0f);
        }

        world.stepAgents(dt);
        world.stepEgoAndPhysics(dt, cmd, ManualInput{});
    }
    logger.close();

    std::ifstream ifs(tracePath);
    ASSERT_TRUE(ifs.good());
    std::string line;
    int lineCount = 0;
    Json::CharReaderBuilder rb;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        Json::Value root;
        std::string errs;
        ASSERT_TRUE(Json::parseFromStream(rb, ss, &root, &errs)) << errs;
        EXPECT_TRUE(root.isMember("step"));
        EXPECT_TRUE(root.isMember("ego"));
        EXPECT_TRUE(root.isMember("soil"));
        EXPECT_TRUE(root.isMember("bevformer"));
        EXPECT_TRUE(root.isMember("eval_metrics"));
        EXPECT_TRUE(root.isMember("gpu_stage_ms"));
        ++lineCount;
    }
    EXPECT_EQ(lineCount, 6);
    std::filesystem::remove(tracePath);

    profiler.destroy();
    perception.destroy();
    rig.destroy();
}

TEST_F(ModelQualityAndPrTest, PerClassPRCurvesAPAndBevRoadPrecision) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.rtdetrLowConfThresh = 0.12f;
    VulkanContext& ctx = s_ctx;

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    PerceptionPipeline perception(cfg);
    perception.init(ctx, rig);
    TrajectoryPlanner planner(cfg);

    testutil::PrAccum prVeh, prPed, prSig;
    std::map<int, std::pair<int, int>> voxCls; // {tp, fp}
    int roadTp = 0, roadFp = 0, roadFn = 0;

    const float dt = 1.0f / cfg.controlHz;
    auto evalScenario = [&](const char* scen, int steps, bool checkRoad) {
        world.reset();
        perception.reset();
        if (std::string(scen) == "jaywalk") world.triggerJaywalkerCutIn();
        else if (std::string(scen) == "hardbrake") world.triggerLeadHardBrake();

        const auto allStatic = world.staticOBBs();
        PerceptionFrameOutput out;
        PlannerCommand cmd;
        std::vector<Vec4> s4(static_cast<size_t>(8 * cfg.featWidth * cfg.featHeight * 4));

        for (int s = 0; s < steps; ++s) {
            rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt, nullptr);
            perception.runFrame(ctx, rig, world, dt, true, out, nullptr);
            planner.plan(world, out, cmd);

            if (s >= 2) {
                const EgoState& ego = world.ego();
                const auto dyn = world.dynamicGroundTruthOBBs();
                std::vector<OBB> gtVeh, gtPed, gtSig, bVeh, bPed, bSig;
                for (const auto& g : dyn) {
                    const float dist = std::hypot(g.cx - ego.pos.x, g.cz - ego.pos.z);
                    if (g.semantic == SemanticClass::Vehicle) {
                        if (dist <= 40.0f) gtVeh.push_back(g);
                        else if (dist <= 46.0f) bVeh.push_back(g);
                    } else if (g.semantic == SemanticClass::Pedestrian) {
                        if (dist <= 40.0f) gtPed.push_back(g);
                        else if (dist <= 46.0f) bPed.push_back(g);
                    }
                }
                for (const auto& st : allStatic) {
                    if (st.semantic == SemanticClass::SignalSign) {
                        const float dist = std::hypot(st.cx - ego.pos.x, st.cz - ego.pos.z);
                        if (dist <= 45.0f) gtSig.push_back(st);
                        else if (dist <= 50.0f) bSig.push_back(st);
                    }
                }

                testutil::accumulatePrCls(SemanticClass::Vehicle, out.detections3D, gtVeh, bVeh, 5.2f, 42.0f, prVeh);
                testutil::accumulatePrCls(SemanticClass::Pedestrian, out.detections3D, gtPed, bPed, 3.2f, 42.0f, prPed);
                testutil::accumulatePrCls(SemanticClass::SignalSign, out.detections3D, gtSig, bSig, 6.0f, 42.0f, prSig);

                for (const auto& v : out.activeVoxels) {
                    if (v.confidence < 0.45f) continue;
                    const int cid = static_cast<int>(v.sem);
                    bool match = false;
                    if (v.sem == SemanticClass::Vehicle || v.sem == SemanticClass::Pedestrian) {
                        for (const auto& g : dyn) {
                            if (g.semantic == v.sem && testutil::isPointInOBB(g, v.wx, v.wz, 1.3f)) { match = true; break; }
                        }
                    } else if (v.sem == SemanticClass::SignalSign || v.sem == SemanticClass::Static ||
                               v.sem == SemanticClass::Overhang || v.sem == SemanticClass::Curb) {
                        for (const auto& st : allStatic) {
                            if (testutil::isPointInOBB(st, v.wx, v.wz, 2.0f)) { match = true; break; }
                        }
                    }
                    if (match) voxCls[cid].first++;
                    else voxCls[cid].second++;
                }

                if (checkRoad && s % 10 == 0) {
                    perception.backboneRtdetr().featS4SSBO().downloadVector(s4);
                    const float sinY = std::sin(ego.yaw), cosY = std::cos(ego.yaw);
                    const int fw = cfg.featWidth, fh = cfg.featHeight;
                    const int cw = rig.camWidth(), ch = rig.camHeight();
                    for (float fwd : {7.0f, 10.0f, 14.0f, 18.0f}) {
                        for (float lat : {-1.2f, 0.0f, 1.2f}) {
                            const Vec3 wp{ego.pos.x + sinY * fwd + cosY * lat, 0.0f,
                                          ego.pos.z + cosY * fwd - sinY * lat};
                            bool blocked = false;
                            for (const auto& g : dyn) {
                                if (testutil::isPointInOBB(g, wp.x, wp.z, 0.6f)) { blocked = true; break; }
                            }
                            if (blocked) continue;
                            Vec2 px{};
                            float dep = 0.0f;
                            if (rig.projectWorldToPixel(0, wp, px, dep)) {
                                const int fx = std::clamp(static_cast<int>(px.x * fw / cw), 0, fw - 1);
                                const int fy = std::clamp(static_cast<int>(px.y * fh / ch), 0, fh - 1);
                                if (s4[static_cast<size_t>(fy * fw + fx) * 4u + 1u].w >= 0.45f) ++roadTp;
                                else ++roadFn;
                            }
                        }
                    }
                    for (int fy = 1; fy < std::max(2, fh / 4); ++fy) {
                        for (int fx = 2; fx < fw - 2; fx += 4) {
                            if (s4[static_cast<size_t>(fy * fw + fx) * 4u + 1u].w >= 0.45f) ++roadFp;
                        }
                    }
                }
            }
            world.stepAgents(dt);
            world.stepEgoAndPhysics(dt, cmd, ManualInput{});
        }
    };

    evalScenario("nominal", 140, true);
    evalScenario("jaywalk", 65, false);
    evalScenario("hardbrake", 70, false);

    double vehAp = 0.0, vehF1 = 0.0, pedAp = 0.0, pedF1 = 0.0, sigAp = 0.0, sigF1 = 0.0;
    testutil::summarizePrCurve(prVeh, vehAp, vehF1);
    testutil::summarizePrCurve(prPed, pedAp, pedF1);
    testutil::summarizePrCurve(prSig, sigAp, sigF1);
    RecordProperty("vehicle_ap", testutil::fmtProp(vehAp));
    RecordProperty("vehicle_max_f1", testutil::fmtProp(vehF1));
    RecordProperty("pedestrian_ap", testutil::fmtProp(pedAp));
    RecordProperty("pedestrian_max_f1", testutil::fmtProp(pedF1));
    RecordProperty("signalsign_ap", testutil::fmtProp(sigAp));
    RecordProperty("signalsign_max_f1", testutil::fmtProp(sigF1));

    const double pedPrecAt18 = (prPed.tp[1] + prPed.fp[1] > 0)
        ? static_cast<double>(prPed.tp[1]) / (prPed.tp[1] + prPed.fp[1]) : 0.0;
    const double roadPrec = (roadTp + roadFp > 0) ? static_cast<double>(roadTp) / (roadTp + roadFp) : 0.0;
    const double roadRec = (roadTp + roadFn > 0) ? static_cast<double>(roadTp) / (roadTp + roadFn) : 0.0;
    auto voxPrec = [&](int cid) {
        const auto p = voxCls[cid];
        return (p.first + p.second > 0) ? static_cast<double>(p.first) / (p.first + p.second) : 0.0;
    };
    RecordProperty("road_prec", testutil::fmtProp(roadPrec));
    RecordProperty("road_rec", testutil::fmtProp(roadRec));
    RecordProperty("vox_veh_prec", testutil::fmtProp(voxPrec(1)));
    RecordProperty("vox_ped_prec", testutil::fmtProp(voxPrec(2)));
    RecordProperty("vox_sig_prec", testutil::fmtProp(voxPrec(3)));

    EXPECT_GE(vehF1, 0.56);
    EXPECT_GE(vehAp, 0.36);
    EXPECT_GE(pedPrecAt18, 0.94);
    EXPECT_GE(pedF1, 0.58);
    EXPECT_GE(pedAp, 0.42);
    EXPECT_GT(prSig.tp[0], 0) << "SignalSign should maintain >0 detections on approach";
    EXPECT_GE(roadPrec, 0.90);
    EXPECT_GE(roadRec, 0.78);
    EXPECT_GE(voxPrec(2), 0.85);
    EXPECT_GE(voxPrec(1), 0.25);
    EXPECT_GE(voxPrec(3), 0.42);

    perception.destroy();
    rig.destroy();
}

} // namespace
} // namespace cybercab
