#include "MetricsEvalHelpers.hpp"
#include "perception/BEVFormer.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace cybercab {
namespace {
using V15VoxelDetHallucTest = testutil::SharedGpuMetricsTest;

constexpr float kOffMap = -350.0f, kOccPadZ = 0.5f, kVoxLocEps = 0.05f;
constexpr float kObbPadM = 1.80f, kMinVoxDynPrec = 0.55f;
constexpr float kEvalRangeM = 42.0f, kBoundRangeM = 48.0f, kPrGateM = 5.0f;
constexpr double kMinRec = 0.50, kMinPrec = 0.70, kMinAp = 0.45;
constexpr double kMaxRmse = 2.5, kMaxGhost = 1.5;
constexpr int kVoxSteps = 16, kEmptySteps = 8, kPrSteps = 40, kWarmSteps = 4;

SimConfig v15Cfg() {
  SimConfig c =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  c.pipelineVersion = "v15";
  return c;
}

TEST_F(V15VoxelDetHallucTest, VoxelBoundsTransformAndSemanticLocationAccuracy) {
  const SimConfig cfg = v15Cfg();
  WorldEngine world(cfg);
  CameraRigRenderer rig; rig.init(s_ctx, cfg, world);
  PerceptionPipeline pipe(cfg); pipe.init(s_ctx, rig);
  TrajectoryPlanner planner(cfg);
  PerceptionFrameOutput out{}; PlannerCommand cmd{}; std::vector<OBB> gt;
  const float dt = 1.0f / cfg.controlHz;
  float maxLocErr = 0.0f;
  size_t dynTotal = 0u, dynInObb = 0u;
  int sumTruckVeh = 0, sumTruckPed = 0, sumSignVox = 0;

  for (int s = 0; s < kVoxSteps; ++s) {
    rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
    pipe.runFrame(s_ctx, rig, world, dt, true, out);
    planner.plan(world, out, cmd);
    const Mat4 wFromIso = BEVFormer::worldFromIsoEgo(world.ego());
    world.collectDynamicGroundTruthOBBs(gt);
    for (const auto& v : out.activeVoxels) {
      EXPECT_GE(v.egoX, cfg.occMinX - kVoxLocEps);
      EXPECT_LE(v.egoX, cfg.occMaxX + kVoxLocEps);
      EXPECT_GE(v.egoY, cfg.occMinY - kVoxLocEps);
      EXPECT_LE(v.egoY, cfg.occMaxY + kVoxLocEps);
      EXPECT_GE(v.wy, 0.0f);
      EXPECT_LE(v.wy, cfg.occMaxZ + kOccPadZ);
      const Vec3 exp = wFromIso.transformPoint({v.egoX, v.egoY, v.wy});
      maxLocErr = std::max(maxLocErr, std::hypot(v.wx - exp.x, v.wz - exp.z));
      if (v.sem == SemanticClass::Vehicle ||
          v.sem == SemanticClass::Pedestrian) {
        ++dynTotal;
        for (const auto& b : gt) {
          if (testutil::isPointInOBB(b, v.wx, v.wz, kObbPadM)) {
            ++dynInObb;
            break;
          }
        }
      }
    }
    if (s >= kWarmSteps) {
      sumTruckVeh += out.eval.truckVehVoxels;
      sumTruckPed += out.eval.truckPedVoxels;
      sumSignVox += out.eval.signPoleVoxels;
    }
    world.stepAgents(dt);
    world.stepEgoAndPhysics(dt, cmd, {});
  }
  const double voxPrec =
    dynTotal > 0u ? static_cast<double>(dynInObb) / dynTotal : 0.0;
  RecordProperty("v15_voxel_loc_max_err_m", testutil::fmtProp(maxLocErr));
  RecordProperty("v15_voxel_dyn_precision", testutil::fmtProp(voxPrec));
  RecordProperty("v15_truck_veh_voxels", std::to_string(sumTruckVeh));
  RecordProperty("v15_sign_pole_voxels", std::to_string(sumSignVox));
  EXPECT_LT(maxLocErr, kVoxLocEps);
  EXPECT_GT(voxPrec, static_cast<double>(kMinVoxDynPrec));
  EXPECT_GT(sumTruckVeh, 0);
  EXPECT_EQ(sumTruckPed, 0);
  EXPECT_GT(sumSignVox, 0);
  pipe.destroy(); rig.destroy();
}

TEST_F(V15VoxelDetHallucTest, EmptyRoadProducesZeroHallucinationsAndFalseAeb) {
  const SimConfig cfg = v15Cfg();
  WorldEngine world(cfg);
  for (auto& v : world.mutableVehicles()) {
    v.pos = {kOffMap, 0.0f, kOffMap}; v.speed = v.vx = v.vz = 0.0f;
  }
  for (auto& p : world.mutablePedestrians()) {
    p.x = p.ax = p.bx = p.targetX = kOffMap;
    p.z = p.az = p.bz = p.targetZ = kOffMap;
  }
  CameraRigRenderer rig; rig.init(s_ctx, cfg, world);
  PerceptionPipeline pipe(cfg); pipe.init(s_ctx, rig);
  TrajectoryPlanner planner(cfg);
  PerceptionFrameOutput out{}; PlannerCommand cmd{};
  const float dt = 1.0f / cfg.controlHz;
  int dynVox = 0, dynGauss = 0, ghostVeh = 0, ghostPed = 0, falseAeb = 0;

  for (int s = 0; s < kEmptySteps; ++s) {
    rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
    pipe.runFrame(s_ctx, rig, world, dt, true, out);
    planner.plan(world, out, cmd);
    ghostVeh += out.eval.ghostVehDetections;
    ghostPed += out.eval.ghostPedDetections;
    for (const auto& v : out.activeVoxels) {
      if (v.sem == SemanticClass::Vehicle || v.sem == SemanticClass::Pedestrian)
        ++dynVox;
    }
    for (const auto& g : out.v15Gaussians) {
      const int sem = static_cast<int>(std::lround(g.scaleSem.w));
      if (sem == static_cast<int>(SemanticClass::Vehicle) ||
          sem == static_cast<int>(SemanticClass::Pedestrian)) ++dynGauss;
    }
    if (cmd.aebActive) ++falseAeb;
    world.stepEgoAndPhysics(dt, cmd, {});
  }
  EXPECT_EQ(ghostVeh, 0); EXPECT_EQ(ghostPed, 0);
  EXPECT_EQ(dynVox, 0); EXPECT_EQ(dynGauss, 0);
  EXPECT_EQ(falseAeb, 0);
  pipe.destroy(); rig.destroy();
}

TEST_F(V15VoxelDetHallucTest,
       ClosedLoopPrecisionRecallApRmseAndZeroTruckMisclass) {
  const SimConfig cfg = v15Cfg();
  WorldEngine world(cfg);
  CameraRigRenderer rig; rig.init(s_ctx, cfg, world);
  PerceptionPipeline pipe(cfg); pipe.init(s_ctx, rig);
  TrajectoryPlanner planner(cfg);
  PerceptionFrameOutput out{}; PlannerCommand cmd{};
  testutil::PrAccum vehAcc{};
  std::vector<OBB> gtAll, reqVeh, boundVeh;
  const float dt = 1.0f / cfg.controlHz;
  double recSum = 0.0, precSum = 0.0, rmseSum = 0.0, ghostSum = 0.0;
  int nEval = 0, misTrucks = 0;

  for (int s = 0; s < kPrSteps; ++s) {
    rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
    pipe.runFrame(s_ctx, rig, world, dt, true, out);
    planner.plan(world, out, cmd);
    if (s >= kWarmSteps) {
      recSum += out.eval.detRecall;
      precSum += out.eval.detPrecision;
      rmseSum += out.eval.trackPosRmseM;
      ghostSum += out.eval.ghostVehDetections + out.eval.ghostPedDetections;
      misTrucks += out.eval.misclassifiedTrucks;
      world.collectDynamicGroundTruthOBBs(gtAll);
      reqVeh.clear(); boundVeh.clear();
      const EgoState& e = world.ego();
      for (const auto& g : gtAll) {
        if (g.semantic != SemanticClass::Vehicle) continue;
        const float d = std::hypot(g.cx - e.pos.x, g.cz - e.pos.z);
        if (d <= kEvalRangeM) reqVeh.push_back(g);
        else if (d <= kBoundRangeM) boundVeh.push_back(g);
      }
      testutil::accumulatePrCls(SemanticClass::Vehicle, out.detections3D,
                                reqVeh, boundVeh, kPrGateM,
                                cfg.rtdetrMaxRangeM, vehAcc);
      ++nEval;
    }
    world.stepAgents(dt);
    world.stepEgoAndPhysics(dt, cmd, {});
  }
  double vehAp = 0.0, vehF1 = 0.0;
  testutil::summarizePrCurve(vehAcc, vehAp, vehF1);
  RecordProperty("v15_mean_recall", testutil::fmtProp(recSum / nEval));
  RecordProperty("v15_mean_prec", testutil::fmtProp(precSum / nEval));
  RecordProperty("v15_veh_ap", testutil::fmtProp(vehAp));
  RecordProperty("v15_pos_rmse_m", testutil::fmtProp(rmseSum / nEval));
  EXPECT_GT(recSum / nEval, kMinRec);
  EXPECT_GT(precSum / nEval, kMinPrec);
  EXPECT_GT(vehAp, kMinAp);
  EXPECT_LT(rmseSum / nEval, kMaxRmse);
  EXPECT_EQ(misTrucks, 0);
  EXPECT_LT(ghostSum / nEval, kMaxGhost);
  pipe.destroy(); rig.destroy();
}
}  // namespace
}  // namespace cybercab
