#include "AvScenarioHierarchy.hpp"
#include "AvValidationMetrics.hpp"
#include "MetricsEvalHelpers.hpp"
#include "core/TraceLogger.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include <gtest/gtest.h>
#include <cstdio>
#include <string>

namespace cybercab {
namespace {
using AvShadowReplayHierarchyTest = testutil::SharedGpuMetricsTest;
using testutil::fmtProp;
using testutil::MultiAxisSliceTable;
using testutil::parseTraceJsonl;
using testutil::ScenarioHierarchyReport;
using testutil::SliceKey;
using testutil::TraceReplaySummary;

constexpr int kReplaySteps = 45;
constexpr int kGpuProfSlots = 64;
constexpr int kHierFrames = 300;
constexpr int kHierCritIdx = 299;
constexpr int kDaySliceN = 40;
constexpr int kNightSliceN = 11;
constexpr int kNightSliceOkN = 9;
constexpr size_t kExpectedSliceCount = 3u;
constexpr double kCritSliceCost = 100.0;
constexpr double kMinHierAcc = 0.95;
constexpr double kMinSingleCritFrameAcc = 0.99;
constexpr double kMinMicroAcc = 0.96;
constexpr double kMaxNightSliceAcc = 0.85;
constexpr double kSliceGateAcc = 0.90;
constexpr double kMaxDetPosDiffM = 1e-4;
constexpr double kDetMetricEps = 1e-5;

TraceReplaySummary runTracedEpisode(VulkanContext& ctx, const char* pipeVer,
                                    const std::string& tracePath) {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = pipeVer;
  WorldEngine w(cfg);
  CameraRigRenderer r;
  r.init(ctx, cfg, w);
  PerceptionPipeline p(cfg);
  p.init(ctx, r);
  TrajectoryPlanner pl(cfg);
  GpuProfiler prof;
  prof.init(ctx, kGpuProfSlots);
  TraceLogger tracer;
  EXPECT_TRUE(tracer.open(tracePath));

  const float dt = 1.0f / cfg.controlHz;
  PerceptionFrameOutput out{};
  PlannerCommand cmd{};
  for (int s = 0; s < kReplaySteps; ++s) {
    const float tSec = static_cast<float>(s) * dt;
    r.renderAllCameras(w, cfg, tSec, &prof);
    p.runFrame(ctx, r, w, dt, true, out, &prof);
    pl.plan(w, out, cmd);
    w.stepAgents(dt);
    w.stepEgoAndPhysics(dt, cmd, {});
    tracer.logFrame(s, tSec, cfg, w.ego(), cmd, out, w.lastContact());
  }
  tracer.close();
  prof.destroy();
  p.destroy();
  r.destroy();
  return parseTraceJsonl(tracePath);
}

TEST_F(AvShadowReplayHierarchyTest,
       ZeroToleranceScenarioHierarchyAndMultiAxisSliceGating) {
  ScenarioHierarchyReport good{};
  ScenarioHierarchyReport singleCrit{};
  for (int i = 0; i < kHierFrames; ++i) {
    ++good.framesEvaluated;
    ++good.frameCorrect;
    ++good.assocTotal;
    ++good.assocCorrect;
    ++good.decisionTotal;
    ++good.decisionCorrect;

    ++singleCrit.framesEvaluated;
    ++singleCrit.assocTotal;
    ++singleCrit.assocCorrect;
    ++singleCrit.decisionTotal;
    if (i == kHierCritIdx) {
      ++singleCrit.criticalFailures;
    } else {
      ++singleCrit.frameCorrect;
      ++singleCrit.decisionCorrect;
    }
  }
  EXPECT_TRUE(good.scenarioPassed(kMinHierAcc, kMinHierAcc, kMinHierAcc));
  EXPECT_GT(singleCrit.frameAcc(), kMinSingleCritFrameAcc);
  EXPECT_FALSE(
    singleCrit.scenarioPassed(kMinHierAcc, kMinHierAcc, kMinHierAcc));

  MultiAxisSliceTable tbl;
  const SliceKey sDayNear{"nominal", "red", "0-20m", "clear", "day", "none",
                          "straight"};
  const SliceKey sDayMid{"nominal", "red", "20-50m", "clear", "day", "none",
                         "straight"};
  const SliceKey sNightLeft{"unprot_left", "red_arrow", "20-50m", "rain",
                            "night", "partial", "left_turn"};
  for (int i = 0; i < kDaySliceN; ++i) {
    tbl.record(sDayNear, true, false, 0.0);
  }
  for (int i = 0; i < kDaySliceN; ++i) {
    tbl.record(sDayMid, true, false, 0.0);
  }
  for (int i = 0; i < kNightSliceN; ++i) {
    const bool ok = (i < kNightSliceOkN);
    tbl.record(sNightLeft, ok, !ok, ok ? 0.0 : kCritSliceCost);
  }
  EXPECT_GT(tbl.microAccuracy(), kMinMicroAcc);
  EXPECT_LT(tbl.worstSliceAccuracy(), kMaxNightSliceAcc);
  EXPECT_FALSE(tbl.allSlicesPass(kSliceGateAcc));
  EXPECT_TRUE(tbl.anyCriticalFailure());
  EXPECT_EQ(tbl.sliceCount(), kExpectedSliceCount);
}

TEST_F(AvShadowReplayHierarchyTest,
       DeterministicShadowReplayAndPairedV14V15AbDiff) {
  const std::string p14 = "/tmp/shadow_v14.jsonl";
  const std::string p15a = "/tmp/shadow_v15_a.jsonl";
  const std::string p15b = "/tmp/shadow_v15_b.jsonl";
  const TraceReplaySummary s14 = runTracedEpisode(s_ctx, "v14", p14);
  const TraceReplaySummary s15a = runTracedEpisode(s_ctx, "v15", p15a);
  const TraceReplaySummary s15b = runTracedEpisode(s_ctx, "v15", p15b);
  std::remove(p14.c_str());
  std::remove(p15a.c_str());
  std::remove(p15b.c_str());

  EXPECT_EQ(s14.frames, kReplaySteps);
  EXPECT_EQ(s15a.frames, kReplaySteps);
  EXPECT_EQ(s15b.frames, kReplaySteps);
  EXPECT_EQ(s14.collisions, 0);
  EXPECT_EQ(s15a.collisions, 0);
  EXPECT_EQ(s14.v15Frames, 0);
  EXPECT_EQ(s15a.v15Frames, kReplaySteps);
  EXPECT_GT(s15a.meanGaussians, 0.0);

  const double posDiffM =
    std::hypot(s15a.finalX - s15b.finalX, s15a.finalZ - s15b.finalZ);
  EXPECT_LT(posDiffM, kMaxDetPosDiffM);
  EXPECT_NEAR(s15a.meanRecall, s15b.meanRecall, kDetMetricEps);
  EXPECT_NEAR(s15a.meanPrec, s15b.meanPrec, kDetMetricEps);
  EXPECT_EQ(s15a.totalTracks, s15b.totalTracks);

  RecordProperty("ab_v14_recall", fmtProp(s14.meanRecall));
  RecordProperty("ab_v15_recall", fmtProp(s15a.meanRecall));
  RecordProperty("ab_v14_prec", fmtProp(s14.meanPrec));
  RecordProperty("ab_v15_prec", fmtProp(s15a.meanPrec));
  RecordProperty("ab_v14_rmse_m", fmtProp(s14.meanRmse));
  RecordProperty("ab_v15_rmse_m", fmtProp(s15a.meanRmse));
  RecordProperty("ab_v14_gpu_ms", fmtProp(s14.meanGpuMs));
  RecordProperty("ab_v15_gpu_ms", fmtProp(s15a.meanGpuMs));
  RecordProperty("ab_det_pos_diff_m", fmtProp(posDiffM));
}

}  // namespace
}  // namespace cybercab
