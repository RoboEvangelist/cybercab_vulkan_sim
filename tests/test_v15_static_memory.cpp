#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/FrequencySplit.hpp"
#include "perception/v15/PatchTokenizer.hpp"
#include "perception/v15/StaticSpatialMemory.hpp"
#include "perception/v15/VirtualCameraRectifier.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace cybercab {
namespace {
using namespace v15;

using V15StaticMemoryTest = testutil::SharedGpuMetricsTest;

TEST_F(V15StaticMemoryTest, GuardsHandleUninitializedStateAndInvalidInit) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    StaticSpatialMemory mem;
    EXPECT_EQ(mem.readbackStats().refreshedCount, 0u);
    EXPECT_FLOAT_EQ(mem.readbackStats().selectiveRefreshSkipRatio, 0.0f);
    std::vector<StaticCellGpu> cells(5);
    mem.readbackCells(cells);
    EXPECT_TRUE(cells.empty());
    mem.reset();

    VirtualCameraRectifier emptyRect;
    PatchTokenizer emptyTok;
    EXPECT_THROW(mem.init(s_ctx, cfg, emptyRect, emptyTok), std::runtime_error);
    VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
    mem.recordDispatch(cmd, cfg, EgoState{}, 1.0f / 36.0f, emptyRect, emptyTok, {});
    s_ctx.endSingleTimeCommands(cmd);
    mem.destroy();
}

TEST_F(V15StaticMemoryTest, LocksStaticCellsSurvivesTruckOcclusionAndWarpsCleanly) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline perc(cfg);
    perc.init(s_ctx, rig);
    GpuProfiler profiler;
    profiler.init(s_ctx, 64);

    const float dt = 1.0f / cfg.controlHz;
    PerceptionFrameOutput out;
    for (int f = 0; f < 6; ++f) {
        rig.renderAllCameras(world, cfg, 0.05f * f, &profiler);
        perc.runFrame(s_ctx, rig, world, dt, true, out, &profiler);
    }
    EXPECT_EQ(out.gpuStageMs.count("5_v15_static_mem"), 1u);
    const StaticSpatialMemory& staticMem = perc.v15()->staticMemory();
    const StaticMemStats warmStats = staticMem.readbackStats();
    EXPECT_GT(warmStats.lockedCount, 50u);
    EXPECT_GT(warmStats.selectiveRefreshSkipRatio, 0.30f);

    std::vector<StaticCellGpu> cells;
    staticMem.readbackCells(cells);
    const int nx = staticMem.nx(), ny = staticMem.ny();
    const float dx = (cfg.bevMaxX - cfg.bevMinX) / static_cast<float>(nx);
    const float dy = (cfg.bevMaxY - cfg.bevMinY) / static_cast<float>(ny);

    int targetIdx = -1;
    float bestOcc = -1.0f;
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const auto& c = cells[static_cast<size_t>(iy) * nx + ix];
            const float xIso = cfg.bevMinX + (ix + 0.5f) * dx;
            const float yIso = cfg.bevMinY + (iy + 0.5f) * dy;
            if (xIso > 5.0f && std::abs(yIso) > 2.5f && c.meta.z > 0.5f && c.featConf.w >= 0.80f && c.featConf.x > bestOcc) {
                bestOcc = c.featConf.x;
                targetIdx = iy * nx + ix;
            }
        }
    }
    ASSERT_GE(targetIdx, 0);
    const float txIso = cfg.bevMinX + ((targetIdx % nx) + 0.5f) * dx;
    const float tyIso = cfg.bevMinY + ((targetIdx / nx) + 0.5f) * dy;
    const Vec3 worldAnchor = BEVFormer::worldFromIsoEgo(world.ego()).transformPoint({txIso, tyIso, 0.0f});

    // Mask target cell + surroundings with synthetic TRUCK_01 TrackPriorGpu for 16 frames.
    TrackPriorGpu truckMask{};
    truckMask.posVel = {txIso, tyIso, 0.0f, 0.0f};
    truckMask.sizeMeta = {12.0f, 12.0f, 0.0f, 0.25f};
    truckMask.yawValid = {0.0f, 0.95f, 101.0f, 1.0f};
    StaticSpatialMemory& mutMem = const_cast<StaticSpatialMemory&>(staticMem);
    for (int f = 0; f < 16; ++f) {
        VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
        mutMem.recordDispatch(cmd, cfg, world.ego(), dt, perc.v15()->rectifier(),
                              perc.v15()->patchTokenizer(), {truckMask});
        s_ctx.endSingleTimeCommands(cmd);
    }
    const StaticMemStats occStats = staticMem.readbackStats();
    EXPECT_GT(occStats.dynamicMaskedCount, 0u);
    staticMem.readbackCells(cells);
    EXPECT_GT(cells[targetIdx].meta.z, 0.5f);
    EXPECT_GT(cells[targetIdx].featConf.w, 0.75f);

    // SE(3) ego translation + yaw rotation while dynamically masked -> exact warped location stays locked.
    EgoState movedEgo = world.ego();
    movedEgo.pos.x += 2.0f;
    movedEgo.pos.z += 0.8f;
    movedEgo.yaw += 0.08f;
    const Vec3 warpedIso = BEVFormer::worldFromIsoEgo(movedEgo).inverseRigid().transformPoint(worldAnchor);
    const int wx = static_cast<int>(std::lround((warpedIso.x - cfg.bevMinX) / dx - 0.5f));
    const int wy = static_cast<int>(std::lround((warpedIso.y - cfg.bevMinY) / dy - 0.5f));
    ASSERT_GE(wx, 0); ASSERT_LT(wx, nx); ASSERT_GE(wy, 0); ASSERT_LT(wy, ny);

    truckMask.posVel = {warpedIso.x, warpedIso.y, 0.0f, 0.0f};
    VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
    mutMem.recordDispatch(cmd, cfg, movedEgo, dt, perc.v15()->rectifier(),
                          perc.v15()->patchTokenizer(), {truckMask});
    s_ctx.endSingleTimeCommands(cmd);
    staticMem.readbackCells(cells);
    const StaticCellGpu& warpedCell = cells[static_cast<size_t>(wy) * nx + wx];
    EXPECT_GT(warpedCell.meta.z, 0.5f);
    EXPECT_NEAR(warpedCell.featConf.x, bestOcc, 1e-4f);

    RecordProperty("m5_locked_cells", static_cast<int>(warmStats.lockedCount));
    RecordProperty("m5_skip_ratio", testutil::fmtProp(warmStats.selectiveRefreshSkipRatio));
    RecordProperty("m5_occluded_conf", testutil::fmtProp(cells[static_cast<size_t>(wy) * nx + wx].featConf.w));
    profiler.destroy();
    perc.destroy();
    rig.destroy();
}

} // namespace
} // namespace cybercab
