#include "MetricsEvalHelpers.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/FrequencySplit.hpp"
#include "perception/v15/KinematicNavEncoder.hpp"
#include "perception/v15/PatchTokenizer.hpp"
#include "perception/v15/VirtualCameraRectifier.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cybercab {
namespace {
using namespace v15;

using V15PatchTokenTest = testutil::SharedGpuMetricsTest;

TEST(V15KinematicNavTest, Substeps100HzRingAndTargetsLaneCenter) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    WorldEngine world(cfg);
    KinematicNavEncoder enc;
    EXPECT_EQ(enc.sampleCount(), 0);
    EXPECT_FLOAT_EQ(enc.latestSample().speed, 0.0f);

    EgoState prev = world.ego(), curr = world.ego();
    const float dt = 1.0f / cfg.controlHz; // ~1/36 s -> 3 substeps at 100 Hz
    for (int i = 0; i < 15; ++i) {
        curr.speed = 8.5f + 0.1f * static_cast<float>(i);
        curr.yawRate = 0.02f * static_cast<float>(i);
        enc.pushControlStep(prev, curr, dt);
        prev = curr;
    }
    EXPECT_EQ(enc.sampleCount(), KinematicNavEncoder::kImuCapacity);
    EXPECT_NEAR(enc.latestSample().dt, dt / 3.0f, 1e-4f);
    EXPECT_GT(enc.meanImu().speed, 8.5f);

    enc.encodeNavigation(world, -1.0f, 5.0f);
    for (const auto& w : enc.navTokens()) {
        EXPECT_FLOAT_EQ(w.laneCenterOffsetM, -1.95f);
        EXPECT_GT(w.dividerClearanceM, 0.90f);    // inner edge well clear of 0.0 m divider
        EXPECT_GT(w.outerLimitClearanceM, 1.20f); // outer edge well clear of 4.23 m curb
        EXPECT_GT(w.tanVeh.y, 0.95f);
    }
    enc.reset();
    EXPECT_EQ(enc.sampleCount(), 0);
}

TEST_F(V15PatchTokenTest, PluckerInvariantsCrossViewEpipolarAndRingAges) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    cfg.photonNoiseEnabled = false;

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    FrequencySplit freq;
    freq.init(s_ctx, rig.cameraLinearSSBO(), cfg.camWidth, cfg.camHeight, 8,
              FreqSplitParams::forConfig(cfg.camWidth, cfg));
    VirtualCameraRectifier rect;
    rect.init(s_ctx, cfg, rig.cameraLinearSSBO(), freq.bandsSSBO(), cfg.camWidth, cfg.camHeight);

    PatchTokenizer tok;
    VirtualCameraRectifier emptyRect;
    EXPECT_THROW(tok.init(s_ctx, emptyRect), std::runtime_error);
    tok.init(s_ctx, rect);

    const float dt = 1.0f / cfg.controlHz;
    for (int f = 0; f < 3; ++f) {
        rig.renderAllCameras(world, cfg, 0.05f * f, nullptr);
        VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
        freq.recordDispatch(cmd);
        rect.recordDispatch(cmd);
        tok.recordDispatch(cmd, dt);
        s_ctx.endSingleTimeCommands(cmd);
        tok.recordSlotEgo(world.ego());
        world.stepAgents(dt);
    }
    EXPECT_EQ(tok.validSlots(), 3u);
    EXPECT_EQ(tok.headSlot(), 3u);
    EXPECT_EQ(tok.latestWrittenSlot(), 2u);

    constexpr size_t kFrameN = PatchTokenizer::kTokensPerFrame;
    std::vector<PatchTokenGpu> ring(PatchTokenizer::kHistoryFrames * kFrameN);
    tok.ringSSBO().downloadVector(ring);

    double maxOrthoErr = 0.0;
    int validCenterCount = 0;
    for (size_t i = 0; i < kFrameN; ++i) {
        const PatchTokenGpu& t = ring[2 * kFrameN + i];
        ASSERT_TRUE(std::isfinite(t.app0.x) && std::isfinite(t.app1.y) &&
                    std::isfinite(t.pluckerD.x) && std::isfinite(t.pluckerM.x));
        const Vec3 d{t.pluckerD.x, t.pluckerD.y, t.pluckerD.z};
        const Vec3 m{t.pluckerM.x, t.pluckerM.y, t.pluckerM.z};
        maxOrthoErr = std::max(maxOrthoErr, static_cast<double>(std::abs(d.dot(m))));
        EXPECT_NEAR(d.norm(), 1.0f, 1e-4f);
        EXPECT_FLOAT_EQ(t.pluckerM.w, 0.0f); // newest slot deltaT == 0
        if (t.app0.w > 0.9f) ++validCenterCount;
    }
    RecordProperty("plucker_max_ortho_err", testutil::fmtProp(maxOrthoErr));
    EXPECT_LT(maxOrthoErr, 1e-5);
    EXPECT_GT(validCenterCount, 5500);

    // Check temporal ring deltaT progression across slots 0, 1, 2
    EXPECT_NEAR(ring[0 * kFrameN].pluckerM.w, 2.0f * dt, 1e-4f);
    EXPECT_NEAR(ring[1 * kFrameN].pluckerM.w, 1.0f * dt, 1e-4f);
    EXPECT_NEAR(ring[2 * kFrameN].pluckerM.w, 0.0f, 1e-6f);

    // Cross-camera Plucker reciprocal product (d1.m2 + d2.m1 == 0 for rays meeting at 3D P_veh)
    auto nearestPatch = [&](int viewIdx, const Vec3& ptVeh) {
        const auto& vs = rect.viewSpec(viewIdx);
        const Vec3 pc = vs.uprightVehFromVirtCam.inverseRigid().transformPoint(ptVeh);
        const float px = vs.intr.cx + vs.intr.fx * pc.x / pc.z;
        const float py = vs.intr.cy + vs.intr.fy * pc.y / pc.z;
        const int gx = std::clamp(static_cast<int>(px / PatchTokenizer::kPatchPx), 0, PatchTokenizer::kGridW - 1);
        const int gy = std::clamp(static_cast<int>(py / PatchTokenizer::kPatchPx), 0, PatchTokenizer::kGridH - 1);
        return ring[2 * kFrameN + static_cast<size_t>((viewIdx * PatchTokenizer::kGridH + gy) * PatchTokenizer::kGridW + gx)];
    };
    const Vec3 pTargetVeh{0.2f, 0.0f, 12.0f};
    const auto t0 = nearestPatch(0, pTargetVeh);
    const auto t2 = nearestPatch(2, pTargetVeh);
    const Vec3 d0{t0.pluckerD.x, t0.pluckerD.y, t0.pluckerD.z}, m0{t0.pluckerM.x, t0.pluckerM.y, t0.pluckerM.z};
    const Vec3 d2{t2.pluckerD.x, t2.pluckerD.y, t2.pluckerD.z}, m2{t2.pluckerM.x, t2.pluckerM.y, t2.pluckerM.z};
    const float recip = std::abs(d0.dot(m2) + d2.dot(m0));
    RecordProperty("plucker_epipolar_recip", testutil::fmtProp(recip));
    EXPECT_LT(recip, 2.5e-3f);

    tok.reset();
    EXPECT_EQ(tok.validSlots(), 0u);
    tok.destroy();
    rect.destroy();
    freq.destroy();
    rig.destroy();
}

} // namespace
} // namespace cybercab
