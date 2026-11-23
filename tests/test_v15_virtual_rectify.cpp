#include "MetricsEvalHelpers.hpp"
#include "perception/v15/FrequencySplit.hpp"
#include "perception/v15/VirtualCameraRectifier.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cybercab {
namespace {
using namespace v15;

using V15VirtRectifyTest = testutil::SharedGpuMetricsTest;

TEST(V15VirtRectifyUnitTest, ViewSpecsBuildCanonicalDualBranchesAndGuardBadInputs) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    EXPECT_THROW(VirtualCameraRectifier::buildViewSpecs(SimConfig{}, 320, 240, Mat4::identity()),
                 std::runtime_error);

    const auto specs = VirtualCameraRectifier::buildViewSpecs(
        cfg, cfg.camWidth, cfg.camHeight, Mat4::identity());
    EXPECT_EQ(specs[VirtualCameraRectifier::kViewVru].srcCamIdx, 2);
    EXPECT_FLOAT_EQ(specs[VirtualCameraRectifier::kViewVru].rpyDeg.y, -12.0f);
    EXPECT_EQ(specs[VirtualCameraRectifier::kViewElevated].srcCamIdx, 0);
    EXPECT_FLOAT_EQ(specs[VirtualCameraRectifier::kViewElevated].rpyDeg.y, 5.0f);

    // Dual branch angular & FOV properties: VRU covers ground denser than Wide (view 2); Elevated captures overhead signal.
    const auto projVirt = [&](int v, const Vec3& ptVeh) -> Vec2 {
        const Vec3 pc = specs[v].uprightVehFromVirtCam.inverseRigid().transformPoint(ptVeh);
        if (pc.z <= 0.1f) return {-1e5f, -1e5f};
        return {specs[v].intr.cx + specs[v].intr.fx * pc.x / pc.z,
                specs[v].intr.cy + specs[v].intr.fy * pc.y / pc.z};
    };
    const Vec2 vruNear = projVirt(VirtualCameraRectifier::kViewVru, {0.0f, 0.25f, 5.5f});
    const Vec2 vruFar = projVirt(VirtualCameraRectifier::kViewVru, {0.0f, 0.25f, 15.0f});
    const Vec2 wideNear = projVirt(2, {0.0f, 0.25f, 5.5f});
    const Vec2 wideFar = projVirt(2, {0.0f, 0.25f, 15.0f});
    const float vruSpan = std::abs(vruNear.y - vruFar.y);
    const float wideSpan = std::abs(wideNear.y - wideFar.y);
    RecordProperty("vru_span_px", testutil::fmtProp(vruSpan));
    RecordProperty("wide_span_px", testutil::fmtProp(wideSpan));
    EXPECT_GT(vruSpan / std::max(0.1f, wideSpan), 1.8f);

    const Vec2 sigElev = projVirt(VirtualCameraRectifier::kViewElevated, {0.0f, 5.4f, 44.5f});
    EXPECT_GE(sigElev.x, 0.0f);
    EXPECT_LT(sigElev.x, static_cast<float>(VirtualCameraRectifier::kVirtW));
    EXPECT_GE(sigElev.y, 0.0f);
    EXPECT_LT(sigElev.y, static_cast<float>(VirtualCameraRectifier::kVirtH));
}

TEST_F(V15VirtRectifyTest, SyntheticRoundTripSubPixelAndRealCalibCompensationOnGpu) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    cfg.photonNoiseEnabled = false;
    const int w = cfg.camWidth, h = cfg.camHeight;
    const VkDeviceSize inBytes = static_cast<VkDeviceSize>(8 * w * h) * sizeof(Vec4);
    const VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VulkanBuffer inLin, inBands;
    inLin.create(s_ctx, inBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    inBands.create(s_ctx, inBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);

    VirtualCameraRectifier rect;
    EXPECT_THROW(rect.init(s_ctx, cfg, inLin, inBands, 0, h), std::runtime_error);
    rect.init(s_ctx, cfg, inLin, inBands, w, h);

    // Paint Gaussian beacon at exact forward-warped physical pixel for virtual center (80, 60) under R_calib.
    const Mat4 rCal = Mat4::rotX(degToRad(2.0f)) * Mat4::rotZ(degToRad(-1.5f));
    rect.updateViews(cfg, w, h, rCal);
    std::vector<Vec4> srcLin(static_cast<size_t>(8 * w * h), Vec4{0.0f, 0.0f, 0.0f, 1.0f});
    std::vector<Vec4> srcBands(static_cast<size_t>(8 * w * h), Vec4{0.0f, 0.0f, 0.0f, 0.0f});
    constexpr int kTx = 76, kTy = 58;
    for (int v = 0; v < VirtualCameraRectifier::kNumViews; ++v) {
        const auto& vs = rect.viewSpec(v);
        const auto& phys = cfg.cameras[vs.srcCamIdx];
        const auto pIntr = phys.intrinsics(w, h);
        const Vec3 rVirt = Vec3{(kTx + 0.5f - vs.intr.cx) / vs.intr.fx,
                                (kTy + 0.5f - vs.intr.cy) / vs.intr.fy, 1.0f}.normalized();
        const Vec3 rPhys = vs.physCamFromVirtCam.transformVector(rVirt);
        const Vec2 xd = distortNormalized(rPhys.x / rPhys.z, rPhys.y / rPhys.z, phys.distortion);
        const float px = xd.x * pIntr.fx + pIntr.cx, py = xd.y * pIntr.fy + pIntr.cy;
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                const int ix = static_cast<int>(std::floor(px)) + dx, iy = static_cast<int>(std::floor(py)) + dy;
                if (ix < 0 || ix >= w || iy < 0 || iy >= h) continue;
                const float d2 = (ix + 0.5f - px) * (ix + 0.5f - px) + (iy + 0.5f - py) * (iy + 0.5f - py);
                const size_t idx = (static_cast<size_t>(vs.srcCamIdx) * h + iy) * w + ix;
                const float amp = std::exp(-0.5f * d2);
                srcLin[idx].x = std::max(srcLin[idx].x, amp);
                srcBands[idx].x = std::max(srcBands[idx].x, amp * 0.5f);
            }
        }
    }
    inLin.uploadVector(srcLin);
    inBands.uploadVector(srcBands);
    VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
    rect.recordDispatch(cmd);
    s_ctx.endSingleTimeCommands(cmd);

    constexpr size_t kViewPx = VirtualCameraRectifier::kVirtW * VirtualCameraRectifier::kVirtH;
    std::vector<Vec4> vLin(VirtualCameraRectifier::kNumViews * kViewPx);
    std::vector<Vec4> vBands(VirtualCameraRectifier::kNumViews * kViewPx);
    rect.virtLinearSSBO().downloadVector(vLin);
    rect.virtBandsSSBO().downloadVector(vBands);
    double maxErrPx = 0.0;
    for (int v = 0; v < VirtualCameraRectifier::kNumViews; ++v) {
        double wx = 0.0, wy = 0.0, wsum = 0.0;
        for (int y = 0; y < VirtualCameraRectifier::kVirtH; ++y) {
            for (int x = 0; x < VirtualCameraRectifier::kVirtW; ++x) {
                const Vec4& px = vLin[static_cast<size_t>(v) * kViewPx + y * VirtualCameraRectifier::kVirtW + x];
                ASSERT_TRUE(std::isfinite(px.x) && std::isfinite(px.y) && std::isfinite(px.z) && std::isfinite(px.w));
                if (px.w > 0.5f && std::abs(x - kTx) <= 3 && std::abs(y - kTy) <= 3) {
                    wx += px.x * x; wy += px.x * y; wsum += px.x;
                }
            }
        }
        ASSERT_GT(wsum, 0.5);
        maxErrPx = std::max(maxErrPx, std::hypot(wx / wsum - kTx, wy / wsum - kTy));
    }
    RecordProperty("roundtrip_max_err_px", testutil::fmtProp(maxErrPx));
    EXPECT_LT(maxErrPx, 0.35);

    // Rendered-world tilt compensation: +2 deg pitch, -1.5 deg roll aligned back by R_calib.
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    FrequencySplit freq;
    freq.init(s_ctx, rig.cameraLinearSSBO(), w, h, 8, FreqSplitParams::forConfig(w, cfg));
    rect.rebind(cfg, rig.cameraLinearSSBO(), freq.bandsSSBO(), w, h);

    auto renderRectifiedView0 = [&](const Mat4& cal) {
        rect.updateViews(cfg, w, h, cal);
        VkCommandBuffer c2 = s_ctx.beginSingleTimeCommands();
        freq.recordDispatch(c2);
        rect.recordDispatch(c2);
        s_ctx.endSingleTimeCommands(c2);
        std::vector<Vec4> out(VirtualCameraRectifier::kNumViews * kViewPx);
        rect.virtLinearSSBO().downloadVector(out);
        return out;
    };
    rig.renderAllCameras(world, cfg, 0.1f, nullptr);
    const auto baseView = renderRectifiedView0(Mat4::identity());
    world.mutableEgo().pitch = degToRad(2.0f);
    world.mutableEgo().roll = degToRad(-1.5f);
    rig.renderAllCameras(world, cfg, 0.2f, nullptr);
    const auto uncalView = renderRectifiedView0(Mat4::identity());
    const auto calView = renderRectifiedView0(rCal);

    double maeUncal = 0.0, maeCal = 0.0;
    int cnt = 0;
    for (int y = 20; y < VirtualCameraRectifier::kVirtH - 20; ++y) {
        for (int x = 20; x < VirtualCameraRectifier::kVirtW - 20; ++x) {
            const size_t i = static_cast<size_t>(y) * VirtualCameraRectifier::kVirtW + x;
            if (baseView[i].w > 0.5f && uncalView[i].w > 0.5f && calView[i].w > 0.5f) {
                maeUncal += std::abs(uncalView[i].y - baseView[i].y);
                maeCal += std::abs(calView[i].y - baseView[i].y);
                ++cnt;
            }
        }
    }
    maeUncal /= std::max(1, cnt); maeCal /= std::max(1, cnt);
    RecordProperty("uncal_mae", testutil::fmtProp(maeUncal));
    RecordProperty("cal_mae", testutil::fmtProp(maeCal));
    EXPECT_LT(maeCal, 0.35 * maeUncal);

    freq.destroy();
    rig.destroy();
    rect.destroy();
    rect.destroy(); // idempotent destroy guard
    inLin.destroy();
    inBands.destroy();
}

} // namespace
} // namespace cybercab
