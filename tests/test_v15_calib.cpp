#include "MetricsEvalHelpers.hpp"
#include "perception/v15/CameraSelfCalib.hpp"
#include "perception/v15/FrequencySplit.hpp"

#include <cmath>
#include <vector>

namespace cybercab {
namespace {

using V15CalibTest = testutil::SharedGpuMetricsTest;
// Rendered-world tolerance: measured over a kWindowFrames sliding window of 8-camera edge chains.
constexpr float kGpuTolDeg = 0.35f;     // half-res (320x240, 8 px tiles)
constexpr float kGpuFullTolDeg = 0.15f; // full-res (640x480, 16 px tiles)

Vec3 trueUpright(float pitchRad, float rollRad) { return v15::calib::uprightVec(pitchRad, rollRad); }

TEST_F(V15CalibTest, GpuSelfCalibRecoversSprungPerturbationsFromRenderedWorld) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    cfg.photonNoiseEnabled = true;

    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    v15::FrequencySplit freq;
    freq.init(s_ctx, rig.cameraLinearSSBO(), cfg.camWidth, cfg.camHeight, 8,
              v15::FreqSplitParams::forConfig(cfg.camWidth, cfg));
    v15::CameraSelfCalib calib;
    calib.init(s_ctx, cfg, freq.bandsSSBO(), cfg.camWidth, cfg.camHeight);

    const float perts[3][2] = {{0.0f, 0.0f}, {2.0f, -2.0f}, {-2.0f, 2.0f}};
    const auto& loop = world.centerline();
    auto placeEgo = [&](float u, float pitchDeg, float rollDeg) {
        EgoState& e = world.mutableEgo();
        const Vec3 tan = loop.getTangentAt(u);
        e.pos = loop.getPointAt(u) + Vec3{tan.z, 0.0f, -tan.x} * -1.9f;
        e.yaw = std::atan2(tan.x, tan.z);
        e.pitch = degToRad(pitchDeg);
        e.roll = degToRad(rollDeg);
    };
    auto runFrame = [&](float u, float pitchDeg, float rollDeg, float t) {
        placeEgo(u, pitchDeg, rollDeg);
        rig.renderAllCameras(world, cfg, t, nullptr);
        VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
        freq.recordDispatch(cmd);
        calib.recordDispatch(cmd);
        s_ctx.endSingleTimeCommands(cmd);
        return calib.solveAndUpdate(cfg);
    };
    for (int p = 0; p < 3; ++p) {
        calib.reset();
        v15::CalibEstimate est{};
        for (int f = 0; f < v15::CameraSelfCalib::kWindowFrames; ++f) {
            est = runFrame(0.03f + 0.025f * static_cast<float>(f), perts[p][0], perts[p][1],
                           0.1f * static_cast<float>(f + 1));
        }
        const std::string tag = "case" + std::to_string(p);
        RecordProperty(tag + "_pitch_deg", testutil::fmtProp(est.pitchDeg));
        RecordProperty(tag + "_roll_deg", testutil::fmtProp(est.rollDeg));
        RecordProperty(tag + "_res_deg", testutil::fmtProp(est.residualDeg));
        RecordProperty(tag + "_inliers", std::to_string(est.inlierCount));
        RecordProperty(tag + "_window_eqs", std::to_string(calib.windowConstraintCount()));

        ASSERT_TRUE(est.converged);
        EXPECT_NEAR(est.pitchDeg, perts[p][0], kGpuTolDeg);
        EXPECT_NEAR(est.rollDeg, perts[p][1], kGpuTolDeg);

        const Vec3 rectifiedUp = calib.vehicleCalibMatrix().transformVector(
            trueUpright(degToRad(perts[p][0]), degToRad(perts[p][1])));
        EXPECT_NEAR(rectifiedUp.x, 0.0f, degToRad(kGpuTolDeg) * 1.5f);
        EXPECT_NEAR(rectifiedUp.y, 1.0f, 1e-3f);
        EXPECT_NEAR(rectifiedUp.z, 0.0f, degToRad(kGpuTolDeg) * 1.5f);
    }
    EXPECT_EQ(calib.windowFrameCount(), v15::CameraSelfCalib::kWindowFrames);
    runFrame(0.95f, perts[2][0], perts[2][1], 4.0f); // window slides: oldest frame evicted
    EXPECT_EQ(calib.windowFrameCount(), v15::CameraSelfCalib::kWindowFrames);

    // Full-res rebind: same window solve must still recover the perturbation.
    rig.resize(cfg.fullCamWidth, cfg.fullCamHeight);
    freq.rebind(rig.cameraLinearSSBO(), cfg.fullCamWidth, cfg.fullCamHeight,
                v15::FreqSplitParams::forConfig(cfg.fullCamWidth, cfg));
    calib.rebind(freq.bandsSSBO(), cfg.fullCamWidth, cfg.fullCamHeight);
    calib.reset();
    v15::CalibEstimate fullEst{};
    for (int f = 0; f < v15::CameraSelfCalib::kWindowFrames; ++f) {
        fullEst = runFrame(0.03f + 0.025f * static_cast<float>(f), perts[2][0], perts[2][1], 0.05f * f);
    }
    RecordProperty("fullres_pitch_deg", testutil::fmtProp(fullEst.pitchDeg));
    RecordProperty("fullres_roll_deg", testutil::fmtProp(fullEst.rollDeg));
    ASSERT_TRUE(fullEst.converged);
    EXPECT_NEAR(fullEst.pitchDeg, perts[2][0], kGpuFullTolDeg);
    EXPECT_NEAR(fullEst.rollDeg, perts[2][1], kGpuFullTolDeg);

    calib.destroy();
    EXPECT_EQ(calib.windowFrameCount(), 0);
    EXPECT_FALSE(calib.solveAndUpdate(cfg).converged); // uninitialized: safe no-op
    VkCommandBuffer cmd = s_ctx.beginSingleTimeCommands();
    calib.recordDispatch(cmd);                         // uninitialized: safe no-op
    s_ctx.endSingleTimeCommands(cmd);
    calib.rebind(freq.bandsSSBO(), cfg.camWidth, cfg.camHeight); // uninitialized: safe no-op
    freq.destroy();
    rig.destroy();
}

} // namespace
} // namespace cybercab
