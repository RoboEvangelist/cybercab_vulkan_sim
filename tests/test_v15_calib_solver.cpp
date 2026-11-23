#include "core/VulkanPipeline.hpp"
#include "perception/v15/CalibSolver.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace cybercab {
namespace {
using namespace v15;

// Synthesizes the tile observation of a world-vertical line through undistorted point (xu, yu):
// edge-plane normal nCam = p x up_cam, pushed forward into distorted pixel gradients.
CalibRayObs syntheticVerticalTile(const SimConfig& cfg, int c, int w, int h, float xu, float yu, const Vec3& upVeh) {
    const auto intr = cfg.cameras[c].intrinsics(w, h);
    const auto& d = cfg.cameras[c].distortion;
    const Vec2 xd = distortNormalized(xu, yu, d);
    const Vec3 nCam = Vec3{xu, yu, 1.0f}.cross(
        cfg.cameras[c].extrinsicVehicleLocalFromCam().inverseRigid().transformVector(upVeh)).normalized();
    const Vec2 jx = (distortNormalized(xu + 1e-4f, yu, d) - distortNormalized(xu - 1e-4f, yu, d)) * 5000.0f;
    const Vec2 jy = (distortNormalized(xu, yu + 1e-4f, d) - distortNormalized(xu, yu - 1e-4f, d)) * 5000.0f;
    const float det = jx.x * jy.y - jx.y * jy.x;
    const float gx = (jy.y * nCam.x - jy.x * nCam.y) / (det * intr.fx);
    const float gy = (-jx.y * nCam.x + jx.x * nCam.y) / (det * intr.fy);
    const float s = 1.0f / std::max(1e-6f, std::hypot(gx, gy));
    return {c, intr.cx + intr.fx * xd.x, intr.cy + intr.fy * xd.y, gx * s, gy * s, 1.2f, 0.95f};
}

TEST(V15CalibSolverTest, UprightVectorMatchesSprungRotation) {
    for (float p : {-0.04f, 0.0f, 0.03f}) {
        for (float r : {-0.035f, 0.0f, 0.02f}) {
            const Vec3 u = calib::uprightVec(p, r);
            const Vec3 back = (Mat4::rotX(p) * Mat4::rotZ(r)).transformVector(u); // must restore world up
            EXPECT_NEAR(back.x, 0.0f, 1e-6f);
            EXPECT_NEAR(back.y, 1.0f, 1e-6f);
            EXPECT_NEAR(back.z, 0.0f, 1e-6f);
        }
    }
}

TEST(V15CalibSolverTest, ValidOpticsMaskRejectsBordersAndOverscanSmear) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    const auto& wide = cfg.cameras.at(2); // FRONT WIDE, barrel k1 < -0.14 -> overscan 1.24
    EXPECT_FLOAT_EQ(wide.renderOverscan(), 1.24f);
    EXPECT_FLOAT_EQ(cfg.cameras.at(0).renderOverscan(), 1.04f);
    const auto intr = wide.intrinsics(320, 240);
    EXPECT_TRUE(calib::insideValidOptics(wide, intr, 160.0f, 120.0f, {0.0f, 0.0f}));
    EXPECT_FALSE(calib::insideValidOptics(wide, intr, 5.0f, 120.0f, {0.0f, 0.0f}));     // left border
    EXPECT_FALSE(calib::insideValidOptics(wide, intr, 315.0f, 120.0f, {0.0f, 0.0f}));   // right border
    EXPECT_FALSE(calib::insideValidOptics(wide, intr, 160.0f, 2.0f, {0.0f, 0.0f}));     // top border
    EXPECT_FALSE(calib::insideValidOptics(wide, intr, 160.0f, 120.0f, {2.5f, 0.0f}));   // pinhole u beyond render
    EXPECT_FALSE(calib::insideValidOptics(wide, intr, 160.0f, 120.0f, {0.0f, -2.0f}));  // pinhole v beyond render
}

TEST(V15CalibSolverTest, SyntheticRaysWithDistortionAndOutliersRecoverSubTenthDeg) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    const int w = cfg.camWidth, h = cfg.camHeight;
    EXPECT_FALSE(calib::solveFromObservations(cfg, w, h, {}).converged);
    EXPECT_FALSE(calib::solveFromObservations(SimConfig{}, w, h, {}).converged); // no cameras
    const float cases[3][2] = {{0.0f, 0.0f}, {2.0f, -2.0f}, {-2.0f, 2.0f}};
    for (const auto& tp : cases) {
        const Vec3 uTrue = calib::uprightVec(degToRad(tp[0]), degToRad(tp[1]));
        std::vector<CalibRayObs> obs;
        for (int c = 0; c < static_cast<int>(cfg.cameras.size()); ++c) {
            const auto intr = cfg.cameras[c].intrinsics(w, h);
            for (float xu : {-0.18f, 0.18f}) obs.push_back(syntheticVerticalTile(cfg, c, w, h, xu, -0.14f, uTrue));
            obs.push_back({c, intr.cx, intr.cy + 0.35f * h, 1.0f, 0.1f, 1.5f, 0.9f});  // below horizon: gated
            obs.push_back({c, intr.cx, intr.cy - 0.15f * h, 0.9f, 0.18f, 1.0f, 0.85f}); // tilted roofline outlier
            obs.push_back({c, intr.cx, intr.cy - 0.15f * h, 1.0f, 0.0f, 1.0f, 0.30f});  // incoherent: gated
            obs.push_back({-1, intr.cx, intr.cy, 1.0f, 0.0f, 1.0f, 0.95f});             // invalid camera: gated
        }
        std::vector<Vec4> chainsOnly;
        calib::extractEdgeNormals(cfg, w, h, obs, 0.0f, 0.0f, chainsOnly, false);
        EXPECT_TRUE(chainsOnly.empty()); // isolated tiles never form vertical chains
        const auto est = calib::solveFromObservations(cfg, w, h, obs);
        ASSERT_TRUE(est.converged);
        EXPECT_NEAR(est.pitchDeg, tp[0], 0.05f);
        EXPECT_NEAR(est.rollDeg, tp[1], 0.05f);
        EXPECT_GE(est.inlierCount, 16);
        EXPECT_GT(est.confidence, 0.3f);
        EXPECT_LT(est.residualDeg, 0.1f);
    }
}

TEST(V15CalibSolverTest, VerticalChainsAreLinkedAndKinkedChainsRejected) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    const int w = cfg.camWidth, h = cfg.camHeight;
    const Vec3 uTrue = calib::uprightVec(degToRad(1.0f), degToRad(-1.5f));
    std::vector<CalibRayObs> obs;
    for (int c = 0; c < 8; ++c) {
        const auto intr = cfg.cameras[c].intrinsics(w, h);
        const Vec3 up = cfg.cameras[c].extrinsicVehicleLocalFromCam().inverseRigid().transformVector(uTrue);
        for (float uPx : {0.3f, 0.5f, 0.7f}) {
            const float x0 = (uPx * w - intr.cx) / intr.fx, y0 = (0.05f * h - intr.cy) / intr.fy;
            for (int k = 0; k < 7; ++k) { // tiles every 10 px down one true world-vertical edge
                const float yu = y0 + 10.0f * static_cast<float>(k) / intr.fy;
                const float s = (yu - y0) / (up.y - yu * up.z);
                const float xu = (x0 + s * up.x) / (1.0f + s * up.z);
                obs.push_back(syntheticVerticalTile(cfg, c, w, h, xu, yu, uTrue));
            }
        }
    }
    std::vector<Vec4> eqs;
    calib::extractEdgeNormals(cfg, w, h, obs, 0.0f, 0.0f, eqs, false);
    RecordProperty("chain_planes", std::to_string(eqs.size()));
    ASSERT_GE(eqs.size(), 12u);
    const auto est = calib::solveUpright(eqs, 0.0f, 0.0f);
    ASSERT_TRUE(est.converged);
    EXPECT_NEAR(est.pitchDeg, 1.0f, 0.05f);
    EXPECT_NEAR(est.rollDeg, -1.5f, 0.05f);

    // Zig-zag columns (odd tiles shifted 2 px): still linked, but the straightness test rejects them.
    std::vector<CalibRayObs> kinked = obs;
    for (size_t i = 0; i < kinked.size(); ++i) if ((i % 7) % 2 == 1) kinked[i].cxPx += 2.0f;
    std::vector<Vec4> eqsKinked;
    calib::extractEdgeNormals(cfg, w, h, kinked, 0.0f, 0.0f, eqsKinked, false);
    EXPECT_TRUE(eqsKinked.empty());
    EXPECT_FALSE(calib::solveUpright({}, 0.0f, 0.0f).converged);
    const std::vector<Vec4> degenerate(6, Vec4{0.0f, 1.0f, 0.0f, 1.0f}); // no x/z information
    EXPECT_FALSE(calib::solveUpright(degenerate, 0.0f, 0.0f).converged);
}

} // namespace
} // namespace cybercab
