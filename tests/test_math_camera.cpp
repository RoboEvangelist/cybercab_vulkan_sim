#include "config/SimConfig.hpp"
#include "core/Math.hpp"
#include "core/VulkanPipeline.hpp"
#include "world/RoadCurve.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace cybercab;

TEST(ConfigAndMathTest, LoadsYamlConfigWithEightCameras) {
    const std::string path = resolveAssetPath("config/cybercab_sim.yaml");
    const SimConfig cfg = SimConfig::loadFromYaml(path);
    EXPECT_EQ(cfg.cameras.size(), 8u);
    EXPECT_FLOAT_EQ(cfg.controlHz, 36.0f);
    EXPECT_EQ(cfg.occNx(), 60);
    EXPECT_EQ(cfg.occNy(), 40);
    EXPECT_EQ(cfg.occNz(), 8);
    EXPECT_EQ(cfg.cameras[0].id, "CAM_FRONT_MAIN");
    EXPECT_NEAR(cfg.cameras[2].hfovDeg, 120.0f, 1e-4f);
}

TEST(ConfigAndMathTest, CameraExtrinsicsAndIntrinsicsAreConsistent) {
    const SimConfig cfg = SimConfig::defaultConfig();

    // Verify rigid inverse identity for all 8 cameras
    for (const auto& cam : cfg.cameras) {
        const Mat4 T = cam.extrinsicEgoFromCamISO();
        const Mat4 Tinv = cam.extrinsicCamFromEgoISO();
        const Mat4 I = T * Tinv;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                const float expected = (r == c) ? 1.0f : 0.0f;
                EXPECT_NEAR(I.at(r, c), expected, 1e-4f);
            }
        }
    }

    // Front main camera: point 10m straight ahead at camera height -> near optical axis, slightly above cy due to -1.5 deg pitch
    const auto& front = cfg.cameras[0];
    const Vec3 pCamF = front.extrinsicCamFromEgoISO().transformPoint({11.48f, 0.0f, 1.42f});
    EXPECT_GT(pCamF.z, 9.5f);
    EXPECT_NEAR(pCamF.x, 0.0f, 1e-3f);
    EXPECT_LT(pCamF.y, 0.0f); // camera pitched down -1.5 deg, so horizontal point appears slightly above optical axis (-Y_cam)

    // Left Pillar (+55 deg yaw in ISO Ego): optical axis points toward (cos 55, sin 55, -)
    const auto& pillarL = cfg.cameras[3];
    const Vec3 dirL = pillarL.extrinsicEgoFromCamISO().transformVector({0.0f, 0.0f, 1.0f});
    EXPECT_NEAR(dirL.x, std::cos(degToRad(55.0f)) * std::cos(degToRad(3.0f)), 1e-2f);
    EXPECT_NEAR(dirL.y, std::sin(degToRad(55.0f)) * std::cos(degToRad(3.0f)), 1e-2f);

    // Rear camera (180 deg yaw): optical axis points toward -X_iso
    const auto& rear = cfg.cameras[7];
    const Vec3 dirR = rear.extrinsicEgoFromCamISO().transformVector({0.0f, 0.0f, 1.0f});
    EXPECT_LT(dirR.x, -0.98f);
}

TEST(ConfigAndMathTest, ClosedLoopRoadSplineIsContinuous) {
    const auto loop = buildCybercabLoopPath(0.0f);
    EXPECT_GT(loop.getLength(), 300.0f);
    const Vec3 p0 = loop.getPointAt(0.0f);
    const Vec3 p1 = loop.getPointAt(0.9999f);
    EXPECT_LT((p0 - p1).norm(), 0.2f);
    const Vec3 t0 = loop.getTangentAt(0.25f);
    EXPECT_NEAR(t0.norm(), 1.0f, 1e-3f);
}

TEST(ConfigAndMathTest, Mat4RigidInverseAndVectorTransform) {
    const Mat4 xform = Mat4::translation({12.5f, -3.0f, 44.0f}) * Mat4::rotY(0.75f) * Mat4::rotX(-0.22f) * Mat4::rotZ(0.11f);
    const Mat4 inv = xform.inverseRigid();
    const Mat4 id = xform * inv;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(id.at(r, c), (r == c) ? 1.0f : 0.0f, 1e-4f);
        }
    }

    const Vec3 pt{2.0f, 5.0f, -1.5f};
    const Vec3 roundTrip = inv.transformPoint(xform.transformPoint(pt));
    EXPECT_NEAR(roundTrip.x, pt.x, 1e-4f);
    EXPECT_NEAR(roundTrip.y, pt.y, 1e-4f);
    EXPECT_NEAR(roundTrip.z, pt.z, 1e-4f);

    // transformVector must ignore translation component
    const Vec3 dir{1.0f, 0.0f, 0.0f};
    const Vec3 rotDir = xform.transformVector(dir);
    EXPECT_NEAR(rotDir.norm(), 1.0f, 1e-4f);
    EXPECT_LT(rotDir.x, 2.0f);
}

TEST(ConfigAndMathTest, VulkanPerspectiveAndLookAtMapDepthAndPixels) {
    const float w = 320.0f, h = 240.0f;
    const float fx = 280.0f, fy = 280.0f, cx = 160.0f, cy = 120.0f;
    const float zNear = 0.5f, zFar = 50.0f;
    const Mat4 proj = Mat4::vulkanPerspectiveCV(fx, fy, cx, cy, w, h, zNear, zFar);

    // Optical axis near-plane point -> z_ndc == 0
    const Vec4 pNear = proj * Vec4{0.0f, 0.0f, zNear, 1.0f};
    EXPECT_NEAR(pNear.z / pNear.w, 0.0f, 1e-4f);

    // Far-plane point -> z_ndc == 1
    const Vec4 pFar = proj * Vec4{0.0f, 0.0f, zFar, 1.0f};
    EXPECT_NEAR(pFar.z / pFar.w, 1.0f, 1e-4f);

    // Off-center point maps to expected normalized screen position
    const Vec4 pObj = proj * Vec4{2.0f, -1.0f, 10.0f, 1.0f};
    const float uPx = ((pObj.x / pObj.w) + 1.0f) * 0.5f * w;
    const float vPx = ((pObj.y / pObj.w) + 1.0f) * 0.5f * h;
    EXPECT_NEAR(uPx, fx * (2.0f / 10.0f) + cx, 1e-2f);
    EXPECT_NEAR(vPx, fy * (-1.0f / 10.0f) + cy, 1e-2f);

    // LookAt Vulkan aligns target along +Z_cam and keeps world up (-Y_cam)
    const Mat4 view = Mat4::lookAtVulkan({0.0f, 4.0f, -10.0f}, {0.0f, 1.0f, 0.0f});
    const Vec3 tgtCam = view.transformPoint({0.0f, 1.0f, 0.0f});
    EXPECT_GT(tgtCam.z, 9.0f);
    EXPECT_NEAR(tgtCam.x, 0.0f, 1e-4f);
    EXPECT_NEAR(tgtCam.y, 0.0f, 1e-4f);
}

TEST(ConfigAndMathTest, UndistortNormalizedInvertsBrownConradyDistortion) {
    const std::array<float, 4> dist{-0.22f, 0.06f, 0.001f, -0.002f};
    const float xuTrue = 0.32f;
    const float yuTrue = -0.18f;
    const float r2 = xuTrue * xuTrue + yuTrue * yuTrue;
    const float r4 = r2 * r2;
    const float rad = 1.0f + dist[0] * r2 + dist[1] * r4;
    const float xd = xuTrue * rad + 2.0f * dist[2] * xuTrue * yuTrue + dist[3] * (r2 + 2.0f * xuTrue * xuTrue);
    const float yd = yuTrue * rad + dist[2] * (r2 + 2.0f * yuTrue * yuTrue) + 2.0f * dist[3] * xuTrue * yuTrue;

    const Vec2 rec = undistortNormalized(xd, yd, dist);
    EXPECT_NEAR(rec.x, xuTrue, 1e-3f);
    EXPECT_NEAR(rec.y, yuTrue, 1e-3f);
}

TEST(ConfigAndMathTest, AllEightCamerasAndLoopLanesValid) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    ASSERT_EQ(cfg.cameras.size(), 8u);
    for (const auto& cam : cfg.cameras) {
        const auto intr = cam.intrinsics(cfg.camWidth, cfg.camHeight);
        EXPECT_GT(intr.fx, 50.0f);
        EXPECT_GT(intr.fy, 50.0f);
        EXPECT_FLOAT_EQ(intr.cx, 160.0f);
        EXPECT_FLOAT_EQ(intr.cy, 120.0f);
        EXPECT_GT(cam.maxRangeM, 15.0f);
    }

    const auto outerLane = buildCybercabLoopPath(-1.9f);
    const auto innerLane = buildCybercabLoopPath(1.9f);
    EXPECT_GT(outerLane.getLength(), innerLane.getLength());
    const Vec3 pOuter = outerLane.getPointAt(0.0f);
    const Vec3 pInner = innerLane.getPointAt(0.0f);
    EXPECT_NEAR((pOuter - pInner).norm(), 3.8f, 0.35f);

    // Along curve at u=0.25, closest separation between lanes equals 3.8m width
    const Vec3 qOuter = outerLane.getPointAt(0.25f);
    float minSep = 99.0f;
    for (int i = 200; i <= 300; ++i) {
        const float u = static_cast<float>(i) / 1000.0f;
        minSep = std::min(minSep, (qOuter - innerLane.getPointAt(u)).norm());
    }
    EXPECT_NEAR(minSep, 3.8f, 0.25f);
}
