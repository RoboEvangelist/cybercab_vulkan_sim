#include "MetricsEvalHelpers.hpp"
#include "app/GaussianOverlayMesh.hpp"
#include "app/VulkanViewer.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/sota/Gaussian3DTypes.hpp"

#include <gtest/gtest.h>
#include <malloc.h>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace cybercab::gaussian_overlay {
namespace {

using GaussianOverlayGpuTest = testutil::SharedGpuMetricsTest;

long getHeapUsedBytes() {
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
  return static_cast<long>(::mallinfo2().uordblks);
#else
  return static_cast<long>(::mallinfo().uordblks);
#endif
}

V15Gaussian3D makeSampleGaussian(Vec3 center, Vec3 scale, float yaw,
                                 float opacity, Vec3 col, bool isDyn,
                                 SemanticClass sem) {
  return V15Gaussian3D{
    {center.x, center.y, center.z, opacity},
    {scale.x, scale.y, scale.z, static_cast<float>(static_cast<int>(sem))},
    {0.0f, 0.0f, yaw, isDyn ? 1.0f : 0.0f},
    {col.x, col.y, col.z, 0.95f}};
}

TEST(GaussianOverlayMeshTest, EllipsoidGeometryWindingNormalsAndYawRotation) {
  static_assert(std::is_trivially_copyable_v<GaussianOverlayStats>);
  const Vec3 center{4.0f, 1.2f, -3.5f};
  const Vec3 scale{1.8f, 0.9f, 3.2f};
  const float yaw = 0.65f;
  const std::vector<V15Gaussian3D> input = {
    makeSampleGaussian(center, scale, yaw, 0.85f, {0.2f, 0.7f, 0.9f},
                       true, SemanticClass::Vehicle)};

  std::vector<SceneVertex> verts;
  std::vector<uint32_t> idx;
  const GaussianOverlayStats stats = appendGaussianEllipsoids(
    input, verts, idx, 160000u, 300000u);

  EXPECT_EQ(stats.inputCount, 1u);
  EXPECT_EQ(stats.renderedCount, 1u);
  EXPECT_EQ(stats.skippedInvalid, 0u);
  EXPECT_EQ(stats.truncatedByBudget, 0u);
  EXPECT_EQ(stats.verticesAdded, kVertsPerEllipsoid);
  EXPECT_EQ(stats.indicesAdded, kIndicesPerEllipsoid);
  ASSERT_EQ(verts.size(), kVertsPerEllipsoid);
  ASSERT_EQ(idx.size(), kIndicesPerEllipsoid);

  const float c = std::cos(yaw);
  const float s = std::sin(yaw);
  for (const auto& v : verts) {
    const float dx = v.px - center.x;
    const float dy = v.py - center.y;
    const float dz = v.pz - center.z;
    const float lx = dx * c - dz * s;
    const float ly = dy;
    const float lz = dx * s + dz * c;
    const float eq = (lx / scale.x) * (lx / scale.x) +
                     (ly / scale.y) * (ly / scale.y) +
                     (lz / scale.z) * (lz / scale.z);
    EXPECT_NEAR(eq, 1.0f, 1e-4f);

    const float nLen = std::sqrt(v.nx * v.nx + v.ny * v.ny + v.nz * v.nz);
    EXPECT_NEAR(nLen, 1.0f, 1e-4f);
    EXPECT_GT(v.nx * dx + v.ny * dy + v.nz * dz, 0.0f);
    EXPECT_FLOAT_EQ(v.emissive, GaussianOverlayConfig{}.emissiveDynamic);
  }

  for (size_t k = 0u; k < idx.size(); k += 3u) {
    const SceneVertex& v0 = verts[idx[k + 0u]];
    const SceneVertex& v1 = verts[idx[k + 1u]];
    const SceneVertex& v2 = verts[idx[k + 2u]];
    const Vec3 e1{v1.px - v0.px, v1.py - v0.py, v1.pz - v0.pz};
    const Vec3 e2{v2.px - v0.px, v2.py - v0.py, v2.pz - v0.pz};
    const Vec3 cross = e1.cross(e2);
    const Vec3 out{(v0.px + v1.px + v2.px) / 3.0f - center.x,
                   (v0.py + v1.py + v2.py) / 3.0f - center.y,
                   (v0.pz + v1.pz + v2.pz) / 3.0f - center.z};
    EXPECT_GT(cross.dot(out), 0.0f);
  }
}

TEST(GaussianOverlayMeshTest, FmeaInvalidOpacityNanScaleAndBudgetCaps) {
  const float qNan = std::numeric_limits<float>::quiet_NaN();
  const float qInf = std::numeric_limits<float>::infinity();
  std::vector<V15Gaussian3D> invalidInputs = {
    makeSampleGaussian({qNan, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, 0.0f, 0.8f,
                       {1, 1, 1}, false, SemanticClass::Vehicle),
    makeSampleGaussian({0.0f, 1.0f, 1.0f}, {qInf, 1.0f, 1.0f}, 0.0f, 0.8f,
                       {1, 1, 1}, false, SemanticClass::Vehicle),
    makeSampleGaussian({0.0f, 1.0f, 1.0f}, {-0.5f, 1.0f, 1.0f}, 0.0f, 0.8f,
                       {1, 1, 1}, false, SemanticClass::Vehicle),
    makeSampleGaussian({0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, 0.0f, 0.8f,
                       {1, 1, 1}, false, SemanticClass::Vehicle),
    makeSampleGaussian({0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, 0.0f, 0.01f,
                       {1, 1, 1}, false, SemanticClass::Vehicle)};

  std::vector<SceneVertex> verts;
  std::vector<uint32_t> idx;
  const GaussianOverlayStats badStats = appendGaussianEllipsoids(
    invalidInputs, verts, idx, 160000u, 300000u);
  EXPECT_EQ(badStats.renderedCount, 0u);
  EXPECT_EQ(badStats.skippedInvalid, 5u);
  EXPECT_TRUE(verts.empty());
  EXPECT_TRUE(idx.empty());

  const V15Gaussian3D sanitized = makeSampleGaussian(
    {0.0f, 2.0f, 0.0f}, {0.01f, 20.0f, 1.0f}, qNan, 0.9f,
    {0.0f, 0.0f, 0.0f}, false, SemanticClass::Vehicle);
  const GaussianOverlayStats sanStats = appendGaussianEllipsoids(
    {sanitized}, verts, idx, 160000u, 300000u);
  ASSERT_EQ(sanStats.renderedCount, 1u);
  const Vec3 expRgb = sota::semanticDefaultColor(SemanticClass::Vehicle);
  EXPECT_FLOAT_EQ(verts[0].r, expRgb.x);
  EXPECT_FLOAT_EQ(verts[0].g, expRgb.y);
  EXPECT_FLOAT_EQ(verts[0].b, expRgb.z);
  EXPECT_FLOAT_EQ(verts[0].emissive, GaussianOverlayConfig{}.emissiveStatic);

  float minX = 1e9f, maxX = -1e9f, maxY = -1e9f;
  for (const auto& v : verts) {
    ASSERT_TRUE(std::isfinite(v.px) && std::isfinite(v.py) && std::isfinite(v.pz));
    minX = std::min(minX, v.px);
    maxX = std::max(maxX, v.px);
    maxY = std::max(maxY, v.py);
  }
  EXPECT_NEAR(maxX - minX, 2.0f * kDefaultMinScaleM, 1e-4f);
  EXPECT_NEAR(maxY - 2.0f, kDefaultMaxScaleM, 1e-4f);

  std::vector<V15Gaussian3D> five(
    5u, makeSampleGaussian({1.0f, 1.0f, 1.0f}, {0.5f, 0.5f, 0.5f}, 0.0f,
                           0.7f, {0.4f, 0.6f, 0.8f}, false,
                           SemanticClass::Static));

  verts.clear();
  idx.clear();
  GaussianOverlayConfig capCfg{};
  capCfg.maxGaussians = 2u;
  const auto capStats = appendGaussianEllipsoids(
    five, verts, idx, 160000u, 300000u, capCfg);
  EXPECT_EQ(capStats.renderedCount, 2u);
  EXPECT_EQ(capStats.truncatedByBudget, 3u);

  verts.clear();
  idx.clear();
  const auto vCapStats = appendGaussianEllipsoids(
    five, verts, idx, kVertsPerEllipsoid * 2u + 10u, 300000u);
  EXPECT_EQ(vCapStats.renderedCount, 2u);
  EXPECT_EQ(vCapStats.truncatedByBudget, 3u);
  EXPECT_LE(verts.size(), kVertsPerEllipsoid * 2u + 10u);

  verts.clear();
  idx.clear();
  const auto iCapStats = appendGaussianEllipsoids(
    five, verts, idx, 160000u, kIndicesPerEllipsoid * 2u + 10u);
  EXPECT_EQ(iCapStats.renderedCount, 2u);
  EXPECT_EQ(iCapStats.truncatedByBudget, 3u);
  EXPECT_LE(idx.size(), kIndicesPerEllipsoid * 2u + 10u);
}

TEST(GaussianOverlayMeshTest, ToggleStateIndependentOfVoxelsAndToastMessages) {
  ViewerToggleState toggles{};
  EXPECT_TRUE(toggles.showVoxels);
  EXPECT_FALSE(toggles.showGaussians);

  const std::string tOn = toggleGaussianOverlay(toggles.showGaussians, 512u);
  EXPECT_TRUE(toggles.showGaussians);
  EXPECT_TRUE(toggles.showVoxels);
  EXPECT_NE(tOn.find("3D Gaussians ON (512 ellipsoids)"), std::string::npos);

  toggles.showVoxels = false;
  const std::string tOff = toggleGaussianOverlay(toggles.showGaussians, 512u);
  EXPECT_FALSE(toggles.showGaussians);
  EXPECT_FALSE(toggles.showVoxels);
  EXPECT_EQ(tOff, "3D Gaussians OFF");

  const std::string tZero = toggleGaussianOverlay(toggles.showGaussians, 0u);
  EXPECT_TRUE(toggles.showGaussians);
  EXPECT_NE(tZero.find("0 active: use v15 or --spatial-rep gaussianformer3d"),
            std::string::npos);
}

TEST(GaussianOverlayMeshTest, ZeroAllocAfterReserveAtFull512Capacity) {
  std::vector<V15Gaussian3D> crowd(
    512u, makeSampleGaussian({2.0f, 1.0f, 5.0f}, {0.6f, 0.5f, 0.8f}, 0.2f,
                             0.75f, {0.2f, 0.8f, 0.9f}, true,
                             SemanticClass::Vehicle));
  std::vector<SceneVertex> verts;
  std::vector<uint32_t> idx;
  verts.reserve(160000u);
  idx.reserve(300000u);

  appendGaussianEllipsoids(crowd, verts, idx, 160000u, 300000u);
  const long bytesBefore = getHeapUsedBytes();
  for (int i = 0; i < 10; ++i) {
    verts.clear();
    idx.clear();
    const auto st = appendGaussianEllipsoids(
      crowd, verts, idx, 160000u, 300000u);
    ASSERT_EQ(st.renderedCount, 512u);
  }
  const long bytesAfter = getHeapUsedBytes();
  EXPECT_EQ(bytesAfter, bytesBefore);
}

TEST_F(GaussianOverlayGpuTest,
       EndToEndPipelineGaussiansFeedOverlayAcrossV14AndV15) {
  auto runPipe = [&](const std::string& pipeVer, const std::string& spatialRep,
                     std::vector<V15Gaussian3D>& outGaussians) {
    SimConfig cfg =
      SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = pipeVer;
    cfg.spatialRepMode = spatialRep;
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(s_ctx, cfg, world);
    PerceptionPipeline pipe(cfg);
    pipe.init(s_ctx, rig);

    PerceptionFrameOutput out{};
    const float dt = 1.0f / cfg.controlHz;
    for (int s = 0; s < 3; ++s) {
      rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
      pipe.runFrame(s_ctx, rig, world, dt, true, out);
      world.stepAgents(dt);
      world.stepEgoAndPhysics(dt, PlannerCommand{}, {});
    }
    outGaussians = out.v15Gaussians;
    pipe.destroy();
    rig.destroy();
  };

  std::vector<SceneVertex> verts;
  std::vector<uint32_t> idx;

  std::vector<V15Gaussian3D> v15GfGauss;
  runPipe("v15", "gaussianformer3d", v15GfGauss);
  EXPECT_GT(v15GfGauss.size(), 0u);
  const auto stV15 = appendGaussianEllipsoids(
    v15GfGauss, verts, idx, 160000u, 300000u);
  EXPECT_GT(stV15.renderedCount, 0u);
  EXPECT_EQ(stV15.skippedInvalid, 0u);
  for (const auto& v : verts) {
    EXPECT_TRUE(std::isfinite(v.px) && std::isfinite(v.py) && std::isfinite(v.pz));
    EXPECT_TRUE(std::isfinite(v.nx) && std::isfinite(v.ny) && std::isfinite(v.nz));
  }

  verts.clear();
  idx.clear();
  std::vector<V15Gaussian3D> v14GfGauss;
  runPipe("v14", "gaussianformer3d", v14GfGauss);
  EXPECT_GT(v14GfGauss.size(), 0u);
  const auto stV14 = appendGaussianEllipsoids(
    v14GfGauss, verts, idx, 160000u, 300000u);
  EXPECT_GT(stV14.renderedCount, 0u);
  EXPECT_EQ(stV14.skippedInvalid, 0u);

  std::vector<V15Gaussian3D> v14LegGauss;
  runPipe("v14", "legacy", v14LegGauss);
  EXPECT_TRUE(v14LegGauss.empty());
}

}  // namespace
}  // namespace cybercab::gaussian_overlay
