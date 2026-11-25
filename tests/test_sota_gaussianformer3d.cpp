#include "app/V15HudOverlay.hpp"
#include "config/SimConfig.hpp"
#include "core/TraceLogger.hpp"
#include "perception/sota/DeformableAttention3D.hpp"
#include "perception/sota/Gaussian3DTypes.hpp"
#include "perception/sota/GaussianFormer3DPipeline.hpp"
#include "perception/sota/GaussianToVoxelSplatter.hpp"
#include "perception/sota/VoxelToGaussianInit.hpp"
#include "perception/v15/GaussianWorldModel.hpp"

#include <gtest/gtest.h>
#include <json/json.h>
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace cybercab::sota {
namespace {

constexpr float kPiTol = 1e-4f;
constexpr int kZeroAllocsTol = 0;

TEST(GaussianFormer3DUnitTest, ConfigValidationAndQuaternionMath) {
    const SimConfig defCfg = SimConfig::defaultConfig();
    EXPECT_EQ(defCfg.spatialRepMode, "legacy");
    EXPECT_FALSE(defCfg.isGaussianFormer3D());
    EXPECT_TRUE(SimConfig::isValidSpatialRepMode("legacy"));
    EXPECT_TRUE(SimConfig::isValidSpatialRepMode("gaussianformer3d"));
    EXPECT_FALSE(SimConfig::isValidSpatialRepMode("voxel_only"));

    const float yaws[] = {-2.35f, -0.75f, 0.0f, 0.92f, 2.18f};
    for (float y : yaws) {
        const Quat3D q = Quat3D::fromYaw(y);
        EXPECT_NEAR(q.normSq(), 1.0f, kPiTol);
        EXPECT_NEAR(q.toYaw(), y, 1e-3f);

        const Vec3 probe{1.4f, -0.3f, 2.1f};
        const Vec3 roundTrip = q.invRotateVector(q.rotateVector(probe));
        EXPECT_NEAR(roundTrip.x, probe.x, kPiTol);
        EXPECT_NEAR(roundTrip.y, probe.y, kPiTol);
        EXPECT_NEAR(roundTrip.z, probe.z, kPiTol);
    }

    Gaussian3DPrimitive g{};
    g.mean = {10.0f, 0.8f, 4.0f};
    g.scale = {2.4f, 0.7f, 0.5f};
    g.rot = Quat3D::fromYaw(0.4f);
    g.opacity = 0.88f;
    g.confidence = 0.9f;
    g.isDynamic = true;
    g.semLogits[static_cast<size_t>(SemanticClass::Vehicle)] = 5.0f;

    const Vec3 alongLong = g.mean + g.rot.rotateVector({1.5f, 0.0f, 0.0f});
    const Vec3 alongShort = g.mean + g.rot.rotateVector({0.0f, 0.0f, 1.5f});
    EXPECT_GT(evalAnisotropicGaussian3D(g, alongLong), evalAnisotropicGaussian3D(g, alongShort));

    const V15Gaussian3D gpuG = toV15Gaussian3D(g);
    const std::vector<V15Gaussian3D> oneVec{gpuG};
    const float gfProb = evalAnisotropicGaussian3D(g, alongLong);
    const float v15Prob = v15::GaussianWorldModel::queryOccupancyProb(oneVec, alongLong, 0.0f, false);
    EXPECT_NEAR(gfProb, v15Prob, 1e-3f);
}

TEST(GaussianFormer3DUnitTest, VoxelToGaussianInitClustersVoxelsAndTracks) {
    const SimConfig cfg = SimConfig::defaultConfig();
    GaussianFormer3DConfig gfCfg{};
    gfCfg.mergeRadiusM = 1.2f;
    gfCfg.maxGaussians = 64u;

    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = 0.0f;

    Track3D car{};
    car.trackId = 7;
    car.semantic = SemanticClass::Vehicle;
    car.state = TrackState::Tracked;
    car.worldPos = {12.0f, 0.7f, 1.0f};
    car.worldVel = {8.0f, 0.0f, 0.0f};
    car.halfExtents = {1.0f, 0.75f, 2.2f};
    car.yaw = 0.05f;
    car.score = 0.92f;

    std::vector<ActiveVoxel> voxels;
    for (int i = 0; i < 6; ++i) {
        const float offset = 0.25f * static_cast<float>(i);
        voxels.push_back(ActiveVoxel{
            18.0f + offset, -4.0f, 1.0f, 18.0f + offset, -4.0f,
            SemanticClass::SignalSign, 0.85f, 0.0f, 0.0f, 0.12f});
    }
    for (int i = 0; i < 4; ++i) {
        voxels.push_back(ActiveVoxel{
            24.0f + 0.2f * static_cast<float>(i), 3.0f, 0.8f,
            24.0f + 0.2f * static_cast<float>(i), 3.0f,
            SemanticClass::Static, 0.80f, 0.0f, 0.0f, 0.15f});
    }

    VoxelToGaussianInit init;
    init.reserve(gfCfg.maxGaussians);
    std::vector<Gaussian3DPrimitive> prims;
    const uint32_t count = init.initialize(
        cfg, gfCfg, ego, voxels, {car}, nullptr, 0, 0, nullptr, prims);

    EXPECT_EQ(count, prims.size());
    EXPECT_LT(count, voxels.size() + 1u);
    EXPECT_GE(count, 3u);

    bool hasVeh = false;
    bool hasSign = false;
    for (const auto& g : prims) {
        if (g.argmaxSemantic() == SemanticClass::Vehicle) hasVeh = true;
        if (g.argmaxSemantic() == SemanticClass::SignalSign) hasSign = true;
    }
    EXPECT_TRUE(hasVeh);
    EXPECT_TRUE(hasSign);
}

TEST(GaussianFormer3DUnitTest, DeformableAttentionReducesEntropyAndRefinesPose) {
    GaussianFormer3DConfig gfCfg{};
    gfCfg.numRefinementLayers = 3u;
    gfCfg.camAttnWeight = 0.60f;
    gfCfg.lidarAttnWeight = 0.40f;

    Gaussian3DPrimitive targetG{};
    targetG.mean = {14.2f, 0.9f, 0.2f};
    targetG.scale = {0.8f, 0.6f, 1.5f};
    targetG.rot = Quat3D::fromYaw(0.0f);
    targetG.opacity = 0.55f;
    targetG.confidence = 0.55f;
    targetG.isDynamic = true;
    for (auto& l : targetG.semLogits) l = 0.0f;
    targetG.semLogits[static_cast<size_t>(SemanticClass::Vehicle)] = 0.25f;
    const float confBefore = targetG.semanticConfidence();

    Gaussian3DPrimitive isolatedG{};
    isolatedG.mean = {-18.0f, 0.9f, -10.0f};
    isolatedG.scale = {0.5f, 0.5f, 0.5f};
    isolatedG.opacity = 0.65f;
    isolatedG.confidence = 0.45f;
    isolatedG.semLogits[static_cast<size_t>(SemanticClass::Static)] = 1.0f;

    Mat4 camFromWorld{};
    // Camera coordinates: +X_cam = -Z_world, +Y_cam = -Y_world, +Z_cam = +X_world.
    camFromWorld.at(0, 2) = -1.0f;
    camFromWorld.at(1, 1) = -1.0f;
    camFromWorld.at(2, 0) = 1.0f;
    camFromWorld.at(3, 3) = 1.0f;

    DeformableCamView frontCam{};
    frontCam.camIdx = 0;
    frontCam.camFromWorld = camFromWorld;
    frontCam.intr = {240.0f, 240.0f, 160.0f, 120.0f};
    frontCam.width = 320.0f;
    frontCam.height = 240.0f;
    frontCam.tokenWeight = 1.0f;

    Detection2D det{};
    det.camIndex = 0;
    det.cxNorm = 0.50f;
    det.cyNorm = 0.47f;
    det.wNorm = 0.22f;
    det.hNorm = 0.22f;
    det.score = 0.95f;
    det.semantic = SemanticClass::Vehicle;
    det.camPatchWeight = 1.0f;

    ActiveVoxel vox{};
    vox.wx = 15.0f;
    vox.wy = 0.85f;
    vox.wz = 0.0f;
    vox.sem = SemanticClass::Vehicle;
    vox.confidence = 0.92f;

    Track3D tr{};
    tr.state = TrackState::Tracked;
    tr.semantic = SemanticClass::Vehicle;
    tr.worldPos = {15.0f, 0.8f, 0.0f};
    tr.halfExtents = {1.0f, 0.8f, 2.2f};
    tr.yaw = 0.20f;
    tr.score = 0.95f;

    std::vector<Gaussian3DPrimitive> prims{targetG, isolatedG};
    DeformableAttention3D attn;
    uint32_t corrobCnt = 0u;
    float meanResidual = 0.0f;
    attn.refineWithViews(gfCfg, {frontCam}, {det}, {vox}, {tr}, prims, corrobCnt, meanResidual);

    EXPECT_EQ(corrobCnt, 1u);
    EXPECT_GT(meanResidual, 0.05f);
    EXPECT_GE(prims[0].camCorrobCount, 1u);
    EXPECT_GT(prims[0].mean.x, targetG.mean.x);
    EXPECT_GT(prims[0].opacity, targetG.opacity);
    EXPECT_GT(prims[0].semanticConfidence(), confBefore);
    EXPECT_EQ(prims[0].argmaxSemantic(), SemanticClass::Vehicle);
    EXPECT_LT(prims[1].opacity, isolatedG.opacity);
}

TEST(GaussianFormer3DUnitTest, PipelineStageSplatTraceHudAndZeroRealloc) {
    const SimConfig cfg = SimConfig::defaultConfig();
    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = 0.0f;
    ego.gear = 1;

    GaussianFormer3DPipeline pipeline;
    pipeline.init(cfg);

    PerceptionFrameOutput out{};
    out.activeVoxels.reserve(static_cast<size_t>(cfg.occTotalVoxels()));
    out.reserveV15Scratch(1024u, 320u * 180u * 4u);

    auto seedInputs = [&]() {
        out.tracks.resize(1u);
        out.tracks[0].trackId = 1;
        out.tracks[0].semantic = SemanticClass::Vehicle;
        out.tracks[0].state = TrackState::Tracked;
        out.tracks[0].worldPos = {10.0f, 0.7f, 0.0f};
        out.tracks[0].halfExtents = {0.95f, 0.72f, 2.1f};
        out.tracks[0].score = 0.9f;

        out.activeVoxels.assign(2u, ActiveVoxel{});
        out.activeVoxels[0] = {10.0f, 0.0f, 0.75f, 10.0f, 0.0f,
                               SemanticClass::Vehicle, 0.85f, 5.0f, 0.0f, 0.25f};
        out.activeVoxels[1] = {16.0f, -5.0f, 1.25f, 16.0f, -5.0f,
                               SemanticClass::SignalSign, 0.82f, 0.0f, 0.0f, 0.18f};
    };

    for (int warm = 0; warm < 2; ++warm) {
        seedInputs();
        pipeline.runStage(cfg, ego, nullptr, nullptr, out);
    }

    EXPECT_TRUE(out.gaussianFormer3D.active);
    EXPECT_TRUE(out.gaussianFormer3D.usedDeformableAttn);
    EXPECT_GE(out.gaussianFormer3D.gaussianCount, 2u);
    EXPECT_GE(out.gaussianFormer3D.splattedVoxelCount, 2u);
    EXPECT_EQ(pipeline.v15Gaussians().size(), out.gaussianFormer3D.gaussianCount);
    EXPECT_FALSE(out.activeVoxels.empty());

    out.v15.active = true;
    out.v15Gaussians = pipeline.v15Gaussians();
    const PlannerCommand cmd{};
    const auto hud = v15_hud::formatV15HudTelemetry(ego, out, cmd);
    EXPECT_NE(std::string(hud.header.data()).find("GF3D:"), std::string::npos);

    const std::string jsonStr = TraceLogger::formatFrameJson(
        4, 0.1f, cfg, ego, cmd, out, ContactEvent{});
    Json::CharReaderBuilder rb;
    Json::Value parsed;
    std::string errs;
    std::istringstream iss(jsonStr);
    ASSERT_TRUE(Json::parseFromStream(rb, iss, &parsed, &errs)) << errs;
    ASSERT_TRUE(parsed.isMember("gaussianformer3d"));
    EXPECT_TRUE(parsed["gaussianformer3d"]["active"].asBool());
    EXPECT_EQ(parsed["gaussianformer3d"]["gaussian_count"].asUInt(),
              out.gaussianFormer3D.gaussianCount);

    const size_t capPrims = pipeline.primitives().capacity();
    const size_t capV15 = pipeline.v15Gaussians().capacity();
    const size_t capOutVox = out.activeVoxels.capacity();
    int growthEvents = 0;
    for (int iter = 0; iter < 40; ++iter) {
        seedInputs();
        pipeline.runStage(cfg, ego, nullptr, nullptr, out);
        if (pipeline.primitives().capacity() != capPrims) ++growthEvents;
        if (pipeline.v15Gaussians().capacity() != capV15) ++growthEvents;
        if (out.activeVoxels.capacity() != capOutVox) ++growthEvents;
    }
    EXPECT_EQ(growthEvents, kZeroAllocsTol);
}

} // namespace
} // namespace cybercab::sota
