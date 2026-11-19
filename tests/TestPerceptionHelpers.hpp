#pragma once

#include "config/SimConfig.hpp"
#include "core/Math.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/BEVFormer.hpp"
#include "perception/BackboneRTDETR.hpp"
#include "perception/OccupancyFlowNet.hpp"
#include "perception/SoilDetector.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "world/WorldEngine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace cybercab::testutil {

inline Detection3D makeDet3D(SemanticClass sem,
                             const Vec3& worldPos,
                             float score,
                             float camWeight = 1.0f,
                             float sigma = 0.25f,
                             int camIdx = 0) {
    Detection3D d{};
    d.camIndex = camIdx;
    d.semantic = sem;
    d.score = score;
    d.worldPos = worldPos;
    d.camWeight = camWeight;
    d.epistemicSigma = sigma;
    d.halfExtents = (sem == SemanticClass::Pedestrian)
        ? Vec3{0.35f, 0.90f, 0.35f}
        : Vec3{0.96f, 0.72f, 2.18f};
    return d;
}

inline TrackPriorGpu makeTrackPrior(const EgoState& ego,
                                    const Vec3& worldPos,
                                    const Vec3& worldVel,
                                    float yaw,
                                    float hx,
                                    float hz,
                                    SemanticClass sem = SemanticClass::Vehicle,
                                    float conf = 0.92f,
                                    int trackId = 1,
                                    float sigma = 0.35f) {
    const float dx = worldPos.x - ego.pos.x;
    const float dz = worldPos.z - ego.pos.z;
    const float s = std::sin(ego.yaw);
    const float c = std::cos(ego.yaw);
    TrackPriorGpu tp{};
    tp.posVel = {dx * s + dz * c, dx * c - dz * s, worldVel.x, worldVel.z};
    tp.sizeMeta = {hx, hz, static_cast<float>(static_cast<int>(sem)), sigma};
    tp.yawValid = {yaw, conf, static_cast<float>(trackId), 1.0f};
    return tp;
}

inline void fillCleanRoadSkyTensor(int w, int h, std::vector<Vec4>& pixels, bool addLaneDashes = false, bool addAmberCurb = false) {
    const size_t perCam = static_cast<size_t>(w) * h;
    pixels.resize(8u * perCam);
    const int horizonY = (h * 46) / 100;
    const Vec4 sky{0.545f, 0.722f, 0.910f, 1.0f};
    const Vec4 asphalt{0.149f, 0.169f, 0.204f, 1.0f};
    const Vec4 whiteDash{0.945f, 0.961f, 0.976f, 1.0f};
    const Vec4 amberCurb{0.961f, 0.620f, 0.043f, 1.0f};

    for (int c = 0; c < 8; ++c) {
        for (int y = 0; y < h; ++y) {
            const bool isGround = (y >= horizonY);
            for (int x = 0; x < w; ++x) {
                Vec4 col = isGround ? asphalt : sky;
                if (addLaneDashes && y >= horizonY + 8 && ((y / 14) % 2 == 0) && std::abs(x - w / 2) <= 2) {
                    col = whiteDash;
                }
                if (addAmberCurb && y >= horizonY + 12 && y <= horizonY + 20 && x > w / 5 && x < (w * 2) / 5) {
                    col = amberCurb;
                }
                pixels[static_cast<size_t>(c) * perCam + static_cast<size_t>(y) * w + x] = col;
            }
        }
    }
}

inline void drawCamRect(std::vector<Vec4>& pixels, int camIdx, int w, int h,
                        int x0, int y0, int x1, int y1, const Vec3& rgb) {
    const size_t perCam = static_cast<size_t>(w) * h;
    const int cx0 = std::max(0, x0);
    const int cy0 = std::max(0, y0);
    const int cx1 = std::min(w - 1, x1);
    const int cy1 = std::min(h - 1, y1);
    for (int y = cy0; y <= cy1; ++y) {
        for (int x = cx0; x <= cx1; ++x) {
            pixels[static_cast<size_t>(camIdx) * perCam + static_cast<size_t>(y) * w + x] =
                Vec4{rgb.x, rgb.y, rgb.z, 1.0f};
        }
    }
}

inline void drawMudGlarePatch(std::vector<Vec4>& pixels, int camIdx, int w, int h,
                              int x0, int y0, int x1, int y1) {
    const size_t perCam = static_cast<size_t>(w) * h;
    const int cx0 = std::max(0, x0);
    const int cy0 = std::max(0, y0);
    const int cx1 = std::min(w - 1, x1);
    const int cy1 = std::min(h - 1, y1);
    for (int y = cy0; y <= cy1; ++y) {
        for (int x = cx0; x <= cx1; ++x) {
            const float noise = static_cast<float>((x * 7 + y * 13) & 15) / 255.0f;
            pixels[static_cast<size_t>(camIdx) * perCam + static_cast<size_t>(y) * w + x] =
                Vec4{88.0f / 255.0f + noise, 66.0f / 255.0f + noise * 0.8f, 44.0f / 255.0f, 1.0f};
        }
    }
}

class IsolatedGpuPerceptionHarness {
public:
    SimConfig cfg;
    VulkanContext ctx;
    std::unique_ptr<WorldEngine> world;
    CameraRigRenderer rig;
    VulkanBuffer synthCamTensorSSBO;
    SoilDetector soil;
    BackboneRTDETR detr;
    BEVFormer bev;
    OccupancyFlowNet occ;

    std::vector<Vec4> synthPixelsScratch;
    std::array<SoilCameraSummary, 8> soilCamsScratch{};
    std::vector<SoilPatchStats> soilPatchesScratch;
    float fwdSigmaScratch = 0.0f;
    std::vector<TrackPriorGpu> tracksScratch;

    void init(const SimConfig& customCfg) {
        cfg = customCfg;
        ctx.init(false, nullptr);
        world = std::make_unique<WorldEngine>(cfg);
        rig.init(ctx, cfg, *world);
        rig.renderAllCameras(*world, cfg, 0.0f, nullptr);

        const VkDeviceSize bytes = static_cast<VkDeviceSize>(8 * rig.camWidth() * rig.camHeight() * 4) * sizeof(float);
        synthCamTensorSSBO.create(ctx, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        fillCleanRoadSkyTensor(rig.camWidth(), rig.camHeight(), synthPixelsScratch);
        synthCamTensorSSBO.uploadVector(synthPixelsScratch);

        soil.init(ctx, cfg, synthCamTensorSSBO);
        detr.init(ctx, cfg, synthCamTensorSSBO, soil.patchSSBO());
        bev.init(ctx, cfg, detr.encMemorySSBO(), soil.patchSSBO());
        occ.init(ctx, cfg, bev.currBevFeatSSBO(), bev.currBevMetaSSBO(), bev.pillarHeightSSBO());
        tracksScratch.resize(static_cast<size_t>(cfg.btMaxTracks));
    }

    ~IsolatedGpuPerceptionHarness() {
        destroy();
    }

    void bindSyntheticTensor() {
        soil.updateCameraTensorBinding(synthCamTensorSSBO);
        detr.updateCameraTensorBinding(synthCamTensorSSBO, rig.camWidth(), rig.camHeight());
    }

    void bindRealRigTensor() {
        soil.updateCameraTensorBinding(rig.cameraTensorSSBO());
        detr.updateCameraTensorBinding(rig.cameraTensorSSBO(), rig.camWidth(), rig.camHeight());
    }

    void uploadSyntheticPixels(const std::vector<Vec4>& pixels) {
        synthCamTensorSSBO.uploadVector(pixels);
    }

    void runDetr(const EgoState& ego, std::vector<Detection2D>& out2D, std::vector<Detection3D>& out3D) {
        detr.updateHorizonFromRig(rig);
        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        soil.recordDispatch(cmd, cfg, rig.camWidth(), rig.camHeight());
        detr.recordBackbone(cmd, rig.camWidth(), rig.camHeight());
        detr.recordRTDETR(cmd);
        detr.copyCurrToPrevTensor(cmd);
        ctx.endSingleTimeCommands(cmd);

        soil.readbackResults(soilCamsScratch, soilPatchesScratch, fwdSigmaScratch);
        detr.decodeAndLift3D(cfg, rig, ego, out2D, out3D);
    }

    void runBevAndOcc(const EgoState& ego, float dt,
                      const std::vector<TrackPriorGpu>& activePriors,
                      bool enableShadows,
                      std::vector<ActiveVoxel>& outOcc,
                      std::vector<ActiveVoxel>& outShadows) {
        std::fill(tracksScratch.begin(), tracksScratch.end(), TrackPriorGpu{});
        const size_t n = std::min(tracksScratch.size(), activePriors.size());
        std::copy_n(activePriors.begin(), n, tracksScratch.begin());

        {
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            soil.recordDispatch(cmd, cfg, rig.camWidth(), rig.camHeight());
            detr.recordBackbone(cmd, rig.camWidth(), rig.camHeight());
            detr.recordRTDETR(cmd);
            detr.copyCurrToPrevTensor(cmd);
            ctx.endSingleTimeCommands(cmd);
        }
        soil.readbackResults(soilCamsScratch, soilPatchesScratch, fwdSigmaScratch);

        {
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            bev.recordDispatch(cmd, cfg, rig, ego, dt, soilCamsScratch);
            occ.recordDispatch(cmd, cfg, ego, dt, tracksScratch, enableShadows);
            bev.copyCurrToPrev(cmd);
            ctx.endSingleTimeCommands(cmd);
        }
        occ.readbackVoxels(cfg, ego, outOcc, outShadows);
    }

    void destroy() {
        occ.destroy();
        bev.destroy();
        detr.destroy();
        soil.destroy();
        synthCamTensorSSBO.destroy();
        rig.destroy();
        world.reset();
        ctx.cleanup();
    }
};

} // namespace cybercab::testutil
