#include "config/SimConfig.hpp"
#include "core/Math.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "TestPerceptionHelpers.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

using namespace cybercab;
using namespace cybercab::testutil;

TEST(RTDETRExtendedTest, EmptyCleanRoadSkyStripeCurbProducesZeroGhostDetections) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch, true, true);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    std::ostringstream ghostLog;
    for (size_t i = 0; i < out2D.size(); ++i) {
        const auto& d = out2D[i];
        ghostLog << "\n  [" << i << "] cam=" << d.camIndex
                 << " sem=" << static_cast<int>(d.semantic)
                 << " score=" << d.score
                 << " box=(" << d.cxNorm << "," << d.cyNorm << "," << d.wNorm << "," << d.hNorm << ")";
    }
    EXPECT_TRUE(out2D.empty())
        << "Empty road/sky/lane-stripe/amber-curb generated " << out2D.size()
        << " ghost 2D detections:" << ghostLog.str();
    EXPECT_TRUE(out3D.empty())
        << "Empty road/sky/lane-stripe/amber-curb generated " << out3D.size() << " ghost 3D detections";
}

TEST(RTDETRExtendedTest, CleanRoadAndSkyOnlyProducesZeroDetections) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    fillCleanRoadSkyTensor(h.rig.camWidth(), h.rig.camHeight(), h.synthPixelsScratch, true, false);
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    EXPECT_TRUE(out2D.empty()) << "Road+sky+white-dashes leaked " << out2D.size() << " detections";
    EXPECT_TRUE(out3D.empty());
}

TEST(RTDETRExtendedTest, BlackCoveredLensReportedAsLowConfidenceHighUncertainty) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    std::fill(h.synthPixelsScratch.begin(), h.synthPixelsScratch.end(), Vec4{0.0f, 0.0f, 0.0f, 1.0f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    EXPECT_LT(h.soilCamsScratch[0].tokenWeight, 0.65f)
        << "Pitch-black covered lens misclassified as healthy asphalt (tokenWeight="
        << h.soilCamsScratch[0].tokenWeight << ")";
    EXPECT_GT(h.soilCamsScratch[0].meanEpistemicVar, 0.25f)
        << "Pitch-black covered lens must report high epistemic variance (actual="
        << h.soilCamsScratch[0].meanEpistemicVar << ")";
}

TEST(RTDETRExtendedTest, BuildingWindowStripsNearHorizonDoNotTriggerGhostDetections) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);

    // Replicate WorldEngine building facade + lower illuminated cyan/gold window strip spanning v in [0.36, 0.49].
    drawCamRect(h.synthPixelsScratch, 0, w, hPx, 10, (hPx * 20) / 100, 120, (hPx * 50) / 100, {0.16f, 0.20f, 0.26f});
    drawCamRect(h.synthPixelsScratch, 0, w, hPx, 16, (hPx * 38) / 100, 114, (hPx * 47) / 100, {0.22f, 0.74f, 0.97f});
    drawCamRect(h.synthPixelsScratch, 3, w, hPx, 180, (hPx * 37) / 100, 290, (hPx * 48) / 100, {0.99f, 0.90f, 0.54f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    int dynFalsePos = 0;
    for (const auto& d : out2D) {
        if (d.semantic == SemanticClass::Vehicle || d.semantic == SemanticClass::Pedestrian) {
            ++dynFalsePos;
        }
    }
    EXPECT_EQ(dynFalsePos, 0)
        << "Building window strips near horizon leaked " << dynFalsePos << " vehicle/pedestrian ghost detections";
}

TEST(RTDETRExtendedTest, SingleSyntheticVehicleYieldsOneTightDetectionNotSplitBoxes) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);

    // Single painted red sedan in CAM_FRONT_MAIN centered at cx=0.50, cy=0.58.
    drawCamRect(h.synthPixelsScratch, 0, w, hPx, 134, 118, 186, 160, {0.88f, 0.11f, 0.28f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    int cam0Count = 0;
    int otherCamCount = 0;
    for (const auto& d : out2D) {
        if (d.camIndex == 0) ++cam0Count;
        else ++otherCamCount;
    }
    EXPECT_EQ(cam0Count, 1) << "Single vehicle produced " << cam0Count << " boxes on cam 0";
    EXPECT_EQ(otherCamCount, 0) << "Empty cameras 1..7 produced " << otherCamCount << " ghost boxes";
    if (cam0Count > 0) {
        EXPECT_NEAR(out2D[0].cxNorm, 0.50f, 0.06f);
        EXPECT_NEAR(out2D[0].cyNorm, 0.58f, 0.08f);
        EXPECT_EQ(out2D[0].semantic, SemanticClass::Vehicle);
        EXPECT_GE(out2D[0].score, cfg.rtdetrHighConfThresh);
    }
}

TEST(RTDETRExtendedTest, DiffuseLowSaliencyWashDoesNotOutscoreCompactSharpTarget) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);

    // Cam 0: large diffuse slightly warm/tinted area (not a distinct object, border gradient only).
    drawCamRect(h.synthPixelsScratch, 0, w, hPx, 40, 120, 200, 190, {0.36f, 0.25f, 0.22f});
    // Cam 1: crisp compact pedestrian target.
    drawCamRect(h.synthPixelsScratch, 1, w, hPx, 150, 122, 166, 168, {0.98f, 0.45f, 0.09f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    float maxGhostScore = 0.0f;
    float pedScore = 0.0f;
    for (const auto& d : out2D) {
        if (d.camIndex == 0) maxGhostScore = std::max(maxGhostScore, d.score);
        if (d.camIndex == 1 && d.semantic == SemanticClass::Pedestrian) pedScore = std::max(pedScore, d.score);
    }
    EXPECT_GT(pedScore, cfg.rtdetrHighConfThresh);
    EXPECT_LT(maxGhostScore, cfg.rtdetrHighConfThresh)
        << "Wide diffuse low-saliency region falsely triggered high-conf ghost score=" << maxGhostScore;
}

TEST(RTDETRExtendedTest, DecoderBarrierAndSharedMemoryIsolationAcrossRuns) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;

    for (int iter = 0; iter < 4; ++iter) {
        // Dense multi-object frame across all 8 cameras populates shared decoder slots.
        fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);
        for (int c = 0; c < 8; ++c) {
            drawCamRect(h.synthPixelsScratch, c, w, hPx, 30, 125, 65, 165, {0.88f, 0.11f, 0.28f});
            drawCamRect(h.synthPixelsScratch, c, w, hPx, 115, 120, 145, 168, {0.06f, 0.72f, 0.51f});
            drawCamRect(h.synthPixelsScratch, c, w, hPx, 210, 126, 260, 164, {0.06f, 0.65f, 0.91f});
        }
        h.uploadSyntheticPixels(h.synthPixelsScratch);
        h.runDetr(h.world->ego(), out2D, out3D);
        ASSERT_FALSE(out2D.empty());

        // Immediately follow with totally empty clean tensor: zero stale slots/NaNs allowed.
        fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);
        h.uploadSyntheticPixels(h.synthPixelsScratch);
        h.runDetr(h.world->ego(), out2D, out3D);
        EXPECT_TRUE(out2D.empty()) << "Stale decoder shared memory leaked " << out2D.size() << " detections on iter " << iter;
        for (const auto& d : out2D) {
            EXPECT_FALSE(std::isnan(d.score) || std::isnan(d.cxNorm) || std::isnan(d.cyNorm));
        }
    }
}

TEST(RTDETRExtendedTest, SoiledPatchReducesScoreIntoStage2TierAndInflatesBoxSigma) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    std::vector<Detection2D> clean2D, soiled2D;
    std::vector<Detection3D> clean3D, soiled3D;

    // Clean bright pedestrian in center of CAM_FRONT_MAIN.
    fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);
    drawCamRect(h.synthPixelsScratch, 0, w, hPx, 142, 118, 178, 174, {0.98f, 0.45f, 0.09f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);
    h.runDetr(h.world->ego(), clean2D, clean3D);
    ASSERT_FALSE(clean2D.empty());

    // Soil only the region covering that pedestrian patch cleanly with mud occlusion.
    drawMudGlarePatch(h.synthPixelsScratch, 0, w, hPx, 120, 100, 200, 190);
    // Keep a small visible corner so detector still emits a low-weight candidate if partial.
    drawCamRect(h.synthPixelsScratch, 0, w, hPx, 152, 125, 168, 165, {0.98f, 0.45f, 0.09f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);
    h.runDetr(h.world->ego(), soiled2D, soiled3D);

    if (!soiled2D.empty()) {
        std::ostringstream diag;
        diag << "cleanN=" << clean2D.size() << " soiledN=" << soiled2D.size()
             << " camW=" << h.soilCamsScratch[0].tokenWeight << " meanP=" << h.soilCamsScratch[0].meanSoilProb;
        for (size_t i = 0; i < soiled2D.size(); ++i) {
            diag << " [d" << i << " cam=" << soiled2D[i].camIndex << " sem=" << static_cast<int>(soiled2D[i].semantic)
                 << " c=(" << soiled2D[i].cxNorm << "," << soiled2D[i].cyNorm << ")"
                 << " w=" << soiled2D[i].camPatchWeight << " sc=" << soiled2D[i].score << "]";
        }
        diag << " patch(4,4)=" << h.soilPatchesScratch[4 * cfg.soilPatchCols + 4].patchConfidence
             << "(p=" << h.soilPatchesScratch[4 * cfg.soilPatchCols + 4].soilProb
             << ",e=" << h.soilPatchesScratch[4 * cfg.soilPatchCols + 4].meanEdge << ")";
        EXPECT_LT(soiled2D[0].camPatchWeight, 0.70f) << diag.str();
        EXPECT_LT(soiled2D[0].score, clean2D[0].score) << diag.str();
        EXPECT_GT(soiled2D[0].boxSigma, clean2D[0].boxSigma) << diag.str();
    } else {
        EXPECT_LT(h.soilCamsScratch[0].tokenWeight, 0.90f);
    }
}

TEST(RTDETRExtendedTest, WideAngleFisheyeProjectWorldToPixelMatchesDistortedImage) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);

    const auto& specWide = cfg.cameras.at(2); // CAM_FRONT_WIDE (k1 = -0.245)
    const auto& cm = h.rig.cameraMatrices(2);
    const EgoState& ego = h.world->ego();

    // Point 10m forward and 5.5m left (off-axis inside wide FOV where distortion shifts pixels strongly).
    const float sinY = std::sin(ego.yaw), cosY = std::cos(ego.yaw);
    const Vec3 ptW{ego.pos.x + 10.0f * sinY + 5.5f * cosY, 1.0f, ego.pos.z + 10.0f * cosY - 5.5f * sinY};

    Vec2 projPx{};
    float depth = 0.0f;
    ASSERT_TRUE(h.rig.projectWorldToPixel(2, ptW, projPx, depth));

    // Exact forward Brown-Conrady distorted pixel location matching cam_isp_soil.comp.
    const Vec3 pc = cm.camFromWorld.transformPoint(ptW);
    const Vec2 distXY = distortNormalized(pc.x / pc.z, pc.y / pc.z, specWide.distortion);
    const Vec2 trueDistPx{cm.intr.fx * distXY.x + cm.intr.cx, cm.intr.fy * distXY.y + cm.intr.cy};

    const float pxErr = (projPx - trueDistPx).norm();
    EXPECT_LT(pxErr, 3.0f)
        << "projectWorldToPixel() omitted Brown-Conrady lens distortion on CAM_FRONT_WIDE (error=" << pxErr << " px)";
}

TEST(RTDETRExtendedTest, FullWorldNominalSingleFrameGhostAndDuplicateAudit) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindRealRigTensor();

    h.rig.renderAllCameras(*h.world, cfg, 0.0f, nullptr);
    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    const auto gtAll = h.world->allGroundTruthOBBs();
    const auto gtDyn = h.world->dynamicGroundTruthOBBs();

    std::vector<int> perGtMatchCount(gtDyn.size(), 0);
    int orphanGhostCount = 0;
    int duplicateDetCount = 0;

    std::ostringstream orphanLog;
    for (const auto& d : out3D) {
        if (d.semantic == SemanticClass::SignalSign || d.rangeM > 45.0f) continue;

        // Check nearest dynamic GT.
        float bestDynDist = std::numeric_limits<float>::infinity();
        int bestDynIdx = -1;
        for (size_t g = 0; g < gtDyn.size(); ++g) {
            const float dist = std::hypot(d.worldPos.x - gtDyn[g].cx, d.worldPos.z - gtDyn[g].cz);
            if (dist < bestDynDist) {
                bestDynDist = dist;
                bestDynIdx = static_cast<int>(g);
            }
        }

        // Check nearest ANY GT (dynamic or static) to classify true orphan ghost.
        float bestAnyDist = bestDynDist;
        for (const auto& g : gtAll) {
            const float dx = std::max(0.0f, std::abs(d.worldPos.x - g.cx) - g.hx);
            const float dz = std::max(0.0f, std::abs(d.worldPos.z - g.cz) - g.hz);
            bestAnyDist = std::min(bestAnyDist, std::hypot(dx, dz));
        }

        if (bestDynIdx >= 0 && bestDynDist < 5.5f) {
            perGtMatchCount[bestDynIdx]++;
            if (perGtMatchCount[bestDynIdx] > 1) {
                ++duplicateDetCount;
                orphanLog << "\n  [DUP] cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
                          << " score=" << d.score << " gt=" << gtDyn[bestDynIdx].id
                          << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm << "," << d.det2d.wNorm << "," << d.det2d.hNorm << ")"
                          << " w=(" << d.worldPos.x << "," << d.worldPos.z << ")";
            }
        } else if (bestAnyDist > 4.0f) {
            ++orphanGhostCount;
            orphanLog << "\n  [GHOST] cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
                      << " score=" << d.score << " range=" << d.rangeM
                      << " box=(" << d.det2d.cxNorm << "," << d.det2d.cyNorm << "," << d.det2d.wNorm << "," << d.det2d.hNorm << ")"
                      << " w=(" << d.worldPos.x << "," << d.worldPos.z << ") minGtDist=" << bestAnyDist;
        }
    }

    RecordProperty("total_3d_detections", static_cast<int>(out3D.size()));
    RecordProperty("cross_camera_duplicate_3d", duplicateDetCount);
    RecordProperty("orphan_ghost_3d", orphanGhostCount);

    EXPECT_EQ(orphanGhostCount, 0)
        << "Nominal frame produced " << orphanGhostCount << " orphan 3D ghost detections (>4m from any GT):" << orphanLog.str();
    EXPECT_LE(duplicateDetCount, 1)
        << "Missing 3D cross-camera NMS produced " << duplicateDetCount << " duplicate 3D detections:" << orphanLog.str();
}

TEST(RTDETRExtendedTest, TrafficSignalDistinguishedFromAmberRoadCurb) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch, false, true);
    // Bright emissive green signal light in upper quadrant of CAM_FRONT_NARROW.
    drawCamRect(h.synthPixelsScratch, 1, w, hPx, 150, 50, 174, 80, {0.08f, 0.90f, 0.52f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    bool foundSignal = false;
    int nonSignalGhosts = 0;
    for (const auto& d : out2D) {
        if (d.camIndex == 1 && d.semantic == SemanticClass::SignalSign && d.cyNorm < 0.45f) {
            foundSignal = true;
        } else {
            ++nonSignalGhosts;
        }
    }
    EXPECT_TRUE(foundSignal);
    EXPECT_EQ(nonSignalGhosts, 0);
}

TEST(RTDETRExtendedTest, GroundRay3DLiftingMatchesFrontMainProjectionGeometry) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const EgoState& ego = h.world->ego();
    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();

    // Project a virtual 1.5m-tall sedan 20m directly ahead of ego into CAM_FRONT_MAIN pixels.
    const float sinY = std::sin(ego.yaw), cosY = std::cos(ego.yaw);
    const Vec3 botW{ego.pos.x + 20.0f * sinY, 0.0f, ego.pos.z + 20.0f * cosY};
    const Vec3 topW{ego.pos.x + 20.0f * sinY, 1.50f, ego.pos.z + 20.0f * cosY};
    Vec2 pxBot{}, pxTop{};
    float dBot = 0.0f, dTop = 0.0f;
    ASSERT_TRUE(h.rig.projectWorldToPixel(0, botW, pxBot, dBot));
    ASSERT_TRUE(h.rig.projectWorldToPixel(0, topW, pxTop, dTop));

    const float boxH = std::max(10.0f, pxBot.y - pxTop.y);
    const float boxW = boxH * 1.35f;
    fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);
    drawCamRect(h.synthPixelsScratch, 0, w, hPx,
                static_cast<int>(pxBot.x - boxW * 0.5f), static_cast<int>(pxTop.y),
                static_cast<int>(pxBot.x + boxW * 0.5f), static_cast<int>(pxBot.y),
                {0.06f, 0.65f, 0.91f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(ego, out2D, out3D);

    ASSERT_FALSE(out3D.empty());
    EXPECT_NEAR(out3D[0].egoPos.x, 20.0f, 5.0f);
    EXPECT_NEAR(out3D[0].egoPos.y, 0.0f, 2.0f);
}

TEST(RTDETRExtendedTest, JaywalkerPedestrianDetectedInFrontCamerasAtCutIn) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindRealRigTensor();

    h.world->reset();
    h.world->triggerJaywalkerCutIn();
    h.rig.renderAllCameras(*h.world, cfg, 0.0f, nullptr);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    // Find PED_08 in world GT
    const auto gtDyn = h.world->dynamicGroundTruthOBBs();
    Vec3 pedW{};
    for (const auto& g : gtDyn) {
        if (g.id == "PED_08") pedW = {g.cx, g.cy, g.cz};
    }

    Vec2 pxMain{}, pxWide{};
    float dMain = 0.0f, dWide = 0.0f;
    h.rig.projectWorldToPixel(0, pedW, pxMain, dMain);
    h.rig.projectWorldToPixel(2, pedW, pxWide, dWide);

    std::ostringstream detLog;
    detLog << "\nPED_08 world=(" << pedW.x << "," << pedW.z << ")"
           << " mainPx=(" << pxMain.x << "," << pxMain.y << ",d=" << dMain << ")"
           << " widePx=(" << pxWide.x << "," << pxWide.y << ",d=" << dWide << ")";
    float bestDist = 1e9f;
    for (const auto& d : out3D) {
        const float dist = std::hypot(d.worldPos.x - pedW.x, d.worldPos.z - pedW.z);
        bestDist = std::min(bestDist, dist);
        detLog << "\n  det3D cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
               << " score=" << d.score << " range=" << d.rangeM
               << " w=(" << d.worldPos.x << "," << d.worldPos.z << ") distToPed=" << dist;
    }
    for (const auto& d : out2D) {
        if (d.camIndex == 0 || d.camIndex == 2) {
            detLog << "\n  det2D cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
                   << " score=" << d.score << " box=(" << d.cxNorm << "," << d.cyNorm << "," << d.wNorm << "," << d.hNorm << ")";
        }
    }

    const auto img0 = h.rig.downloadCameraRGBA(0);
    const auto img2 = h.rig.downloadCameraRGBA(2);
    std::vector<Vec4> s4(static_cast<size_t>(8 * cfg.featWidth * cfg.featHeight * 4));
    h.detr.featS4SSBO().downloadVector(s4);
    const int w = h.rig.camWidth();
    const int p0x = clampf(pxMain.x, 0.0f, static_cast<float>(w - 1));
    const int p0y = clampf(pxMain.y, 0.0f, static_cast<float>(h.rig.camHeight() - 1));
    const int p2x = clampf(pxWide.x, 0.0f, static_cast<float>(w - 1));
    const int p2y = clampf(pxWide.y, 0.0f, static_cast<float>(h.rig.camHeight() - 1));
    const size_t i0 = (static_cast<size_t>(p0y) * w + p0x) * 4;
    const size_t i2 = (static_cast<size_t>(p2y) * w + p2x) * 4;
    const int fx0 = (p0x * cfg.featWidth) / w;
    const int fy0 = (p0y * cfg.featHeight) / h.rig.camHeight();
    const size_t fBase = static_cast<size_t>(fy0 * cfg.featWidth + fx0) * 4;

    detLog << "\n  cam0@ped RGB=(" << int(img0[i0]) << "," << int(img0[i0+1]) << "," << int(img0[i0+2]) << ")"
           << " S4 c0=(" << s4[fBase].x << "," << s4[fBase].y << "," << s4[fBase].z << ")"
           << " c1=(" << s4[fBase+1].x << "," << s4[fBase+1].y << "," << s4[fBase+1].z << "," << s4[fBase+1].w << ")"
           << " c2=(" << s4[fBase+2].x << "," << s4[fBase+2].y << "," << s4[fBase+2].z << "," << s4[fBase+2].w << ")"
           << " c3.obj=" << s4[fBase+3].w
           << "\n  cam2@ped RGB=(" << int(img2[i2]) << "," << int(img2[i2+1]) << "," << int(img2[i2+2]) << ")";

    EXPECT_LT(bestDist, 3.5f) << "Jaywalker PED_08 not detected within 3.5m:" << detLog.str();
}

TEST(RTDETRExtendedTest, SimultaneousVehiclePedestrianAndSignalDetectedWithoutSwallowing) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch, true, true);
    // Vehicle (wide), Pedestrian (tall), and overhead SignalSign (green) in CAM_FRONT_MAIN
    drawCamRect(h.synthPixelsScratch, 0, w, hPx, 190, 122, 246, 162, {0.88f, 0.11f, 0.28f});
    drawCamRect(h.synthPixelsScratch, 0, w, hPx, 78, 122, 98, 168, {0.98f, 0.45f, 0.09f});
    drawCamRect(h.synthPixelsScratch, 0, w, hPx, 148, 52, 174, 80, {0.08f, 0.90f, 0.52f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    int nVeh = 0, nPed = 0, nSig = 0;
    std::ostringstream log;
    for (const auto& d : out2D) {
        log << "\n  [d2] cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
            << " c=(" << d.cxNorm << "," << d.cyNorm << ") wh=(" << d.wNorm << "," << d.hNorm << ") sc=" << d.score;
        if (d.camIndex != 0) continue;
        if (d.semantic == SemanticClass::Vehicle) ++nVeh;
        else if (d.semantic == SemanticClass::Pedestrian) ++nPed;
        else if (d.semantic == SemanticClass::SignalSign) ++nSig;
    }
    EXPECT_EQ(nVeh, 1) << log.str();
    EXPECT_EQ(nPed, 1) << log.str();
    EXPECT_EQ(nSig, 1) << log.str();
}

TEST(RTDETRExtendedTest, CrimsonPedestrianNotBoostedAsVehicleTaillight) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);
    // Shaded crimson-pink torso of PED_08 (~18x18 px on CAM_PILLAR_RIGHT) must not trigger vehicle taillight bias
    drawCamRect(h.synthPixelsScratch, 4, w, hPx, 150, 120, 168, 138, {0.358f, 0.092f, 0.138f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    ASSERT_FALSE(out2D.empty());
    EXPECT_EQ(out2D[0].camIndex, 4);
    EXPECT_EQ(out2D[0].semantic, SemanticClass::Pedestrian)
        << "Crimson/magenta shirt was misclassified as Vehicle taillight (w=" << out2D[0].wNorm << ", h=" << out2D[0].hNorm << ")";
}

TEST(RTDETRExtendedTest, BorderTruncatedAndOutOfRangeHorizonDetectionsRejected) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    IsolatedGpuPerceptionHarness h;
    h.init(cfg);
    h.bindSyntheticTensor();

    const int w = h.rig.camWidth();
    const int hPx = h.rig.camHeight();
    fillCleanRoadSkyTensor(w, hPx, h.synthPixelsScratch);
    // 1. Half-cropped vehicle clipped at extreme left edge of CAM_FRONT_NARROW (cx < 0.045)
    drawCamRect(h.synthPixelsScratch, 1, w, hPx, 0, 114, 14, 134, {0.06f, 0.65f, 0.91f});
    // 2. Distant horizon building window beyond CAM_FRONT_WIDE 36m maxRangeM (y=102..115, tGroundRaw > 50m)
    drawCamRect(h.synthPixelsScratch, 2, w, hPx, 75, 102, 125, 115, {0.22f, 0.74f, 0.97f});
    h.uploadSyntheticPixels(h.synthPixelsScratch);

    std::vector<Detection2D> out2D;
    std::vector<Detection3D> out3D;
    h.runDetr(h.world->ego(), out2D, out3D);

    int leaked = 0;
    std::ostringstream log;
    for (const auto& d : out3D) {
        if (d.semantic == SemanticClass::Vehicle || d.semantic == SemanticClass::Pedestrian) {
            ++leaked;
            log << "\n  cam=" << d.camIndex << " sem=" << static_cast<int>(d.semantic)
                << " cx=" << d.det2d.cxNorm << " cy=" << d.det2d.cyNorm << " r=" << d.rangeM;
        }
    }
    EXPECT_EQ(leaked, 0) << "Truncated/out-of-range artifacts leaked into 3D detections:" << log.str();
}
