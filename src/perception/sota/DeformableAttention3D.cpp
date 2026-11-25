#include "perception/sota/DeformableAttention3D.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace cybercab::sota {
namespace {

constexpr float kMaxDynLayerShiftM = 0.25f;
constexpr float kMaxStaticLayerShiftM = 0.45f;
constexpr float kGeomSearchRadM = 2.40f;
constexpr float kTrackSearchRadM = 3.20f;
constexpr float kMeanStepGain = 0.38f;
constexpr float kScaleAdaptGain = 0.25f;
constexpr float kYawAdaptGain = 0.28f;
constexpr float kCamLogitBoost = 1.10f;
constexpr float kGeomLogitBoost = 0.75f;
constexpr float kCorrobOpacityStep = 0.08f;
constexpr float kUncorrobOpacityDecay = 0.96f;
constexpr float kMinWeightEps = 1e-5f;

void attendTo3DGeometry(const Gaussian3DPrimitive& g,
                        const SpatialHashIndex3D& voxelHash,
                        const std::vector<ActiveVoxel>& voxels3D,
                        const std::vector<Track3D>& tracks3D,
                        Vec3& outGeomMean,
                        Vec3& outTargetScale,
                        float& outTrackYaw,
                        bool& outHasTrackYaw,
                        std::array<float, kNumGaussianSemClasses>& outSemBoost,
                        float& outGeomConf) {
    outGeomMean = g.mean;
    outTargetScale = g.scale;
    outTrackYaw = g.rot.toYaw();
    outHasTrackYaw = false;
    outSemBoost.fill(0.0f);
    outGeomConf = 0.0f;

    float wSum = 0.0f;
    Vec3 sumPos{};
    Vec3 maxLocal{g.scale.x * (g.isDynamic ? 1.0f : 0.6f),
                  g.scale.y * (g.isDynamic ? 1.0f : 0.6f),
                  g.scale.z * (g.isDynamic ? 1.0f : 0.6f)};
    const float radSq = kGeomSearchRadM * kGeomSearchRadM;
    const SemanticClass gSem = g.argmaxSemantic();
    const float invScaleSq = 1.0f / std::max(0.35f, g.scale.normSq());

    voxelHash.forEachNeighbor(g.mean.x, g.mean.z, kGeomSearchRadM, [&](int vi) {
        const auto& v = voxels3D[static_cast<size_t>(vi)];
        if (v.sem != gSem) return true;
        const Vec3 vp{v.wx, v.wy, v.wz};
        const Vec3 diff = vp - g.mean;
        const float dSq = diff.normSq();
        if (dSq > radSq) return true;

        const float semMatch = (v.sem == gSem) ? 1.25f : 0.45f;
        const float w = std::exp(-0.5f * dSq * invScaleSq) *
                        clampf(v.confidence, 0.1f, 1.0f) * semMatch;
        wSum += w;
        sumPos += vp * w;
        const Vec3 local = g.rot.invRotateVector(diff);
        maxLocal.x = std::max(maxLocal.x, std::abs(local.x));
        maxLocal.y = std::max(maxLocal.y, std::abs(local.y));
        maxLocal.z = std::max(maxLocal.z, std::abs(local.z));
        const uint32_t cls = std::min(static_cast<uint32_t>(v.sem), kNumGaussianSemClasses - 1u);
        outSemBoost[cls] += w;
        return true;
    });

    const float trRadSq = kTrackSearchRadM * kTrackSearchRadM;
    float bestTrackDistSq = trRadSq;
    for (const auto& tr : tracks3D) {
        if (tr.state == TrackState::Removed) continue;
        if (g.isDynamic && tr.semantic != gSem) continue;
        const Vec3 tp{tr.worldPos.x, std::max(0.55f, tr.halfExtents.y), tr.worldPos.z};
        const float dSq = (tp - g.mean).normSq();
        if (dSq > trRadSq) continue;

        const float w = (1.5f / (0.25f + dSq)) * clampf(tr.score, 0.2f, 1.0f);
        wSum += w;
        sumPos += tp * w;
        const uint32_t cls = std::min(static_cast<uint32_t>(tr.semantic), kNumGaussianSemClasses - 1u);
        outSemBoost[cls] += w * 1.4f;
        if (dSq < bestTrackDistSq) {
            bestTrackDistSq = dSq;
            outTrackYaw = tr.yaw;
            outHasTrackYaw = true;
            maxLocal.x = std::max(maxLocal.x, tr.halfExtents.x);
            maxLocal.y = std::max(maxLocal.y, tr.halfExtents.y);
            maxLocal.z = std::max(maxLocal.z, tr.halfExtents.z);
        }
    }

    if (wSum > kMinWeightEps) {
        outGeomMean = sumPos * (1.0f / wSum);
        const float scaleCapMul = g.fromTrack ? 1.05f : 1.15f;
        maxLocal.x = std::min(maxLocal.x, g.scale.x * scaleCapMul);
        maxLocal.y = std::min(maxLocal.y, g.scale.y * scaleCapMul);
        maxLocal.z = std::min(maxLocal.z, g.scale.z * scaleCapMul);
        outTargetScale = {
            clampf(maxLocal.x, kDefaultMinScaleM, kDefaultMaxScaleM),
            clampf(maxLocal.y, kDefaultMinScaleM, kDefaultMaxScaleM),
            clampf(maxLocal.z, kDefaultMinScaleM, kDefaultMaxScaleM)
        };
        outGeomConf = clampf(wSum / (1.0f + wSum), 0.0f, 1.0f);
    }
}

} // namespace

void DeformableAttention3D::reserve(uint32_t /*maxGaussians*/) {
    viewsScratch_.reserve(kMaxCamBuckets);
    voxelHash_.reserve(20480u);
    detNextScratch_.reserve(256u);
}

void DeformableAttention3D::refine(const SimConfig& cfg,
                                   const GaussianFormer3DConfig& gfCfg,
                                   const EgoState& /*ego*/,
                                   const CameraRigRenderer* rig,
                                   const PerceptionFrameOutput& frameIn,
                                   std::vector<Gaussian3DPrimitive>& gaussians,
                                   uint32_t& outCamCorrobCount,
                                   float& outMeanResidualM) {
    viewsScratch_.clear();
    if (rig) {
        const size_t numCams = cfg.cameras.size();
        for (size_t c = 0u; c < numCams; ++c) {
            const CameraMatrices& mats = rig->cameraMatrices(static_cast<int>(c));
            const float tokW = (c < frameIn.soilCameras.size())
                ? frameIn.soilCameras[c].tokenWeight : 1.0f;
            viewsScratch_.push_back(DeformableCamView{
                static_cast<int>(c),
                mats.camFromWorld,
                mats.intr,
                static_cast<float>(rig->camWidth()),
                static_cast<float>(rig->camHeight()),
                tokW
            });
        }
    }
    refineWithViews(gfCfg, viewsScratch_, frameIn.detections2D,
                    frameIn.activeVoxels, frameIn.tracks,
                    gaussians, outCamCorrobCount, outMeanResidualM);
}

void DeformableAttention3D::refineWithViews(const GaussianFormer3DConfig& gfCfg,
                                            const std::vector<DeformableCamView>& views,
                                            const std::vector<Detection2D>& detections2D,
                                            const std::vector<ActiveVoxel>& voxels3D,
                                            const std::vector<Track3D>& tracks3D,
                                            std::vector<Gaussian3DPrimitive>& gaussians,
                                            uint32_t& outCamCorrobCount,
                                            float& outMeanResidualM) {
    outCamCorrobCount = 0u;
    outMeanResidualM = 0.0f;
    if (gaussians.empty()) return;

    const uint32_t layers = std::max(1u, gfCfg.numRefinementLayers);
    const float camW = clampf(gfCfg.camAttnWeight, 0.0f, 1.0f);
    const float lidW = clampf(gfCfg.lidarAttnWeight, 0.0f, 1.0f);
    const float normModal = std::max(kMinWeightEps, camW + lidW);

    camDetHead_.fill(-1);
    if (detNextScratch_.capacity() < detections2D.size()) detNextScratch_.reserve(detections2D.size());
    detNextScratch_.assign(detections2D.size(), -1);
    for (size_t i = 0u; i < detections2D.size(); ++i) {
        const int cIdx = detections2D[i].camIndex;
        if (cIdx < 0 || static_cast<size_t>(cIdx) >= kMaxCamBuckets) continue;
        detNextScratch_[i] = camDetHead_[static_cast<size_t>(cIdx)];
        camDetHead_[static_cast<size_t>(cIdx)] = static_cast<int32_t>(i);
    }

    voxelHash_.reset(voxels3D.size(), kGeomSearchRadM);
    for (size_t i = 0u; i < voxels3D.size(); ++i) {
        const auto& v = voxels3D[i];
        if (v.sem == SemanticClass::Empty || v.sem == SemanticClass::OccShadow) continue;
        voxelHash_.insert(static_cast<int>(i), v.wx, v.wz);
    }

    std::array<Vec3, kMaxDeformableSamples> samples{};
    std::array<float, kNumGaussianSemClasses> camSemBoost{};
    std::array<float, kNumGaussianSemClasses> geomSemBoost{};
    float sumResidual = 0.0f;

    size_t refinedCount = 0u;
    for (auto& g : gaussians) {
        if (g.fromStaticMem) continue;
        ++refinedCount;
        const Vec3 initMean = g.mean;
        uint32_t maxMatchedViews = 0u;

        for (uint32_t layer = 0u; layer < layers; ++layer) {
            uint32_t sampleCnt = 0u;
            buildDeformableSamples(g, gfCfg.numDeformableSamplePts, samples, sampleCnt);

            Vec3 camPull{};
            uint32_t matchedViews = 0u;
            float camConf = 0.0f;
            attendToCameraViews(g, samples, sampleCnt, views, detections2D,
                                camDetHead_, detNextScratch_,
                                camPull, camSemBoost, matchedViews, camConf);

            Vec3 geomMean{};
            Vec3 targetScale{};
            float trackYaw = 0.0f;
            bool hasTrackYaw = false;
            float geomConf = 0.0f;
            attendTo3DGeometry(g, voxelHash_, voxels3D, tracks3D,
                               geomMean, targetScale, trackYaw, hasTrackYaw, geomSemBoost, geomConf);

            const Vec3 geomDelta = geomMean - g.mean;
            const Vec3 fusedDelta = (camPull * camW + geomDelta * lidW) * (1.0f / normModal);
            Vec3 step = fusedDelta * kMeanStepGain;
            const float stepCap = g.fromTrack ? 0.15f
                : (g.isDynamic ? kMaxDynLayerShiftM : kMaxStaticLayerShiftM);
            const float stepLen = step.norm();
            if (stepLen > stepCap) step = step * (stepCap / stepLen);
            g.mean += step;
            g.scale = {
                clampf(lerpf(g.scale.x, targetScale.x, kScaleAdaptGain * geomConf),
                       kDefaultMinScaleM, kDefaultMaxScaleM),
                clampf(lerpf(g.scale.y, targetScale.y, kScaleAdaptGain * geomConf),
                       kDefaultMinScaleM, kDefaultMaxScaleM),
                clampf(lerpf(g.scale.z, targetScale.z, kScaleAdaptGain * geomConf),
                       kDefaultMinScaleM, kDefaultMaxScaleM)
            };

            if (hasTrackYaw) {
                const float curYaw = g.rot.toYaw();
                const float updYaw = curYaw + kYawAdaptGain * wrapAngle(trackYaw - curYaw);
                g.rot = Quat3D::fromYaw(updYaw);
            } else {
                g.rot.normalize();
            }

            for (uint32_t c = 0u; c < kNumGaussianSemClasses; ++c) {
                const float deltaLogit = kCamLogitBoost * camW * camSemBoost[c] +
                                         kGeomLogitBoost * lidW * geomSemBoost[c];
                g.semLogits[c] = clampf(g.semLogits[c] + deltaLogit, kSemLogitMin, kSemLogitMax);
            }

            if (matchedViews > 0u) {
                g.opacity = clampf(g.opacity + kCorrobOpacityStep * static_cast<float>(matchedViews),
                                   kMinGaussianOpacity, kMaxGaussianOpacity);
                g.confidence = clampf(std::max(g.confidence, 0.5f * camConf + 0.5f * geomConf), 0.0f, 1.0f);
            } else if (geomConf < 0.25f) {
                g.opacity = clampf(g.opacity * kUncorrobOpacityDecay, kMinGaussianOpacity, kMaxGaussianOpacity);
            }
            maxMatchedViews = std::max(maxMatchedViews, matchedViews);
        }

        g.camCorrobCount = maxMatchedViews;
        g.lastDisplacementM = (g.mean - initMean).norm();
        sumResidual += g.lastDisplacementM;
        if (maxMatchedViews > 0u) ++outCamCorrobCount;
    }

    outMeanResidualM = (refinedCount > 0u) ? (sumResidual / static_cast<float>(refinedCount)) : 0.0f;
}

} // namespace cybercab::sota
