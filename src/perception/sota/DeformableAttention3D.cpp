#include "perception/sota/DeformableAttention3D.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace cybercab::sota {
namespace {

constexpr uint32_t kMaxDeformableSamples = 7u;
constexpr float kSampleRadiusFrac = 0.72f;
constexpr float kMinProjDepthM = 0.35f;
constexpr float kCamBoxPadNorm = 0.045f;
constexpr float kCamRayPullGain = 0.25f;
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

void buildDeformableSamples(const Gaussian3DPrimitive& g,
                            uint32_t requestedPts,
                            std::array<Vec3, kMaxDeformableSamples>& outPts,
                            uint32_t& outCount) {
    outCount = std::clamp(requestedPts, 1u, kMaxDeformableSamples);
    outPts[0] = g.mean;
    if (outCount == 1u) return;

    const Vec3 ax = g.rot.rotateVector({kSampleRadiusFrac * g.scale.x, 0.0f, 0.0f});
    const Vec3 ay = g.rot.rotateVector({0.0f, kSampleRadiusFrac * g.scale.y, 0.0f});
    const Vec3 az = g.rot.rotateVector({0.0f, 0.0f, kSampleRadiusFrac * g.scale.z});
    const Vec3 offsets[kMaxDeformableSamples] = {
        {}, ax, -ax, ay, -ay, az, -az
    };
    for (uint32_t k = 1u; k < outCount; ++k) {
        outPts[k] = g.mean + offsets[k];
    }
}

bool projectCamPoint(const DeformableCamView& view, const Vec3& worldPt, Vec2& outNormUv) {
    const Vec3 pc = view.camFromWorld.transformPoint(worldPt);
    if (pc.z <= kMinProjDepthM) return false;
    const float invZ = 1.0f / pc.z;
    const float px = view.intr.cx + view.intr.fx * pc.x * invZ;
    const float py = view.intr.cy + view.intr.fy * pc.y * invZ;
    const float w = std::max(1.0f, view.width);
    const float h = std::max(1.0f, view.height);
    outNormUv = {px / w, py / h};
    return outNormUv.x >= -0.05f && outNormUv.x <= 1.05f &&
           outNormUv.y >= -0.05f && outNormUv.y <= 1.05f;
}

void attendToCameraViews(const Gaussian3DPrimitive& g,
                         const std::array<Vec3, kMaxDeformableSamples>& samples,
                         uint32_t numSamples,
                         const std::vector<DeformableCamView>& views,
                         const std::vector<Detection2D>& detections2D,
                         Vec3& outCamPull,
                         std::array<float, kNumGaussianSemClasses>& outSemBoost,
                         uint32_t& outMatchedViews,
                         float& outCamConf) {
    outCamPull = {};
    outSemBoost.fill(0.0f);
    outMatchedViews = 0u;
    outCamConf = 0.0f;
    if (views.empty() || detections2D.empty()) return;

    float pullWSum = 0.0f;
    for (const auto& view : views) {
        if (view.tokenWeight <= 0.01f) continue;

        float bestViewScore = 0.0f;
        Vec3 viewWorldRight = {view.camFromWorld.at(0, 0),
                               view.camFromWorld.at(0, 1),
                               view.camFromWorld.at(0, 2)};
        float bestOffsetU = 0.0f;

        for (const auto& det : detections2D) {
            if (det.camIndex != view.camIdx || det.score < 0.15f) continue;
            const float halfW = 0.5f * det.wNorm + kCamBoxPadNorm;
            const float halfH = 0.5f * det.hNorm + kCamBoxPadNorm;
            if (halfW <= 0.01f || halfH <= 0.01f) continue;

            float detAttnSum = 0.0f;
            float meanDu = 0.0f;
            int hitSamples = 0;
            for (uint32_t k = 0u; k < numSamples; ++k) {
                Vec2 uv{};
                if (!projectCamPoint(view, samples[k], uv)) continue;
                const float du = (uv.x - det.cxNorm) / halfW;
                const float dv = (uv.y - det.cyNorm) / halfH;
                if (std::abs(du) > 1.25f || std::abs(dv) > 1.25f) continue;
                const float w = std::exp(-1.5f * (du * du + dv * dv));
                detAttnSum += w;
                meanDu += (det.cxNorm - uv.x);
                ++hitSamples;
            }
            if (hitSamples == 0) continue;

            const float avgAttn = detAttnSum / static_cast<float>(numSamples);
            const float score = avgAttn * det.score * det.camPatchWeight * view.tokenWeight;
            const uint32_t cls = std::min(static_cast<uint32_t>(det.semantic), kNumGaussianSemClasses - 1u);
            outSemBoost[cls] += score;
            if (score > bestViewScore) {
                bestViewScore = score;
                bestOffsetU = meanDu / static_cast<float>(hitSamples);
            }
        }

        if (bestViewScore > 0.08f) {
            ++outMatchedViews;
            outCamConf += bestViewScore;
            const float dist = std::max(1.5f, g.mean.norm());
            outCamPull += viewWorldRight * (bestOffsetU * dist * kCamRayPullGain * bestViewScore);
            pullWSum += bestViewScore;
        }
    }
    if (pullWSum > kMinWeightEps) {
        outCamPull = outCamPull * (1.0f / pullWSum);
        outCamConf = clampf(outCamConf / static_cast<float>(outMatchedViews), 0.0f, 1.0f);
    }
}

void attendTo3DGeometry(const Gaussian3DPrimitive& g,
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
    Vec3 maxLocal{g.scale.x * 0.6f, g.scale.y * 0.6f, g.scale.z * 0.6f};
    const float radSq = kGeomSearchRadM * kGeomSearchRadM;
    const SemanticClass gSem = g.argmaxSemantic();

    for (const auto& v : voxels3D) {
        if (v.sem == SemanticClass::Empty || v.sem == SemanticClass::OccShadow) continue;
        const Vec3 vp{v.wx, v.wy, v.wz};
        const Vec3 diff = vp - g.mean;
        const float dSq = diff.normSq();
        if (dSq > radSq) continue;

        const float semMatch = (v.sem == gSem) ? 1.25f : 0.45f;
        const float w = std::exp(-0.5f * dSq / std::max(0.35f, g.scale.normSq())) *
                        clampf(v.confidence, 0.1f, 1.0f) * semMatch;
        wSum += w;
        sumPos += vp * w;
        const Vec3 local = g.rot.invRotateVector(diff);
        maxLocal.x = std::max(maxLocal.x, std::abs(local.x));
        maxLocal.y = std::max(maxLocal.y, std::abs(local.y));
        maxLocal.z = std::max(maxLocal.z, std::abs(local.z));
        const uint32_t cls = std::min(static_cast<uint32_t>(v.sem), kNumGaussianSemClasses - 1u);
        outSemBoost[cls] += w;
    }

    const float trRadSq = kTrackSearchRadM * kTrackSearchRadM;
    float bestTrackDistSq = trRadSq;
    for (const auto& tr : tracks3D) {
        if (tr.state == TrackState::Removed) continue;
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
    viewsScratch_.reserve(16u);
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

    std::array<Vec3, kMaxDeformableSamples> samples{};
    std::array<float, kNumGaussianSemClasses> camSemBoost{};
    std::array<float, kNumGaussianSemClasses> geomSemBoost{};
    float sumResidual = 0.0f;

    for (auto& g : gaussians) {
        const Vec3 initMean = g.mean;
        uint32_t maxMatchedViews = 0u;

        for (uint32_t layer = 0u; layer < layers; ++layer) {
            uint32_t sampleCnt = 0u;
            buildDeformableSamples(g, gfCfg.numDeformableSamplePts, samples, sampleCnt);

            Vec3 camPull{};
            uint32_t matchedViews = 0u;
            float camConf = 0.0f;
            attendToCameraViews(g, samples, sampleCnt, views, detections2D,
                                camPull, camSemBoost, matchedViews, camConf);

            Vec3 geomMean{};
            Vec3 targetScale{};
            float trackYaw = 0.0f;
            bool hasTrackYaw = false;
            float geomConf = 0.0f;
            attendTo3DGeometry(g, voxels3D, tracks3D,
                               geomMean, targetScale, trackYaw, hasTrackYaw, geomSemBoost, geomConf);

            const Vec3 geomDelta = geomMean - g.mean;
            const Vec3 fusedDelta = (camPull * camW + geomDelta * lidW) * (1.0f / normModal);
            g.mean += fusedDelta * kMeanStepGain;
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

    outMeanResidualM = sumResidual / static_cast<float>(gaussians.size());
}

} // namespace cybercab::sota
