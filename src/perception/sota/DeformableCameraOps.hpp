#pragma once
// Stage 1b Camera Cross-Attention helpers shared by DeformableAttention3D.
// Generates 3D ellipsoid reference offsets, projects them into calibrated 2D views,
// and aggregates multi-view detection confidence & ray alignment residuals.

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/sota/Gaussian3DTypes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cybercab::sota {

inline constexpr uint32_t kMaxDeformableSamples = 7u;
inline constexpr size_t kMaxCamBuckets = 16u;
inline constexpr float kSampleRadiusFrac = 0.72f;
inline constexpr float kMinProjDepthM = 0.35f;
inline constexpr float kCamBoxPadNorm = 0.045f;
inline constexpr float kCamRayPullGain = 0.25f;
inline constexpr float kMaxCamRayDepthM = 35.0f;

struct DeformableCamView {
    int camIdx = 0;
    Mat4 camFromWorld{};
    CameraIntrinsics intr{};
    float width = 320.0f;
    float height = 240.0f;
    float tokenWeight = 1.0f;
};

inline void buildDeformableSamples(const Gaussian3DPrimitive& g,
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

inline bool projectCamPoint(const DeformableCamView& view, const Vec3& worldPt, Vec2& outNormUv) {
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

inline void attendToCameraViews(const Gaussian3DPrimitive& g,
                                const std::array<Vec3, kMaxDeformableSamples>& samples,
                                uint32_t numSamples,
                                const std::vector<DeformableCamView>& views,
                                const std::vector<Detection2D>& detections2D,
                                const std::array<int32_t, kMaxCamBuckets>& camHead,
                                const std::vector<int32_t>& detNext,
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
        if (view.camIdx < 0 || static_cast<size_t>(view.camIdx) >= camHead.size()) continue;
        const int32_t firstDet = camHead[static_cast<size_t>(view.camIdx)];
        if (firstDet < 0) continue;

        const Vec3 pcMean = view.camFromWorld.transformPoint(g.mean);
        if (pcMean.z <= kMinProjDepthM) continue;
        const float camDepth = clampf(pcMean.z, 1.5f, kMaxCamRayDepthM);

        float bestViewScore = 0.0f;
        const Vec3 viewWorldRight = {view.camFromWorld.at(0, 0),
                                     view.camFromWorld.at(0, 1),
                                     view.camFromWorld.at(0, 2)};
        float bestOffsetU = 0.0f;

        for (int32_t di = firstDet; di >= 0; di = detNext[static_cast<size_t>(di)]) {
            const auto& det = detections2D[static_cast<size_t>(di)];
            if (det.score < 0.15f) continue;
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
            outCamPull += viewWorldRight * (bestOffsetU * camDepth * kCamRayPullGain * bestViewScore);
            pullWSum += bestViewScore;
        }
    }
    if (pullWSum > 1e-5f) {
        outCamPull = outCamPull * (1.0f / pullWSum);
        outCamConf = clampf(outCamConf / static_cast<float>(outMatchedViews), 0.0f, 1.0f);
    }
}

} // namespace cybercab::sota
