#pragma once
// Stage 1 (SOTA): 3D Gaussian primitive & multi-modal GaussianFormer3D data types.
// Implements the continuous SO(3) anisotropic 3D Gaussian representation and conversion
// helpers shared across initialization, 3D deformable attention, and voxel splatting.

#include "core/Math.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldTypes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace cybercab::sota {

inline constexpr uint32_t kNumGaussianSemClasses = 8u;
inline constexpr float kDefaultMinScaleM = 0.14f;
inline constexpr float kDefaultMaxScaleM = 4.50f;
inline constexpr float kMinGaussianOpacity = 0.02f;
inline constexpr float kMaxGaussianOpacity = 0.99f;
inline constexpr float kQuatNormEps = 1e-6f;
inline constexpr float kMahalanobisSqCutoff = 9.0f;
inline constexpr float kGaussianExpHalf = -0.5f;
inline constexpr float kSemLogitMin = -6.0f;
inline constexpr float kSemLogitMax = 8.0f;

// Unit quaternion in (w, x, y, z) convention with Y-up right-handed yaw matching GaussianWorldModel.
struct Quat3D {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    float normSq() const { return w * w + x * x + y * y + z * z; }

    Quat3D normalized() const {
        const float nSq = normSq();
        if (nSq <= kQuatNormEps) return {};
        const float inv = 1.0f / std::sqrt(nSq);
        return {w * inv, x * inv, y * inv, z * inv};
    }

    void normalize() { *this = normalized(); }

    static Quat3D fromYaw(float yawRad) {
        const float half = 0.5f * yawRad;
        return Quat3D{std::cos(half), 0.0f, std::sin(half), 0.0f}.normalized();
    }

    // Extracts heading yaw consistent with axes[0]=(cos, 0, -sin) and axes[2]=(sin, 0, cos).
    float toYaw() const {
        const Quat3D q = normalized();
        const float sinYaw = 2.0f * (q.w * q.y - q.x * q.z);
        const float cosYaw = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
        return std::atan2(sinYaw, cosYaw);
    }

    Vec3 rotateVector(const Vec3& v) const {
        const Quat3D q = normalized();
        const float tx = 2.0f * (q.y * v.z - q.z * v.y);
        const float ty = 2.0f * (q.z * v.x - q.x * v.z);
        const float tz = 2.0f * (q.x * v.y - q.y * v.x);
        return {
            v.x + q.w * tx + (q.y * tz - q.z * ty),
            v.y + q.w * ty + (q.z * tx - q.x * tz),
            v.z + q.w * tz + (q.x * ty - q.y * tx)
        };
    }

    Vec3 invRotateVector(const Vec3& v) const {
        const Quat3D conj{w, -x, -y, -z};
        return conj.rotateVector(v);
    }
};

struct Gaussian3DPrimitive {
    Vec3 mean{0.0f, 0.0f, 0.0f};
    Vec3 scale{0.5f, 0.5f, 0.5f};
    Quat3D rot{};
    float opacity = 0.0f;
    std::array<float, kNumGaussianSemClasses> semLogits{};
    Vec3 velWorld{0.0f, 0.0f, 0.0f};
    float confidence = 0.0f;
    bool isDynamic = false;
    bool fromStaticMem = false;
    bool fromTrack = false;
    bool staticMemLocked = false;
    uint32_t camCorrobCount = 0u;
    float lastDisplacementM = 0.0f;

    SemanticClass argmaxSemantic() const {
        uint32_t bestIdx = 0u;
        float bestVal = semLogits[0];
        for (uint32_t i = 1u; i < kNumGaussianSemClasses; ++i) {
            if (semLogits[i] > bestVal) {
                bestVal = semLogits[i];
                bestIdx = i;
            }
        }
        return static_cast<SemanticClass>(bestIdx);
    }

    float semanticConfidence() const {
        float maxL = semLogits[0];
        for (uint32_t i = 1u; i < kNumGaussianSemClasses; ++i) {
            maxL = std::max(maxL, semLogits[i]);
        }
        float sumExp = 0.0f;
        for (uint32_t i = 0u; i < kNumGaussianSemClasses; ++i) {
            sumExp += std::exp(semLogits[i] - maxL);
        }
        return (sumExp > 1e-6f) ? (1.0f / sumExp) : 0.0f;
    }

    void softmaxProbs(std::array<float, kNumGaussianSemClasses>& outProbs) const {
        float maxL = semLogits[0];
        for (uint32_t i = 1u; i < kNumGaussianSemClasses; ++i) {
            maxL = std::max(maxL, semLogits[i]);
        }
        float sumExp = 0.0f;
        for (uint32_t i = 0u; i < kNumGaussianSemClasses; ++i) {
            outProbs[i] = std::exp(semLogits[i] - maxL);
            sumExp += outProbs[i];
        }
        const float inv = (sumExp > 1e-6f) ? (1.0f / sumExp) : (1.0f / kNumGaussianSemClasses);
        for (uint32_t i = 0u; i < kNumGaussianSemClasses; ++i) {
            outProbs[i] *= inv;
        }
    }
};

struct GaussianFormer3DConfig {
    uint32_t maxGaussians = 512u;
    uint32_t numRefinementLayers = 2u;
    uint32_t numDeformableSamplePts = 7u;
    float mergeRadiusM = 0.75f;
    float minInitConf = 0.25f;
    float voxelScaleX = 0.72f;
    float voxelScaleY = 0.80f;
    float voxelScaleZ = 0.72f;
    float splatRadiusSigmaMul = 1.25f;
    float camAttnWeight = 0.55f;
    float lidarAttnWeight = 0.45f;
    float temporalSmoothAlpha = 0.30f;
    float splatSigmaCutoffSq = 2.50f;
    float minVoxelOccProb = 0.38f;
    float splatVoxelStepX = 0.0f;
    float splatVoxelStepY = 0.0f;
    float splatVoxelStepZ = 0.0f;
    bool enableDeformableAttn = true;
};

inline Vec3 semanticDefaultColor(SemanticClass sem) {
    switch (sem) {
        case SemanticClass::Vehicle:    return {0.95f, 0.28f, 0.16f};
        case SemanticClass::Pedestrian: return {0.18f, 0.92f, 0.42f};
        case SemanticClass::SignalSign: return {0.96f, 0.78f, 0.18f};
        case SemanticClass::Overhang:   return {0.72f, 0.45f, 0.92f};
        case SemanticClass::Curb:       return {0.58f, 0.62f, 0.68f};
        case SemanticClass::Static:
        default:                        return {0.42f, 0.48f, 0.56f};
    }
}

inline float evalAnisotropicGaussian3D(const Gaussian3DPrimitive& g, const Vec3& worldPt,
                                       float horizonSec = 0.0f) {
    const float t = (g.isDynamic && horizonSec > 0.0f) ? horizonSec : 0.0f;
    const Vec3 center{g.mean.x + g.velWorld.x * t, g.mean.y, g.mean.z + g.velWorld.z * t};
    const Vec3 local = g.rot.invRotateVector(worldPt - center);
    const float sx = std::max(kDefaultMinScaleM, g.scale.x);
    const float sy = std::max(kDefaultMinScaleM, g.scale.y);
    const float sz = std::max(kDefaultMinScaleM, g.scale.z);
    const float mSq = (local.x * local.x) / (sx * sx) +
                      (local.y * local.y) / (sy * sy) +
                      (local.z * local.z) / (sz * sz);
    if (mSq > kMahalanobisSqCutoff) return 0.0f;
    return clampf(g.opacity * std::exp(kGaussianExpHalf * mSq), 0.0f, kMaxGaussianOpacity);
}

inline V15Gaussian3D toV15Gaussian3D(const Gaussian3DPrimitive& g) {
    const SemanticClass sem = g.argmaxSemantic();
    const Vec3 rgb = semanticDefaultColor(sem);
    return V15Gaussian3D{
        {g.mean.x, g.mean.y, g.mean.z, clampf(g.opacity, 0.0f, kMaxGaussianOpacity)},
        {std::max(kDefaultMinScaleM, g.scale.x),
         std::max(kDefaultMinScaleM, g.scale.y),
         std::max(kDefaultMinScaleM, g.scale.z),
         static_cast<float>(static_cast<int>(sem))},
        {g.velWorld.x, g.velWorld.z, g.rot.toYaw(), g.isDynamic ? 1.0f : 0.0f},
        {rgb.x, rgb.y, rgb.z, clampf(g.confidence, 0.0f, 1.0f)}
    };
}

} // namespace cybercab::sota
