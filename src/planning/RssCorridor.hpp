#pragma once

#include "core/Math.hpp"
#include "world/RoadCurve.hpp"
#include "world/WorldTypes.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace cybercab::rss {

inline Vec3 sampleRoadCorridorPoint(const ClosedCatmullRomCurve3& centerline, float totalLoopLen, float loopU,
                                    float lookaheadDist, float d0, float d1, float sFromEgo, float sFromBumper,
                                    Vec3* outTan = nullptr) {
    float u = std::fmod(loopU + sFromEgo / totalLoopLen, 1.0f);
    if (u < 0.0f) u += 1.0f;
    const Vec3 cPt = centerline.getPointAt(u);
    const Vec3 tan = centerline.getTangentAt(u);
    const Vec3 left{tan.z, 0.0f, -tan.x};
    // Lateral lane-change transition begins physically at the front bumper rather than teleporting sideways at s=0..4m.
    const float tNorm = clampf(sFromBumper / std::max(lookaheadDist, 20.0f), 0.0f, 1.0f);
    const float poly = tNorm * tNorm * tNorm * (10.0f + tNorm * (-15.0f + 6.0f * tNorm));
    const float lat = d0 + (d1 - d0) * poly;
    if (outTan) *outTan = tan;
    return {cPt.x + left.x * lat, 0.08f, cPt.z + left.z * lat};
}

/// Builds a C1-continuous curved RSS corridor centerline anchored at the front bumper.
inline void buildRssCorridorPoints(const EgoState& ego, const ClosedCatmullRomCurve3& centerline,
                                   float loopLen, float loopU, float lookaheadDist,
                                   float currentLat, float targetLat, float wheelbaseM,
                                   bool manualMode, float maxLen,
                                   std::vector<Vec3>& ptsOut, std::vector<float>& cumSOut) {
    const float sinE = std::sin(ego.yaw), cosE = std::cos(ego.yaw);
    const float bumperFwd = ego.hz + ego.centerOffsetForward;
    const Vec3 frontBumper{ego.pos.x + sinE * bumperFwd, 0.08f, ego.pos.z + cosE * bumperFwd};
    const Vec3 egoFwd{sinE, 0.0f, cosE};

    constexpr float kStepM = 1.0f;
    const int nPts = std::max(4, static_cast<int>(std::ceil(maxLen / kStepM)) + 1);
    ptsOut.resize(static_cast<size_t>(nPts));
    cumSOut.resize(static_cast<size_t>(nPts));

    Vec3 roadTanAtBumper{};
    sampleRoadCorridorPoint(centerline, loopLen, loopU, lookaheadDist, currentLat, targetLat,
                            bumperFwd, 0.0f, &roadTanAtBumper);
    const bool useBicycleArc = manualMode &&
                               (std::abs(ego.steerAngle) > 0.03f || egoFwd.dot(roadTanAtBumper) < 0.25f);
    if (useBicycleArc) {
        const float kappa = std::tan(ego.steerAngle) / std::max(wheelbaseM, 0.5f);
        Vec3 cur = frontBumper;
        float yaw = ego.yaw;
        for (int i = 0; i < nPts; ++i) {
            ptsOut[i] = cur;
            cumSOut[i] = static_cast<float>(i) * kStepM;
            yaw += kappa * kStepM;
            cur.x += std::sin(yaw) * kStepM;
            cur.z += std::cos(yaw) * kStepM;
        }
        return;
    }

    constexpr float kBlendLen = 4.5f;
    Vec3 p1Tan{};
    const Vec3 p1 = sampleRoadCorridorPoint(centerline, loopLen, loopU, lookaheadDist,
                                            currentLat, targetLat, bumperFwd + kBlendLen, kBlendLen, &p1Tan);
    for (int i = 0; i < nPts; ++i) {
        const float s = static_cast<float>(i) * kStepM;
        if (s < kBlendLen) {
            const float t = s / kBlendLen;
            const float t2 = t * t, t3 = t2 * t;
            const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
            const float h10 = t3 - 2.0f * t2 + t;
            const float h01 = -2.0f * t3 + 3.0f * t2;
            const float h11 = t3 - t2;
            ptsOut[i] = {
                h00 * frontBumper.x + h10 * (kBlendLen * egoFwd.x) + h01 * p1.x + h11 * (kBlendLen * p1Tan.x),
                0.08f,
                h00 * frontBumper.z + h10 * (kBlendLen * egoFwd.z) + h01 * p1.z + h11 * (kBlendLen * p1Tan.z)
            };
        } else {
            ptsOut[i] = sampleRoadCorridorPoint(centerline, loopLen, loopU, lookaheadDist,
                                                currentLat, targetLat, bumperFwd + s, s);
        }
        cumSOut[i] = (i == 0)
            ? 0.0f
            : (cumSOut[i - 1] + std::hypot(ptsOut[i].x - ptsOut[i - 1].x, ptsOut[i].z - ptsOut[i - 1].z));
    }
}

/// Projects (px, pz) onto an arc-length-parameterized XZ polyline, returning signed arc-length and lateral offset.
inline void projectOntoPolyline(const std::vector<Vec3>& pts, const std::vector<float>& cumS,
                                float px, float pz,
                                float& outS, float& outLat, float& outTanX, float& outTanZ) {
    outS = 1e6f;
    outLat = 1e6f;
    outTanX = 0.0f;
    outTanZ = 1.0f;
    const size_t n = pts.size();
    if (n < 2) return;

    float bestDistSq = std::numeric_limits<float>::infinity();
    size_t bestSeg = 0;
    float bestT = 0.0f;
    float bestSegLen = 1.0f;
    for (size_t i = 0; i + 1 < n; ++i) {
        const float ax = pts[i].x, az = pts[i].z;
        const float vx = pts[i + 1].x - ax, vz = pts[i + 1].z - az;
        const float lenSq = vx * vx + vz * vz;
        const float len = std::sqrt(std::max(lenSq, 1e-8f));
        const float t = (lenSq > 1e-8f) ? clampf(((px - ax) * vx + (pz - az) * vz) / lenSq, 0.0f, 1.0f) : 0.0f;
        const float dx = px - (ax + t * vx), dz = pz - (az + t * vz);
        const float dSq = dx * dx + dz * dz;
        if (dSq < bestDistSq) {
            bestDistSq = dSq;
            bestSeg = i;
            bestT = t;
            bestSegLen = len;
        }
    }

    const float vx = pts[bestSeg + 1].x - pts[bestSeg].x;
    const float vz = pts[bestSeg + 1].z - pts[bestSeg].z;
    outTanX = vx / bestSegLen;
    outTanZ = vz / bestSegLen;
    outS = cumS[bestSeg] + bestT * bestSegLen;

    if (bestSeg == 0 && bestT <= 0.0f) {
        const float rawProj = (px - pts[0].x) * outTanX + (pz - pts[0].z) * outTanZ;
        if (rawProj < 0.0f) outS = rawProj;
    } else if (bestSeg + 2 == n && bestT >= 1.0f) {
        const float rawProj = (px - pts[bestSeg + 1].x) * outTanX + (pz - pts[bestSeg + 1].z) * outTanZ;
        if (rawProj > 0.0f) outS = cumS.back() + rawProj;
    }
    const float cx = pts[bestSeg].x + bestT * vx;
    const float cz = pts[bestSeg].z + bestT * vz;
    outLat = std::hypot(px - cx, pz - cz);
}

inline void resamplePolyline(const std::vector<Vec3>& src, const std::vector<float>& cumS,
                             float targetLen, size_t numSamples, std::vector<Vec3>& dst) {
    dst.resize(numSamples);
    if (src.size() < 2 || cumS.size() != src.size()) return;
    const float endS = clampf(targetLen, 0.5f, cumS.back());
    size_t seg = 0;
    for (size_t i = 0; i < numSamples; ++i) {
        const float s = (numSamples > 1) ? (endS * static_cast<float>(i) / static_cast<float>(numSamples - 1)) : 0.0f;
        while (seg + 2 < src.size() && cumS[seg + 1] < s) ++seg;
        const float segSpan = std::max(cumS[seg + 1] - cumS[seg], 1e-5f);
        const float a = clampf((s - cumS[seg]) / segSpan, 0.0f, 1.0f);
        dst[i] = {lerpf(src[seg].x, src[seg + 1].x, a), 0.08f, lerpf(src[seg].z, src[seg + 1].z, a)};
    }
}

} // namespace cybercab::rss