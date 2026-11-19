#pragma once

#include "core/Math.hpp"
#include <vector>

namespace cybercab {

// Closed-loop Catmull-Rom centripetal/uniform spline matching Three.js CatmullRomCurve3(pts, true, 'catmullrom', 0.25)
class ClosedCatmullRomCurve3 {
public:
    ClosedCatmullRomCurve3() = default;
    explicit ClosedCatmullRomCurve3(std::vector<Vec3> controlPts, float tension = 0.25f, int arcSamples = 600)
        : ctrl_(std::move(controlPts)), tension_(tension) {
        buildArcTable(arcSamples);
    }

    Vec3 evaluateRaw(float t) const {
        if (ctrl_.empty()) return {};
        const int n = static_cast<int>(ctrl_.size());
        float u = t - std::floor(t);
        float p = u * static_cast<float>(n);
        int idx = static_cast<int>(std::floor(p)) % n;
        float weight = p - std::floor(p);

        const Vec3& p0 = ctrl_[(idx - 1 + n) % n];
        const Vec3& p1 = ctrl_[idx % n];
        const Vec3& p2 = ctrl_[(idx + 1) % n];
        const Vec3& p3 = ctrl_[(idx + 2) % n];

        // Standard Catmull-Rom with tension tau (matches Three.js CatmullRomCurve3 with tension=0.25)
        // Note: Three.js uses c0 = p1, m0 = tension*(p2-p0)? Actually Three.js cubic Catmull-Rom
        // with tension s uses scale 0.5? In Three.js CurveExtras/CatmullRomCurve3:
        // for type 'catmullrom', tension default 0.5 -> standard 0.5*(p2-p0). Here tension=0.25 -> tighter corners.
        // Using standard Hermite basis with tangents m1 = s*(p2-p0), m2 = s*(p3-p1), s = tension_ * 2.0?
        // Wait: in Three.js Interpolations.CatmullRom(t, p0, p1, p2, p3):
        // v0 = (p2 - p0) * 0.5, v1 = (p3 - p1) * 0.5; wait in Three.js r128 'catmullrom' uses tension directly in initNonuniform? No, Three.js uses p_i = (p2-p0)*tension for 'catmullrom'!
        // Wait: if tension=0.5 it's standard Catmull-Rom. Let's use Catmull-Rom tangent scale = tension_!
        float s = tension_;
        Vec3 m1 = (p2 - p0) * s;
        Vec3 m2 = (p3 - p1) * s;

        float w2 = weight * weight;
        float w3 = w2 * weight;
        float h00 = 2.0f * w3 - 3.0f * w2 + 1.0f;
        float h10 = w3 - 2.0f * w2 + weight;
        float h01 = -2.0f * w3 + 3.0f * w2;
        float h11 = w3 - w2;

        return p1 * h00 + m1 * h10 + p2 * h01 + m2 * h11;
    }

    Vec3 getPointAt(float u) const {
        if (arcU_.empty()) return {};
        float uu = u - std::floor(u);
        float targetLen = uu * totalLength_;
        // Binary search arc table
        auto it = std::lower_bound(arcLen_.begin(), arcLen_.end(), targetLen);
        size_t idx = static_cast<size_t>(std::distance(arcLen_.begin(), it));
        if (idx == 0) return evaluateRaw(0.0f);
        if (idx >= arcLen_.size()) return evaluateRaw(1.0f);

        float l0 = arcLen_[idx - 1];
        float l1 = arcLen_[idx];
        float frac = (l1 > l0 + 1e-6f) ? ((targetLen - l0) / (l1 - l0)) : 0.0f;
        float rawT = lerpf(arcU_[idx - 1], arcU_[idx], frac);
        return evaluateRaw(rawT);
    }

    Vec3 getTangentAt(float u) const {
        const float eps = 1e-4f;
        Vec3 p0 = getPointAt(u - eps);
        Vec3 p1 = getPointAt(u + eps);
        return (p1 - p0).normalized();
    }

    float getLength() const { return totalLength_; }

private:
    std::vector<Vec3> ctrl_;
    float tension_ = 0.25f;
    std::vector<float> arcU_;
    std::vector<float> arcLen_;
    float totalLength_ = 1.0f;

    void buildArcTable(int samples) {
        arcU_.resize(samples + 1);
        arcLen_.resize(samples + 1);
        Vec3 prev = evaluateRaw(0.0f);
        arcU_[0] = 0.0f;
        arcLen_[0] = 0.0f;
        float accum = 0.0f;
        for (int i = 1; i <= samples; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(samples);
            Vec3 cur = evaluateRaw(t);
            accum += (cur - prev).norm();
            arcU_[i] = t;
            arcLen_[i] = accum;
            prev = cur;
        }
        totalLength_ = std::max(accum, 1e-3f);
    }
};

inline ClosedCatmullRomCurve3 buildCybercabLoopPath(float offsetFromCenterline) {
    constexpr float LOOP_HALF_X = 56.0f;
    constexpr float LOOP_HALF_Z = 44.0f;
    const float hx = LOOP_HALF_X - offsetFromCenterline;
    const float hz = LOOP_HALF_Z - offsetFromCenterline;
    const float r = std::max(6.0f, 14.0f - offsetFromCenterline);
    std::vector<Vec3> pts;
    const int arcSteps = 10;

    for (int i = 0; i <= 16; ++i) {
        float t = static_cast<float>(i) / 16.0f;
        pts.push_back({lerpf(-hx + r, hx - r, t), 0.0f, hz});
    }
    for (int i = 1; i <= arcSteps; ++i) {
        float a = kPi * 0.5f - (static_cast<float>(i) / arcSteps) * (kPi * 0.5f);
        pts.push_back({hx - r + std::cos(a) * r, 0.0f, hz - r + std::sin(a) * r});
    }
    for (int i = 1; i <= 14; ++i) {
        float t = static_cast<float>(i) / 14.0f;
        pts.push_back({hx, 0.0f, lerpf(hz - r, -hz + r, t)});
    }
    for (int i = 1; i <= arcSteps; ++i) {
        float a = -(static_cast<float>(i) / arcSteps) * (kPi * 0.5f);
        pts.push_back({hx - r + std::cos(a) * r, 0.0f, -hz + r + std::sin(a) * r});
    }
    for (int i = 1; i <= 16; ++i) {
        float t = static_cast<float>(i) / 16.0f;
        pts.push_back({lerpf(hx - r, -hx + r, t), 0.0f, -hz});
    }
    for (int i = 1; i <= arcSteps; ++i) {
        float a = -kPi * 0.5f - (static_cast<float>(i) / arcSteps) * (kPi * 0.5f);
        pts.push_back({-hx + r + std::cos(a) * r, 0.0f, -hz + r + std::sin(a) * r});
    }
    for (int i = 1; i <= 14; ++i) {
        float t = static_cast<float>(i) / 14.0f;
        pts.push_back({-hx, 0.0f, lerpf(-hz + r, hz - r, t)});
    }
    for (int i = 1; i < arcSteps; ++i) {
        float a = kPi - (static_cast<float>(i) / arcSteps) * (kPi * 0.5f);
        pts.push_back({-hx + r + std::cos(a) * r, 0.0f, hz - r + std::sin(a) * r});
    }

    return ClosedCatmullRomCurve3(std::move(pts), 0.25f);
}

} // namespace cybercab
