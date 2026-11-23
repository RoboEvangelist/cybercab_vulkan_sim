#pragma once

#include "TestPerceptionHelpers.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cybercab::testutil {

class SharedGpuMetricsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { s_ctx.init(false, nullptr); }
    static void TearDownTestSuite() { s_ctx.cleanup(); }
    static VulkanContext s_ctx;
};

inline VulkanContext SharedGpuMetricsTest::s_ctx;

inline bool isPointInOBB(const OBB& b, float wx, float wz, float pad) {
    const float c = std::cos(b.yaw), s = std::sin(b.yaw);
    const float dx = wx - b.cx, dz = wz - b.cz;
    const float lx = dx * c - dz * s;
    const float lz = dx * s + dz * c;
    return std::abs(lx) <= (b.hx + pad) && std::abs(lz) <= (b.hz + pad);
}

inline std::string fmtProp(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << v;
    return oss.str();
}

constexpr int kPrThreshCount = 10;
constexpr std::array<float, kPrThreshCount> kPrThresh = {
    0.12f, 0.18f, 0.24f, 0.32f, 0.42f, 0.52f, 0.65f, 0.78f, 0.88f, 0.94f
};

struct PrAccum {
    int tp[kPrThreshCount]{};
    int fp[kPrThreshCount]{};
    int fn[kPrThreshCount]{};
};

// Accumulates per-threshold TP/FP/FN while treating boundary GTs just beyond evalRange as non-FP ignores.
inline void accumulatePrCls(SemanticClass cls,
                              const std::vector<Detection3D>& dets,
                              const std::vector<OBB>& reqGts,
                              const std::vector<OBB>& boundaryGts,
                              float gateM,
                              float maxDetRangeM,
                              PrAccum& acc) {
    for (int k = 0; k < kPrThreshCount; ++k) {
        std::vector<uint8_t> used(reqGts.size(), 0);
        int tp = 0, fp = 0;
        for (const auto& d : dets) {
            if (d.semantic != cls || d.score < kPrThresh[k] || d.rangeM > maxDetRangeM) continue;
            int bestIdx = -1;
            float bestDist = gateM;
            for (size_t gi = 0; gi < reqGts.size(); ++gi) {
                if (used[gi]) continue;
                const float dist = std::hypot(d.worldPos.x - reqGts[gi].cx, d.worldPos.z - reqGts[gi].cz);
                if (dist < bestDist) { bestDist = dist; bestIdx = static_cast<int>(gi); }
            }
            if (bestIdx >= 0) {
                used[bestIdx] = 1;
                ++tp;
                continue;
            }
            bool matchedBoundary = false;
            for (const auto& bg : boundaryGts) {
                if (std::hypot(d.worldPos.x - bg.cx, d.worldPos.z - bg.cz) < gateM) {
                    matchedBoundary = true;
                    break;
                }
            }
            if (!matchedBoundary) ++fp;
        }
        acc.tp[k] += tp;
        acc.fp[k] += fp;
        acc.fn[k] += static_cast<int>(reqGts.size()) - tp;
    }
}

inline void summarizePrCurve(const PrAccum& a, double& apOut, double& maxF1Out) {
    double maxF1 = 0.0;
    std::vector<std::pair<double, double>> pts;
    pts.reserve(kPrThreshCount);
    for (int k = 0; k < kPrThreshCount; ++k) {
        const double p = (a.tp[k] + a.fp[k] > 0) ? static_cast<double>(a.tp[k]) / (a.tp[k] + a.fp[k]) : 1.0;
        const double r = (a.tp[k] + a.fn[k] > 0) ? static_cast<double>(a.tp[k]) / (a.tp[k] + a.fn[k]) : 0.0;
        const double f1 = (p + r > 0.0) ? (2.0 * p * r / (p + r)) : 0.0;
        maxF1 = std::max(maxF1, f1);
        pts.push_back({r, p});
    }
    std::sort(pts.begin(), pts.end());
    double ap = 0.0, prevR = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        double pMax = 0.0;
        for (size_t j = i; j < pts.size(); ++j) pMax = std::max(pMax, pts[j].second);
        ap += std::max(0.0, pts[i].first - prevR) * pMax;
        prevR = pts[i].first;
    }
    apOut = ap;
    maxF1Out = maxF1;
}

} // namespace cybercab::testutil
