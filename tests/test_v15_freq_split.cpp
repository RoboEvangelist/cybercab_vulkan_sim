// FSD v15 Module 1c: frequency separation (illumination vs. reflectance detail) of the photon tensor.
// (1) GPU kernel == CPU reference; (2) exposure invariance: a 4x darker scene moves only the low band
// by log(4) while high band + log-gradients stay put; (3) sanity on a real v15 render.
#include "config/SimConfig.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/v15/FrequencySplit.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "sensors/PhotonModel.hpp"
#include "world/WorldEngine.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace cybercab {
namespace {

std::string fmt(double v) {
    std::ostringstream os;
    os.precision(5);
    os << std::fixed << v;
    return os.str();
}

constexpr int kW = 128, kH = 96;

// Checkerboard reflectance {0.25, 0.75} under an exponential illumination ramp 1 -> 1/4 (tunnel entrance).
std::vector<Vec4> syntheticScene(float exposure) {
    std::vector<Vec4> px(static_cast<size_t>(kW * kH));
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const float refl = (((x / 8) + (y / 8)) % 2) ? 0.75f : 0.25f;
            const float illum = std::exp(-std::log(4.0f) * static_cast<float>(x) / (kW - 1));
            const float rad = refl * illum * exposure / photon::kFullWellRadiance;
            px[static_cast<size_t>(y * kW + x)] = Vec4{rad, rad, rad, 1.0f};
        }
    }
    return px;
}

// CPU reference of v15_freq_split.comp (same clamp-to-edge borders, same weights).
std::vector<Vec4> cpuReference(const std::vector<Vec4>& in, const v15::FreqSplitParams& p) {
    auto at = [&](int x, int y) {
        x = std::clamp(x, 0, kW - 1);
        y = std::clamp(y, 0, kH - 1);
        const Vec4& c = in[static_cast<size_t>(y * kW + x)];
        const float Y = (0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z) * p.fullWell;
        return std::log(Y + p.logEps);
    };
    const int r = p.radius();
    std::vector<float> w(static_cast<size_t>(2 * r + 1));
    float ws = 0.0f;
    for (int k = -r; k <= r; ++k) ws += (w[static_cast<size_t>(k + r)] = std::exp(-0.5f * k * k / (p.sigmaPx * p.sigmaPx)));
    std::vector<float> hb(static_cast<size_t>(kW * kH));
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            float a = 0.0f;
            for (int k = -r; k <= r; ++k) a += w[static_cast<size_t>(k + r)] * at(x + k, y);
            hb[static_cast<size_t>(y * kW + x)] = a / ws;
        }
    std::vector<Vec4> out(in.size());
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            float a = 0.0f;
            for (int k = -r; k <= r; ++k) a += w[static_cast<size_t>(k + r)] * hb[static_cast<size_t>(std::clamp(y + k, 0, kH - 1) * kW + x)];
            const float low = a / ws;
            out[static_cast<size_t>(y * kW + x)] = Vec4{low, at(x, y) - low, 0.5f * (at(x + 1, y) - at(x - 1, y)),
                                                       0.5f * (at(x, y + 1) - at(x, y - 1))};
        }
    return out;
}

std::vector<Vec4> runGpu(VulkanContext& ctx, const std::vector<Vec4>& img) {
    VulkanBuffer in;
    in.create(ctx, img.size() * sizeof(Vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    in.uploadVector(img);
    v15::FrequencySplit fs;
    fs.init(ctx, in, kW, kH, 1);
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    fs.recordDispatch(cmd);
    ctx.endSingleTimeCommands(cmd);
    std::vector<Vec4> out(img.size());
    fs.bandsSSBO().downloadVector(out);
    fs.destroy();
    in.destroy();
    return out;
}

class V15FreqSplitTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { s_ctx = new VulkanContext(); s_ctx->init(false, nullptr); }
    static void TearDownTestSuite() { s_ctx->cleanup(); delete s_ctx; s_ctx = nullptr; }
    static VulkanContext* s_ctx;
};
VulkanContext* V15FreqSplitTest::s_ctx = nullptr;

TEST_F(V15FreqSplitTest, GpuKernelMatchesCpuReference) {
    const auto img = syntheticScene(1.0f);
    const auto gpu = runGpu(*s_ctx, img);
    const auto ref = cpuReference(img, v15::FreqSplitParams{});
    double maxErr = 0.0;
    for (size_t i = 0; i < img.size(); ++i) {
        maxErr = std::max({maxErr, std::abs(double(gpu[i].x - ref[i].x)), std::abs(double(gpu[i].y - ref[i].y)),
                           std::abs(double(gpu[i].z - ref[i].z)), std::abs(double(gpu[i].w - ref[i].w))});
    }
    std::ostringstream sci;
    sci << std::scientific << maxErr;
    RecordProperty("gpu_vs_cpu_max_abs_err", sci.str());
    EXPECT_LT(maxErr, 2e-4) << "GPU bands must match the CPU reference (float rounding only)";
}

TEST_F(V15FreqSplitTest, ExposureChangeMovesOnlyLowBandHighAndGradientsInvariant) {
    const auto bright = runGpu(*s_ctx, syntheticScene(1.0f));
    const auto dark = runGpu(*s_ctx, syntheticScene(0.25f));
    const int margin = v15::FreqSplitParams{}.radius() + 1;
    double sumHigh = 0.0, maxHigh = 0.0, sumLowShift = 0.0, maxLowDev = 0.0, sumHighAbs = 0.0;
    double gradRatioSum = 0.0, linRatioSum = 0.0;
    int n = 0, nEdge = 0;
    const auto lin = syntheticScene(1.0f);
    const auto linDark = syntheticScene(0.25f);
    for (int y = margin; y < kH - margin; ++y) {
        for (int x = margin; x < kW - margin; ++x) {
            const size_t i = static_cast<size_t>(y * kW + x);
            const double dh = std::abs(double(bright[i].y - dark[i].y));
            sumHigh += dh;
            maxHigh = std::max(maxHigh, dh);
            sumHighAbs += std::abs(bright[i].y);
            const double shift = double(bright[i].x - dark[i].x);
            sumLowShift += shift;
            maxLowDev = std::max(maxLowDev, std::abs(shift - std::log(4.0)));
            ++n;
            if (std::abs(bright[i].z) > 0.2f) { // checker edge column
                gradRatioSum += dark[i].z / bright[i].z;
                const size_t l = i - 1, rr = i + 1;
                linRatioSum += (lin[rr].x - lin[l].x) / (linDark[rr].x - linDark[l].x);
                ++nEdge;
            }
        }
    }
    const double meanHighDiff = sumHigh / n, meanShift = sumLowShift / n;
    const double gradRatio = gradRatioSum / std::max(1, nEdge), linRatio = linRatioSum / std::max(1, nEdge);
    RecordProperty("mean_abs_high_band", fmt(sumHighAbs / n));
    RecordProperty("high_band_mean_abs_diff", fmt(meanHighDiff));
    RecordProperty("high_band_max_abs_diff", fmt(maxHigh));
    RecordProperty("low_band_mean_shift", fmt(meanShift));
    RecordProperty("low_band_max_dev_from_ln4", fmt(maxLowDev));
    RecordProperty("log_grad_ratio_dark_over_bright", fmt(gradRatio));
    RecordProperty("linear_grad_ratio_bright_over_dark", fmt(linRatio));
    RecordProperty("edge_pixels", nEdge);

    ASSERT_GT(nEdge, 50);
    EXPECT_GT(sumHighAbs / n, 0.3) << "High band must carry the checker detail (|log 3|/2 ~ 0.55)";
    EXPECT_LT(meanHighDiff, 0.02) << "High band must be exposure invariant (residual from logEps only)";
    EXPECT_LT(maxHigh, 0.05);
    EXPECT_NEAR(meanShift, std::log(4.0), 0.03) << "A 4x exposure drop must appear as log(4) in the low band";
    EXPECT_LT(maxLowDev, 0.06);
    EXPECT_NEAR(gradRatio, 1.0, 0.10) << "Log-gradients at edges must be exposure invariant";
    EXPECT_NEAR(linRatio, 4.0, 0.05) << "Reference: linear-domain gradients scale with exposure";
}

TEST_F(V15FreqSplitTest, RealV15RenderProducesFiniteZeroMeanDetailBand) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(*s_ctx, cfg, world);
    rig.renderAllCameras(world, cfg, 0.0f, nullptr);
    v15::FrequencySplit fs;
    fs.init(*s_ctx, rig.cameraLinearSSBO(), rig.camWidth(), rig.camHeight(), 8);
    VkCommandBuffer cmd = s_ctx->beginSingleTimeCommands();
    fs.recordDispatch(cmd);
    s_ctx->endSingleTimeCommands(cmd);
    std::vector<Vec4> bands(static_cast<size_t>(8 * rig.camWidth() * rig.camHeight()));
    fs.bandsSSBO().downloadVector(bands);
    double sumHigh = 0.0, sumAbsGrad = 0.0;
    size_t finite = 0;
    for (const auto& b : bands) {
        if (std::isfinite(b.x) && std::isfinite(b.y) && std::isfinite(b.z) && std::isfinite(b.w)) ++finite;
        sumHigh += b.y;
        sumAbsGrad += std::hypot(b.z, b.w);
    }
    RecordProperty("real_mean_high", fmt(sumHigh / bands.size()));
    RecordProperty("real_mean_log_grad", fmt(sumAbsGrad / bands.size()));
    EXPECT_EQ(finite, bands.size());
    EXPECT_LT(std::abs(sumHigh / bands.size()), 0.05) << "High band is a residual: mean must be ~0";
    fs.destroy();
    rig.destroy();
}

} // namespace
} // namespace cybercab
