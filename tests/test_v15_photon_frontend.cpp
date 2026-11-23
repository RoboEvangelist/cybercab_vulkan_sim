// FSD v15 Module 1a: ISP-bypass linear 14-bit photon-count tensor.
// Validates (1) the sensor model math (tutorial exercise 1.3), (2) v14 mode is untouched,
// (3) the v15 tensor is linear w.r.t. the display tensor and keeps highlight headroom the v14 path clips.
#include "config/SimConfig.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "sensors/PhotonModel.hpp"
#include "world/WorldEngine.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <vector>

namespace cybercab {
namespace {

std::string fmt(double v) {
    std::ostringstream os;
    os.precision(4);
    os << std::fixed << v;
    return os.str();
}

// Tutorial 1.3: radiance 12 vs 18 (of 16384 full scale) next to a 15000 highlight.
TEST(V15PhotonModelTest, Linear14BitPreservesDarkContrastThatGamma8Crushes) {
    const float fullScale = 16384.0f;
    auto toModel = [&](float counts) { return counts / fullScale * photon::kFullWellRadiance; };

    std::set<uint32_t> linCodes, gamCodes;
    double linMaxErr = 0.0, gamMaxErr = 0.0;
    for (float counts = 12.0f; counts <= 18.0f; counts += 1.0f) {
        const float r = toModel(counts);
        const uint32_t lc = photon::linearCode(r);
        const uint32_t gc = photon::gamma8Code(counts, fullScale);
        linCodes.insert(lc);
        gamCodes.insert(gc);
        linMaxErr = std::max<double>(linMaxErr, std::abs(photon::linearCodeToRadiance(lc) / photon::kFullWellRadiance * fullScale - counts));
        gamMaxErr = std::max<double>(gamMaxErr, std::abs(photon::gamma8CodeToRadiance(gc, fullScale) - counts));
    }
    RecordProperty("linear_distinct_codes_12_18", static_cast<int>(linCodes.size()));
    RecordProperty("gamma8_distinct_codes_12_18", static_cast<int>(gamCodes.size()));
    EXPECT_EQ(linCodes.size(), 7u) << "14-bit linear must resolve every count step 12..18";
    EXPECT_LE(gamCodes.size(), 4u) << "8-bit gamma should merge neighbouring dark levels";
    EXPECT_LE(linMaxErr, 0.51) << "Linear ADC reconstruction error must be <= half an LSB";
    EXPECT_GT(gamMaxErr, linMaxErr);

    // Contrast ratio 18/12 = 1.5 survives linear quantization; the 15000 highlight does not saturate.
    const double linRatio = static_cast<double>(photon::linearCode(toModel(18))) / photon::linearCode(toModel(12));
    EXPECT_NEAR(linRatio, 1.5, 0.06);
    EXPECT_LT(photon::linearCode(toModel(15000.0f)), static_cast<uint32_t>(photon::kAdcMaxCode));

    // Over a wide range the linear ADC error stays <= 0.5 LSB, gamma-8 error explodes in highlights.
    double gamWide = 0.0;
    for (float counts = 8.0f; counts <= 4096.0f; counts *= 1.25f) {
        gamWide = std::max<double>(gamWide, std::abs(photon::gamma8CodeToRadiance(photon::gamma8Code(counts, fullScale), fullScale) - counts));
    }
    EXPECT_GT(gamWide, 10.0);
}

TEST(V15PhotonGpuTest, V14ModeLeavesPhotonTensorUnallocatedAndRawClipped) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    ASSERT_FALSE(cfg.isV15());
    VulkanContext ctx;
    ctx.init(false, nullptr);
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    EXPECT_FALSE(rig.linearHdrEnabled());
    EXPECT_EQ(rig.cameraLinearSSBO().size(), 16u) << "v14 must not pay memory for the v15 tensor";
    rig.renderAllCameras(world, cfg, 0.0f, nullptr);
    std::vector<Vec4> disp(static_cast<size_t>(8 * rig.camWidth() * rig.camHeight()));
    rig.cameraTensorSSBO().downloadVector(disp);
    float maxV = 0.0f;
    for (const auto& p : disp) maxV = std::max({maxV, p.x, p.y, p.z});
    EXPECT_LE(maxV, 1.0f);
    rig.destroy();
    ctx.cleanup();
}

TEST(V15PhotonGpuTest, LinearTensorTracksDisplayAndKeepsHighlightHeadroom) {
    SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    cfg.pipelineVersion = "v15";
    cfg.photonNoiseEnabled = false; // M1a scope: noiseless 14-bit ADC (noise is validated in test_v15_photon_noise)
    VulkanContext ctx;
    ctx.init(false, nullptr);
    WorldEngine world(cfg);
    CameraRigRenderer rig;
    rig.init(ctx, cfg, world);
    ASSERT_TRUE(rig.linearHdrEnabled());
    const size_t n = static_cast<size_t>(8 * rig.camWidth() * rig.camHeight());
    ASSERT_EQ(rig.cameraLinearSSBO().size(), n * sizeof(Vec4));

    rig.renderAllCameras(world, cfg, 0.0f, nullptr);
    std::vector<Vec4> disp(n), lin(n);
    rig.cameraTensorSSBO().downloadVector(disp);
    rig.cameraLinearSSBO().downloadVector(lin);

    const float fw = photon::kFullWellRadiance;
    const float halfLsb = 0.5f * fw / photon::kAdcMaxCode;
    double sumErr = 0.0, maxErr = 0.0, maxRad = 0.0, minRad = 1e9;
    size_t unclipped = 0, dispClipped = 0, linClipped = 0, headroomPx = 0;
    for (size_t i = 0; i < n; ++i) {
        const float d[3] = {disp[i].x, disp[i].y, disp[i].z};
        const float l[3] = {lin[i].x * fw, lin[i].y * fw, lin[i].z * fw};
        const float dMax = std::max({d[0], d[1], d[2]});
        const float lMax = std::max({l[0], l[1], l[2]});
        ASSERT_TRUE(std::isfinite(lMax));
        ASSERT_GE(std::min({lin[i].x, lin[i].y, lin[i].z}), 0.0f);
        ASSERT_LE(std::max({lin[i].x, lin[i].y, lin[i].z}), 1.0f);
        maxRad = std::max<double>(maxRad, lMax);
        if (lMax > 1e-3f) minRad = std::min<double>(minRad, lMax);
        if (dMax >= 0.999f) ++dispClipped;
        if (std::max({lin[i].x, lin[i].y, lin[i].z}) >= 1.0f) ++linClipped;
        if (lMax > 1.0f) ++headroomPx;
        if (dMax < 0.98f) {
            for (int c = 0; c < 3; ++c) {
                const double e = std::abs(l[c] - d[c]);
                sumErr += e;
                maxErr = std::max(maxErr, e);
            }
            ++unclipped;
        }
    }
    const double meanErr = sumErr / std::max<size_t>(1, unclipped * 3);
    RecordProperty("lin_vs_display_mean_abs_err", fmt(meanErr));
    RecordProperty("lin_vs_display_max_abs_err", fmt(maxErr));
    RecordProperty("max_radiance", fmt(maxRad));
    RecordProperty("dynamic_range_db", fmt(20.0 * std::log10(maxRad / std::max(minRad, 1e-6))));
    RecordProperty("display_clipped_px", static_cast<int>(dispClipped));
    RecordProperty("linear_clipped_px", static_cast<int>(linClipped));
    RecordProperty("headroom_px_above_white", static_cast<int>(headroomPx));

    ASSERT_GT(unclipped, n / 2);
    EXPECT_LE(maxErr, halfLsb * 1.05) << "Below white, photon tensor must equal display radiance to 1/2 LSB";
    EXPECT_GT(maxRad, 1.02) << "Scene highlights/emissives must exceed v14 display white";
    EXPECT_GT(headroomPx, 0u);
    EXPECT_EQ(linClipped, 0u) << "Nominal daylight scene must not saturate the 16x full well";
    rig.destroy();
    ctx.cleanup();
}

} // namespace
} // namespace cybercab
