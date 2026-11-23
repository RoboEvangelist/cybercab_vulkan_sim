// FSD v15 Module 1b: Poisson-Gaussian sensor noise on the 14-bit linear photon tensor.
// Model (Foi et al. 2008): var_DN(mu) = a*mu + b. Validated empirically on the GPU output:
// slope/intercept recovery, unbiasedness, determinism, and spatial/temporal/channel independence.
#include "config/SimConfig.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanImage.hpp"
#include "core/VulkanPipeline.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "sensors/PhotonModel.hpp"
#include "world/WorldEngine.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cmath>
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

TEST(V15PhotonNoiseModelTest, PhysicalSensorParamsMapToConfigDefaults) {
    const photon::NoiseParams p = photon::noiseParamsFromPhysical(40000.0f, 3.0f);
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    EXPECT_NEAR(p.a, cfg.photonShotNoiseA, 1e-3f) << "a = 1/gain must match 40000 e- full well";
    EXPECT_NEAR(p.b, cfg.photonReadNoiseB, 1e-2f) << "b = (3 e- / gain)^2";
    EXPECT_NEAR(photon::noiseVarianceDn(1000.0f, p), 0.4096f * 1000.0f + 1.51f, 0.05f);
    EXPECT_NEAR(photon::noiseVarianceDn(-5.0f, p), p.b, 1e-6f) << "negative means clamp to read-noise floor";
}

struct NoiseRig {
    SimConfig cfg;
    VulkanContext ctx;
    WorldEngine* world = nullptr;
    CameraRigRenderer rig;
    size_t n = 0;

    NoiseRig() : cfg(SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"))) {
        cfg.pipelineVersion = "v15";
        ctx.init(false, nullptr);
        world = new WorldEngine(cfg);
        rig.init(ctx, cfg, *world);
        n = static_cast<size_t>(8 * rig.camWidth() * rig.camHeight());
    }
    ~NoiseRig() {
        rig.destroy();
        delete world;
        ctx.cleanup();
    }
    std::vector<Vec4> capture(bool noise, float timeSec) {
        cfg.photonNoiseEnabled = noise;
        rig.renderAllCameras(*world, cfg, timeSec, nullptr);
        std::vector<Vec4> out(n);
        rig.cameraLinearSSBO().downloadVector(out);
        return out;
    }
};

float ch(const Vec4& v, int c) { return c == 0 ? v.x : (c == 1 ? v.y : v.z); }

TEST(V15PhotonNoiseGpuTest, VarianceFollowsPoissonGaussianModelAndIsUnbiased) {
    NoiseRig r;
    const auto clean = r.capture(false, 0.0f);
    const auto noisy = r.capture(true, 0.0f);
    const double A = photon::kAdcMaxCode;

    const std::array<double, 8> edges{20, 60, 150, 300, 600, 1000, 1600, 2600};
    std::array<double, 7> sMu{}, sR{}, sR2{};
    std::array<size_t, 7> cnt{};
    for (size_t i = 0; i < r.n; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double mu = ch(clean[i], c) * A;
            const double res = ch(noisy[i], c) * A - mu;
            for (size_t b = 0; b + 1 < edges.size(); ++b) {
                if (mu >= edges[b] && mu < edges[b + 1]) {
                    sMu[b] += mu; sR[b] += res; sR2[b] += res * res; ++cnt[b];
                    break;
                }
            }
        }
    }
    // Least-squares fit var = a_hat * mu + b_hat over populated bins; quantization adds ~1/6 DN^2.
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int used = 0;
    std::ostringstream bins;
    for (size_t b = 0; b < cnt.size(); ++b) {
        if (cnt[b] < 2000) continue;
        const double mu = sMu[b] / cnt[b];
        const double mean = sR[b] / cnt[b];
        const double var = sR2[b] / cnt[b] - mean * mean;
        const double expect = r.cfg.photonShotNoiseA * mu + r.cfg.photonReadNoiseB + 1.0 / 6.0;
        bins << "[mu=" << fmt(mu) << " n=" << cnt[b] << " var=" << fmt(var) << " exp=" << fmt(expect)
             << " bias=" << fmt(mean) << "] ";
        EXPECT_NEAR(var / expect, 1.0, 0.08) << "Bin variance off model at mu=" << mu;
        EXPECT_LT(std::abs(mean), 0.15) << "Noise must be zero-mean away from the black clip, mu=" << mu;
        sx += mu; sy += var; sxx += mu * mu; sxy += mu * var; ++used;
    }
    ASSERT_GE(used, 4) << "Need >= 4 populated intensity bins: " << bins.str();
    const double aHat = (used * sxy - sx * sy) / (used * sxx - sx * sx);
    const double bHat = (sy - aHat * sx) / used;
    RecordProperty("fit_a_hat", fmt(aHat));
    RecordProperty("fit_b_hat", fmt(bHat));
    RecordProperty("bins", bins.str());
    RecordProperty("snr_db_at_1000dn", fmt(20.0 * std::log10(1000.0 / std::sqrt(photon::noiseVarianceDn(
        1000.0f, {r.cfg.photonShotNoiseA, r.cfg.photonReadNoiseB})))));
    EXPECT_NEAR(aHat, r.cfg.photonShotNoiseA, 0.08 * r.cfg.photonShotNoiseA) << bins.str();
    EXPECT_NEAR(bHat, r.cfg.photonReadNoiseB + 1.0 / 6.0, 3.0) << bins.str();
}

TEST(V15PhotonNoiseGpuTest, DeterministicPerFrameAndIndependentAcrossTimeAndChannels) {
    NoiseRig r;
    const auto clean = r.capture(false, 0.0f);
    const auto n0a = r.capture(true, 0.0f);
    const auto n0b = r.capture(true, 0.0f);
    const auto n1 = r.capture(true, 0.1f);
    const double A = photon::kAdcMaxCode;

    size_t identical = 0, differ = 0, bright = 0;
    double sTT = 0, sT0 = 0, sT1 = 0, sRG = 0, sR = 0, sG = 0, sRR = 0, sGG = 0;
    size_t m = 0;
    for (size_t i = 0; i < r.n; ++i) {
        if (n0a[i].x == n0b[i].x && n0a[i].y == n0b[i].y && n0a[i].z == n0b[i].z) ++identical;
        const double mu = clean[i].y * A;
        if (mu < 100.0 || clean[i].x * A < 100.0) continue;
        ++bright;
        if (n0a[i].y != n1[i].y) ++differ;
        const double e0 = n0a[i].y * A - mu, e1 = n1[i].y * A - mu;
        const double er = n0a[i].x * A - clean[i].x * A;
        sTT += e0 * e1; sT0 += e0 * e0; sT1 += e1 * e1;
        sRG += er * e0; sR += er; sG += e0; sRR += er * er; sGG += e0 * e0;
        ++m;
    }
    ASSERT_GT(m, 10000u);
    const double rhoTime = sTT / std::sqrt(sT0 * sT1);
    const double rhoChan = (sRG / m - (sR / m) * (sG / m)) /
                           std::sqrt((sRR / m - (sR / m) * (sR / m)) * (sGG / m - (sG / m) * (sG / m)));
    RecordProperty("rho_time", fmt(rhoTime));
    RecordProperty("rho_channel_rg", fmt(rhoChan));
    EXPECT_EQ(identical, r.n) << "Same (seed, timeSec) must reproduce the tensor bit-exactly";
    EXPECT_GT(static_cast<double>(differ) / bright, 0.90) << "New frame time must draw new noise";
    EXPECT_LT(std::abs(rhoTime), 0.02) << "Temporal noise must be uncorrelated";
    EXPECT_LT(std::abs(rhoChan), 0.02) << "R/G noise must be uncorrelated";
}

TEST(V15PhotonNoiseGpuTest, DisabledNoiseEqualsNoiselessAdc) {
    NoiseRig r;
    const auto a = r.capture(false, 0.0f);
    const auto b = r.capture(false, 0.37f);
    size_t same = 0;
    for (size_t i = 0; i < r.n; ++i) {
        if (a[i].x == b[i].x && a[i].y == b[i].y && a[i].z == b[i].z) ++same;
    }
    EXPECT_EQ(same, r.n) << "With noise off the tensor must not depend on frame time";
}

TEST(V15PhotonNoiseGpuTest, DarkScenePinsReadNoiseInterceptB) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    VulkanContext ctx;
    ctx.init(false, nullptr);
    constexpr int kS = 128;
    struct IspPC { uint32_t c, w, h, soil; Vec4 dist, intr; float rs, t; uint32_t lin; float fw, a, b; uint32_t seed, en; };
    VulkanComputePipeline pipe;
    pipe.create(ctx, "shaders/cam_isp_soil.comp.spv",
                {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                 {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                 {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                 {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1}}, sizeof(IspPC));
    VulkanImage raw, dst;
    raw.create(ctx, kS, kS, VK_FORMAT_R16G16B16A16_SFLOAT,
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, true);
    dst.create(ctx, kS, kS, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false);
    dst.transitionLayoutImmediate(VK_IMAGE_LAYOUT_GENERAL);
    VulkanBuffer dispBuf, linBuf;
    const size_t n = kS * kS;
    dispBuf.create(ctx, n * sizeof(Vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    linBuf.create(ctx, n * sizeof(Vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDescriptorSet set = pipe.allocateDescriptorSet();
    VkDescriptorImageInfo i0{raw.sampler(), raw.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo i1{VK_NULL_HANDLE, dst.view(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo b2 = dispBuf.descriptorInfo(), b3 = linBuf.descriptorInfo();
    VkWriteDescriptorSet w[4]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &i0, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &i1, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b2, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b3, nullptr},
    };
    vkUpdateDescriptorSets(ctx.device(), 4, w, 0, nullptr);

    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    const float darkDns[4] = {8.0f, 16.0f, 24.0f, 36.0f};
    std::vector<Vec4> clean(n), noisy(n);
    for (float dn : darkDns) {
        const float r = dn / photon::kAdcMaxCode * photon::kFullWellRadiance;
        const VkClearColorValue clr{{r, r, r, 1.0f}};
        const VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (int pass = 0; pass < 2; ++pass) {
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            VulkanImage::cmdTransitionLayout(cmd, raw.handle(), VK_IMAGE_ASPECT_COLOR_BIT,
                                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdClearColorImage(cmd, raw.handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &rng);
            VulkanImage::cmdTransitionLayout(cmd, raw.handle(), VK_IMAGE_ASPECT_COLOR_BIT,
                                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            raw.setLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            IspPC pc{0, kS, kS, 0, {}, {140, 140, kS * 0.5f, kS * 0.5f}, 1.0f, 0.0f,
                     1u, photon::kFullWellRadiance, cfg.photonShotNoiseA, cfg.photonReadNoiseB, 42u, uint32_t(pass)};
            pipe.bindAndDispatch(cmd, set, kS / 16, kS / 16, 1, &pc, sizeof(pc));
            ctx.endSingleTimeCommands(cmd);
            linBuf.downloadVector(pass == 0 ? clean : noisy);
        }
        double sMu = 0, sE = 0, sE2 = 0;
        for (size_t i = 0; i < n; ++i) {
            for (int c = 0; c < 3; ++c) {
                const double mu = ch(clean[i], c) * photon::kAdcMaxCode;
                const double e = ch(noisy[i], c) * photon::kAdcMaxCode - mu;
                sMu += mu; sE += e; sE2 += e * e;
            }
        }
        const double cnt = static_cast<double>(n * 3);
        const double mu = sMu / cnt, mean = sE / cnt, var = sE2 / cnt - mean * mean;
        sx += mu; sy += var; sxx += mu * mu; sxy += mu * var;
    }
    const double aHat = (4 * sxy - sx * sy) / (4 * sxx - sx * sx);
    const double bHat = (sy - aHat * sx) / 4.0;
    const double bExpect = cfg.photonReadNoiseB + 1.0 / 6.0;
    RecordProperty("dark_fit_a_hat", fmt(aHat));
    RecordProperty("dark_fit_b_hat", fmt(bHat));
    RecordProperty("dark_b_expect", fmt(bExpect));
    EXPECT_NEAR(aHat, cfg.photonShotNoiseA, 0.04);
    EXPECT_NEAR(bHat, bExpect, 0.35);

    pipe.destroy(); dispBuf.destroy(); linBuf.destroy(); raw.destroy(); dst.destroy();
    ctx.cleanup();
}

} // namespace
} // namespace cybercab
