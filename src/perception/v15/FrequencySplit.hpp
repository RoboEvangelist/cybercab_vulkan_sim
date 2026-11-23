#pragma once
// FSD v15 Module 1c: pre-network frequency separation of the linear photon tensor.
// Output bands per pixel: vec4(low = G*logY, high = logY - low, d/dx logY, d/dy logY).
// See shaders/v15_freq_split.comp and docs/v15/step04_m1c_frequency_split.md.
#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "sensors/PhotonModel.hpp"

#include <algorithm>
#include <cmath>

namespace cybercab::v15 {

struct FreqSplitParams {
    float sigmaPx = 4.0f;     // illumination scale (Gaussian std-dev, pixels)
    float logEps = 1.0e-3f;   // radiance floor inside log (~ read-noise level, avoids log(0))
    float fullWell = photon::kFullWellRadiance; // radiance at code 1.0 (single source of truth)
    int radius() const { return static_cast<int>(3.0f * sigmaPx + 0.5f); }

    // Scales sigmaPx with image width (320px baseline -> sigma 4px) and ties logEps to 2*sigma_read.
    static FreqSplitParams forConfig(int camW, const SimConfig& cfg) {
        FreqSplitParams p;
        p.sigmaPx = 4.0f * std::max(1.0f, static_cast<float>(camW) / 320.0f);
        const float readRad = std::sqrt(std::max(0.0f, cfg.photonReadNoiseB)) /
                              photon::kAdcMaxCode * photon::kFullWellRadiance;
        p.logEps = std::max(1.0e-3f, 2.0f * readRad);
        return p;
    }
};

class FrequencySplit {
public:
    FrequencySplit() = default;
    ~FrequencySplit();
    FrequencySplit(const FrequencySplit&) = delete;
    FrequencySplit& operator=(const FrequencySplit&) = delete;

    void init(const VulkanContext& ctx, const VulkanBuffer& linearIn, int camW, int camH, int numCams,
              const FreqSplitParams& params = {});
    // Re-allocates outputs, rebinds the input, and optionally updates params (e.g. after resize).
    void rebind(const VulkanBuffer& linearIn, int camW, int camH);
    void rebind(const VulkanBuffer& linearIn, int camW, int camH, const FreqSplitParams& params);
    void destroy();

    // Records both passes plus trailing compute barriers; bands are ready for the next stage.
    void recordDispatch(VkCommandBuffer cmd) const;

    const VulkanBuffer& bandsSSBO() const { return bands_; }
    const FreqSplitParams& params() const { return params_; }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    const VulkanContext* ctx_ = nullptr;
    FreqSplitParams params_{};
    int w_ = 0, h_ = 0, cams_ = 0;
    VulkanComputePipeline pipe_;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VulkanBuffer tmp_;
    VulkanBuffer bands_;
};

} // namespace cybercab::v15
