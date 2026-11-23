#pragma once
// FSD v15 Module 1c: pre-network frequency separation of the linear photon tensor.
// Output bands per pixel: vec4(low = G*logY, high = logY - low, d/dx logY, d/dy logY).
// See shaders/v15_freq_split.comp and docs/v15/step04_m1c_frequency_split.md.
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "sensors/PhotonModel.hpp"

namespace cybercab::v15 {

struct FreqSplitParams {
    float sigmaPx = 4.0f;     // illumination scale (Gaussian std-dev, pixels)
    float logEps = 1.0e-3f;   // radiance floor inside log (~ read-noise level, avoids log(0))
    float fullWell = photon::kFullWellRadiance; // radiance at code 1.0 (single source of truth)
    int radius() const { return static_cast<int>(3.0f * sigmaPx + 0.5f); }
};

class FrequencySplit {
public:
    FrequencySplit() = default;
    ~FrequencySplit();
    FrequencySplit(const FrequencySplit&) = delete;
    FrequencySplit& operator=(const FrequencySplit&) = delete;

    void init(const VulkanContext& ctx, const VulkanBuffer& linearIn, int camW, int camH, int numCams,
              const FreqSplitParams& params = {});
    // Re-allocates outputs and rebinds the input after a camera resize.
    void rebind(const VulkanBuffer& linearIn, int camW, int camH);
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
