#pragma once

#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionTypes.hpp"
#include <array>
#include <vector>

namespace cybercab {

class SoilDetector {
public:
    SoilDetector() = default;
    ~SoilDetector();

    void init(const VulkanContext& ctx, const SimConfig& cfg, const VulkanBuffer& camTensorSSBO);
    void updateCameraTensorBinding(const VulkanBuffer& camTensorSSBO);
    void destroy();

    void recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg, int camW, int camH) const;
    void readbackResults(std::array<SoilCameraSummary, 8>& outCameras,
                         std::vector<SoilPatchStats>& outPatches,
                         float& outForwardSigma) const;

    const VulkanBuffer& patchSSBO() const { return patchSSBO_; }
    const VulkanBuffer& camSummarySSBO() const { return camSummarySSBO_; }

private:
    const VulkanContext* ctx_ = nullptr;
    int patchCols_ = 10;
    int patchRows_ = 8;

    VulkanComputePipeline pipeline_;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;
    VulkanBuffer patchSSBO_;
    VulkanBuffer camSummarySSBO_;
    mutable std::vector<Vec4> patchReadbackScratch_;
};

} // namespace cybercab
