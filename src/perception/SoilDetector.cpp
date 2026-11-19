#include "perception/SoilDetector.hpp"
#include <algorithm>
#include <cmath>

namespace cybercab {

namespace {

struct SoilPushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t patchCols;
    uint32_t patchRows;
    float edgeThresh;
    float glareThresh;
    float minTokenWeight;
    float baseEpistemicVar;
    float soilVarGain;
};

} // namespace

SoilDetector::~SoilDetector() {
    destroy();
}

void SoilDetector::init(const VulkanContext& ctx, const SimConfig& cfg, const VulkanBuffer& camTensorSSBO) {
    destroy();
    ctx_ = &ctx;
    patchCols_ = cfg.soilPatchCols;
    patchRows_ = cfg.soilPatchRows;

    pipeline_.create(*ctx_, "shaders/01_soil_detect.comp.spv",
                     {
                         {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                         {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                         {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                     },
                     sizeof(SoilPushConstants));

    const VkDeviceSize patchBytes = static_cast<VkDeviceSize>(8 * patchCols_ * patchRows_) * 4 * sizeof(float);
    patchSSBO_.create(*ctx_, patchBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    patchReadbackScratch_.resize(static_cast<size_t>(8 * patchCols_ * patchRows_));

    camSummarySSBO_.create(*ctx_, 8 * 4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    descSet_ = pipeline_.allocateDescriptorSet();
    updateCameraTensorBinding(camTensorSSBO);
}

void SoilDetector::updateCameraTensorBinding(const VulkanBuffer& camTensorSSBO) {
    if (!ctx_ || descSet_ == VK_NULL_HANDLE) return;
    VkDescriptorBufferInfo b0 = camTensorSSBO.descriptorInfo();
    VkDescriptorBufferInfo b1 = patchSSBO_.descriptorInfo();
    VkDescriptorBufferInfo b2 = camSummarySSBO_.descriptorInfo();

    VkWriteDescriptorSet writes[3]{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b0, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b1, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b2, nullptr};
    vkUpdateDescriptorSets(ctx_->device(), 3, writes, 0, nullptr);
}

void SoilDetector::destroy() {
    if (!ctx_) return;
    pipeline_.destroy();
    patchSSBO_.destroy();
    camSummarySSBO_.destroy();
    ctx_ = nullptr;
}

void SoilDetector::recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg, int camW, int camH) const {
    SoilPushConstants pc{
        static_cast<uint32_t>(camW),
        static_cast<uint32_t>(camH),
        static_cast<uint32_t>(patchCols_),
        static_cast<uint32_t>(patchRows_),
        cfg.soilEdgeThreshold,
        cfg.soilGlareThreshold,
        cfg.soilMinTokenWeight,
        cfg.soilBaseEpistemicVar,
        cfg.soilVarGain
    };
    pipeline_.bindAndDispatch(cmd, descSet_, 1, 1, 8, &pc, sizeof(pc));
    cmdComputeBarrier(cmd);
}

void SoilDetector::readbackResults(std::array<SoilCameraSummary, 8>& outCameras,
                                   std::vector<SoilPatchStats>& outPatches,
                                   float& outForwardSigma) const {
    std::array<Vec4, 8> rawCam{};
    camSummarySSBO_.download(rawCam.data(), sizeof(rawCam));
    for (int c = 0; c < 8; ++c) {
        outCameras[c].camIndex = c;
        outCameras[c].meanSoilProb = rawCam[c].x;
        outCameras[c].tokenWeight = rawCam[c].y;
        outCameras[c].meanEpistemicVar = rawCam[c].z;
        outCameras[c].meanEdge = rawCam[c].w;
    }

    const size_t totalPatches = static_cast<size_t>(8 * patchCols_ * patchRows_);
    patchReadbackScratch_.resize(totalPatches);
    patchSSBO_.downloadVector(patchReadbackScratch_);
    outPatches.resize(totalPatches);
    for (size_t i = 0; i < totalPatches; ++i) {
        outPatches[i] = {patchReadbackScratch_[i].x, patchReadbackScratch_[i].y,
                         patchReadbackScratch_[i].z, patchReadbackScratch_[i].w};
    }

    // Forward epistemic uncertainty across Front Main (0), Front Narrow (1), Front Wide (2)
    float varSum = 0.0f;
    for (int c = 0; c < 3; ++c) {
        varSum += outCameras[c].meanEpistemicVar;
    }
    outForwardSigma = std::sqrt(varSum / 3.0f);
}

} // namespace cybercab
