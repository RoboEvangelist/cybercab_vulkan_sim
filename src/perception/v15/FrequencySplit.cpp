#include "perception/v15/FrequencySplit.hpp"

namespace cybercab::v15 {

namespace {
struct FreqPushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t numCams;
    uint32_t pass;
    float fullWell;
    float logEps;
    float sigmaPx;
    int32_t radius;
};
static_assert(sizeof(FreqPushConstants) == 32, "FreqPushConstants must match v15_freq_split.comp");
} // namespace

FrequencySplit::~FrequencySplit() {
    destroy();
}

void FrequencySplit::init(const VulkanContext& ctx, const VulkanBuffer& linearIn, int camW, int camH,
                          int numCams, const FreqSplitParams& params) {
    destroy();
    ctx_ = &ctx;
    params_ = params;
    cams_ = numCams;
    pipe_.create(*ctx_, "shaders/v15_freq_split.comp.spv",
                 {
                     {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                     {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                     {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                 },
                 sizeof(FreqPushConstants));
    set_ = pipe_.allocateDescriptorSet();
    rebind(linearIn, camW, camH);
}

void FrequencySplit::rebind(const VulkanBuffer& linearIn, int camW, int camH, const FreqSplitParams& params) {
    params_ = params;
    rebind(linearIn, camW, camH);
}

void FrequencySplit::rebind(const VulkanBuffer& linearIn, int camW, int camH) {
    if (!ctx_) return;
    w_ = camW;
    h_ = camH;
    const VkDeviceSize pixels = static_cast<VkDeviceSize>(cams_) * w_ * h_;
    tmp_.create(*ctx_, pixels * 2 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    bands_.create(*ctx_, pixels * 4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDescriptorBufferInfo infos[3] = {linearIn.descriptorInfo(), tmp_.descriptorInfo(), bands_.descriptorInfo()};
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t b = 0; b < 3; ++b) {
        writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[b].dstSet = set_;
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    vkUpdateDescriptorSets(ctx_->device(), 3, writes, 0, nullptr);
}

void FrequencySplit::destroy() {
    if (!ctx_) return;
    pipe_.destroy();
    tmp_.destroy();
    bands_.destroy();
    set_ = VK_NULL_HANDLE;
    ctx_ = nullptr;
}

void FrequencySplit::recordDispatch(VkCommandBuffer cmd) const {
    if (!ctx_) return;
    const uint32_t gx = static_cast<uint32_t>((w_ + 15) / 16);
    const uint32_t gy = static_cast<uint32_t>((h_ + 15) / 16);
    for (uint32_t pass = 0; pass < 2; ++pass) {
        const FreqPushConstants pc{static_cast<uint32_t>(w_), static_cast<uint32_t>(h_),
                                   static_cast<uint32_t>(cams_), pass, params_.fullWell,
                                   params_.logEps, params_.sigmaPx, params_.radius()};
        pipe_.bindAndDispatch(cmd, set_, gx, gy, static_cast<uint32_t>(cams_), &pc, sizeof(pc));
        cmdComputeBarrier(cmd);
    }
}

} // namespace cybercab::v15
