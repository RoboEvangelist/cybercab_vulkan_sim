#include "perception/v15/PatchTokenizer.hpp"

#include <algorithm>
#include <stdexcept>

namespace cybercab::v15 {

PatchTokenizer::~PatchTokenizer() { destroy(); }

void PatchTokenizer::init(const VulkanContext& ctx, const VirtualCameraRectifier& rectifier) {
    destroy();
    if (rectifier.virtLinearSSBO().handle() == VK_NULL_HANDLE ||
        rectifier.virtBandsSSBO().handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("PatchTokenizer::init requires initialized VirtualCameraRectifier");
    }
    ctx_ = &ctx;
    pipe_.create(ctx, "shaders/v15_patch_tokenize.comp.spv",
                 {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1}},
                 sizeof(PatchPC));
    set_ = pipe_.allocateDescriptorSet();

    const VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    specsSSBO_.create(ctx, static_cast<VkDeviceSize>(kNumViews) * sizeof(VirtPluckerSpecGpu),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    ringSSBO_.create(ctx, static_cast<VkDeviceSize>(kHistoryFrames) * kTokensPerFrame * sizeof(PatchTokenGpu),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);

    const VkDescriptorBufferInfo infos[4] = {
        rectifier.virtLinearSSBO().descriptorInfo(),
        rectifier.virtBandsSSBO().descriptorInfo(),
        specsSSBO_.descriptorInfo(),
        ringSSBO_.descriptorInfo(),
    };
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, i, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[i], nullptr};
    }
    vkUpdateDescriptorSets(ctx.device(), 4, writes, 0, nullptr);
    reset();
    updateSpecs(rectifier);
}

void PatchTokenizer::reset() {
    headSlot_ = 0;
    lastWrittenSlot_ = 0;
    validSlots_ = 0;
    slotEgos_ = {};
}

void PatchTokenizer::destroy() {
    if (!ctx_) return;
    ringSSBO_.destroy();
    specsSSBO_.destroy();
    pipe_.destroy();
    set_ = VK_NULL_HANDLE;
    ctx_ = nullptr;
    reset();
}

void PatchTokenizer::updateSpecs(const VirtualCameraRectifier& rectifier) {
    if (!ctx_) return;
    std::array<VirtPluckerSpecGpu, kNumViews> specs{};
    for (int v = 0; v < kNumViews; ++v) {
        const auto& vs = rectifier.viewSpec(v);
        specs[v] = {vs.uprightVehFromVirtCam, {vs.intr.fx, vs.intr.fy, vs.intr.cx, vs.intr.cy}};
    }
    specsSSBO_.upload(specs.data(), sizeof(specs));
}

void PatchTokenizer::recordDispatch(VkCommandBuffer cmd, float dt) {
    if (!ctx_) return;
    const PatchPC pc{
        static_cast<uint32_t>(kGridW),
        static_cast<uint32_t>(kGridH),
        static_cast<uint32_t>(VirtualCameraRectifier::kVirtW),
        static_cast<uint32_t>(VirtualCameraRectifier::kVirtH),
        static_cast<uint32_t>(kNumViews),
        static_cast<uint32_t>(kPatchPx),
        headSlot_,
        validSlots_,
        dt,
    };
    pipe_.bindAndDispatch(cmd, set_,
                          (static_cast<uint32_t>(kGridW) + 15u) / 16u,
                          (static_cast<uint32_t>(kGridH) + 15u) / 16u,
                          static_cast<uint32_t>(kNumViews), &pc, sizeof(pc));
    cmdComputeBarrier(cmd);
    lastWrittenSlot_ = headSlot_;
    headSlot_ = (headSlot_ + 1u) % static_cast<uint32_t>(kHistoryFrames);
    validSlots_ = std::min<uint32_t>(static_cast<uint32_t>(kHistoryFrames), validSlots_ + 1u);
}

} // namespace cybercab::v15
