#pragma once
// FSD v15 Module 3b: Spatiotemporal 5x5 Plucker patch tokenizer with 4-frame deltaT ring buffer.
// Partitions 10 160x120 virtual views into 32x24 patches = 7,680 tokens/frame.
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/v15/VirtualCameraRectifier.hpp"
#include "world/WorldTypes.hpp"

#include <array>
#include <cstdint>

namespace cybercab::v15 {

struct alignas(16) PatchTokenGpu {
    Vec4 app0;     // meanR, meanG, meanB, validFrac
    Vec4 app1;     // meanLow, rmsHigh, meanGx, meanGy
    Vec4 pluckerD; // d_veh.xyz, float(viewIdx)
    Vec4 pluckerM; // cross(o_veh, d_veh).xyz, deltaT_sec
};

struct alignas(16) VirtPluckerSpecGpu {
    Mat4 uprightVehFromVirtCam;
    Vec4 virtIntr; // fx, fy, cx, cy
};

class PatchTokenizer {
public:
    static constexpr int kPatchPx = 5;
    static constexpr int kGridW = VirtualCameraRectifier::kVirtW / kPatchPx; // 32
    static constexpr int kGridH = VirtualCameraRectifier::kVirtH / kPatchPx; // 24
    static constexpr int kNumViews = VirtualCameraRectifier::kNumViews;      // 10
    static constexpr int kTokensPerFrame = kNumViews * kGridW * kGridH;      // 7680
    static constexpr int kHistoryFrames = 4;

    PatchTokenizer() = default;
    ~PatchTokenizer();
    PatchTokenizer(const PatchTokenizer&) = delete;
    PatchTokenizer& operator=(const PatchTokenizer&) = delete;

    void init(const VulkanContext& ctx, const VirtualCameraRectifier& rectifier);
    void reset();
    void destroy();

    void updateSpecs(const VirtualCameraRectifier& rectifier);
    void recordDispatch(VkCommandBuffer cmd, float dt);

    const VulkanBuffer& ringSSBO() const { return ringSSBO_; }
    uint32_t headSlot() const { return headSlot_; }
    uint32_t latestWrittenSlot() const { return lastWrittenSlot_; }
    uint32_t validSlots() const { return validSlots_; }
    const std::array<EgoState, kHistoryFrames>& slotEgoStates() const { return slotEgos_; }
    void recordSlotEgo(const EgoState& ego) { slotEgos_[lastWrittenSlot_] = ego; }

private:
    struct PatchPC {
        uint32_t gridW, gridH, virtW, virtH;
        uint32_t numViews, patchPx, headSlot, validSlots;
        float dt;
    };

    const VulkanContext* ctx_ = nullptr;
    VulkanComputePipeline pipe_;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VulkanBuffer specsSSBO_, ringSSBO_;
    uint32_t headSlot_ = 0, lastWrittenSlot_ = 0, validSlots_ = 0;
    std::array<EgoState, kHistoryFrames> slotEgos_{};
};

} // namespace cybercab::v15
