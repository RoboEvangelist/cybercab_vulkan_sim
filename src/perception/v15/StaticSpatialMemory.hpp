#pragma once
// FSD v15 Module 5: Persistent SE(3) ego-warped static spatial memory (US 2026/0285363 A1).
// Warps 60x40 BEV grid via BEVFormer::worldFromIsoEgo, lifts Plucker tokens, and skips locked rewrites.
#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/v15/PatchTokenizer.hpp"
#include "perception/v15/VirtualCameraRectifier.hpp"
#include <cstdint>
#include <vector>

namespace cybercab::v15 {

struct StaticMemStats {
    uint32_t refreshedCount = 0, skippedCount = 0, lockedCount = 0, dynamicMaskedCount = 0;
    float selectiveRefreshSkipRatio = 0.0f;
};

struct alignas(16) StaticCellGpu {
    Vec4 featConf; // occProb, maxHeightM, semanticF, confidence
    Vec4 meta;     // ageSec, viewConsensus, staticLocked, refreshFlag
};

class StaticSpatialMemory {
public:
    StaticSpatialMemory() = default;
    ~StaticSpatialMemory();
    StaticSpatialMemory(const StaticSpatialMemory&) = delete;
    StaticSpatialMemory& operator=(const StaticSpatialMemory&) = delete;

    void init(const VulkanContext& ctx, const SimConfig& cfg,
              const VirtualCameraRectifier& rectifier, const PatchTokenizer& patches);
    void reset();
    void destroy();
    void recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg, const EgoState& ego, float dt,
                        const VirtualCameraRectifier& rectifier, const PatchTokenizer& patches,
                        const std::vector<TrackPriorGpu>& tracks);
    StaticMemStats readbackStats() const;
    void readbackCells(std::vector<StaticCellGpu>& out) const;

    const VulkanBuffer& currMemSSBO() const { return currMem_; }
    const VulkanBuffer& statsSSBO() const { return statsSSBO_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }

private:
    struct alignas(16) VirtViewCamGpu { Mat4 camFromIsoEgo; Vec4 intr; };
    struct StaticMemPC {
        Mat4 prevIsoFromCurrIso;
        Vec4 zAnchors;
        uint32_t nx, ny, pass, hasHistory, currSlot, numViews, gridW, gridH, numTracks;
        float minX, maxX, minY, maxY, dt, lockThresh, decayRate;
    };
    const VulkanContext* ctx_ = nullptr;
    int nx_ = 60, ny_ = 40;
    bool hasPrevEgo_ = false;
    EgoState prevEgo_{};
    VulkanComputePipeline pipe_;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VulkanBuffer prevMem_, warpedMem_, currMem_, viewsSSBO_, tracksSSBO_, statsSSBO_;
};
} // namespace cybercab::v15
