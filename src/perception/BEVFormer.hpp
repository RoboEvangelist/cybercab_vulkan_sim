#pragma once

#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionTypes.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "world/WorldTypes.hpp"
#include <array>

namespace cybercab {

class BEVFormer {
public:
    BEVFormer() = default;
    ~BEVFormer();

    void init(const VulkanContext& ctx, const SimConfig& cfg,
              const VulkanBuffer& encMemorySSBO, const VulkanBuffer& soilPatchSSBO);
    void resetHistory();
    void destroy();

    void recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg,
                        const CameraRigRenderer& rig,
                        const EgoState& ego, float dt,
                        const std::array<SoilCameraSummary, 8>& soilCameras);

    void copyCurrToPrev(VkCommandBuffer cmd) const;
    BevSummary readbackSummary() const;

    const VulkanBuffer& currBevFeatSSBO() const { return currBevFeatSSBO_; }
    const VulkanBuffer& currBevMetaSSBO() const { return currBevMetaSSBO_; }
    const VulkanBuffer& pillarHeightSSBO() const { return pillarHeightSSBO_; }

    static Mat4 worldFromIsoEgo(const EgoState& ego);

private:
    struct alignas(16) CamProjGpu {
        Mat4 camFromEgoISO;
        Vec4 intr; // fx, fy, cx, cy
        Vec4 dist; // k1, k2, p1, p2
        Vec4 dims; // width, height, maxRange, camWeight
    };

    const VulkanContext* ctx_ = nullptr;
    int nx_ = 60;
    int ny_ = 40;
    int featW_ = 40;
    int featH_ = 30;
    int patchCols_ = 10;
    int patchRows_ = 8;

    bool hasPrevEgo_ = false;
    EgoState prevEgo_{};

    VulkanComputePipeline temporalPipe_;
    VulkanComputePipeline spatialPipe_;
    VkDescriptorSet tempSet_ = VK_NULL_HANDLE;
    VkDescriptorSet spatSet_ = VK_NULL_HANDLE;

    VulkanBuffer camProjSSBO_;
    VulkanBuffer prevBevFeatSSBO_;
    VulkanBuffer prevBevMetaSSBO_;
    VulkanBuffer warpedBevFeatSSBO_;
    VulkanBuffer warpedBevMetaSSBO_;
    VulkanBuffer currBevFeatSSBO_;
    VulkanBuffer currBevMetaSSBO_;
    VulkanBuffer pillarHeightSSBO_;
    mutable std::vector<Vec4> metaReadbackScratch_;
};

} // namespace cybercab
