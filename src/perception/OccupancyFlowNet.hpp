#pragma once

#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldTypes.hpp"
#include <vector>

namespace cybercab {

class OccupancyFlowNet {
public:
    OccupancyFlowNet() = default;
    ~OccupancyFlowNet();

    void init(const VulkanContext& ctx, const SimConfig& cfg,
              const VulkanBuffer& bevFeatSSBO,
              const VulkanBuffer& bevMetaSSBO,
              const VulkanBuffer& bevPillarSSBO);
    void resetVoxels();
    void destroy();

    void recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg,
                        const EgoState& ego, float dt,
                        const std::vector<TrackPriorGpu>& tracks,
                        bool enableShadows);

    void readbackVoxels(const SimConfig& cfg, const EgoState& ego,
                        std::vector<ActiveVoxel>& outObserved,
                        std::vector<ActiveVoxel>& outShadows) const;

    const VulkanBuffer& voxelStateSSBO() const { return voxelStateSSBO_; }
    const VulkanBuffer& voxelUncertSSBO() const { return voxelUncertSSBO_; }
    const VulkanBuffer& trackPriorsSSBO() const { return trackPriorsSSBO_; }

private:
    const VulkanContext* ctx_ = nullptr;
    int nx_ = 60;
    int ny_ = 40;
    int nz_ = 8;

    VulkanComputePipeline occPipe_;
    VulkanComputePipeline shadowPipe_;
    VkDescriptorSet occSet_ = VK_NULL_HANDLE;
    VkDescriptorSet shadowSet_ = VK_NULL_HANDLE;

    VulkanBuffer trackPriorsSSBO_;
    VulkanBuffer voxelStateSSBO_;
    VulkanBuffer voxelUncertSSBO_;
    mutable std::vector<Vec4> statesScratch_;
    mutable std::vector<Vec4> uncertScratch_;
};

} // namespace cybercab
