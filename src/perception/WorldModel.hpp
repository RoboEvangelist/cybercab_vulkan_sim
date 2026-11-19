#pragma once

#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldTypes.hpp"
#include <vector>

namespace cybercab {

class WorldModel {
public:
    static constexpr int kMaxGhosts = 640;

    WorldModel() = default;
    ~WorldModel();

    void init(const VulkanContext& ctx, const SimConfig& cfg, const VulkanBuffer& trackPriorsSSBO);
    void destroy();

    void recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg, const EgoState& ego, uint32_t trackBufSize);
    void readbackGhosts(std::vector<WorldGhost>& out) const;

private:
    const VulkanContext* ctx_ = nullptr;
    VulkanComputePipeline pipe_;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VulkanBuffer ghostCounterSSBO_;
    VulkanBuffer worldGhostsSSBO_;
};

} // namespace cybercab
