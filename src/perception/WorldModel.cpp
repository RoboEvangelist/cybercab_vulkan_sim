#include "perception/WorldModel.hpp"

namespace cybercab {

namespace {

struct alignas(16) WorldModelPC {
    Vec4 horizons;
    uint32_t numTracks;
    uint32_t maxGhosts;
    float uncertaintyGrowthRate;
    float egoX;
    float egoZ;
    float egoYaw;
};

} // namespace

WorldModel::~WorldModel() {
    destroy();
}

void WorldModel::init(const VulkanContext& ctx, const SimConfig& cfg, const VulkanBuffer& trackPriorsSSBO) {
    destroy();
    ctx_ = &ctx;

    pipe_.create(*ctx_, "shaders/06_world_model.comp.spv",
                 {
                     {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                     {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                     {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                 },
                 sizeof(WorldModelPC));

    ghostCounterSSBO_.create(*ctx_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    worldGhostsSSBO_.create(*ctx_, static_cast<VkDeviceSize>(kMaxGhosts) * sizeof(WorldGhost),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    set_ = pipe_.allocateDescriptorSet();
    VkDescriptorBufferInfo b0 = trackPriorsSSBO.descriptorInfo();
    VkDescriptorBufferInfo b1 = ghostCounterSSBO_.descriptorInfo();
    VkDescriptorBufferInfo b2 = worldGhostsSSBO_.descriptorInfo();
    VkWriteDescriptorSet writes[3]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b0, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b1, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b2, nullptr},
    };
    vkUpdateDescriptorSets(ctx_->device(), 3, writes, 0, nullptr);
    (void)cfg;
}

void WorldModel::destroy() {
    if (!ctx_) return;
    pipe_.destroy();
    ghostCounterSSBO_.destroy();
    worldGhostsSSBO_.destroy();
    ctx_ = nullptr;
}

void WorldModel::recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg, const EgoState& ego, uint32_t trackBufSize) {
    uint32_t zero = 0;
    ghostCounterSSBO_.upload(&zero, sizeof(zero));

    Vec4 horizons{0.5f, 1.0f, 1.5f, 2.0f};
    if (cfg.wmHorizonsSec.size() >= 4) {
        horizons = {cfg.wmHorizonsSec[0], cfg.wmHorizonsSec[1], cfg.wmHorizonsSec[2], cfg.wmHorizonsSec[3]};
    }
    WorldModelPC pc{
        horizons,
        trackBufSize,
        static_cast<uint32_t>(kMaxGhosts),
        cfg.wmUncertaintyGrowthRate,
        ego.pos.x, ego.pos.z, ego.yaw
    };
    uint32_t gx = (trackBufSize + 15) / 16;
    pipe_.bindAndDispatch(cmd, set_, gx, 1, 1, &pc, sizeof(pc));
    cmdComputeBarrier(cmd);
}

void WorldModel::readbackGhosts(std::vector<WorldGhost>& out) const {
    uint32_t cnt = 0;
    ghostCounterSSBO_.download(&cnt, sizeof(cnt));
    cnt = std::min<uint32_t>(cnt, kMaxGhosts);
    out.resize(cnt);
    if (cnt > 0) {
        worldGhostsSSBO_.download(out.data(), static_cast<VkDeviceSize>(cnt) * sizeof(WorldGhost), 0);
    }
}

} // namespace cybercab
