#include "perception/OccupancyFlowNet.hpp"

namespace cybercab {

namespace {

struct OccPC {
    uint32_t nx, ny, nz, numTracks;
    float minX, maxX, minY, maxY, minZ, maxZ;
    float decayRate, dt;
    float egoX, egoZ, egoYaw, occThresh;
};

struct ShadowPC {
    uint32_t nx, ny, nz;
    float minX, maxX, minY, maxY;
    uint32_t enableShadows;
};

} // namespace

OccupancyFlowNet::~OccupancyFlowNet() {
    destroy();
}

void OccupancyFlowNet::init(const VulkanContext& ctx, const SimConfig& cfg,
                            const VulkanBuffer& bevFeatSSBO,
                            const VulkanBuffer& bevMetaSSBO,
                            const VulkanBuffer& bevPillarSSBO) {
    destroy();
    ctx_ = &ctx;
    nx_ = cfg.occNx();
    ny_ = cfg.occNy();
    nz_ = cfg.occNz();

    occPipe_.create(*ctx_, "shaders/05_occupancy_flow.comp.spv",
                    {
                        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                    },
                    sizeof(OccPC));

    shadowPipe_.create(*ctx_, "shaders/05_occlusion_shadow.comp.spv",
                       {
                           {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                           {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                       },
                       sizeof(ShadowPC));

    const VkDeviceSize voxBytes = static_cast<VkDeviceSize>(nx_ * ny_ * nz_) * sizeof(Vec4);
    trackPriorsSSBO_.create(*ctx_, static_cast<VkDeviceSize>(cfg.btMaxTracks) * sizeof(TrackPriorGpu),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    voxelStateSSBO_.create(*ctx_, voxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    voxelUncertSSBO_.create(*ctx_, voxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    statesScratch_.resize(static_cast<size_t>(nx_ * ny_ * nz_));
    uncertScratch_.resize(static_cast<size_t>(nx_ * ny_ * nz_));

    occSet_ = occPipe_.allocateDescriptorSet();
    shadowSet_ = shadowPipe_.allocateDescriptorSet();

    VkDescriptorBufferInfo o0 = bevFeatSSBO.descriptorInfo();
    VkDescriptorBufferInfo o1 = bevMetaSSBO.descriptorInfo();
    VkDescriptorBufferInfo o2 = bevPillarSSBO.descriptorInfo();
    VkDescriptorBufferInfo o3 = trackPriorsSSBO_.descriptorInfo();
    VkDescriptorBufferInfo o4 = voxelStateSSBO_.descriptorInfo();
    VkDescriptorBufferInfo o5 = voxelUncertSSBO_.descriptorInfo();
    VkWriteDescriptorSet oWrites[6]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o0, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o1, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o2, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o3, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o4, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o5, nullptr},
    };
    vkUpdateDescriptorSets(ctx_->device(), 6, oWrites, 0, nullptr);

    VkWriteDescriptorSet sWrites[2]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, shadowSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o4, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, shadowSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o5, nullptr},
    };
    vkUpdateDescriptorSets(ctx_->device(), 2, sWrites, 0, nullptr);

    resetVoxels();
}

void OccupancyFlowNet::resetVoxels() {
    if (!ctx_) return;
    std::vector<Vec4> zeros(static_cast<size_t>(nx_ * ny_ * nz_), Vec4{0, 0, 0, 0});
    voxelStateSSBO_.uploadVector(zeros);
    voxelUncertSSBO_.uploadVector(zeros);
}

void OccupancyFlowNet::destroy() {
