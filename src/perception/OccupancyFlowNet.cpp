#include "perception/OccupancyFlowNet.hpp"
#include "perception/BEVFormer.hpp"

#include <algorithm>

namespace cybercab {

namespace {

struct alignas(16) OccPC {
    Mat4 prevEgoFromCurrEgo;
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
                        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
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
    prevVoxelStateSSBO_.create(*ctx_, voxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    prevVoxelUncertSSBO_.create(*ctx_, voxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    statesScratch_.resize(static_cast<size_t>(nx_ * ny_ * nz_));
    uncertScratch_.resize(static_cast<size_t>(nx_ * ny_ * nz_));

    occSet_ = occPipe_.allocateDescriptorSet();
    shadowSet_ = shadowPipe_.allocateDescriptorSet();

    const VkDescriptorBufferInfo o0 = bevFeatSSBO.descriptorInfo();
    const VkDescriptorBufferInfo o1 = bevMetaSSBO.descriptorInfo();
    const VkDescriptorBufferInfo o2 = bevPillarSSBO.descriptorInfo();
    const VkDescriptorBufferInfo o3 = trackPriorsSSBO_.descriptorInfo();
    const VkDescriptorBufferInfo o4 = voxelStateSSBO_.descriptorInfo();
    const VkDescriptorBufferInfo o5 = voxelUncertSSBO_.descriptorInfo();
    const VkDescriptorBufferInfo o6 = prevVoxelStateSSBO_.descriptorInfo();
    const VkDescriptorBufferInfo o7 = prevVoxelUncertSSBO_.descriptorInfo();
    const VkWriteDescriptorSet oWrites[8]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o0, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o1, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o2, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o3, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o4, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o5, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o6, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, occSet_, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o7, nullptr},
    };
    vkUpdateDescriptorSets(ctx_->device(), 8, oWrites, 0, nullptr);

    VkWriteDescriptorSet sWrites[2]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, shadowSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o4, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, shadowSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &o5, nullptr},
    };
    vkUpdateDescriptorSets(ctx_->device(), 2, sWrites, 0, nullptr);

    resetVoxels();
}

void OccupancyFlowNet::resetVoxels() {
    hasPrevEgo_ = false;
    if (!ctx_) return;
    const std::vector<Vec4> zeros(static_cast<size_t>(nx_ * ny_ * nz_), Vec4{0, 0, 0, 0});
    voxelStateSSBO_.uploadVector(zeros);
    voxelUncertSSBO_.uploadVector(zeros);
    prevVoxelStateSSBO_.uploadVector(zeros);
    prevVoxelUncertSSBO_.uploadVector(zeros);
}

void OccupancyFlowNet::destroy() {
    if (!ctx_) return;
    occPipe_.destroy();
    shadowPipe_.destroy();
    trackPriorsSSBO_.destroy();
    voxelStateSSBO_.destroy();
    voxelUncertSSBO_.destroy();
    prevVoxelStateSSBO_.destroy();
    prevVoxelUncertSSBO_.destroy();
    ctx_ = nullptr;
}

void OccupancyFlowNet::recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg,
                                      const EgoState& ego, float dt,
                                      const std::vector<TrackPriorGpu>& tracks,
                                      bool enableShadows) {
    trackPriorsSSBO_.uploadVector(tracks);

    const Mat4 prevEgoFromCurrEgo = hasPrevEgo_
        ? (BEVFormer::worldFromIsoEgo(prevEgo_).inverseRigid() * BEVFormer::worldFromIsoEgo(ego))
        : Mat4::identity();
    prevEgo_ = ego;
    hasPrevEgo_ = true;

    const OccPC opc{
        prevEgoFromCurrEgo,
        static_cast<uint32_t>(nx_),
        static_cast<uint32_t>(ny_),
        static_cast<uint32_t>(nz_),
        static_cast<uint32_t>(std::min<size_t>(tracks.size(), 64)),
        cfg.occMinX, cfg.occMaxX,
        cfg.occMinY, cfg.occMaxY,
        cfg.occMinZ, cfg.occMaxZ,
        cfg.occDecayRate, dt,
        ego.pos.x, ego.pos.z, ego.yaw,
        cfg.occThreshold
    };
    occPipe_.bindAndDispatch(cmd, occSet_, (nx_ + 7) / 8, (ny_ + 7) / 8, (nz_ + 1) / 2, &opc, sizeof(opc));

    // Persist pre-shadow voxel state and uncertainty for next-frame ego-motion warping
    const VkMemoryBarrier c2t{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                              VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &c2t, 0, nullptr, 0, nullptr);
    const VkBufferCopy sCopy{0, 0, voxelStateSSBO_.size()};
    vkCmdCopyBuffer(cmd, voxelStateSSBO_.handle(), prevVoxelStateSSBO_.handle(), 1, &sCopy);
    const VkBufferCopy uCopy{0, 0, voxelUncertSSBO_.size()};
    vkCmdCopyBuffer(cmd, voxelUncertSSBO_.handle(), prevVoxelUncertSSBO_.handle(), 1, &uCopy);
    const VkMemoryBarrier t2c{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                              VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &t2c, 0, nullptr, 0, nullptr);

    const ShadowPC spc{
        static_cast<uint32_t>(nx_),
        static_cast<uint32_t>(ny_),
        static_cast<uint32_t>(nz_),
        cfg.occMinX, cfg.occMaxX,
        cfg.occMinY, cfg.occMaxY,
        enableShadows ? 1u : 0u
    };
    shadowPipe_.bindAndDispatch(cmd, shadowSet_, (nx_ + 7) / 8, (ny_ + 7) / 8, 1, &spc, sizeof(spc));
    cmdComputeBarrier(cmd);
}

void OccupancyFlowNet::readbackVoxels(const SimConfig& cfg, const EgoState& ego,
                                      std::vector<ActiveVoxel>& outObserved,
                                      std::vector<ActiveVoxel>& outShadows) const {
    outObserved.clear();
    outShadows.clear();

    const size_t total = static_cast<size_t>(nx_ * ny_ * nz_);
    statesScratch_.resize(total);
    uncertScratch_.resize(total);
    voxelStateSSBO_.downloadVector(statesScratch_);
    voxelUncertSSBO_.downloadVector(uncertScratch_);

    const float sinEgo = std::sin(ego.yaw);
    const float cosEgo = std::cos(ego.yaw);

    for (int ix = 0; ix < nx_; ++ix) {
        const float egoX = cfg.occMinX + (static_cast<float>(ix) + 0.5f) * cfg.occStepX;
        for (int iy = 0; iy < ny_; ++iy) {
            const float egoY = cfg.occMinY + (static_cast<float>(iy) + 0.5f) * cfg.occStepY;
            const float wx = ego.pos.x + egoX * sinEgo + egoY * cosEgo;
            const float wz = ego.pos.z + egoX * cosEgo - egoY * sinEgo;

            for (int iz = 0; iz < nz_; ++iz) {
                const size_t idx = (static_cast<size_t>(ix) * ny_ + iy) * nz_ + iz;
                const float conf = statesScratch_[idx].y;
                if (conf <= 0.08f) continue;

                const int semInt = static_cast<int>(std::round(statesScratch_[idx].x));
                const SemanticClass sem = static_cast<SemanticClass>(semInt);
                const float wy = cfg.occMinZ + (static_cast<float>(iz) + 0.5f) * cfg.occStepZ;

                const ActiveVoxel v{
                    egoX, egoY, wy, wx, wz, sem, conf,
                    statesScratch_[idx].z, statesScratch_[idx].w, uncertScratch_[idx].x
                };
                if (sem == SemanticClass::OccShadow) {
                    outShadows.push_back(v);
                } else if (sem != SemanticClass::Empty) {
                    outObserved.push_back(v);
                }
            }
        }
    }
}

} // namespace cybercab
