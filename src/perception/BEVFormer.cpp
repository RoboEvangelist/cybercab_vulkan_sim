#include "perception/BEVFormer.hpp"

namespace cybercab {

namespace {

struct alignas(16) BevTemporalPC {
    Mat4 prevEgoFromCurrEgo;
    uint32_t nx;
    uint32_t ny;
    float minX;
    float maxX;
    float minY;
    float maxY;
    float temporalBlend;
    float dt;
};

struct alignas(16) BevSpatialPC {
    Vec4 zAnchors;
    uint32_t nx;
    uint32_t ny;
    uint32_t featW;
    uint32_t featH;
    uint32_t patchCols;
    uint32_t patchRows;
    float minX;
    float maxX;
    float minY;
    float maxY;
};

} // namespace

BEVFormer::~BEVFormer() {
    destroy();
}

Mat4 BEVFormer::worldFromIsoEgo(const EgoState& ego) {
    float s = std::sin(ego.yaw), c = std::cos(ego.yaw);
    Mat4 m = Mat4::identity();
    // +X_iso (fwd) -> (s, 0, c)
    m.at(0, 0) = s;  m.at(1, 0) = 0.0f; m.at(2, 0) = c;
    // +Y_iso (left) -> (c, 0, -s)
    m.at(0, 1) = c;  m.at(1, 1) = 0.0f; m.at(2, 1) = -s;
    // +Z_iso (up) -> (0, 1, 0)
    m.at(0, 2) = 0.0f; m.at(1, 2) = 1.0f; m.at(2, 2) = 0.0f;
    m.at(0, 3) = ego.pos.x;
    m.at(1, 3) = ego.pos.y;
    m.at(2, 3) = ego.pos.z;
    return m;
}

void BEVFormer::init(const VulkanContext& ctx, const SimConfig& cfg,
                     const VulkanBuffer& encMemorySSBO, const VulkanBuffer& soilPatchSSBO) {
    destroy();
    ctx_ = &ctx;
    nx_ = cfg.bevNx;
    ny_ = cfg.bevNy;
    featW_ = cfg.featWidth;
    featH_ = cfg.featHeight;
    patchCols_ = cfg.soilPatchCols;
    patchRows_ = cfg.soilPatchRows;

    temporalPipe_.create(*ctx_, "shaders/04_bevformer_temporal.comp.spv",
                         {
                             {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                             {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                             {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                             {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                         },
                         sizeof(BevTemporalPC));

    spatialPipe_.create(*ctx_, "shaders/04_bevformer_spatial.comp.spv",
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
                        sizeof(BevSpatialPC));

    VkDeviceSize featBytes = static_cast<VkDeviceSize>(nx_ * ny_ * 4) * sizeof(Vec4);
    VkDeviceSize metaBytes = static_cast<VkDeviceSize>(nx_ * ny_) * sizeof(Vec4);

    camProjSSBO_.create(*ctx_, 8 * sizeof(CamProjGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    prevBevFeatSSBO_.create(*ctx_, featBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    prevBevMetaSSBO_.create(*ctx_, metaBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    warpedBevFeatSSBO_.create(*ctx_, featBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    warpedBevMetaSSBO_.create(*ctx_, metaBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    currBevFeatSSBO_.create(*ctx_, featBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    currBevMetaSSBO_.create(*ctx_, metaBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    pillarHeightSSBO_.create(*ctx_, metaBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    metaReadbackScratch_.resize(static_cast<size_t>(nx_ * ny_));

    tempSet_ = temporalPipe_.allocateDescriptorSet();
    spatSet_ = spatialPipe_.allocateDescriptorSet();

    VkDescriptorBufferInfo t0 = prevBevFeatSSBO_.descriptorInfo();
    VkDescriptorBufferInfo t1 = prevBevMetaSSBO_.descriptorInfo();
    VkDescriptorBufferInfo t2 = warpedBevFeatSSBO_.descriptorInfo();
    VkDescriptorBufferInfo t3 = warpedBevMetaSSBO_.descriptorInfo();
    VkWriteDescriptorSet tWrites[4]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tempSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &t0, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tempSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &t1, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tempSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &t2, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tempSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &t3, nullptr},
    };
    vkUpdateDescriptorSets(ctx_->device(), 4, tWrites, 0, nullptr);

    VkDescriptorBufferInfo s0 = camProjSSBO_.descriptorInfo();
    VkDescriptorBufferInfo s1 = encMemorySSBO.descriptorInfo();
    VkDescriptorBufferInfo s2 = soilPatchSSBO.descriptorInfo();
    VkDescriptorBufferInfo s3 = warpedBevFeatSSBO_.descriptorInfo();
    VkDescriptorBufferInfo s4 = warpedBevMetaSSBO_.descriptorInfo();
    VkDescriptorBufferInfo s5 = currBevFeatSSBO_.descriptorInfo();
    VkDescriptorBufferInfo s6 = currBevMetaSSBO_.descriptorInfo();
    VkDescriptorBufferInfo s7 = pillarHeightSSBO_.descriptorInfo();
    VkWriteDescriptorSet sWrites[8]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &s0, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &s1, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &s2, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &s3, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatSet_, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &s4, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatSet_, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &s5, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatSet_, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &s6, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatSet_, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &s7, nullptr},
    };
    vkUpdateDescriptorSets(ctx_->device(), 8, sWrites, 0, nullptr);

    resetHistory();
}

void BEVFormer::resetHistory() {
    if (!ctx_) return;
    hasPrevEgo_ = false;
    std::vector<Vec4> zeroFeat(static_cast<size_t>(nx_ * ny_ * 4), Vec4{0, 0, 0, 0});
    std::vector<Vec4> initMeta(static_cast<size_t>(nx_ * ny_), Vec4{0.0f, 1.0f, 0.0f, 0.0f});
    prevBevFeatSSBO_.uploadVector(zeroFeat);
    prevBevMetaSSBO_.uploadVector(initMeta);
}

void BEVFormer::destroy() {
    if (!ctx_) return;
    temporalPipe_.destroy();
    spatialPipe_.destroy();
    camProjSSBO_.destroy();
    prevBevFeatSSBO_.destroy();
    prevBevMetaSSBO_.destroy();
    warpedBevFeatSSBO_.destroy();
    warpedBevMetaSSBO_.destroy();
    currBevFeatSSBO_.destroy();
    currBevMetaSSBO_.destroy();
    pillarHeightSSBO_.destroy();
    ctx_ = nullptr;
}

void BEVFormer::recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg,
                               const CameraRigRenderer& rig,
                               const EgoState& ego, float dt,
                               const std::array<SoilCameraSummary, 8>& soilCameras) {
    Mat4 worldFromCurrIso = worldFromIsoEgo(ego);
    Mat4 prevEgoFromCurrEgo = hasPrevEgo_
        ? (worldFromIsoEgo(prevEgo_).inverseRigid() * worldFromCurrIso)
        : Mat4::identity();
    prevEgo_ = ego;
    hasPrevEgo_ = true;

    std::array<CamProjGpu, 8> camProj{};
    for (int c = 0; c < 8; ++c) {
        const auto& spec = cfg.cameras.at(c);
        const auto& cm = rig.cameraMatrices(c);
        camProj[c].camFromEgoISO = cm.camFromWorld * worldFromCurrIso;
        camProj[c].intr = {cm.intr.fx, cm.intr.fy, cm.intr.cx, cm.intr.cy};
        camProj[c].dist = {spec.distortion[0], spec.distortion[1], spec.distortion[2], spec.distortion[3]};
        camProj[c].dims = {static_cast<float>(cm.intr.width), static_cast<float>(cm.intr.height),
                           spec.maxRangeM, soilCameras[c].tokenWeight};
    }
    camProjSSBO_.upload(camProj.data(), sizeof(camProj));

    uint32_t gx = (nx_ + 7) / 8;
    uint32_t gy = (ny_ + 7) / 8;

    BevTemporalPC tpc{
        prevEgoFromCurrEgo,
        static_cast<uint32_t>(nx_),
        static_cast<uint32_t>(ny_),
        cfg.bevMinX, cfg.bevMaxX,
        cfg.bevMinY, cfg.bevMaxY,
        cfg.bevTemporalBlend,
        dt
    };
    temporalPipe_.bindAndDispatch(cmd, tempSet_, gx, gy, 1, &tpc, sizeof(tpc));
    cmdComputeBarrier(cmd);

    Vec4 zAnc{0.45f, 1.05f, 1.75f, 2.65f};
    if (cfg.bevZAnchors.size() >= 4) {
        zAnc = {cfg.bevZAnchors[0], cfg.bevZAnchors[1], cfg.bevZAnchors[2], cfg.bevZAnchors[3]};
    }
    BevSpatialPC spc{
        zAnc,
        static_cast<uint32_t>(nx_),
        static_cast<uint32_t>(ny_),
        static_cast<uint32_t>(featW_),
        static_cast<uint32_t>(featH_),
        static_cast<uint32_t>(patchCols_),
        static_cast<uint32_t>(patchRows_),
        cfg.bevMinX, cfg.bevMaxX,
        cfg.bevMinY, cfg.bevMaxY
    };
    spatialPipe_.bindAndDispatch(cmd, spatSet_, gx, gy, 1, &spc, sizeof(spc));
    cmdComputeBarrier(cmd);
}

void BEVFormer::copyCurrToPrev(VkCommandBuffer cmd) const {
    VkBufferCopy fCopy{0, 0, currBevFeatSSBO_.size()};
    vkCmdCopyBuffer(cmd, currBevFeatSSBO_.handle(), prevBevFeatSSBO_.handle(), 1, &fCopy);
    VkBufferCopy mCopy{0, 0, currBevMetaSSBO_.size()};
    vkCmdCopyBuffer(cmd, currBevMetaSSBO_.handle(), prevBevMetaSSBO_.handle(), 1, &mCopy);
}

BevSummary BEVFormer::readbackSummary() const {
    metaReadbackScratch_.resize(static_cast<size_t>(nx_ * ny_));
    currBevMetaSSBO_.downloadVector(metaReadbackScratch_);
    double sumObs = 0.0;
    double sumVar = 0.0;
    int obsCnt = 0;
    for (const auto& m : metaReadbackScratch_) {
        sumObs += m.x;
        sumVar += m.y;
        if (m.x > 0.25f) obsCnt++;
    }
    BevSummary out;
    out.meanObservability = static_cast<float>(sumObs / std::max<size_t>(1, metaReadbackScratch_.size()));
    out.meanEpistemicVar = static_cast<float>(sumVar / std::max<size_t>(1, metaReadbackScratch_.size()));
    out.observedCells = obsCnt;
    return out;
}

} // namespace cybercab
