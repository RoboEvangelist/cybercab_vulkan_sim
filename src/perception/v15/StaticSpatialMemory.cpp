#include "perception/v15/StaticSpatialMemory.hpp"
#include "perception/BEVFormer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace cybercab::v15 {
namespace {

Mat4 vehLocalFromIsoEgo() {
    Mat4 m{};
    m.at(0, 1) = 1.0f; // left = ego Y
    m.at(1, 2) = 1.0f; // up   = ego Z
    m.at(2, 0) = 1.0f; // fwd  = ego X
    m.at(3, 3) = 1.0f;
    return m;
}

} // namespace

StaticSpatialMemory::~StaticSpatialMemory() { destroy(); }

void StaticSpatialMemory::init(const VulkanContext& ctx, const SimConfig& cfg,
                               const VirtualCameraRectifier& rectifier, const PatchTokenizer& patches) {
    (void)rectifier;
    destroy();
    if (patches.ringSSBO().handle() == VK_NULL_HANDLE || cfg.bevNx <= 0 || cfg.bevNy <= 0) {
        throw std::runtime_error("StaticSpatialMemory::init invalid PatchTokenizer or BEV grid dims");
    }
    ctx_ = &ctx;
    nx_ = cfg.bevNx;
    ny_ = cfg.bevNy;
    pipe_.create(ctx, "shaders/v15_static_memory.comp.spv",
                 {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1}},
                 sizeof(StaticMemPC));
    set_ = pipe_.allocateDescriptorSet();

    const VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkBufferUsageFlags copyBuf = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkDeviceSize memBytes = static_cast<VkDeviceSize>(nx_) * ny_ * sizeof(StaticCellGpu);
    prevMem_.create(ctx, memBytes, copyBuf, mem);
    warpedMem_.create(ctx, memBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    currMem_.create(ctx, memBytes, copyBuf, mem);
    viewsSSBO_.create(ctx, static_cast<VkDeviceSize>(VirtualCameraRectifier::kNumViews) * sizeof(VirtViewCamGpu),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    tracksSSBO_.create(ctx, static_cast<VkDeviceSize>(std::max(1, cfg.btMaxTracks)) * sizeof(TrackPriorGpu),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    statsSSBO_.create(ctx, 4 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);

    const VkDescriptorBufferInfo infos[7] = {
        prevMem_.descriptorInfo(), warpedMem_.descriptorInfo(), currMem_.descriptorInfo(),
        patches.ringSSBO().descriptorInfo(), viewsSSBO_.descriptorInfo(),
        tracksSSBO_.descriptorInfo(), statsSSBO_.descriptorInfo(),
    };
    VkWriteDescriptorSet w[7]{};
    for (uint32_t i = 0; i < 7; ++i) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, i, 0, 1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[i], nullptr};
    }
    vkUpdateDescriptorSets(ctx.device(), 7, w, 0, nullptr);
    reset();
}

void StaticSpatialMemory::reset() {
    hasPrevEgo_ = false;
    prevEgo_ = {};
    if (!ctx_) return;
    std::vector<StaticCellGpu> zeros(static_cast<size_t>(nx_) * ny_);
    prevMem_.uploadVector(zeros);
    currMem_.uploadVector(zeros);
    const uint32_t zeroStats[4]{};
    statsSSBO_.upload(zeroStats, sizeof(zeroStats));
}

void StaticSpatialMemory::destroy() {
    if (!ctx_) return;
    statsSSBO_.destroy();
    tracksSSBO_.destroy();
    viewsSSBO_.destroy();
    currMem_.destroy();
    warpedMem_.destroy();
    prevMem_.destroy();
    pipe_.destroy();
    set_ = VK_NULL_HANDLE;
    ctx_ = nullptr;
    hasPrevEgo_ = false;
}

void StaticSpatialMemory::recordDispatch(VkCommandBuffer cmd, const SimConfig& cfg, const EgoState& ego, float dt,
                                         const VirtualCameraRectifier& rectifier, const PatchTokenizer& patches,
                                         const std::vector<TrackPriorGpu>& tracks) {
    if (!ctx_) return;
    const Mat4 vehFromIso = vehLocalFromIsoEgo();
    std::array<VirtViewCamGpu, VirtualCameraRectifier::kNumViews> views{};
    for (int v = 0; v < VirtualCameraRectifier::kNumViews; ++v) {
        const auto& vs = rectifier.viewSpec(v);
        views[v] = {vs.uprightVehFromVirtCam.inverseRigid() * vehFromIso,
                    {vs.intr.fx, vs.intr.fy, vs.intr.cx, vs.intr.cy}};
    }
    viewsSSBO_.upload(views.data(), sizeof(views));

    const uint32_t nTracks = static_cast<uint32_t>(std::min(static_cast<int>(tracks.size()), cfg.bevNx > 0 ? cfg.btMaxTracks : 0));
    if (nTracks > 0u) tracksSSBO_.upload(tracks.data(), nTracks * sizeof(TrackPriorGpu));

    const Vec4 zAnchors = (cfg.bevZAnchors.size() >= 4)
        ? Vec4{cfg.bevZAnchors[0], cfg.bevZAnchors[1], cfg.bevZAnchors[2], cfg.bevZAnchors[3]}
        : Vec4{0.35f, 0.90f, 1.50f, 2.20f};
    const Mat4 prevIsoFromCurr = hasPrevEgo_
        ? (BEVFormer::worldFromIsoEgo(prevEgo_).inverseRigid() * BEVFormer::worldFromIsoEgo(ego))
        : Mat4::identity();
    StaticMemPC pc{
        prevIsoFromCurr, zAnchors, static_cast<uint32_t>(nx_), static_cast<uint32_t>(ny_),
        0u, hasPrevEgo_ ? 1u : 0u, patches.latestWrittenSlot(),
        static_cast<uint32_t>(VirtualCameraRectifier::kNumViews),
        static_cast<uint32_t>(PatchTokenizer::kGridW), static_cast<uint32_t>(PatchTokenizer::kGridH),
        nTracks, cfg.bevMinX, cfg.bevMaxX, cfg.bevMinY, cfg.bevMaxY, dt, 0.80f, cfg.occDecayRate,
    };
    const uint32_t gx = (static_cast<uint32_t>(nx_) + 7u) / 8u;
    const uint32_t gy = (static_cast<uint32_t>(ny_) + 7u) / 8u;
    pipe_.bindAndDispatch(cmd, set_, gx, gy, 1u, &pc, sizeof(pc));
    cmdComputeBarrier(cmd);
    pc.pass = 1u;
    pipe_.bindAndDispatch(cmd, set_, gx, gy, 1u, &pc, sizeof(pc));
    cmdComputeBarrier(cmd);

    const VkBufferCopy copyRegion{0, 0, currMem_.size()};
    vkCmdCopyBuffer(cmd, currMem_.handle(), prevMem_.handle(), 1, &copyRegion);
    cmdComputeBarrier(cmd);
    prevEgo_ = ego;
    hasPrevEgo_ = true;
}

StaticMemStats StaticSpatialMemory::readbackStats() const {
    if (!ctx_) return {};
    uint32_t raw[4]{};
    statsSSBO_.download(raw, sizeof(raw));
    StaticMemStats s{raw[0], raw[1], raw[2], raw[3], 0.0f};
    const uint32_t totalObs = s.refreshedCount + s.skippedCount;
    s.selectiveRefreshSkipRatio = (totalObs > 0u) ? static_cast<float>(s.skippedCount) / static_cast<float>(totalObs) : 0.0f;
    return s;
}

void StaticSpatialMemory::readbackCells(std::vector<StaticCellGpu>& out) const {
    if (!ctx_) { out.clear(); return; }
    out.resize(static_cast<size_t>(nx_) * ny_);
    currMem_.downloadVector(out);
}

} // namespace cybercab::v15
