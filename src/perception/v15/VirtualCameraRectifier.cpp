#include "perception/v15/VirtualCameraRectifier.hpp"

#include <cstring>
#include <stdexcept>

namespace cybercab::v15 {
namespace {

Mat4 pureRot(Mat4 m) {
    m.at(0, 3) = 0.0f;
    m.at(1, 3) = 0.0f;
    m.at(2, 3) = 0.0f;
    return m;
}

} // namespace

VirtualCameraRectifier::~VirtualCameraRectifier() { destroy(); }

void VirtualCameraRectifier::init(const VulkanContext& ctx, const SimConfig& cfg,
                                  const VulkanBuffer& linearIn, const VulkanBuffer& bandsIn,
                                  int srcW, int srcH) {
    destroy();
    if (cfg.cameras.size() < 8 || srcW <= 0 || srcH <= 0) {
        throw std::runtime_error("VirtualCameraRectifier::init invalid config or source dimensions");
    }
    ctx_ = &ctx;
    pipe_.create(ctx, "shaders/v15_virtual_rectify.comp.spv",
                 {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1}},
                 sizeof(RectPC));
    set_ = pipe_.allocateDescriptorSet();

    const VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkDeviceSize pxBytes = static_cast<VkDeviceSize>(kNumViews) * kVirtW * kVirtH * sizeof(Vec4);
    specsSSBO_.create(ctx, static_cast<VkDeviceSize>(kNumViews) * sizeof(VirtViewSpecGpu),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    virtLinear_.create(ctx, pxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    virtBands_.create(ctx, pxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    rebind(cfg, linearIn, bandsIn, srcW, srcH);
}

void VirtualCameraRectifier::rebind(const SimConfig& cfg, const VulkanBuffer& linearIn,
                                    const VulkanBuffer& bandsIn, int srcW, int srcH) {
    if (!ctx_) return;
    const VkDescriptorBufferInfo infos[5] = {
        linearIn.descriptorInfo(),
        bandsIn.descriptorInfo(),
        specsSSBO_.descriptorInfo(),
        virtLinear_.descriptorInfo(),
        virtBands_.descriptorInfo(),
    };
    VkWriteDescriptorSet writes[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, i, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[i], nullptr};
    }
    vkUpdateDescriptorSets(ctx_->device(), 5, writes, 0, nullptr);
    updateViews(cfg, srcW, srcH, Mat4::identity());
}

std::array<VirtualViewSpec, VirtualCameraRectifier::kNumViews>
VirtualCameraRectifier::buildViewSpecs(const SimConfig& cfg, int srcW, int srcH,
                                       const Mat4& vehicleCalibMatrix,
                                       std::array<VirtViewSpecGpu, kNumViews>* outGpu) {
    if (cfg.cameras.size() < 8) {
        throw std::runtime_error("VirtualCameraRectifier requires 8 physical cameras");
    }
    const Mat4 rCalInv = pureRot(vehicleCalibMatrix).transposed();
    std::array<VirtualViewSpec, kNumViews> views{};
    for (int v = 0; v < kNumViews; ++v) {
        const int srcIdx = (v == kViewVru) ? 2 : (v == kViewElevated) ? 0 : v;
        const CameraSpec& phys = cfg.cameras.at(srcIdx);
        CameraSpec virt = phys;
        virt.distortion = {0.0f, 0.0f, 0.0f, 0.0f};
        if (v == kViewVru) {
            virt.id = "VIRT_VRU_LOW";
            virt.hfovDeg = 70.0f;
            virt.rpyDeg = {0.0f, -12.0f, 0.0f};
        } else if (v == kViewElevated) {
            virt.id = "VIRT_ELEVATED_MACRO";
            virt.hfovDeg = 36.0f;
            virt.rpyDeg = {0.0f, 5.0f, 0.0f};
        }

        const CameraIntrinsics vIntr = virt.intrinsics(kVirtW, kVirtH);
        const CameraIntrinsics pIntr = phys.intrinsics(srcW, srcH);
        const Mat4 rPhysInv = pureRot(phys.extrinsicVehicleLocalFromCam()).transposed();
        const Mat4 tUpright = virt.extrinsicVehicleLocalFromCam();
        const Mat4 physFromVirt = rPhysInv * rCalInv * pureRot(tUpright);

        views[v] = {srcIdx, virt.id, vIntr, virt.posIso, virt.rpyDeg, tUpright, physFromVirt};
        if (outGpu) {
            VirtViewSpecGpu g{};
            g.physCamFromVirtCam = physFromVirt;
            g.virtIntr = {vIntr.fx, vIntr.fy, vIntr.cx, vIntr.cy};
            g.physIntr = {pIntr.fx, pIntr.fy, pIntr.cx, pIntr.cy};
            g.physDist = {phys.distortion[0], phys.distortion[1], phys.distortion[2], phys.distortion[3]};
            g.srcCamIdx = static_cast<uint32_t>(srcIdx);
            g.srcW = static_cast<uint32_t>(srcW);
            g.srcH = static_cast<uint32_t>(srcH);
            const float rs = phys.renderOverscan();
            std::memcpy(&g.renderOverscanBits, &rs, sizeof(uint32_t));
            (*outGpu)[v] = g;
        }
    }
    return views;
}

void VirtualCameraRectifier::updateViews(const SimConfig& cfg, int srcW, int srcH,
                                         const Mat4& vehicleCalibMatrix) {
    views_ = buildViewSpecs(cfg, srcW, srcH, vehicleCalibMatrix, &gpuSpecs_);
    if (specsSSBO_.handle() != VK_NULL_HANDLE) {
        specsSSBO_.upload(gpuSpecs_.data(), sizeof(gpuSpecs_));
    }
}

void VirtualCameraRectifier::recordDispatch(VkCommandBuffer cmd) const {
    if (!ctx_) return;
    const RectPC pc{static_cast<uint32_t>(kVirtW), static_cast<uint32_t>(kVirtH),
                    static_cast<uint32_t>(kNumViews), 0u};
    pipe_.bindAndDispatch(cmd, set_,
                          (static_cast<uint32_t>(kVirtW) + 15u) / 16u,
                          (static_cast<uint32_t>(kVirtH) + 15u) / 16u,
                          static_cast<uint32_t>(kNumViews), &pc, sizeof(pc));
    cmdComputeBarrier(cmd);
}

void VirtualCameraRectifier::destroy() {
    if (!ctx_) return;
    virtBands_.destroy();
    virtLinear_.destroy();
    specsSSBO_.destroy();
    pipe_.destroy();
    set_ = VK_NULL_HANDLE;
    ctx_ = nullptr;
}

} // namespace cybercab::v15
