#include "perception/BackboneRTDETR.hpp"
#include "core/VulkanUtils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cybercab {

namespace {

struct BackbonePC {
    uint32_t camWidth;
    uint32_t camHeight;
    uint32_t featWidth;
    uint32_t featHeight;
    uint32_t patchCols;
    uint32_t patchRows;
};

struct DetrEncPC {
    uint32_t featW;
    uint32_t featH;
    uint32_t s5W;
    uint32_t s5H;
    uint32_t numQueries;
};

struct DetrDecPC {
    uint32_t featW;
    uint32_t featH;
    uint32_t numQueries;
    uint32_t numLayers;
};

float iou2D(const Detection2D& a, const Detection2D& b) {
    float ax1 = a.cxNorm - a.wNorm * 0.5f, ay1 = a.cyNorm - a.hNorm * 0.5f;
    float ax2 = a.cxNorm + a.wNorm * 0.5f, ay2 = a.cyNorm + a.hNorm * 0.5f;
    float bx1 = b.cxNorm - b.wNorm * 0.5f, by1 = b.cyNorm - b.hNorm * 0.5f;
    float bx2 = b.cxNorm + b.wNorm * 0.5f, by2 = b.cyNorm + b.hNorm * 0.5f;
    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float areaA = a.wNorm * a.hNorm;
    float areaB = b.wNorm * b.hNorm;
    float uni = areaA + areaB - inter;
    return (uni > 1e-6f) ? (inter / uni) : 0.0f;
}

} // namespace

BackboneRTDETR::~BackboneRTDETR() {
    destroy();
}

void BackboneRTDETR::init(const VulkanContext& ctx, const SimConfig& cfg,
                          const VulkanBuffer& camTensorSSBO, const VulkanBuffer& soilPatchSSBO) {
    destroy();
    ctx_ = &ctx;
    featW_ = cfg.featWidth;
    featH_ = cfg.featHeight;
    s5W_ = featW_ / 2;
    s5H_ = featH_ / 2;
    patchCols_ = cfg.soilPatchCols;
    patchRows_ = cfg.soilPatchRows;
    numQueries_ = cfg.rtdetrNumQueries;
    numLayers_ = cfg.rtdetrDecoderLayers;
    nmsIouThresh_ = cfg.rtdetrNmsIouThresh;
    maxRangeM_ = cfg.rtdetrMaxRangeM;
    lowConfThresh_ = cfg.rtdetrLowConfThresh;
    currCamTensorRef_ = &camTensorSSBO;
    soilPatchRef_ = &soilPatchSSBO;

    backbonePipe_.create(*ctx_, "shaders/02_backbone_fpn.comp.spv",
                         {
                             {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                             {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                             {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                             {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                             {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                         },
                         sizeof(BackbonePC));

    detrEncPipe_.create(*ctx_, "shaders/03_rtdetr_encoder.comp.spv",
                        {
                            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                        },
                        sizeof(DetrEncPC));

    detrDecPipe_.create(*ctx_, "shaders/03_rtdetr_decoder.comp.spv",
                        {
                            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                        },
                        sizeof(DetrDecPC));

    VkDeviceSize s4Bytes = static_cast<VkDeviceSize>(8 * featW_ * featH_ * 4) * sizeof(Vec4);
    VkDeviceSize s5Bytes = static_cast<VkDeviceSize>(8 * s5W_ * s5H_ * 4) * sizeof(Vec4);
    VkDeviceSize qInitBytes = static_cast<VkDeviceSize>(8 * numQueries_) * sizeof(Vec4);
    VkDeviceSize detOutBytes = static_cast<VkDeviceSize>(8 * numQueries_ * 2) * sizeof(Vec4);

    featS4SSBO_.create(*ctx_, s4Bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    featS5SSBO_.create(*ctx_, s5Bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    encMemorySSBO_.create(*ctx_, s4Bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    queryInitSSBO_.create(*ctx_, qInitBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    detrOutSSBO_.create(*ctx_, detOutBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    detrRawScratch_.resize(static_cast<size_t>(8 * numQueries_ * 2));
    camCandsScratch_.reserve(static_cast<size_t>(numQueries_));
    keptScratch_.reserve(static_cast<size_t>(numQueries_));

    bbSet_ = backbonePipe_.allocateDescriptorSet();
    encSet_ = detrEncPipe_.allocateDescriptorSet();
    decSet_ = detrDecPipe_.allocateDescriptorSet();

    int initW = (cfg.initialResMode == "full") ? cfg.fullCamWidth : cfg.camWidth;
    int initH = (cfg.initialResMode == "full") ? cfg.fullCamHeight : cfg.camHeight;
    updateCameraTensorBinding(camTensorSSBO, initW, initH);
}

void BackboneRTDETR::updateCameraTensorBinding(const VulkanBuffer& camTensorSSBO, int camW, int camH) {
    if (!ctx_) return;
    currCamTensorRef_ = &camTensorSSBO;
    tensorBytes_ = static_cast<VkDeviceSize>(8 * camW * camH * 4) * sizeof(float);
    prevCamTensorSSBO_.create(*ctx_, tensorBytes_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::vector<float> zeros(static_cast<size_t>(8 * camW * camH * 4), 0.0f);
    prevCamTensorSSBO_.uploadVector(zeros);

    VkDescriptorBufferInfo b0 = currCamTensorRef_->descriptorInfo();
    VkDescriptorBufferInfo b1 = prevCamTensorSSBO_.descriptorInfo();
    VkDescriptorBufferInfo b2 = soilPatchRef_->descriptorInfo();
    VkDescriptorBufferInfo b3 = featS4SSBO_.descriptorInfo();
    VkDescriptorBufferInfo b4 = featS5SSBO_.descriptorInfo();

    VkWriteDescriptorSet bbWrites[5]{};
    bbWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, bbSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b0, nullptr};
    bbWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, bbSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b1, nullptr};
    bbWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, bbSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b2, nullptr};
    bbWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, bbSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b3, nullptr};
    bbWrites[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, bbSet_, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b4, nullptr};
    vkUpdateDescriptorSets(ctx_->device(), 5, bbWrites, 0, nullptr);

    VkDescriptorBufferInfo e0 = featS4SSBO_.descriptorInfo();
    VkDescriptorBufferInfo e1 = featS5SSBO_.descriptorInfo();
    VkDescriptorBufferInfo e2 = encMemorySSBO_.descriptorInfo();
    VkDescriptorBufferInfo e3 = queryInitSSBO_.descriptorInfo();

    VkWriteDescriptorSet encWrites[4]{};
    encWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, encSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &e0, nullptr};
    encWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, encSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &e1, nullptr};
    encWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, encSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &e2, nullptr};
    encWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, encSet_, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &e3, nullptr};
    vkUpdateDescriptorSets(ctx_->device(), 4, encWrites, 0, nullptr);

    VkDescriptorBufferInfo d0 = encMemorySSBO_.descriptorInfo();
    VkDescriptorBufferInfo d1 = queryInitSSBO_.descriptorInfo();
    VkDescriptorBufferInfo d2 = detrOutSSBO_.descriptorInfo();

    VkWriteDescriptorSet decWrites[3]{};
    decWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, decSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &d0, nullptr};
    decWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, decSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &d1, nullptr};
    decWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, decSet_, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &d2, nullptr};
    vkUpdateDescriptorSets(ctx_->device(), 3, decWrites, 0, nullptr);
}

void BackboneRTDETR::destroy() {
    if (!ctx_) return;
    backbonePipe_.destroy();
    detrEncPipe_.destroy();
    detrDecPipe_.destroy();
    prevCamTensorSSBO_.destroy();
    featS4SSBO_.destroy();
    featS5SSBO_.destroy();
    encMemorySSBO_.destroy();
    queryInitSSBO_.destroy();
    detrOutSSBO_.destroy();
    ctx_ = nullptr;
}

void BackboneRTDETR::recordBackbone(VkCommandBuffer cmd, int camW, int camH) const {
    BackbonePC pc{
        static_cast<uint32_t>(camW),
        static_cast<uint32_t>(camH),
        static_cast<uint32_t>(featW_),
        static_cast<uint32_t>(featH_),
        static_cast<uint32_t>(patchCols_),
        static_cast<uint32_t>(patchRows_)
    };
    uint32_t gx = (featW_ + 7) / 8;
    uint32_t gy = (featH_ + 7) / 8;
    backbonePipe_.bindAndDispatch(cmd, bbSet_, gx, gy, 8, &pc, sizeof(pc));
    cmdComputeBarrier(cmd);
}

void BackboneRTDETR::recordRTDETR(VkCommandBuffer cmd) const {
    DetrEncPC encPc{
        static_cast<uint32_t>(featW_),
        static_cast<uint32_t>(featH_),
        static_cast<uint32_t>(s5W_),
        static_cast<uint32_t>(s5H_),
        static_cast<uint32_t>(numQueries_)
    };
    detrEncPipe_.bindAndDispatch(cmd, encSet_, 1, 1, 8, &encPc, sizeof(encPc));
    cmdComputeBarrier(cmd);

    DetrDecPC decPc{
        static_cast<uint32_t>(featW_),
        static_cast<uint32_t>(featH_),
        static_cast<uint32_t>(numQueries_),
        static_cast<uint32_t>(numLayers_)
    };
    detrDecPipe_.bindAndDispatch(cmd, decSet_, 1, 1, 8, &decPc, sizeof(decPc));
    cmdComputeBarrier(cmd);
}

void BackboneRTDETR::copyCurrToPrevTensor(VkCommandBuffer cmd) const {
    if (!currCamTensorRef_ || tensorBytes_ == 0) return;
    VkBufferCopy copy{0, 0, tensorBytes_};
    vkCmdCopyBuffer(cmd, currCamTensorRef_->handle(), prevCamTensorSSBO_.handle(), 1, &copy);
}

void BackboneRTDETR::decodeAndLift3D(const SimConfig& cfg,
                                     const CameraRigRenderer& rig,
                                     const EgoState& ego,
                                     std::vector<Detection2D>& out2D,
                                     std::vector<Detection3D>& out3D) const {
    out2D.clear();
    out3D.clear();

    detrRawScratch_.resize(static_cast<size_t>(8 * numQueries_ * 2));
    detrOutSSBO_.downloadVector(detrRawScratch_);

    const float sinEgo = std::sin(ego.yaw);
    const float cosEgo = std::cos(ego.yaw);

    for (int c = 0; c < 8; ++c) {
        camCandsScratch_.clear();
        for (int q = 0; q < numQueries_; ++q) {
            const int base = (c * numQueries_ + q) * 2;
            const Vec4& box = detrRawScratch_[base + 0];
            const Vec4& meta = detrRawScratch_[base + 1];
            if (meta.x < lowConfThresh_) continue;

            Detection2D d;
            d.camIndex = c;
            d.cxNorm = clampf(box.x, 0.02f, 0.98f);
            d.cyNorm = clampf(box.y, 0.02f, 0.98f);
            d.wNorm = clampf(box.z, 0.02f, 0.65f);
            d.hNorm = clampf(box.w, 0.03f, 0.75f);
            d.score = meta.x;
            const int cls = static_cast<int>(std::round(meta.y));
            if (cls == 2) d.semantic = SemanticClass::Pedestrian;
            else if (cls == 3) d.semantic = SemanticClass::SignalSign;
            else d.semantic = SemanticClass::Vehicle;
            d.camPatchWeight = meta.z;
            d.boxSigma = meta.w;
            camCandsScratch_.push_back(d);
        }

        std::sort(camCandsScratch_.begin(), camCandsScratch_.end(),
                  [](const Detection2D& a, const Detection2D& b) { return a.score > b.score; });

        keptScratch_.clear();
        for (const auto& cand : camCandsScratch_) {
            bool suppress = false;
            for (const auto& k : keptScratch_) {
                if (iou2D(cand, k) > nmsIouThresh_) {
                    suppress = true;
                    break;
                }
            }
