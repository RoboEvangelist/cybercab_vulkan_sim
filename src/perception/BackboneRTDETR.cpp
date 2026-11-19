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
