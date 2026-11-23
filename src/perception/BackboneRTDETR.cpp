#include "perception/BackboneRTDETR.hpp"
#include "perception/MultiCamCorroboration.hpp"
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
    uint32_t hasPrevTensor;
    uint32_t pad0;
    Vec4 horizon03;
    Vec4 horizon47;
    Vec4 horizonSlope03;
    Vec4 horizonSlope47;
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
  float aspectScale;
};

float ioa2D(const Detection2D& a, const Detection2D& b) {
    const float ax1 = a.cxNorm - a.wNorm * 0.5f, ay1 = a.cyNorm - a.hNorm * 0.5f;
    const float ax2 = a.cxNorm + a.wNorm * 0.5f, ay2 = a.cyNorm + a.hNorm * 0.5f;
    const float bx1 = b.cxNorm - b.wNorm * 0.5f, by1 = b.cyNorm - b.hNorm * 0.5f;
    const float bx2 = b.cxNorm + b.wNorm * 0.5f, by2 = b.cyNorm + b.hNorm * 0.5f;
    const float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    const float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    const float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    const float areaA = a.wNorm * a.hNorm;
    const float areaB = b.wNorm * b.hNorm;
    const float minArea = std::min(areaA, areaB);
    return (minArea > 1e-6f) ? (inter / minArea) : 0.0f;
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
    for (int c = 0; c < 8; ++c) {
        const auto& spec = cfg.cameras.at(c);
        const Mat4 egoFromCam = spec.extrinsicEgoFromCamISO();
        const Mat4 camFromEgo = spec.extrinsicCamFromEgoISO();
        const Vec3 fwdEgo = egoFromCam.transformVector({0.0f, 0.0f, 1.0f});
        const Vec3 horizEgo = Vec3{fwdEgo.x, fwdEgo.y, 0.0f}.normalized();
        const Vec3 horizCam = camFromEgo.transformVector(horizEgo);
        const CameraIntrinsics intr = spec.intrinsics(cfg.camWidth, cfg.camHeight);
        camHorizonV_[c] = clampf((intr.cy + intr.fy * (horizCam.y / std::max(0.1f, horizCam.z))) /
                                 static_cast<float>(cfg.camHeight), 0.42f, 0.50f);
    }
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
    raw3dScratch_.reserve(static_cast<size_t>(8 * numQueries_));
    suppressed3dScratch_.reserve(static_cast<size_t>(8 * numQueries_));

    bbSet_ = backbonePipe_.allocateDescriptorSet();
    encSet_ = detrEncPipe_.allocateDescriptorSet();
    decSet_ = detrDecPipe_.allocateDescriptorSet();

  const auto initRes = cfg.resolveInitialRes();
  updateCameraTensorBinding(camTensorSSBO, initRes.width, initRes.height);
}

void BackboneRTDETR::updateCameraTensorBinding(const VulkanBuffer& camTensorSSBO, int camW, int camH) {
    if (!ctx_) return;
  camW_ = std::max(1, camW);
  camH_ = std::max(1, camH);
    hasPrevTensor_ = false;
    currCamTensorRef_ = &camTensorSSBO;
    tensorBytes_ = static_cast<VkDeviceSize>(8 * camW * camH * 4) * sizeof(float);
    prevCamTensorSSBO_.create(*ctx_, tensorBytes_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
  vkCmdFillBuffer(cmd, prevCamTensorSSBO_.handle(), 0, tensorBytes_, 0u);
  ctx_->endSingleTimeCommands(cmd);

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

void BackboneRTDETR::resetHistory() {
    hasPrevTensor_ = false;
}

void BackboneRTDETR::updateHorizonFromRig(const CameraRigRenderer& rig) {
    for (int c = 0; c < 8; ++c) {
        const auto& cm = rig.cameraMatrices(c);
        const Vec3 fwdW = cm.worldFromCam.transformVector({0.0f, 0.0f, 1.0f});
        const float hLen = std::hypot(fwdW.x, fwdW.z);
        if (hLen < 1e-4f) continue;
        const Vec3 horizFwd{fwdW.x / hLen, 0.0f, fwdW.z / hLen};
        const Vec3 horizCam = cm.camFromWorld.transformVector(horizFwd);
        if (horizCam.z < 0.1f) continue;
        const float hv = (cm.intr.cy + cm.intr.fy * (horizCam.y / horizCam.z)) /
                         std::max(1.0f, static_cast<float>(cm.intr.height));
        camHorizonV_[c] = clampf(hv, 0.34f, 0.62f);

        const Vec3 camRightW = cm.worldFromCam.transformVector({1.0f, 0.0f, 0.0f});
        const float rLen = std::hypot(camRightW.x, camRightW.z);
        const Vec3 horizSide = (rLen > 1e-4f) ? Vec3{camRightW.x / rLen, 0.0f, camRightW.z / rLen}
                                              : Vec3{horizFwd.z, 0.0f, -horizFwd.x};
        auto projHorizUV = [&](const Vec3& dirW, float& outU, float& outV) -> bool {
            const Vec3 pc = cm.camFromWorld.transformVector(dirW);
            if (pc.z < 0.1f) return false;
            outU = (cm.intr.cx + cm.intr.fx * (pc.x / pc.z)) / std::max(1.0f, static_cast<float>(cm.intr.width));
            outV = (cm.intr.cy + cm.intr.fy * (pc.y / pc.z)) / std::max(1.0f, static_cast<float>(cm.intr.height));
            return true;
        };
        float uL = 0.0f, vL = hv, uR = 1.0f, vR = hv;
        const Vec3 dL{horizFwd.x - horizSide.x * 0.25f, 0.0f, horizFwd.z - horizSide.z * 0.25f};
        const Vec3 dR{horizFwd.x + horizSide.x * 0.25f, 0.0f, horizFwd.z + horizSide.z * 0.25f};
        if ((c == 3 || c == 4 || c == 5 || c == 6) &&
            projHorizUV(dL, uL, vL) && projHorizUV(dR, uR, vR) && std::abs(uR - uL) > 0.10f) {
            const float rawSlope = (vR - vL) / (uR - uL);
            camHorizonSlope_[c] = (std::abs(rawSlope) > 0.008f) ? clampf(rawSlope, -0.35f, 0.35f) : 0.0f;
        } else {
            camHorizonSlope_[c] = 0.0f;
        }
    }
}

void BackboneRTDETR::destroy() {
    if (!ctx_) return;
    hasPrevTensor_ = false;
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
        static_cast<uint32_t>(patchRows_),
        hasPrevTensor_ ? 1u : 0u,
        0u,
        {camHorizonV_[0], camHorizonV_[1], camHorizonV_[2], camHorizonV_[3]},
        {camHorizonV_[4], camHorizonV_[5], camHorizonV_[6], camHorizonV_[7]},
        {camHorizonSlope_[0], camHorizonSlope_[1], camHorizonSlope_[2], camHorizonSlope_[3]},
        {camHorizonSlope_[4], camHorizonSlope_[5], camHorizonSlope_[6], camHorizonSlope_[7]}
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

  constexpr float kRefCamAspect =
    static_cast<float>(kLegacyHalfCamW) /
    static_cast<float>(kLegacyHalfCamH);
  const float aspectScale =
    (static_cast<float>(camW_) / static_cast<float>(camH_)) /
    kRefCamAspect;
  DetrDecPC decPc{
    static_cast<uint32_t>(featW_),
    static_cast<uint32_t>(featH_),
    static_cast<uint32_t>(numQueries_),
    static_cast<uint32_t>(numLayers_),
    aspectScale
  };
    detrDecPipe_.bindAndDispatch(cmd, decSet_, 1, 1, 8, &decPc, sizeof(decPc));
    cmdComputeBarrier(cmd);
}

void BackboneRTDETR::copyCurrToPrevTensor(VkCommandBuffer cmd) const {
    if (!currCamTensorRef_ || tensorBytes_ == 0) return;
    VkBufferCopy copy{0, 0, tensorBytes_};
    vkCmdCopyBuffer(cmd, currCamTensorRef_->handle(), prevCamTensorSSBO_.handle(), 1, &copy);
    hasPrevTensor_ = true;
}

void BackboneRTDETR::decodeAndLift3D(const SimConfig& cfg,
                                     const CameraRigRenderer& rig,
                                     const EgoState& ego,
                                     std::vector<Detection2D>& out2D,
                                     std::vector<Detection3D>& out3D) const {
    out2D.clear();
    out3D.clear();
    raw3dScratch_.clear();

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
            d.wNorm = clampf(box.z, 0.008f, 0.94f);
            d.hNorm = clampf(box.w, 0.010f, 0.92f);
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
                if (cand.semantic == k.semantic && ioa2D(cand, k) > nmsIouThresh_) {
                    suppress = true;
                    break;
                }
                const float distNorm = std::hypot((cand.cxNorm - k.cxNorm) / std::max(0.05f, std::max(cand.wNorm, k.wNorm)),
                                                  (cand.cyNorm - k.cyNorm) / std::max(0.05f, std::max(cand.hNorm, k.hNorm)));
                if (distNorm < 0.75f && ioa2D(cand, k) > 0.25f) {
                    suppress = true;
                    break;
                }
            }
            if (!suppress) keptScratch_.push_back(cand);
        }

        const auto& spec = cfg.cameras.at(c);
        const auto& cm = rig.cameraMatrices(c);
        const float W = static_cast<float>(cm.intr.width);
        const float H = static_cast<float>(cm.intr.height);

        for (const auto& d : keptScratch_) {
            // Skip truncated detections clipped near sensor periphery and low-conf tiny distortion slivers in wide fisheyes
            const float edgeMargin = (spec.hfovDeg > 80.0f) ? 0.080f : 0.045f;
            if (d.cxNorm < edgeMargin || d.cxNorm > 1.0f - edgeMargin) {
                continue;
            }
            if (spec.hfovDeg > 80.0f && (d.cxNorm < 0.115f || d.cxNorm > 0.885f) &&
                d.hNorm < 0.030f && d.score < 0.32f) {
                continue;
            }
            // Reject flat bottom-edge tire/bumper slivers truncated at the very base of the sensor.
            if (d.semantic == SemanticClass::Vehicle && d.cyNorm > 0.90f &&
                d.hNorm < 0.18f && (d.cyNorm + d.hNorm * 0.5f) > 0.985f) {
                continue;
            }

            // Compute calibrated world rays for both box center and bottom edge
            const Vec2 xuCenter = undistortNormalized((d.cxNorm * W - cm.intr.cx) / cm.intr.fx,
                                                      (d.cyNorm * H - cm.intr.cy) / cm.intr.fy,
                                                      spec.distortion);
            const Vec3 rayCenterW = cm.worldFromCam.transformVector({xuCenter.x, xuCenter.y, 1.0f}).normalized();

            const float uBot = d.cxNorm * W;
            const float vBot = (d.cyNorm + d.hNorm * 0.5f) * H;
            const float xd = (uBot - cm.intr.cx) / cm.intr.fx;
            const float yd = (vBot - cm.intr.cy) / cm.intr.fy;
            const Vec2 xu = undistortNormalized(xd, yd, spec.distortion);

            const Vec3 camOriginW = cm.worldFromCam.transformPoint({0.0f, 0.0f, 0.0f});
            const Vec3 rayDirW = cm.worldFromCam.transformVector({xu.x, xu.y, 1.0f}).normalized();

            const float hPx = std::max(d.hNorm * H, 2.5f);
            const float wPx = std::max(d.wNorm * W, 2.5f);
            const float vTop = d.cyNorm - d.hNorm * 0.5f;
            const float aboveHoriz = camHorizonV_[c] - vTop;
            // Calibrate vehicle roof height in world frame using top/bottom rays so pitch and camera focal zoom
            // never misclassify a 1.38-1.58m sedan/SUV as a 2.68m tall truck.
            const Vec2 xuTop = undistortNormalized((d.cxNorm * W - cm.intr.cx) / cm.intr.fx,
                                                   (vTop * H - cm.intr.cy) / cm.intr.fy,
                                                   spec.distortion);
            const Vec3 rayTopW = cm.worldFromCam.transformVector({xuTop.x, xuTop.y, 1.0f}).normalized();
            const float tRoofRef = (rayDirW.y < -0.004f)
                                   ? clampf(-camOriginW.y / rayDirW.y, 1.8f, spec.maxRangeM)
                                   : clampf((cm.intr.fy * 2.50f) / hPx, 1.8f, spec.maxRangeM);
            const float topWorldY = camOriginW.y + rayTopW.y * tRoofRef;
            const bool geomTallRoof = (topWorldY >= 2.15f) && (rayTopW.y > 0.006f || vTop < camHorizonV_[c] - 0.015f);
            const bool isTallTruck = (d.semantic == SemanticClass::Vehicle) &&
                                     (d.hNorm > 0.070f) &&
                                     geomTallRoof &&
                                     ((hPx >= wPx * 0.68f) || (aboveHoriz > 0.045f && wPx >= hPx * 0.55f));
            // Sub-cell 1px chromatic pedestrian boxes capture lit torso (WorldEngine y=[0.75, 1.33]m, height 0.58m)
            const float sigPrior = (d.cyNorm < camHorizonV_[c] - 0.08f && d.hNorm < 0.065f) ? 0.25f : 1.05f;
            const float hPrior = (d.semantic == SemanticClass::Pedestrian) ? 0.56f :
                                 (d.semantic == SemanticClass::SignalSign) ? sigPrior :
                                 isTallTruck ? 2.68f : 1.50f;
            const bool borderClipX = (d.cxNorm - d.wNorm * 0.5f < 0.035f) || (d.cxNorm + d.wNorm * 0.5f > 0.965f);
            const float tHeight = clampf((cm.intr.fy * hPrior) / hPx, 1.5f, std::min(spec.maxRangeM, maxRangeM_));
            const float tWidth = clampf((cm.intr.fx * 2.38f) / wPx, 1.5f, std::min(spec.maxRangeM, maxRangeM_));
            const float tSize = isTallTruck
                              ? ((vTop < 0.04f && !borderClipX)
                                     ? tWidth
                                     : ((borderClipX || wPx > hPx * 1.28f) ? tHeight : (0.35f * tHeight + 0.65f * tWidth)))
                              : tHeight;

            float tEst = tSize;
            const float planeY = (d.semantic == SemanticClass::Pedestrian) ? ((spec.hfovDeg < 40.0f) ? 0.80f : 0.76f) :
                                 isTallTruck ? 0.35f : 0.0f;
            if (d.semantic != SemanticClass::SignalSign) {
                const float maxCenterY = isTallTruck ? 0.085f : (d.hNorm < 0.11f ? 0.005f : 0.025f);
                const float maxBotRayY = isTallTruck ? 0.025f : -0.003f;
                // Non-signal dynamic road users must have plausible center elevation and (for ground-lifted cars/peds) downward bottom ray
                if (rayCenterW.y > maxCenterY || camOriginW.y <= planeY + 0.05f || rayDirW.y >= maxBotRayY) {
                    continue;
                }
                const float tGroundRaw = (rayDirW.y < -0.003f) ? (-(camOriginW.y - planeY) / rayDirW.y) : spec.maxRangeM;
                if (!isTallTruck && tGroundRaw > spec.maxRangeM * 1.15f) {
                    continue;
                }
                // Reject flat road-paint pancakes whose world-implied height at ground range is < 0.58m.
                if (d.semantic == SemanticClass::Vehicle && !isTallTruck &&
                    vTop >= camHorizonV_[c] - 0.01f && d.hNorm < 0.045f && tGroundRaw > 6.0f &&
                    (tGroundRaw * hPx / std::max(1.0f, cm.intr.fy)) < 0.58f) {
                    continue;
                }
                const float tGround = clampf(tGroundRaw, 1.2f, spec.maxRangeM);
                // Calibrated ground-ray range dominates beyond ~15m for cars;
                // pedestrians blend 1px sub-cell angular torso height with shirt-bottom ground ray;
                // tall cargo trucks rely primarily on yaw-invariant angular height + pitch-calibrated ground ray.
                const float gWeight = isTallTruck
                                    ? clampf((-rayDirW.y - 0.045f) / 0.08f, 0.0f, 0.35f)
                                    : (d.semantic == SemanticClass::Pedestrian)
                                        ? clampf((tGroundRaw - 4.0f) / 10.0f, 0.78f, 0.985f)
                                        : clampf((tGroundRaw - 5.0f) / 15.0f, 0.65f, 0.98f);
                tEst = lerpf(tSize, tGround, gWeight);
                if (tEst > maxRangeM_) {
                    continue;
                }
                // Reject sub-6px Stage-2 extreme-fringe corner slivers in wide-FOV cameras whose monocular ground-ray depth inflates > 14m.
                if (spec.hfovDeg > 65.0f && d.semantic == SemanticClass::Pedestrian &&
                    d.score < cfg.rtdetrHighConfThresh && d.hNorm < 0.024f && tEst > 14.0f &&
                    (d.cxNorm < 0.16f || d.cxNorm > 0.84f)) {
                    continue;
                }
            }

            Vec3 wp = camOriginW + rayDirW * tEst;
            if (isTallTruck) {
                // Offset from visible rear cargo plane inward toward truck OBB center
                const float hLen = std::hypot(rayDirW.x, rayDirW.z);
                if (hLen > 1e-4f) {
                    wp.x += (rayDirW.x / hLen) * 1.65f;
                    wp.z += (rayDirW.z / hLen) * 1.65f;
                }
            }
            const Vec3 he = (d.semantic == SemanticClass::Pedestrian) ? Vec3{0.35f, 0.90f, 0.35f} :
                            (d.semantic == SemanticClass::SignalSign) ? Vec3{0.45f, 1.50f, 0.45f} :
                            isTallTruck ? Vec3{1.22f, 1.55f, 2.90f} :
                                          Vec3{0.96f, 0.72f, 2.18f};
            if (d.semantic != SemanticClass::SignalSign) wp.y = he.y;

            const float relX = wp.x - ego.pos.x;
            const float relZ = wp.z - ego.pos.z;
            const Vec3 egoP{relX * sinEgo + relZ * cosEgo, relX * cosEgo - relZ * sinEgo, wp.y};

            // Filter self-body detections (hood, fenders, rear decklid)
            if (d.semantic != SemanticClass::SignalSign &&
                egoP.x > -1.35f && egoP.x < 3.70f && std::abs(egoP.y) < 1.30f) {
                continue;
            }
            out2D.push_back(d);

            // Epistemic + Jacobian depth uncertainty propagation
            const float rangeUncert = (tEst * tEst / (cm.intr.fy * hPx)) * 2.5f + d.boxSigma * 0.8f;

            Detection3D d3;
            d3.camIndex = c;
            d3.semantic = d.semantic;
            d3.score = d.score;
            d3.worldPos = wp;
            d3.egoPos = egoP;
            d3.halfExtents = he;
            d3.yaw = ego.yaw;
            d3.rangeM = tEst;
            d3.camWeight = d.camPatchWeight;
            d3.epistemicSigma = clampf(rangeUncert, 0.15f, 8.0f);
            d3.det2d = d;
            raw3dScratch_.push_back(d3);
        }
    }

    // Cross-camera 3D NMS / fusion: anisotropic range-vs-lateral gate handles monocular depth spread along viewing rays
    // while keeping parallel-lane targets distinct and preserving per-camera out2D boxes for the HUD.
    auto isBorderClippedX = [](const Detection2D& det) {
        return (det.cxNorm - det.wNorm * 0.5f) < 0.025f || (det.cxNorm + det.wNorm * 0.5f) > 0.975f;
    };
    auto camSharpnessWeight = [&](const Detection3D& d) {
        const float fov = cfg.cameras.at(d.camIndex).hfovDeg;
        const bool clippedX = isBorderClippedX(d.det2d);
        if (fov < 40.0f) {
            // Telephoto wins on distant/mid-range compact targets, yielding to wider main cams when close, overflowing, or border-clipped.
            if (clippedX) return 0.32f;
            const bool tallMidTele = (d.semantic == SemanticClass::Vehicle && d.halfExtents.y > 1.20f) &&
                                     (d.det2d.wNorm > 0.205f && d.det2d.cyNorm < 0.46f);
            const bool saturatedTele = d.det2d.wNorm > 0.45f || d.det2d.hNorm > 0.50f;
            if (saturatedTele) return 0.48f;
            const bool closeTele = d.det2d.wNorm > (tallMidTele ? 0.30f : 0.20f) ||
                                   d.det2d.hNorm > (tallMidTele ? 0.36f : 0.26f) ||
                                   (d.det2d.cyNorm > 0.60f && d.det2d.hNorm > 0.12f) ||
                                   (d.det2d.cyNorm + d.det2d.hNorm * 0.5f) > 0.94f;
            return closeTele ? 0.50f : 2.15f;
        }
        if (fov > 65.0f) {
            const bool edgeFringe = clippedX || d.det2d.cxNorm < 0.16f || d.det2d.cxNorm > 0.84f;
            if (fov > 95.0f) {
                return edgeFringe ? 0.36f : 0.55f;
            }
            return edgeFringe ? 0.40f : 0.80f;
        }
        return clippedX ? 0.85f : 1.15f;
    };
    std::sort(raw3dScratch_.begin(), raw3dScratch_.end(),
              [&](const Detection3D& a, const Detection3D& b) {
                  const bool aHi = a.score >= cfg.rtdetrHighConfThresh;
                  const bool bHi = b.score >= cfg.rtdetrHighConfThresh;
                  if (aHi != bHi) return aHi;
                  return (a.score * std::max(0.2f, a.camWeight) * camSharpnessWeight(a)) >
                         (b.score * std::max(0.2f, b.camWeight) * camSharpnessWeight(b));
              });
    suppressed3dScratch_.assign(raw3dScratch_.size(), 0);
    for (size_t i = 0; i < raw3dScratch_.size(); ++i) {
        if (suppressed3dScratch_[i]) continue;
    Detection3D& keep = raw3dScratch_[i];
    keep.corroboratingCamMask = multicam::camBit(keep.camIndex);
    keep.corroboratingCamCount = multicam::kMinCorrobCams;
        const float relX = keep.worldPos.x - ego.pos.x;
        const float relZ = keep.worldPos.z - ego.pos.z;
        const float rayLen = std::hypot(relX, relZ);
        const float rx = (rayLen > 1e-3f) ? (relX / rayLen) : 0.0f;
        const float rz = (rayLen > 1e-3f) ? (relZ / rayLen) : 1.0f;
        const bool keepTallTruck = (keep.semantic == SemanticClass::Vehicle && keep.halfExtents.y > 1.20f);
        const float maxLat = (keep.semantic == SemanticClass::Pedestrian) ? 1.8f : (keepTallTruck ? 2.8f : 2.2f);
        const float maxRad = std::max(maxLat, 0.34f * std::max(keep.rangeM, rayLen));
        const float keepFov = cfg.cameras.at(keep.camIndex).hfovDeg;
        const bool keepIsSharp = keepFov <= 65.0f;

        for (size_t j = i + 1; j < raw3dScratch_.size(); ++j) {
            if (suppressed3dScratch_[j]) continue;
            const auto& cand = raw3dScratch_[j];
            if ((keep.semantic == SemanticClass::SignalSign) != (cand.semantic == SemanticClass::SignalSign)) {
                continue;
            }
            const float dx = cand.worldPos.x - keep.worldPos.x;
            const float dz = cand.worldPos.z - keep.worldPos.z;
            const float dRad = std::abs(dx * rx + dz * rz);
            const float dLat = std::abs(-dx * rz + dz * rx);
            const float cRelX = cand.worldPos.x - ego.pos.x;
            const float cRelZ = cand.worldPos.z - ego.pos.z;
            const float cLen = std::hypot(cRelX, cRelZ);
            const float candFov = cfg.cameras.at(cand.camIndex).hfovDeg;
            const bool candClipX = isBorderClippedX(cand.det2d);
            const bool candWidePeripheral = (candFov > 65.0f) && (candClipX || cand.det2d.cxNorm < 0.16f || cand.det2d.cxNorm > 0.84f);
            const bool candUltrawideRay = (candFov > 95.0f) && (cand.det2d.cxNorm < 0.22f || cand.det2d.cxNorm > 0.78f);
      constexpr float kTeleMaxFovDeg = 40.0f;
      constexpr float kTeleSatSpan = 0.65f;
      constexpr float kTeleCloseCy = 0.60f;
      constexpr float kTeleCloseH = 0.12f;
      const bool candTeleCloseSliver =
        (candFov < kTeleMaxFovDeg) && (cand.det2d.cyNorm > kTeleCloseCy) &&
        (cand.det2d.hNorm > kTeleCloseH) && (cand.rangeM > keep.rangeM) &&
        (cand.score < keep.score);
            // Distance from unclipped keep center to cand's viewing ray is invariant to cand's clipped/peripheral depth inflation.
            float dLatEff = dLat;
      if ((candClipX || candWidePeripheral || candUltrawideRay ||
           candTeleCloseSliver) &&
          cLen > 1e-3f) {
                const float crx = cRelX / cLen;
                const float crz = cRelZ / cLen;
                if ((relX * crx + relZ * crz) > 0.0f) {
                    const float dLatCandRay = std::abs(-dx * crz + dz * crx);
                    dLatEff = std::min(dLat, dLatCandRay);
                }
            }
            const bool anyTallTruck = keepTallTruck || (cand.semantic == SemanticClass::Vehicle && cand.halfExtents.y > 1.20f);
            float latGate = (cand.semantic == keep.semantic) ? maxLat : 1.8f;
            float radGate = (cand.semantic == keep.semantic) ? maxRad : std::max(2.5f, 0.34f * rayLen);
            if (anyTallTruck && cand.semantic == SemanticClass::Vehicle && keep.semantic == SemanticClass::Vehicle) {
                latGate = std::max(latGate, 2.8f);
                radGate = std::max(radGate, 4.4f + 0.24f * rayLen);
                if (keepIsSharp && (candFov > 65.0f || candClipX) &&
                    (candClipX || cand.det2d.cxNorm < 0.14f || cand.det2d.cxNorm > 0.86f)) {
                    latGate = std::max(latGate, 4.0f);
                } else if (keepIsSharp && (candFov > 65.0f || candClipX) &&
                           (cand.det2d.cxNorm < 0.18f || cand.det2d.cxNorm > 0.82f)) {
                    latGate = std::max(latGate, 3.6f);
                }
                if (keepTallTruck && keepIsSharp && keep.score >= cfg.rtdetrHighConfThresh &&
                    cand.score < cfg.rtdetrHighConfThresh && (candFov > 65.0f || candClipX)) {
                    latGate = std::max(latGate, (isBorderClippedX(keep.det2d) && candFov > 95.0f) ? 6.8f : 5.8f);
                    radGate = std::max(radGate, 6.2f + 0.25f * rayLen);
                }
            }
      const bool candClippedOrSat =
        candClipX || candTeleCloseSliver ||
        ((candFov < kTeleMaxFovDeg) &&
         (cand.det2d.wNorm > kTeleSatSpan ||
          cand.det2d.hNorm > kTeleSatSpan));
            if (cand.semantic == SemanticClass::Vehicle && keep.semantic == SemanticClass::Vehicle &&
                ((keepIsSharp && (candFov > 95.0f || candWidePeripheral || (keepFov < 40.0f && candFov >= 45.0f && cand.rangeM < keep.rangeM))) ||
                 candClippedOrSat)) {
                const float refR = std::max(rayLen, cand.rangeM);
                radGate = std::max(radGate, ((candFov > 95.0f || candWidePeripheral || candClippedOrSat) ? 0.72f : 0.48f) * refR);
            }
            if (dLatEff <= latGate && dRad <= radGate) {
              multicam::applyDetectionCorroboration(keep, cand);
              suppressed3dScratch_[j] = 1;
            }
        }
        out3D.push_back(keep);
    }
}

std::vector<uint8_t> BackboneRTDETR::downloadBackboneHeatmapRGBA(int camIdx, int outW, int outH) const {
    std::vector<Vec4> raw(static_cast<size_t>(8 * featW_ * featH_ * 4));
    featS4SSBO_.downloadVector(raw);

    std::vector<uint8_t> img(static_cast<size_t>(outW) * outH * 4, 255);
    for (int y = 0; y < outH; ++y) {
        int fy = clampf(static_cast<float>(y * featH_) / outH, 0.0f, static_cast<float>(featH_ - 1));
        for (int x = 0; x < outW; ++x) {
            int fx = clampf(static_cast<float>(x * featW_) / outW, 0.0f, static_cast<float>(featW_ - 1));
            size_t base = (static_cast<size_t>((camIdx * featH_ + fy) * featW_ + fx)) * 4;
            float edgeEnergy = raw[base + 0].z;
            float objEnergy = raw[base + 3].w;
            float patchW = raw[base + 3].y;

            float e = clampf((edgeEnergy * 0.5f + objEnergy * 0.9f) * 1.3f, 0.0f, 1.0f);
            uint8_t r = static_cast<uint8_t>(clampf(e * 1.45f, 0.0f, 1.0f) * 255.0f);
            uint8_t g = static_cast<uint8_t>(clampf((e - 0.30f) * 1.6f, 0.0f, 1.0f) * 255.0f);
            uint8_t b = static_cast<uint8_t>(clampf(0.22f + (1.0f - e) * 0.55f, 0.0f, 1.0f) * 255.0f);
            // Tint orange on soiled low-weight patches so user sees SoilDetector output in heatmap!
            if (patchW < 0.55f) {
                r = static_cast<uint8_t>(std::min(255, r + 90));
                g = static_cast<uint8_t>(g * 0.6f);
            }
            size_t di = (static_cast<size_t>(y) * outW + x) * 4;
            img[di + 0] = r;
            img[di + 1] = g;
            img[di + 2] = b;
            img[di + 3] = 255;
        }
    }
    return img;
}

} // namespace cybercab
