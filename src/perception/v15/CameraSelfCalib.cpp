#include "perception/v15/CameraSelfCalib.hpp"
#include <algorithm>
#include <cmath>

namespace cybercab::v15 {
namespace {
constexpr float kMinGradPx = 0.06f;     // per-pixel |grad log Y| floor inside the shader
constexpr float kMinTileMag = 0.35f;    // tile filter before readback-side processing
constexpr float kMinTileCoher = 0.75f;
constexpr size_t kMaxFrameEqReserve = 512u;
constexpr size_t kRingFrameCap =
  static_cast<size_t>(CameraSelfCalib::kWindowFrames) + 2u;
} // namespace

CameraSelfCalib::~CameraSelfCalib() { destroy(); }

void CameraSelfCalib::init(const VulkanContext& ctx, const SimConfig& cfg, const VulkanBuffer& bandsSSBO,
                           int camW, int camH) {
    destroy();
    ctx_ = &ctx;
    numCams_ = cfg.cameras.empty() ? 8 : static_cast<int>(cfg.cameras.size());
    pipe_.create(ctx, "shaders/v15_self_calib.comp.spv",
                 {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1}},
                 sizeof(SelfCalibPC));
    set_ = pipe_.allocateDescriptorSet();
    rebind(bandsSSBO, camW, camH);
}

void CameraSelfCalib::rebind(const VulkanBuffer& bandsSSBO, int camW, int camH) {
    if (!ctx_) return;
    w_ = camW;
    h_ = camH;
  constexpr uint32_t kBaseTilePx = 8u;
  constexpr float kRefCalibW = 320.0f;
  const float wScale = static_cast<float>(camW) / kRefCalibW;
  tilePx_ = kBaseTilePx *
    std::max(1u, static_cast<uint32_t>(std::lround(wScale)));
    tilesX_ = (static_cast<uint32_t>(camW) + tilePx_ - 1) / tilePx_;
    tilesY_ = (static_cast<uint32_t>(camH) + tilePx_ - 1) / tilePx_;
    const size_t nTiles = static_cast<size_t>(numCams_) * tilesX_ * tilesY_;
    tilesSSBO_.destroy();
    tilesSSBO_.create(*ctx_, nTiles * sizeof(TileGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    tilesScratch_.resize(nTiles);
    obsScratch_.reserve(nTiles);
  frameEqScratch_.reserve(kMaxFrameEqReserve);
  eqRing_.reserve(kMaxFrameEqReserve * kRingFrameCap);
  eqFrameCounts_.reserve(kRingFrameCap);

    const VkDescriptorBufferInfo inInfo = bandsSSBO.descriptorInfo();
    const VkDescriptorBufferInfo outInfo = tilesSSBO_.descriptorInfo();
    const VkWriteDescriptorSet w[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &inInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &outInfo, nullptr},
    };
    vkUpdateDescriptorSets(ctx_->device(), 2, w, 0, nullptr);
}

void CameraSelfCalib::reset() {
    state_ = {};
    eqRing_.clear();
    eqFrameCounts_.clear();
}

void CameraSelfCalib::destroy() {
    if (!ctx_) return;
    tilesSSBO_.destroy();
    pipe_.destroy();
    set_ = VK_NULL_HANDLE;
    ctx_ = nullptr;
    reset();
}

void CameraSelfCalib::recordDispatch(VkCommandBuffer cmd) const {
    if (!ctx_ || tilesX_ == 0 || tilesY_ == 0) return;
    const SelfCalibPC pc{static_cast<uint32_t>(w_), static_cast<uint32_t>(h_),
                         static_cast<uint32_t>(numCams_), tilePx_, tilesX_, tilesY_, kMinGradPx};
    pipe_.bindAndDispatch(cmd, set_, tilesX_, tilesY_, static_cast<uint32_t>(numCams_), &pc, sizeof(pc));
    cmdComputeBarrier(cmd);
}

CalibEstimate CameraSelfCalib::solveAndUpdate(const SimConfig& cfg) {
    if (!ctx_ || tilesScratch_.empty()) return state_;
    tilesSSBO_.downloadVector(tilesScratch_);
    obsScratch_.clear();
    for (const auto& t : tilesScratch_) {
        if (t.gradMag.z >= kMinTileMag && t.gradMag.w >= kMinTileCoher) {
            obsScratch_.push_back({static_cast<int>(t.centroidCam.z + 0.5f), t.centroidCam.x, t.centroidCam.y,
                                   t.gradMag.x, t.gradMag.y, t.gradMag.z, t.gradMag.w});
        }
    }
    // Chain planes only: noisy per-tile fallback planes would dominate a multi-frame window.
    calib::extractEdgeNormals(cfg, w_, h_, obsScratch_, state_.pitchRad, state_.rollRad, frameEqScratch_, false);
    eqRing_.insert(eqRing_.end(), frameEqScratch_.begin(), frameEqScratch_.end());
    eqFrameCounts_.push_back(static_cast<int>(frameEqScratch_.size()));
    if (static_cast<int>(eqFrameCounts_.size()) > kWindowFrames) {
        eqRing_.erase(eqRing_.begin(), eqRing_.begin() + eqFrameCounts_.front());
        eqFrameCounts_.erase(eqFrameCounts_.begin());
    }
    const CalibEstimate meas = calib::solveUpright(eqRing_, state_.pitchRad, state_.rollRad);
    if (meas.converged) state_ = meas;
    return state_;
}

} // namespace cybercab::v15
