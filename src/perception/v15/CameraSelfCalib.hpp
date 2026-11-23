#pragma once
// FSD v15 Module 2a (GPU front-end): reduces M1c FrequencySplit gradient bands into coherent
// edge tiles (v15_self_calib.comp), then feeds a sliding window of edge planes to calib::solveUpright.
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/v15/CalibSolver.hpp"

#include <vector>

namespace cybercab::v15 {

class CameraSelfCalib {
public:
    static constexpr int kWindowFrames = 36; // 1 s at 36 Hz

    CameraSelfCalib() = default;
    ~CameraSelfCalib();
    CameraSelfCalib(const CameraSelfCalib&) = delete;
    CameraSelfCalib& operator=(const CameraSelfCalib&) = delete;

    void init(const VulkanContext& ctx, const SimConfig& cfg, const VulkanBuffer& bandsSSBO, int camW, int camH);
    void rebind(const VulkanBuffer& bandsSSBO, int camW, int camH);
    void reset();
    void destroy();

    void recordDispatch(VkCommandBuffer cmd) const;
    // Reads back tiles, appends this frame's chain planes to the window, re-solves the whole window.
    CalibEstimate solveAndUpdate(const SimConfig& cfg);

    const CalibEstimate& state() const { return state_; }
    int windowConstraintCount() const { return static_cast<int>(eqRing_.size()); }
    int windowFrameCount() const { return static_cast<int>(eqFrameCounts_.size()); }
    // R such that R * up_measured = (0, 1, 0): removes sprung pitch/roll from vehicle-frame rays.
    Mat4 vehicleCalibMatrix() const { return Mat4::rotX(state_.pitchRad) * Mat4::rotZ(state_.rollRad); }

private:
    struct TileGpu { Vec4 gradMag; Vec4 centroidCam; };
    struct SelfCalibPC { uint32_t width, height, numCams, tilePx, tilesX, tilesY; float minGrad; };

    const VulkanContext* ctx_ = nullptr;
    int w_ = 0, h_ = 0, numCams_ = 8;
    uint32_t tilePx_ = 8, tilesX_ = 0, tilesY_ = 0;
    VulkanComputePipeline pipe_;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VulkanBuffer tilesSSBO_;
    std::vector<TileGpu> tilesScratch_;
    std::vector<CalibRayObs> obsScratch_;
    std::vector<Vec4> frameEqScratch_;
    std::vector<Vec4> eqRing_;
    std::vector<int> eqFrameCounts_;
    CalibEstimate state_{};
};

} // namespace cybercab::v15
