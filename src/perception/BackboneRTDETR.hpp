#pragma once

#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionTypes.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include <array>
#include <vector>

namespace cybercab {

class BackboneRTDETR {
public:
    BackboneRTDETR() = default;
    ~BackboneRTDETR();

    void init(const VulkanContext& ctx, const SimConfig& cfg,
              const VulkanBuffer& camTensorSSBO, const VulkanBuffer& soilPatchSSBO);
    void updateCameraTensorBinding(const VulkanBuffer& camTensorSSBO, int camW, int camH);
    void updateHorizonFromRig(const CameraRigRenderer& rig);
    void resetHistory();
    void destroy();

    void recordBackbone(VkCommandBuffer cmd, int camW, int camH) const;
    void recordRTDETR(VkCommandBuffer cmd) const;
    void copyCurrToPrevTensor(VkCommandBuffer cmd) const;

    // Reads back 128 compact RT-DETR queries, runs per-camera NMS, and lifts to 3D World/Ego space.
    void decodeAndLift3D(const SimConfig& cfg,
                         const CameraRigRenderer& rig,
                         const EgoState& ego,
                         std::vector<Detection2D>& out2D,
                         std::vector<Detection3D>& out3D) const;

    // Readback helper for HUD "Backbone Heatmap" mode
    std::vector<uint8_t> downloadBackboneHeatmapRGBA(int camIdx, int outW, int outH) const;

    const VulkanBuffer& featS4SSBO() const { return featS4SSBO_; }
    const VulkanBuffer& encMemorySSBO() const { return encMemorySSBO_; }

private:
    const VulkanContext* ctx_ = nullptr;
    int featW_ = 40;
    int featH_ = 30;
    int s5W_ = 20;
    int s5H_ = 15;
    int patchCols_ = 10;
    int patchRows_ = 8;
    int numQueries_ = 16;
    int numLayers_ = 2;
    float nmsIouThresh_ = 0.45f;
    float maxRangeM_ = 65.0f;
    float lowConfThresh_ = 0.24f;
    VkDeviceSize tensorBytes_ = 0;

    VulkanComputePipeline backbonePipe_;
    VulkanComputePipeline detrEncPipe_;
    VulkanComputePipeline detrDecPipe_;

    VkDescriptorSet bbSet_ = VK_NULL_HANDLE;
    VkDescriptorSet encSet_ = VK_NULL_HANDLE;
    VkDescriptorSet decSet_ = VK_NULL_HANDLE;

    VulkanBuffer prevCamTensorSSBO_;
    VulkanBuffer featS4SSBO_;
    VulkanBuffer featS5SSBO_;
    VulkanBuffer encMemorySSBO_;
    VulkanBuffer queryInitSSBO_;
    VulkanBuffer detrOutSSBO_;

    const VulkanBuffer* currCamTensorRef_ = nullptr;
    const VulkanBuffer* soilPatchRef_ = nullptr;
    mutable bool hasPrevTensor_ = false;
    std::array<float, 8> camHorizonV_{0.462f, 0.462f, 0.476f, 0.465f, 0.465f, 0.469f, 0.469f, 0.464f};
    std::array<float, 8> camHorizonSlope_{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    mutable std::vector<Vec4> detrRawScratch_;
    mutable std::vector<Detection2D> camCandsScratch_;
    mutable std::vector<Detection2D> keptScratch_;
    mutable std::vector<Detection3D> raw3dScratch_;
    mutable std::vector<uint8_t> suppressed3dScratch_;
};

} // namespace cybercab
