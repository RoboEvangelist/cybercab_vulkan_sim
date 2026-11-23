#pragma once
// FSD v15 pipeline orchestrator attached conditionally when cfg.isV15() is active.
// Owns v15-only GPU/CPU modules so v14 executes with zero v15 allocations or dispatches.
#include "config/SimConfig.hpp"
#include "core/GpuProfiler.hpp"
#include "core/VulkanContext.hpp"
#include "perception/v15/CameraSelfCalib.hpp"
#include "perception/v15/FrequencySplit.hpp"
#include "perception/v15/VirtualCameraRectifier.hpp"
#include "sensors/CameraRigRenderer.hpp"

namespace cybercab::v15 {

class V15Pipeline {
public:
    V15Pipeline() = default;
    ~V15Pipeline();
    V15Pipeline(const V15Pipeline&) = delete;
    V15Pipeline& operator=(const V15Pipeline&) = delete;

    void init(const VulkanContext& ctx, const SimConfig& cfg, const CameraRigRenderer& rig);
    void onCameraResized(const CameraRigRenderer& rig);
    void reset();
    void destroy();

    // Dispatches M1 frequency split, M2a self-calib tile reduction, and M2b virtual rectification.
    void recordFrontEnd(VkCommandBuffer cmd, GpuProfiler* profiler = nullptr);
    // Consumes Batch 1 GPU readbacks (e.g. M2a tile gradients) after queue wait.
    void finishBatch1Cpu();

    bool isInitialized() const { return initialized_; }
    const FrequencySplit& frequencySplit() const { return freqSplit_; }
    const CameraSelfCalib& selfCalib() const { return selfCalib_; }
    const VirtualCameraRectifier& rectifier() const { return rectifier_; }

private:
    const SimConfig* cfg_ = nullptr;
    bool initialized_ = false;
    FrequencySplit freqSplit_;
    CameraSelfCalib selfCalib_;
    VirtualCameraRectifier rectifier_;
};

} // namespace cybercab::v15
