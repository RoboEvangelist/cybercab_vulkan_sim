#pragma once
// FSD v15 pipeline orchestrator attached conditionally when cfg.isV15() is active.
// Owns v15-only GPU/CPU modules so v14 executes with zero v15 allocations or dispatches.
#include "config/SimConfig.hpp"
#include "core/GpuProfiler.hpp"
#include "core/VulkanContext.hpp"
#include "perception/v15/FrequencySplit.hpp"
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

    // Dispatches M1 pre-network frequency split into Batch 1 before downstream extractors.
    void recordFrontEnd(VkCommandBuffer cmd, GpuProfiler* profiler = nullptr) const;

    bool isInitialized() const { return initialized_; }
    const FrequencySplit& frequencySplit() const { return freqSplit_; }

private:
    const SimConfig* cfg_ = nullptr;
    bool initialized_ = false;
    FrequencySplit freqSplit_;
};

} // namespace cybercab::v15
