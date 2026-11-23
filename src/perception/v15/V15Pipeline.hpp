#pragma once
// FSD v15 pipeline orchestrator attached conditionally when cfg.isV15() is active.
// Owns v15-only GPU/CPU modules so v14 executes with zero v15 allocations or dispatches.
#include "config/SimConfig.hpp"
#include "core/GpuProfiler.hpp"
#include "core/VulkanContext.hpp"
#include "perception/v15/AudioTokenizer.hpp"
#include "perception/v15/CameraSelfCalib.hpp"
#include "perception/v15/FrequencySplit.hpp"
#include "perception/v15/GaussianWorldModel.hpp"
#include "perception/v15/KinematicNavEncoder.hpp"
#include "perception/v15/LatentPredictor.hpp"
#include "perception/v15/PatchTokenizer.hpp"
#include "perception/v15/StaticSpatialMemory.hpp"
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

    // Updates 48 kHz audio token + 100 Hz IMU/nav tokens prior to Batch 1.
    void beginFrameCpu(const WorldEngine& world, float dt);
    // Dispatches M1 frequency split, M2a self-calib, M2b virtual rectify, and M3b patch tokenizer.
    void recordFrontEnd(VkCommandBuffer cmd, GpuProfiler* profiler = nullptr, float dt = 1.0f / 36.0f);
    // Consumes Batch 1 GPU readbacks (e.g. M2a tile gradients) after queue wait.
    void finishBatch1Cpu();
    // Dispatches Batch 2 v15 stages (M5 static spatial memory).
    void recordBackEnd(VkCommandBuffer cmd, const EgoState& ego, float dt,
                       const std::vector<TrackPriorGpu>& tracks, GpuProfiler* profiler = nullptr);
    void finishBackEnd(const VulkanContext& ctx, const EgoState& ego,
                       PerceptionFrameOutput& out, GpuProfiler* profiler = nullptr);

    bool isInitialized() const { return initialized_; }
    const FrequencySplit& frequencySplit() const { return freqSplit_; }
    const CameraSelfCalib& selfCalib() const { return selfCalib_; }
    const VirtualCameraRectifier& rectifier() const { return rectifier_; }
    const PatchTokenizer& patchTokenizer() const { return patchTokenizer_; }
    const StaticSpatialMemory& staticMemory() const { return staticMem_; }
    const GaussianWorldModel& gaussianWorld() const { return gaussianWorld_; }
    const LatentPredictor& latentPredictor() const { return latentPred_; }
    const AudioToken& audioToken() const { return audioToken_; }
    const KinematicNavEncoder& kinematicNav() const { return kinNav_; }

private:
    const SimConfig* cfg_ = nullptr;
    bool initialized_ = false;
    FrequencySplit freqSplit_;
    CameraSelfCalib selfCalib_;
    VirtualCameraRectifier rectifier_;
    PatchTokenizer patchTokenizer_;
    StaticSpatialMemory staticMem_;
    GaussianWorldModel gaussianWorld_;
    LatentPredictor latentPred_;
    AudioToken audioToken_{};
    KinematicNavEncoder kinNav_;
    uint32_t frameIdx_ = 0;
    bool hasPrevEgo_ = false;
    EgoState prevEgo_{};
};

} // namespace cybercab::v15
