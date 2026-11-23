#include "perception/v15/V15Pipeline.hpp"

#include <stdexcept>

namespace cybercab::v15 {
namespace {

template <typename Fn>
void runStage(VkCommandBuffer cmd, GpuProfiler* profiler, const char* name, Fn&& fn) {
    if (profiler) profiler->beginStage(cmd, name, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    fn();
    if (profiler) profiler->endStage(cmd, name, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

} // namespace

V15Pipeline::~V15Pipeline() {
    destroy();
}

void V15Pipeline::init(const VulkanContext& ctx, const SimConfig& cfg, const CameraRigRenderer& rig) {
    destroy();
    if (!cfg.isV15()) {
        throw std::runtime_error("V15Pipeline::init requires simulation.pipeline_version == v15");
    }
    if (!rig.linearHdrEnabled()) {
        throw std::runtime_error("V15Pipeline::init requires HDR linear cameraLinearSSBO()");
    }
    cfg_ = &cfg;
    const int numCams = cfg.cameras.empty() ? 8 : static_cast<int>(cfg.cameras.size());
    const int w = rig.camWidth(), h = rig.camHeight();
    freqSplit_.init(ctx, rig.cameraLinearSSBO(), w, h, numCams, FreqSplitParams::forConfig(w, cfg));
    selfCalib_.init(ctx, cfg, freqSplit_.bandsSSBO(), w, h);
    rectifier_.init(ctx, cfg, rig.cameraLinearSSBO(), freqSplit_.bandsSSBO(), w, h);
    patchTokenizer_.init(ctx, rectifier_);
    staticMem_.init(ctx, cfg, rectifier_, patchTokenizer_);
    gaussianWorld_.initSplatGpu(ctx);
    initialized_ = true;
}

void V15Pipeline::onCameraResized(const CameraRigRenderer& rig) {
    if (!initialized_ || !cfg_) return;
    const int w = rig.camWidth(), h = rig.camHeight();
    freqSplit_.rebind(rig.cameraLinearSSBO(), w, h, FreqSplitParams::forConfig(w, *cfg_));
    selfCalib_.rebind(freqSplit_.bandsSSBO(), w, h);
    rectifier_.rebind(*cfg_, rig.cameraLinearSSBO(), freqSplit_.bandsSSBO(), w, h);
    rectifier_.updateViews(*cfg_, w, h, selfCalib_.vehicleCalibMatrix());
    patchTokenizer_.updateSpecs(rectifier_);
}

void V15Pipeline::reset() {
    selfCalib_.reset();
    patchTokenizer_.reset();
    staticMem_.reset();
    gaussianWorld_.reset();
    kinNav_.reset();
    audioToken_ = {};
    frameIdx_ = 0;
    hasPrevEgo_ = false;
    if (initialized_ && cfg_) {
        rectifier_.updateViews(*cfg_, freqSplit_.width(), freqSplit_.height(), selfCalib_.vehicleCalibMatrix());
        patchTokenizer_.updateSpecs(rectifier_);
    }
}

void V15Pipeline::destroy() {
    gaussianWorld_.destroy();
    staticMem_.destroy();
    patchTokenizer_.destroy();
    rectifier_.destroy();
    selfCalib_.destroy();
    freqSplit_.destroy();
    kinNav_.reset();
    audioToken_ = {};
    frameIdx_ = 0;
    hasPrevEgo_ = false;
    initialized_ = false;
    cfg_ = nullptr;
}

void V15Pipeline::beginFrameCpu(const WorldEngine& world, float dt) {
    if (!initialized_) return;
    audioToken_ = AudioTokenizer::stepFromWorld(world, frameIdx_++);
    const EgoState& curr = world.ego();
    kinNav_.pushControlStep(hasPrevEgo_ ? prevEgo_ : curr, curr, dt);
    kinNav_.encodeNavigation(world, (curr.currentLateralOffset >= 0.0f) ? 1.0f : -1.0f);
    prevEgo_ = curr;
    hasPrevEgo_ = true;
}

void V15Pipeline::recordFrontEnd(VkCommandBuffer cmd, GpuProfiler* profiler, float dt) {
    if (!initialized_ || !cfg_) return;
    rectifier_.updateViews(*cfg_, freqSplit_.width(), freqSplit_.height(), selfCalib_.vehicleCalibMatrix());
    patchTokenizer_.updateSpecs(rectifier_);
    runStage(cmd, profiler, "1c_v15_freq_split", [&] { freqSplit_.recordDispatch(cmd); });
    runStage(cmd, profiler, "2a_v15_self_calib", [&] { selfCalib_.recordDispatch(cmd); });
    runStage(cmd, profiler, "2b_v15_virt_rectify", [&] { rectifier_.recordDispatch(cmd); });
    runStage(cmd, profiler, "3_v15_patch_tokens", [&] { patchTokenizer_.recordDispatch(cmd, dt); });
}

void V15Pipeline::finishBatch1Cpu() {
    if (!initialized_ || !cfg_) return;
    selfCalib_.solveAndUpdate(*cfg_);
    if (hasPrevEgo_) patchTokenizer_.recordSlotEgo(prevEgo_);
}

void V15Pipeline::recordBackEnd(VkCommandBuffer cmd, const EgoState& ego, float dt,
                                const std::vector<TrackPriorGpu>& tracks, GpuProfiler* profiler) {
    if (!initialized_ || !cfg_) return;
    runStage(cmd, profiler, "5_v15_static_mem",
             [&] { staticMem_.recordDispatch(cmd, *cfg_, ego, dt, rectifier_, patchTokenizer_, tracks); });
}

void V15Pipeline::finishBackEnd(const VulkanContext& ctx, const EgoState& ego,
                                PerceptionFrameOutput& out, GpuProfiler* profiler) {
    if (!initialized_ || !cfg_) return;
    gaussianWorld_.updateFromPerception(*cfg_, ego, staticMem_, patchTokenizer_, out);
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    runStage(cmd, profiler, "6_v15_gaussian_splat",
             [&] { gaussianWorld_.dispatchRealitySplat(cmd, *cfg_, ego); });
    ctx.endSingleTimeCommands(cmd);
}

} // namespace cybercab::v15
