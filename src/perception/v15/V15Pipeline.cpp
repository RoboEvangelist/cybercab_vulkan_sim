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
    initialized_ = true;
}

void V15Pipeline::onCameraResized(const CameraRigRenderer& rig) {
    if (!initialized_ || !cfg_) return;
    const int w = rig.camWidth(), h = rig.camHeight();
    freqSplit_.rebind(rig.cameraLinearSSBO(), w, h, FreqSplitParams::forConfig(w, *cfg_));
    selfCalib_.rebind(freqSplit_.bandsSSBO(), w, h);
    rectifier_.rebind(*cfg_, rig.cameraLinearSSBO(), freqSplit_.bandsSSBO(), w, h);
    rectifier_.updateViews(*cfg_, w, h, selfCalib_.vehicleCalibMatrix());
}

void V15Pipeline::reset() {
    selfCalib_.reset();
    if (initialized_ && cfg_) {
        rectifier_.updateViews(*cfg_, freqSplit_.width(), freqSplit_.height(), selfCalib_.vehicleCalibMatrix());
    }
}

void V15Pipeline::destroy() {
    rectifier_.destroy();
    selfCalib_.destroy();
    freqSplit_.destroy();
    initialized_ = false;
    cfg_ = nullptr;
}

void V15Pipeline::recordFrontEnd(VkCommandBuffer cmd, GpuProfiler* profiler) {
    if (!initialized_ || !cfg_) return;
    rectifier_.updateViews(*cfg_, freqSplit_.width(), freqSplit_.height(), selfCalib_.vehicleCalibMatrix());
    runStage(cmd, profiler, "1c_v15_freq_split", [&] { freqSplit_.recordDispatch(cmd); });
    runStage(cmd, profiler, "2a_v15_self_calib", [&] { selfCalib_.recordDispatch(cmd); });
    runStage(cmd, profiler, "2b_v15_virt_rectify", [&] { rectifier_.recordDispatch(cmd); });
}

void V15Pipeline::finishBatch1Cpu() {
    if (!initialized_ || !cfg_) return;
    selfCalib_.solveAndUpdate(*cfg_);
}

} // namespace cybercab::v15
