#include "perception/v15/V15Pipeline.hpp"

#include <stdexcept>

namespace cybercab::v15 {

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
    freqSplit_.init(ctx,
                    rig.cameraLinearSSBO(),
                    rig.camWidth(),
                    rig.camHeight(),
                    numCams,
                    FreqSplitParams::forConfig(rig.camWidth(), cfg));
    initialized_ = true;
}

void V15Pipeline::onCameraResized(const CameraRigRenderer& rig) {
    if (!initialized_ || !cfg_) return;
    freqSplit_.rebind(rig.cameraLinearSSBO(),
                      rig.camWidth(),
                      rig.camHeight(),
                      FreqSplitParams::forConfig(rig.camWidth(), *cfg_));
}

void V15Pipeline::reset() {
    // M1 FrequencySplit is stateless per frame; M2/M3 temporal buffers will reset here.
}

void V15Pipeline::destroy() {
    freqSplit_.destroy();
    initialized_ = false;
    cfg_ = nullptr;
}

void V15Pipeline::recordFrontEnd(VkCommandBuffer cmd, GpuProfiler* profiler) const {
    if (!initialized_) return;
    if (profiler) {
        profiler->beginStage(cmd, "1c_v15_freq_split", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    freqSplit_.recordDispatch(cmd);
    if (profiler) {
        profiler->endStage(cmd, "1c_v15_freq_split", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

} // namespace cybercab::v15
