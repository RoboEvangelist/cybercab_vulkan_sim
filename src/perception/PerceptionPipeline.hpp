#pragma once

#include "config/SimConfig.hpp"
#include "core/GpuProfiler.hpp"
#include "core/VulkanContext.hpp"
#include "perception/BEVFormer.hpp"
#include "perception/BackboneRTDETR.hpp"
#include "perception/ByteTrack.hpp"
#include "perception/OccupancyFlowNet.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/SoilDetector.hpp"
#include "perception/WorldModel.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "world/WorldEngine.hpp"
#include <vector>

namespace cybercab {

class PerceptionPipeline {
public:
    explicit PerceptionPipeline(const SimConfig& cfg);

    void init(const VulkanContext& ctx, const CameraRigRenderer& rig);
    void onCameraResized(const CameraRigRenderer& rig);
    void reset();
    void destroy();

    // Hot-path zero-allocation overload writing into caller-owned PerceptionFrameOutput
    void runFrame(const VulkanContext& ctx,
                  const CameraRigRenderer& rig,
                  const WorldEngine& world,
                  float dt,
                  bool enableShadows,
                  PerceptionFrameOutput& out,
                  GpuProfiler* profiler = nullptr);

    PerceptionFrameOutput runFrame(const VulkanContext& ctx,
                                   const CameraRigRenderer& rig,
                                   const WorldEngine& world,
                                   float dt,
                                   bool enableShadows,
                                   GpuProfiler* profiler = nullptr);

    const SoilDetector& soilDetector() const { return soil_; }
    const BackboneRTDETR& backboneRtdetr() const { return detr_; }
    const ByteTracker& byteTracker() const { return tracker_; }
    const BEVFormer& bevFormer() const { return bev_; }
    const OccupancyFlowNet& occupancyFlow() const { return occ_; }
    const WorldModel& worldModel() const { return wm_; }

private:
    const SimConfig& cfg_;
    SoilDetector soil_;
    BackboneRTDETR detr_;
    ByteTracker tracker_;
    BEVFormer bev_;
    OccupancyFlowNet occ_;
    WorldModel wm_;

    // Pre-allocated hot-path scratch buffers
    std::vector<TrackPriorGpu> trackPriorsScratch_;
    mutable std::vector<OBB> gtDynScratch_;
    mutable std::vector<OBB> gtInRangeScratch_;

    EvalMetrics evaluateAgainstGroundTruth(const WorldEngine& world, const PerceptionFrameOutput& out) const;
};

} // namespace cybercab
