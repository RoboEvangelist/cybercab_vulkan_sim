#include "perception/PerceptionPipeline.hpp"
#include <cmath>
#include <limits>

namespace cybercab {

PerceptionPipeline::PerceptionPipeline(const SimConfig& cfg)
    : cfg_(cfg), tracker_(cfg) {
    trackPriorsScratch_.resize(static_cast<size_t>(cfg.btMaxTracks));
    gtDynScratch_.reserve(32);
    gtInRangeScratch_.reserve(32);
}

void PerceptionPipeline::init(const VulkanContext& ctx, const CameraRigRenderer& rig) {
    soil_.init(ctx, cfg_, rig.cameraTensorSSBO());
    detr_.init(ctx, cfg_, rig.cameraTensorSSBO(), soil_.patchSSBO());
    bev_.init(ctx, cfg_, detr_.encMemorySSBO(), soil_.patchSSBO());
    occ_.init(ctx, cfg_, bev_.currBevFeatSSBO(), bev_.currBevMetaSSBO(), bev_.pillarHeightSSBO());
    wm_.init(ctx, cfg_, occ_.trackPriorsSSBO());
}

void PerceptionPipeline::onCameraResized(const CameraRigRenderer& rig) {
    soil_.updateCameraTensorBinding(rig.cameraTensorSSBO());
    detr_.updateCameraTensorBinding(rig.cameraTensorSSBO(), rig.camWidth(), rig.camHeight());
}

void PerceptionPipeline::reset() {
    tracker_.reset();
    bev_.resetHistory();
    occ_.resetVoxels();
}

void PerceptionPipeline::destroy() {
    wm_.destroy();
    occ_.destroy();
    bev_.destroy();
    detr_.destroy();
    soil_.destroy();
}

void PerceptionPipeline::runFrame(const VulkanContext& ctx,
                                  const CameraRigRenderer& rig,
                                  const WorldEngine& world,
                                  float dt,
                                  bool enableShadows,
                                  PerceptionFrameOutput& out,
                                  GpuProfiler* profiler) {
    const EgoState& ego = world.ego();

    // GPU Batch 1: SoilDetector -> Backbone + FPN -> RT-DETR Encoder & Decoder
    {
        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        if (profiler) profiler->beginStage(cmd, "2_soil_detect", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        soil_.recordDispatch(cmd, cfg_, rig.camWidth(), rig.camHeight());
        if (profiler) {
            profiler->endStage(cmd, "2_soil_detect", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            profiler->beginStage(cmd, "3_backbone_fpn", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        detr_.recordBackbone(cmd, rig.camWidth(), rig.camHeight());
        if (profiler) {
            profiler->endStage(cmd, "3_backbone_fpn", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            profiler->beginStage(cmd, "4_rtdetr", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        detr_.recordRTDETR(cmd);
        detr_.copyCurrToPrevTensor(cmd);
        if (profiler) {
            profiler->endStage(cmd, "4_rtdetr", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        ctx.endSingleTimeCommands(cmd);
    }

    soil_.readbackResults(out.soilCameras, out.soilPatches, out.forwardEpistemicSigma);
    detr_.decodeAndLift3D(cfg_, rig, ego, out.detections2D, out.detections3D);
    tracker_.update(dt, ego, out.detections3D);
    tracker_.collectActiveTracks(out.tracks);
    tracker_.exportTrackPriorsGpu(ego, trackPriorsScratch_);

    // GPU Batch 2: BEVFormer (Temporal + Spatial) -> Occupancy + Flow + Shadows -> World Model
    {
        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        if (profiler) profiler->beginStage(cmd, "5_bevformer", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        bev_.recordDispatch(cmd, cfg_, rig, ego, dt, out.soilCameras);
        if (profiler) {
            profiler->endStage(cmd, "5_bevformer", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            profiler->beginStage(cmd, "6_occ_flow", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        occ_.recordDispatch(cmd, cfg_, ego, dt, trackPriorsScratch_, enableShadows);
        if (profiler) {
            profiler->endStage(cmd, "6_occ_flow", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            profiler->beginStage(cmd, "7_world_model", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        wm_.recordDispatch(cmd, cfg_, ego, static_cast<uint32_t>(trackPriorsScratch_.size()));
        bev_.copyCurrToPrev(cmd);
        if (profiler) {
            profiler->endStage(cmd, "7_world_model", VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        ctx.endSingleTimeCommands(cmd);
    }

    out.bevSummary = bev_.readbackSummary();
    occ_.readbackVoxels(cfg_, ego, out.activeVoxels, out.shadowVoxels);
    wm_.readbackGhosts(out.worldGhosts);

    if (profiler) {
        out.gpuStageMs = profiler->resolveTimingsMs();
    }
    out.eval = evaluateAgainstGroundTruth(world, out);
}

PerceptionFrameOutput PerceptionPipeline::runFrame(const VulkanContext& ctx,
                                                   const CameraRigRenderer& rig,
                                                   const WorldEngine& world,
                                                   float dt,
                                                   bool enableShadows,
                                                   GpuProfiler* profiler) {
    PerceptionFrameOutput out;
    runFrame(ctx, rig, world, dt, enableShadows, out, profiler);
    return out;
}

EvalMetrics PerceptionPipeline::evaluateAgainstGroundTruth(const WorldEngine& world, const PerceptionFrameOutput& out) const {
    EvalMetrics m{};
    world.collectDynamicGroundTruthOBBs(gtDynScratch_);
    const EgoState& ego = world.ego();
    gtInRangeScratch_.clear();
    for (const auto& g : gtDynScratch_) {
        if (std::hypot(g.cx - ego.pos.x, g.cz - ego.pos.z) <= 42.0f) {
            gtInRangeScratch_.push_back(g);
        }
    }
    m.totalGtDynamic = static_cast<int>(gtInRangeScratch_.size());
    if (gtInRangeScratch_.empty()) return m;

    int matchedGt = 0;
    double sqErrSum = 0.0;
    for (const auto& g : gtInRangeScratch_) {
        float bestD = std::numeric_limits<float>::infinity();
        for (const auto& d : out.detections3D) {
            if (d.semantic == SemanticClass::SignalSign) continue;
            const float dist = std::hypot(d.worldPos.x - g.cx, d.worldPos.z - g.cz);
            bestD = std::min(bestD, dist);
        }
        if (bestD < 6.0f) {
            matchedGt++;
            sqErrSum += bestD * bestD;
        }
    }

    int validDets = 0;
    int tpDets = 0;
    for (const auto& d : out.detections3D) {
        if (d.semantic == SemanticClass::SignalSign || d.rangeM > 42.0f) continue;
        validDets++;
        for (const auto& g : gtInRangeScratch_) {
            if (std::hypot(d.worldPos.x - g.cx, d.worldPos.z - g.cz) < 6.0f) {
                tpDets++;
                break;
            }
        }
    }

    m.matchedGtObjects = matchedGt;
    m.detRecall = static_cast<float>(matchedGt) / static_cast<float>(gtInRangeScratch_.size());
    m.detPrecision = (validDets > 0) ? (static_cast<float>(tpDets) / static_cast<float>(validDets)) : 0.0f;
    m.trackPosRmseM = (matchedGt > 0) ? static_cast<float>(std::sqrt(sqErrSum / matchedGt)) : 0.0f;
    return m;
}

} // namespace cybercab
