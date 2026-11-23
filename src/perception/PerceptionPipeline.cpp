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
    detr_.resetHistory();
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
    detr_.updateHorizonFromRig(rig);

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
    double sumMs = 0.0;
    for (const auto& kv : out.gpuStageMs) {
        sumMs += kv.second;
    }
    out.eval.totalPipelineMs = sumMs;
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
        bool matchedSameClass = false;
        for (const auto& g : gtInRangeScratch_) {
            const float dist = std::hypot(d.worldPos.x - g.cx, d.worldPos.z - g.cz);
            if (g.semantic == d.semantic && dist < ((d.semantic == SemanticClass::Pedestrian) ? 3.5f : 5.0f)) {
                matchedSameClass = true;
            }
        }
        if (matchedSameClass) tpDets++;
        if (!matchedSameClass) {
            if (d.semantic == SemanticClass::Pedestrian) m.ghostPedDetections++;
            else if (d.semantic == SemanticClass::Vehicle) m.ghostVehDetections++;
        }
    }

    if (!world.vehicles().empty()) {
        const auto& truck = world.vehicles().at(0);
        const float truckRange = std::hypot(truck.pos.x - ego.pos.x, truck.pos.z - ego.pos.z);
        if (truckRange <= 42.0f) {
            for (const auto& d : out.detections3D) {
                if (d.semantic == SemanticClass::Pedestrian &&
                    std::hypot(d.worldPos.x - truck.pos.x, d.worldPos.z - truck.pos.z) < 5.5f) {
                    m.misclassifiedTrucks++;
                }
            }
            const float cosY = std::cos(truck.yaw), sinY = std::sin(truck.yaw);
            for (const auto& v : out.activeVoxels) {
                const float dx = v.wx - truck.pos.x;
                const float dz = v.wz - truck.pos.z;
                const float locX = dx * cosY - dz * sinY;
                const float locZ = dx * sinY + dz * cosY;
                if (std::abs(locX) <= truck.halfExtents.x + 0.6f &&
                    std::abs(locZ) <= truck.halfExtents.z + 0.6f &&
                    v.wy <= truck.halfExtents.y * 2.0f + 0.6f) {
                    if (v.sem == SemanticClass::Vehicle) m.truckVehVoxels++;
                    else if (v.sem == SemanticClass::Pedestrian) m.truckPedVoxels++;
                }
            }
        }
    }

    const Vec2 signAnchors[] = {
        {6.5f, WorldEngine::kLoopHalfZ - 6.2f},
        {12.5f, -WorldEngine::kLoopHalfZ - 5.6f},
        {-34.0f, WorldEngine::kLoopHalfZ + 5.6f},
        {WorldEngine::kLoopHalfX - 6.0f, 0.0f},
        {WorldEngine::kLoopHalfX + 6.0f, 0.0f},
    };
    for (const auto& v : out.activeVoxels) {
        if (v.sem != SemanticClass::SignalSign && v.sem != SemanticClass::Static) continue;
        for (const auto& sa : signAnchors) {
            if (std::hypot(v.wx - sa.x, v.wz - sa.y) <= 2.2f) {
                m.signPoleVoxels++;
                break;
            }
        }
    }

    m.matchedGtObjects = matchedGt;
    m.detRecall = (!gtInRangeScratch_.empty()) ? (static_cast<float>(matchedGt) / static_cast<float>(gtInRangeScratch_.size())) : 0.0f;
    m.detPrecision = (validDets > 0) ? (static_cast<float>(tpDets) / static_cast<float>(validDets)) : 0.0f;
    m.trackPosRmseM = (matchedGt > 0) ? static_cast<float>(std::sqrt(sqErrSum / matchedGt)) : 0.0f;
    return m;
}

} // namespace cybercab
