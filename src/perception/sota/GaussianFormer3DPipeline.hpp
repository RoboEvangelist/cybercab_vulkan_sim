#pragma once
// Stage 1 Orchestrator: GaussianFormer3D (ICRA 2026, arXiv:2505.10685)
// Coordinates Voxel-to-Gaussian Init, Multi-Modal 3D Deformable Attention, and
// Gaussian-to-Voxel Probabilistic Splatting behind the Stage-1 launch-time selector.

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/sota/DeformableAttention3D.hpp"
#include "perception/sota/Gaussian3DTypes.hpp"
#include "perception/sota/GaussianToVoxelSplatter.hpp"
#include "perception/sota/VoxelToGaussianInit.hpp"
#include "perception/v15/StaticSpatialMemory.hpp"
#include "sensors/CameraRigRenderer.hpp"

#include <vector>

namespace cybercab::sota {

class GaussianFormer3DPipeline {
public:
    void init(const SimConfig& cfg);
    void init(const SimConfig& cfg, const GaussianFormer3DConfig& gfCfg);
    void reset();

    void runStage(const SimConfig& cfg,
                  const EgoState& ego,
                  const CameraRigRenderer* rig,
                  const v15::StaticSpatialMemory* staticMem,
                  PerceptionFrameOutput& out);

    const GaussianFormer3DConfig& config() const { return gfCfg_; }
    void setConfig(const GaussianFormer3DConfig& cfg) { gfCfg_ = cfg; }

    const std::vector<Gaussian3DPrimitive>& primitives() const { return currGaussians_; }
    const std::vector<V15Gaussian3D>& v15Gaussians() const { return v15ExportScratch_; }
    const std::vector<ActiveVoxel>& voxelScratch() const { return outVoxelsScratch_; }

private:
    static GaussianFormer3DSummary buildSummary(uint32_t initCount,
                                                uint32_t corrobCount,
                                                uint32_t splattedCount,
                                                float attnResidualM,
                                                bool usedAttn,
                                                const std::vector<Gaussian3DPrimitive>& prims);

    GaussianFormer3DConfig gfCfg_{};
    VoxelToGaussianInit initializer_;
    DeformableAttention3D attn_;
    GaussianToVoxelSplatter splatter_;

    std::vector<Gaussian3DPrimitive> currGaussians_;
    std::vector<Gaussian3DPrimitive> prevGaussians_;
    std::vector<V15Gaussian3D> v15ExportScratch_;
    std::vector<v15::StaticCellGpu> staticCellScratch_;
    std::vector<ActiveVoxel> outVoxelsScratch_;
};

} // namespace cybercab::sota
