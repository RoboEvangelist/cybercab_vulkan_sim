#pragma once
// Stage 1a: Voxel-to-Gaussian seed initialization (GaussianFormer3D Sec. 3.2).
// Converts 3D occupancy voxels, dynamic tracks, and optional static spatial memory cells
// into an explicit compact set of SO(3) anisotropic 3D Gaussians with zero hot-path allocations.

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/sota/Gaussian3DTypes.hpp"
#include "perception/sota/SpatialHashIndex3D.hpp"
#include "perception/v15/StaticSpatialMemory.hpp"

#include <vector>

namespace cybercab::sota {

class VoxelToGaussianInit {
public:
    void reserve(uint32_t maxGaussians);

    uint32_t initialize(const SimConfig& cfg,
                        const GaussianFormer3DConfig& gfCfg,
                        const EgoState& ego,
                        const std::vector<ActiveVoxel>& voxels,
                        const std::vector<Track3D>& tracks,
                        const std::vector<v15::StaticCellGpu>* staticCells,
                        int staticNx,
                        int staticNy,
                        const std::vector<Gaussian3DPrimitive>* prevGaussians,
                        std::vector<Gaussian3DPrimitive>& outGaussians);

private:
    static Gaussian3DPrimitive makeSeedFromTrack(const Track3D& tr, uint32_t maxClasses);
    static void seedSemanticLogits(Gaussian3DPrimitive& g, SemanticClass sem, float conf);
    static void applyTemporalPrior(const GaussianFormer3DConfig& gfCfg,
                                   const SpatialHashIndex3D& prevHash,
                                   const std::vector<Gaussian3DPrimitive>& prevGaussians,
                                   Gaussian3DPrimitive& g);

    std::vector<uint8_t> visitedScratch_;
    SpatialHashIndex3D voxelHash_;
    SpatialHashIndex3D gaussianHash_;
    SpatialHashIndex3D prevHash_;
};

} // namespace cybercab::sota
