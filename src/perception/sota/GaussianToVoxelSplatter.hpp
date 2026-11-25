#pragma once
// Stage 1c: Gaussian-to-Voxel Probabilistic Splatting (GaussianFormer3D Sec. 3.4).
// Evaluates anisotropic 3D Gaussian mixtures over spatial voxel coordinates to produce
// refined semantic occupancy probabilities and merged ActiveVoxel outputs.

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/sota/Gaussian3DTypes.hpp"

#include <cstdint>
#include <vector>

namespace cybercab::sota {

class GaussianToVoxelSplatter {
public:
    void reserve(size_t maxVoxels);

    // Evaluates P_occ(x) = 1 - prod_i (1 - alpha_i exp(-0.5 * d_M^2)) and fused semantic class.
    static float queryVoxelOccupancyAndSemantic(const std::vector<Gaussian3DPrimitive>& gaussians,
                                                const Vec3& worldPt,
                                                float cutoffSq,
                                                SemanticClass& outSem,
                                                Vec3& outVel,
                                                float& outEpistemicSigma);

    // Refines existing activeVoxels and splats additional corroborated Gaussian support cells.
    uint32_t splatToVoxels(const SimConfig& cfg,
                           const GaussianFormer3DConfig& gfCfg,
                           const EgoState& ego,
                           const std::vector<Gaussian3DPrimitive>& gaussians,
                           std::vector<ActiveVoxel>& inOutVoxels);

private:
    struct CellAccum {
        int ix = 0;
        int iy = 0;
        int iz = 0;
        double emptyProd = 1.0;
        std::array<float, kNumGaussianSemClasses> semWeighted{};
        float velXSum = 0.0f;
        float velZSum = 0.0f;
        float wSum = 0.0f;
        float priorConf = 0.0f;
        float maxStaticOcc = 0.0f;
        uint32_t maxCorrob = 0u;
        uint32_t dynSplatCount = 0u;
        uint32_t staticSplatCount = 0u;
    };

    static int packCellKey(int ix, int iy, int iz, int nx, int ny);

    std::vector<int> cellIndexMap_;
    std::vector<CellAccum> accumScratch_;
};

} // namespace cybercab::sota
