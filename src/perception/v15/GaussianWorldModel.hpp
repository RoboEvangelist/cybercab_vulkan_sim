#pragma once
// FSD v15 Module 6: 3D/4D Gaussian World Model & GaussianFormer-2 probabilistic occupancy.
#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/v15/PatchTokenizer.hpp"
#include "perception/v15/StaticSpatialMemory.hpp"
#include <vector>

namespace cybercab::v15 {

struct alignas(16) Gaussian3DGpu {
    Vec4 meanOpacity; // wx, wy, wz, alpha (Y up, world space)
    Vec4 scaleSem;    // sx, sy, sz, semF
    Vec4 velYaw;      // vx, vz, yaw, isDynamic
    Vec4 colorConf;   // r, g, b, confidence
};

class GaussianWorldModel {
public:
    static constexpr int kMaxGaussians = 1024;
    static constexpr int kRealityW = 320;
    static constexpr int kRealityH = 180;

    void reset();
    void updateFromPerception(const SimConfig& cfg, const EgoState& ego,
                              const StaticSpatialMemory& staticMem,
                              const PatchTokenizer& patches,
                              PerceptionFrameOutput& out);

    // Zwicker/Kerbl EWA Jacobian projection: Sigma2D = J W Sigma3D W^T J^T + 0.3 I_2 (xx, xy, yy).
    static void projectEwa2D(const Gaussian3DGpu& g, const Mat4& camFromWorld,
                             const CameraIntrinsics& intr, Vec2& uvOut,
                             float cov2dOut[3], float& depthOut);

    // GaussianFormer-2 probabilistic superposition P(x, t) = 1 - prod_i (1 - alpha_i f_i(x, t)).
    float queryOccupancyProb(const Vec3& worldPt, float horizonSec = 0.0f) const;
    void rollout4D(float horizonSec, std::vector<Gaussian3DGpu>& outPred) const;
    void setGaussiansForTest(std::vector<Gaussian3DGpu> gaussians) { gaussians_ = std::move(gaussians); }
    const std::vector<Gaussian3DGpu>& gaussians() const { return gaussians_; }

private:
    std::vector<Gaussian3DGpu> gaussians_;
    mutable std::vector<StaticCellGpu> cellScratch_;
    mutable std::vector<PatchTokenGpu> vruScratch_;
};

} // namespace cybercab::v15
