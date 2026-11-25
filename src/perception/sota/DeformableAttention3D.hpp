#pragma once
// Stage 1b: Multi-Modal 3D Deformable Attention (GaussianFormer3D Sec. 3.3).
// Projects 3D anisotropic Gaussian reference points onto multi-camera 2D detection/patch
// evidence and aggregates 3D voxel/track geometric support to iteratively refine mean,
// scale, quaternion orientation, opacity, and semantic logits.

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "perception/sota/Gaussian3DTypes.hpp"
#include "sensors/CameraRigRenderer.hpp"

#include <cstdint>
#include <vector>

namespace cybercab::sota {

struct DeformableCamView {
    int camIdx = 0;
    Mat4 camFromWorld{};
    CameraIntrinsics intr{};
    float width = 320.0f;
    float height = 240.0f;
    float tokenWeight = 1.0f;
};

class DeformableAttention3D {
public:
    void reserve(uint32_t maxGaussians);

    // Full pipeline entry point using CameraRigRenderer camera extrinsics/intrinsics.
    void refine(const SimConfig& cfg,
                const GaussianFormer3DConfig& gfCfg,
                const EgoState& ego,
                const CameraRigRenderer* rig,
                const PerceptionFrameOutput& frameIn,
                std::vector<Gaussian3DPrimitive>& gaussians,
                uint32_t& outCamCorrobCount,
                float& outMeanResidualM);

    // Pure-CPU overload allowing synthetic multi-camera projections in unit tests without Vulkan.
    void refineWithViews(const GaussianFormer3DConfig& gfCfg,
                         const std::vector<DeformableCamView>& views,
                         const std::vector<Detection2D>& detections2D,
                         const std::vector<ActiveVoxel>& voxels3D,
                         const std::vector<Track3D>& tracks3D,
                         std::vector<Gaussian3DPrimitive>& gaussians,
                         uint32_t& outCamCorrobCount,
                         float& outMeanResidualM);

private:
    std::vector<DeformableCamView> viewsScratch_;
};

} // namespace cybercab::sota
