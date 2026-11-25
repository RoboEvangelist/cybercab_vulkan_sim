#include "perception/sota/GaussianFormer3DPipeline.hpp"
#include "perception/v15/GaussianWorldModel.hpp"

#include <algorithm>

namespace cybercab::sota {

void GaussianFormer3DPipeline::init(const SimConfig& cfg, const GaussianFormer3DConfig& gfCfg) {
    gfCfg_ = gfCfg;
    const size_t maxG = static_cast<size_t>(std::max(gfCfg_.maxGaussians, 64u));
    const size_t maxVox = static_cast<size_t>(std::max(cfg.occTotalVoxels(), 1024));

    initializer_.reserve(gfCfg_.maxGaussians);
    attn_.reserve(gfCfg_.maxGaussians);
    splatter_.reserve(maxVox);

    currGaussians_.reserve(maxG);
    prevGaussians_.reserve(maxG);
    v15ExportScratch_.reserve(maxG);
    outVoxelsScratch_.reserve(maxVox);
    staticCellScratch_.reserve(static_cast<size_t>(std::max(cfg.bevNx * cfg.bevNy, 2400)));
    reset();
}

void GaussianFormer3DPipeline::reset() {
    currGaussians_.clear();
    prevGaussians_.clear();
    v15ExportScratch_.clear();
    outVoxelsScratch_.clear();
    staticCellScratch_.clear();
}

GaussianFormer3DSummary GaussianFormer3DPipeline::buildSummary(
    uint32_t initCount,
    uint32_t corrobCount,
    uint32_t splattedCount,
    float attnResidualM,
    bool usedAttn,
    const std::vector<Gaussian3DPrimitive>& prims) {
    GaussianFormer3DSummary s{};
    s.active = true;
    s.usedDeformableAttn = usedAttn;
    s.initSeedCount = initCount;
    s.gaussianCount = static_cast<uint32_t>(prims.size());
    s.camCorroboratedCount = corrobCount;
    s.splattedVoxelCount = splattedCount;
    s.attnResidualM = attnResidualM;
    if (prims.empty()) return s;

    float opSum = 0.0f;
    float confSum = 0.0f;
    float scaleSum = 0.0f;
    for (const auto& g : prims) {
        opSum += g.opacity;
        confSum += g.semanticConfidence();
        scaleSum += (g.scale.x + g.scale.y + g.scale.z) / 3.0f;
    }
    const float invN = 1.0f / static_cast<float>(prims.size());
    s.meanOpacity = opSum * invN;
    s.meanSemanticConf = confSum * invN;
    s.meanScaleM = scaleSum * invN;
    return s;
}

void GaussianFormer3DPipeline::runStage(const SimConfig& cfg,
                                        const EgoState& ego,
                                        const CameraRigRenderer* rig,
                                        const v15::StaticSpatialMemory* staticMem,
                                        PerceptionFrameOutput& out) {
    const std::vector<v15::StaticCellGpu>* staticPtr = nullptr;
    int staticNx = 0;
    int staticNy = 0;
    if (staticMem && staticMem->currMemSSBO().handle() != VK_NULL_HANDLE &&
        staticMem->nx() > 0 && staticMem->ny() > 0) {
        staticMem->readbackCells(staticCellScratch_);
        staticPtr = &staticCellScratch_;
        staticNx = staticMem->nx();
        staticNy = staticMem->ny();
    }

    const uint32_t initCount = initializer_.initialize(
        cfg, gfCfg_, ego, out.activeVoxels, out.tracks,
        staticPtr, staticNx, staticNy,
        prevGaussians_.empty() ? nullptr : &prevGaussians_,
        currGaussians_);

    uint32_t corrobCount = 0u;
    float attnResidualM = 0.0f;
    if (gfCfg_.enableDeformableAttn && !currGaussians_.empty()) {
        attn_.refine(cfg, gfCfg_, ego, rig, out, currGaussians_, corrobCount, attnResidualM);
    }

    const uint32_t splattedCount = splatter_.splatToVoxels(
        cfg, gfCfg_, ego, currGaussians_, out.activeVoxels);

    v15ExportScratch_.clear();
    const size_t capExport = std::min(currGaussians_.size(),
                                      static_cast<size_t>(v15::GaussianWorldModel::kMaxGaussians));
    for (size_t i = 0u; i < capExport; ++i) {
        v15ExportScratch_.push_back(toV15Gaussian3D(currGaussians_[i]));
    }

    out.gaussianFormer3D = buildSummary(
        initCount, corrobCount, splattedCount, attnResidualM,
        gfCfg_.enableDeformableAttn, currGaussians_);

    prevGaussians_.assign(currGaussians_.begin(), currGaussians_.end());
}

} // namespace cybercab::sota
