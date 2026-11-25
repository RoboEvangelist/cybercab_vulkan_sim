#include "perception/sota/GaussianFormer3DPipeline.hpp"
#include "perception/v15/GaussianWorldModel.hpp"

#include <algorithm>
#include <cmath>

namespace cybercab::sota {
namespace {

GaussianFormer3DConfig mapFromSimParams(const SimConfig::GaussianFormer3DParams& p) {
    GaussianFormer3DConfig c{};
    c.maxGaussians = p.maxGaussians;
    c.numRefinementLayers = p.numRefinementLayers;
    c.numDeformableSamplePts = p.numDeformableSamplePts;
    c.mergeRadiusM = p.mergeRadiusM;
    c.minInitConf = p.minInitConf;
    c.voxelScaleX = p.voxelScaleX;
    c.voxelScaleY = p.voxelScaleY;
    c.voxelScaleZ = p.voxelScaleZ;
    c.splatRadiusSigmaMul = p.splatRadiusSigmaMul;
    c.splatSigmaCutoffSq = p.splatSigmaCutoffSq;
    c.minVoxelOccProb = p.minVoxelOccProb;
    c.splatVoxelStepX = p.splatVoxelStepX;
    c.splatVoxelStepY = p.splatVoxelStepY;
    c.splatVoxelStepZ = p.splatVoxelStepZ;
    c.camAttnWeight = p.camAttnWeight;
    c.lidarAttnWeight = p.lidarAttnWeight;
    c.temporalSmoothAlpha = p.temporalSmoothAlpha;
    c.enableDeformableAttn = p.enableDeformableAttn;
    return c;
}

} // namespace

void GaussianFormer3DPipeline::init(const SimConfig& cfg) {
    init(cfg, mapFromSimParams(cfg.gf3d));
}

void GaussianFormer3DPipeline::init(const SimConfig& cfg, const GaussianFormer3DConfig& gfCfg) {
    gfCfg_ = gfCfg;
    const size_t maxG = static_cast<size_t>(std::max(gfCfg_.maxGaussians, 64u));
    const float sx = (gfCfg_.splatVoxelStepX > 0.05f) ? gfCfg_.splatVoxelStepX : cfg.occStepX;
    const float sy = (gfCfg_.splatVoxelStepY > 0.05f) ? gfCfg_.splatVoxelStepY : cfg.occStepY;
    const float sz = (gfCfg_.splatVoxelStepZ > 0.05f) ? gfCfg_.splatVoxelStepZ : cfg.occStepZ;
    const int snx = std::max(1, static_cast<int>(std::lround((cfg.occMaxX - cfg.occMinX) / sx)));
    const int sny = std::max(1, static_cast<int>(std::lround((cfg.occMaxY - cfg.occMinY) / sy)));
    const int snz = std::max(1, static_cast<int>(std::lround((cfg.occMaxZ - cfg.occMinZ) / sz)));
    const size_t splatVox = static_cast<size_t>(snx) * static_cast<size_t>(sny) * static_cast<size_t>(snz);
    const size_t maxVox = std::max(static_cast<size_t>(std::max(cfg.occTotalVoxels(), 1024)), splatVox);

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
