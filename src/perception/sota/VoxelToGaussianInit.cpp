#include "perception/sota/VoxelToGaussianInit.hpp"

#include "perception/BEVFormer.hpp"

#include <algorithm>
#include <cmath>

namespace cybercab::sota {
namespace {

constexpr float kTrackSeedScaleFloorXZ = 0.35f;
constexpr float kTrackSeedScaleFloorY = 0.50f;
constexpr float kTrackSeedCenterYFloor = 0.55f;
constexpr float kTrackOpacityBase = 0.42f;
constexpr float kTrackOpacityGain = 0.54f;
constexpr float kTrackDuplicateRadScale = 0.85f;
constexpr float kDominantLogitScale = 4.0f;
constexpr float kBackgroundLogit = -1.5f;
constexpr float kStaticCellMinConf = 0.50f;
constexpr float kStaticCellHalfExtentXZ = 0.52f;
constexpr float kStaticCellMinHeightM = 0.70f;
constexpr float kStaticCellOpacityBase = 0.45f;
constexpr float kStaticCellOpacityGain = 0.50f;
constexpr float kTemporalMatchMaxDistM = 1.60f;
constexpr float kVoxelStepExpandFrac = 0.50f;
constexpr float kVoxelClusterExpandFrac = 0.35f;
constexpr float kMaxVoxelClusterScaleXZ = 0.90f;
constexpr float kMaxClusterExtentM = 4.20f;

bool areVoxelClassesMergeable(SemanticClass a, SemanticClass b) {
    if (a == b) return true;
    const bool aStaticLike = (a == SemanticClass::Static || a == SemanticClass::Curb);
    const bool bStaticLike = (b == SemanticClass::Static || b == SemanticClass::Curb);
    return aStaticLike && bStaticLike;
}

bool nearExistingGaussian(const SpatialHashIndex3D& gaussianHash,
                          const std::vector<Gaussian3DPrimitive>& existing,
                          const Vec3& pos,
                          float radiusM,
                          SemanticClass sem) {
    const float rSq = radiusM * radiusM;
    bool found = false;
    const float searchRad = std::max(radiusM, kMaxClusterExtentM);
    gaussianHash.forEachNeighbor(pos.x, pos.z, searchRad, [&](int idx) {
        const auto& g = existing[static_cast<size_t>(idx)];
        const SemanticClass gSem = g.argmaxSemantic();
        if (gSem != sem && !areVoxelClassesMergeable(gSem, sem)) return true;
        const float dx = g.mean.x - pos.x;
        const float dy = g.mean.y - pos.y;
        const float dz = g.mean.z - pos.z;
        const float scaleR = std::max(radiusM, kTrackDuplicateRadScale * std::max(g.scale.x, g.scale.z));
        if ((dx * dx + dy * dy + dz * dz) <= std::max(rSq, scaleR * scaleR)) {
            found = true;
            return false;
        }
        return true;
    });
    return found;
}

} // namespace

void VoxelToGaussianInit::reserve(uint32_t maxGaussians) {
    const size_t cap = static_cast<size_t>(std::max(maxGaussians, 256u));
    visitedScratch_.reserve(cap * 16u);
    voxelHash_.reserve(cap * 32u);
    gaussianHash_.reserve(cap);
    prevHash_.reserve(cap);
}

void VoxelToGaussianInit::seedSemanticLogits(Gaussian3DPrimitive& g,
                                             SemanticClass sem,
                                             float conf) {
    const uint32_t target = std::min(static_cast<uint32_t>(sem), kNumGaussianSemClasses - 1u);
    const float peak = clampf(kDominantLogitScale * (0.45f + 0.55f * conf), 0.5f, kSemLogitMax);
    for (uint32_t c = 0u; c < kNumGaussianSemClasses; ++c) {
        g.semLogits[c] = (c == target) ? peak : kBackgroundLogit;
    }
}

Gaussian3DPrimitive VoxelToGaussianInit::makeSeedFromTrack(const Track3D& tr, uint32_t /*maxClasses*/) {
    Gaussian3DPrimitive g{};
    const float hy = std::max(kTrackSeedScaleFloorY, tr.halfExtents.y);
    g.mean = {tr.worldPos.x, std::max(kTrackSeedCenterYFloor, hy), tr.worldPos.z};
    g.scale = {
        clampf(std::max(kTrackSeedScaleFloorXZ, tr.halfExtents.x), kDefaultMinScaleM, kDefaultMaxScaleM),
        clampf(hy, kDefaultMinScaleM, kDefaultMaxScaleM),
        clampf(std::max(kTrackSeedScaleFloorXZ, tr.halfExtents.z), kDefaultMinScaleM, kDefaultMaxScaleM)
    };
    g.rot = Quat3D::fromYaw(tr.yaw);
    g.opacity = clampf(kTrackOpacityBase + kTrackOpacityGain * tr.score, kMinGaussianOpacity, kMaxGaussianOpacity);
    g.velWorld = {tr.worldVel.x, 0.0f, tr.worldVel.z};
    g.confidence = clampf(tr.score, 0.0f, 1.0f);
    g.isDynamic = true;
    g.fromTrack = true;
    seedSemanticLogits(g, tr.semantic, g.confidence);
    return g;
}

void VoxelToGaussianInit::applyTemporalPrior(const GaussianFormer3DConfig& gfCfg,
                                             const SpatialHashIndex3D& prevHash,
                                             const std::vector<Gaussian3DPrimitive>& prevGaussians,
                                             Gaussian3DPrimitive& g) {
    if (prevGaussians.empty() || gfCfg.temporalSmoothAlpha <= 0.0f) return;
    const SemanticClass targetSem = g.argmaxSemantic();
    float bestDistSq = kTemporalMatchMaxDistM * kTemporalMatchMaxDistM;
    const Gaussian3DPrimitive* bestPrev = nullptr;
    prevHash.forEachNeighbor(g.mean.x, g.mean.z, kTemporalMatchMaxDistM, [&](int idx) {
        const auto& prev = prevGaussians[static_cast<size_t>(idx)];
        if (prev.argmaxSemantic() != targetSem) return true;
        const float dx = prev.mean.x - g.mean.x;
        const float dy = prev.mean.y - g.mean.y;
        const float dz = prev.mean.z - g.mean.z;
        const float dSq = dx * dx + dy * dy + dz * dz;
        if (dSq < bestDistSq) {
            bestDistSq = dSq;
            bestPrev = &prev;
        }
        return true;
    });
    if (!bestPrev) return;
    const float wPrev = clampf(gfCfg.temporalSmoothAlpha * bestPrev->confidence, 0.0f, 0.65f);
    const float wCurr = 1.0f - wPrev;
    g.scale = g.scale * wCurr + bestPrev->scale * wPrev;
    g.opacity = clampf(wCurr * g.opacity + wPrev * bestPrev->opacity, kMinGaussianOpacity, kMaxGaussianOpacity);
    for (uint32_t c = 0u; c < kNumGaussianSemClasses; ++c) {
        g.semLogits[c] = clampf(wCurr * g.semLogits[c] + wPrev * bestPrev->semLogits[c],
                                kSemLogitMin, kSemLogitMax);
    }
}

uint32_t VoxelToGaussianInit::initialize(const SimConfig& cfg,
                                         const GaussianFormer3DConfig& gfCfg,
                                         const EgoState& ego,
                                         const std::vector<ActiveVoxel>& voxels,
                                         const std::vector<Track3D>& tracks,
                                         const std::vector<v15::StaticCellGpu>* staticCells,
                                         int staticNx,
                                         int staticNy,
                                         const std::vector<Gaussian3DPrimitive>* prevGaussians,
                                         std::vector<Gaussian3DPrimitive>& outGaussians) {
    outGaussians.clear();
    const uint32_t cap = std::max(1u, gfCfg.maxGaussians);
    gaussianHash_.reset(cap, std::max(1.5f, gfCfg.mergeRadiusM * 2.0f));
    const bool hasPrev = (prevGaussians && !prevGaussians->empty() && gfCfg.temporalSmoothAlpha > 0.0f);
    if (hasPrev) {
        prevHash_.reset(prevGaussians->size(), kTemporalMatchMaxDistM);
        for (size_t i = 0u; i < prevGaussians->size(); ++i) {
            prevHash_.insert(static_cast<int>(i), (*prevGaussians)[i].mean.x, (*prevGaussians)[i].mean.z);
        }
    }

    auto pushGaussian = [&](const Gaussian3DPrimitive& g) {
        const int idx = static_cast<int>(outGaussians.size());
        outGaussians.push_back(g);
        gaussianHash_.insert(idx, g.mean.x, g.mean.z);
    };

    // 1. Seed tracked dynamic actors first so salient traffic participants never get dropped.
    for (const auto& tr : tracks) {
        if (outGaussians.size() >= cap) break;
        if (tr.state == TrackState::Removed) continue;
        Gaussian3DPrimitive g = makeSeedFromTrack(tr, kNumGaussianSemClasses);
        if (hasPrev) applyTemporalPrior(gfCfg, prevHash_, *prevGaussians, g);
        pushGaussian(g);
    }

    // 2. Cluster non-empty active 3D voxels via O(1) spatial hash lookup.
    if (visitedScratch_.capacity() < voxels.size()) {
        visitedScratch_.reserve(voxels.size());
    }
    visitedScratch_.assign(voxels.size(), 0u);
    voxelHash_.reset(voxels.size(), std::max(0.75f, gfCfg.mergeRadiusM));
    for (size_t i = 0u; i < voxels.size(); ++i) {
        const auto& v = voxels[i];
        if (v.sem == SemanticClass::Empty || v.sem == SemanticClass::OccShadow) continue;
        if (v.confidence < gfCfg.minInitConf) continue;
        voxelHash_.insert(static_cast<int>(i), v.wx, v.wz);
    }

    const float mergeRadSq = gfCfg.mergeRadiusM * gfCfg.mergeRadiusM;
    const Quat3D egoYawRot = Quat3D::fromYaw(ego.yaw);

    for (size_t i = 0; i < voxels.size() && outGaussians.size() < cap; ++i) {
        if (visitedScratch_[i] != 0u) continue;
        const auto& seed = voxels[i];
        if (seed.sem == SemanticClass::Empty || seed.sem == SemanticClass::OccShadow) continue;
        if (seed.confidence < gfCfg.minInitConf) continue;

        visitedScratch_[i] = 1u;
        int clusterCount = 1;
        float wSum = std::max(0.05f, seed.confidence);
        Vec3 posSum = Vec3{seed.wx, seed.wy, seed.wz} * wSum;
        Vec3 velSum = Vec3{seed.vx, 0.0f, seed.vz} * wSum;
        Vec3 minP{seed.wx, seed.wy, seed.wz};
        Vec3 maxP = minP;
        std::array<float, kNumGaussianSemClasses> classWeight{};
        const uint32_t seedCls = std::min(static_cast<uint32_t>(seed.sem), kNumGaussianSemClasses - 1u);
        classWeight[seedCls] += wSum;

        voxelHash_.forEachNeighbor(seed.wx, seed.wz, gfCfg.mergeRadiusM, [&](int jIdx) {
            const size_t j = static_cast<size_t>(jIdx);
            if (j <= i || visitedScratch_[j] != 0u) return true;
            const auto& v = voxels[j];
            if (!areVoxelClassesMergeable(seed.sem, v.sem)) return true;
            const float dx = v.wx - seed.wx;
            const float dy = v.wy - seed.wy;
            const float dz = v.wz - seed.wz;
            if (dx * dx + dy * dy + dz * dz > mergeRadSq) return true;

            visitedScratch_[j] = 1u;
            ++clusterCount;
            const float w = std::max(0.05f, v.confidence);
            wSum += w;
            posSum += Vec3{v.wx, v.wy, v.wz} * w;
            velSum += Vec3{v.vx, 0.0f, v.vz} * w;
            minP.x = std::min(minP.x, v.wx);
            minP.y = std::min(minP.y, v.wy);
            minP.z = std::min(minP.z, v.wz);
            maxP.x = std::max(maxP.x, v.wx);
            maxP.y = std::max(maxP.y, v.wy);
            maxP.z = std::max(maxP.z, v.wz);
            const uint32_t cls = std::min(static_cast<uint32_t>(v.sem), kNumGaussianSemClasses - 1u);
            classWeight[cls] += w;
            return true;
        });

        const Vec3 mean = posSum * (1.0f / wSum);
        uint32_t domCls = seedCls;
        for (uint32_t c = 0u; c < kNumGaussianSemClasses; ++c) {
            if (classWeight[c] > classWeight[domCls]) domCls = c;
        }
        const SemanticClass domSem = static_cast<SemanticClass>(domCls);
        if (nearExistingGaussian(gaussianHash_, outGaussians, mean, 0.75f * gfCfg.mergeRadiusM, domSem)) continue;

        const float spanX = std::min(kMaxClusterExtentM, maxP.x - minP.x);
        const float spanY = std::min(kMaxClusterExtentM, maxP.y - minP.y);
        const float spanZ = std::min(kMaxClusterExtentM, maxP.z - minP.z);
        Gaussian3DPrimitive g{};
        g.mean = mean;
        g.scale = {
            clampf(0.5f * spanX + kVoxelClusterExpandFrac * cfg.occStepX, kDefaultMinScaleM, kMaxVoxelClusterScaleXZ),
            clampf(0.5f * spanY + kVoxelStepExpandFrac * cfg.occStepZ, kDefaultMinScaleM, kDefaultMaxScaleM),
            clampf(0.5f * spanZ + kVoxelClusterExpandFrac * cfg.occStepY, kDefaultMinScaleM, kMaxVoxelClusterScaleXZ)
        };
        g.rot = egoYawRot;
        const float avgVoxelConf = clampf(wSum / static_cast<float>(clusterCount), 0.0f, 1.0f);
        g.confidence = avgVoxelConf;
        g.opacity = clampf(avgVoxelConf, kMinGaussianOpacity, kMaxGaussianOpacity);
        g.velWorld = velSum * (1.0f / wSum);
        g.isDynamic = (domSem == SemanticClass::Vehicle || domSem == SemanticClass::Pedestrian);
        for (uint32_t c = 0u; c < kNumGaussianSemClasses; ++c) {
            const float share = classWeight[c] / std::max(0.05f, wSum);
            g.semLogits[c] = (share > 0.01f)
                ? clampf(kDominantLogitScale * share, kSemLogitMin, kSemLogitMax)
                : kBackgroundLogit;
        }
        if (hasPrev) applyTemporalPrior(gfCfg, prevHash_, *prevGaussians, g);
        pushGaussian(g);
    }

    // 3. Backfill persistent static spatial memory structures not already covered above.
    const float staticOccThresh = std::min(0.35f, gfCfg.minVoxelOccProb);
    if (staticCells && staticNx > 0 && staticNy > 0 &&
        static_cast<int>(staticCells->size()) == staticNx * staticNy) {
        const Mat4 wFromIso = BEVFormer::worldFromIsoEgo(ego);
        const float cellDx = (cfg.bevMaxX - cfg.bevMinX) / static_cast<float>(staticNx);
        const float cellDy = (cfg.bevMaxY - cfg.bevMinY) / static_cast<float>(staticNy);
        for (int pass = 0; pass < 2 && outGaussians.size() < cap; ++pass) {
            const bool wantLocked = (pass == 0);
            for (int iy = 0; iy < staticNy && outGaussians.size() < cap; ++iy) {
                for (int ix = 0; ix < staticNx && outGaussians.size() < cap; ++ix) {
                    const auto& c = (*staticCells)[static_cast<size_t>(iy) * staticNx + ix];
                    if (c.featConf.x < staticOccThresh || c.featConf.w < kStaticCellMinConf) continue;
                    const bool isLocked = (c.meta.z > 0.5f);
                    if (isLocked != wantLocked) continue;
                    const int rawSem = static_cast<int>(std::lround(c.featConf.z));
                    const SemanticClass sem = (rawSem == static_cast<int>(SemanticClass::SignalSign))
                        ? SemanticClass::SignalSign : SemanticClass::Static;
                    const float xIso = cfg.bevMinX + (static_cast<float>(ix) + 0.5f) * cellDx;
                    const float yIso = cfg.bevMinY + (static_cast<float>(iy) + 0.5f) * cellDy;
                    const Vec3 wp = wFromIso.transformPoint({xIso, yIso, 0.0f});
                    const float h = std::max(kStaticCellMinHeightM, c.featConf.y);
                    const Vec3 center{wp.x, 0.5f * h, wp.z};
                    if (!isLocked &&
                        nearExistingGaussian(gaussianHash_, outGaussians, center, gfCfg.mergeRadiusM, sem)) {
                        continue;
                    }

                    Gaussian3DPrimitive g{};
                    g.mean = center;
                    g.scale = {kStaticCellHalfExtentXZ, 0.5f * h, kStaticCellHalfExtentXZ};
                    g.rot = egoYawRot;
                    g.confidence = clampf(c.featConf.w, 0.0f, 1.0f);
                    g.opacity = clampf(kStaticCellOpacityBase + kStaticCellOpacityGain * g.confidence,
                                       kMinGaussianOpacity, kMaxGaussianOpacity);
                    g.fromStaticMem = true;
                    g.staticMemLocked = isLocked;
                    seedSemanticLogits(g, sem, g.confidence);
                    pushGaussian(g);
                }
            }
        }
    }

    return static_cast<uint32_t>(outGaussians.size());
}

} // namespace cybercab::sota
