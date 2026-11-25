#include "perception/sota/GaussianToVoxelSplatter.hpp"

#include "perception/BEVFormer.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace cybercab::sota {
namespace {

constexpr int kMaxSplatSpanXY = 2;
constexpr int kMaxSplatSpanZ = 2;
constexpr float kBaseEpistemicSigmaM = 0.24f;
constexpr float kMinDynEpistemicSigmaM = 0.05f;
constexpr float kMinStaticEpistemicSigmaM = 0.11f;
constexpr float kMaxEpistemicSigmaM = 0.85f;
constexpr float kCorrobSigmaDecay = 0.35f;
constexpr float kPriorVoxelWeight = 0.30f;
constexpr float kSplatVoxelWeight = 1.00f;
constexpr float kMinWeightEps = 1e-5f;

} // namespace

void GaussianToVoxelSplatter::reserve(size_t maxVoxels) {
    const size_t cap = std::max<size_t>(maxVoxels, 1024u);
    cellIndexMap_.assign(cap, -1);
    accumScratch_.reserve(cap);
}

int GaussianToVoxelSplatter::packCellKey(int ix, int iy, int iz, int nx, int ny) {
    return (iz * ny + iy) * nx + ix;
}

float GaussianToVoxelSplatter::queryVoxelOccupancyAndSemantic(
    const std::vector<Gaussian3DPrimitive>& gaussians,
    const Vec3& worldPt,
    float cutoffSq,
    SemanticClass& outSem,
    Vec3& outVel,
    float& outEpistemicSigma) {
    double emptyProd = 1.0;
    std::array<float, kNumGaussianSemClasses> semWeighted{};
    std::array<float, kNumGaussianSemClasses> probs{};
    float velX = 0.0f;
    float velZ = 0.0f;
    float wSum = 0.0f;
    uint32_t maxCorrob = 0u;

    for (const auto& g : gaussians) {
        if (g.opacity < kMinGaussianOpacity) continue;
        const Vec3 local = g.rot.invRotateVector(worldPt - g.mean);
        const float sx = std::max(kDefaultMinScaleM, g.scale.x);
        const float sy = std::max(kDefaultMinScaleM, g.scale.y);
        const float sz = std::max(kDefaultMinScaleM, g.scale.z);
        const float mSq = (local.x * local.x) / (sx * sx) +
                          (local.y * local.y) / (sy * sy) +
                          (local.z * local.z) / (sz * sz);
        if (mSq > cutoffSq) continue;

        const float contrib = clampf(g.opacity * std::exp(kGaussianExpHalf * mSq),
                                     0.0f, kMaxGaussianOpacity);
        emptyProd *= (1.0 - static_cast<double>(contrib));
        g.softmaxProbs(probs);
        for (uint32_t c = 0u; c < kNumGaussianSemClasses; ++c) {
            semWeighted[c] += contrib * probs[c];
        }
        velX += g.velWorld.x * contrib;
        velZ += g.velWorld.z * contrib;
        wSum += contrib;
        maxCorrob = std::max(maxCorrob, g.camCorrobCount);
    }

    const float occProb = clampf(static_cast<float>(1.0 - emptyProd), 0.0f, 1.0f);
    uint32_t bestCls = 0u;
    float bestScore = -1.0f;
    for (uint32_t c = 1u; c < kNumGaussianSemClasses; ++c) {
        if (semWeighted[c] > bestScore) {
            bestScore = semWeighted[c];
            bestCls = c;
        }
    }
    outSem = (bestScore > kMinWeightEps) ? static_cast<SemanticClass>(bestCls) : SemanticClass::Empty;
    outVel = (wSum > kMinWeightEps) ? Vec3{velX / wSum, 0.0f, velZ / wSum} : Vec3{};
    outEpistemicSigma = clampf(
        kBaseEpistemicSigmaM / (1.0f + kCorrobSigmaDecay * static_cast<float>(maxCorrob)),
        kMinDynEpistemicSigmaM, kMaxEpistemicSigmaM);
    return occProb;
}

uint32_t GaussianToVoxelSplatter::splatToVoxels(const SimConfig& cfg,
                                              const GaussianFormer3DConfig& gfCfg,
                                              const EgoState& ego,
                                              const std::vector<Gaussian3DPrimitive>& gaussians,
                                              std::vector<ActiveVoxel>& inOutVoxels) {
    const int nx = std::max(1, cfg.occNx());
    const int ny = std::max(1, cfg.occNy());
    const int nz = std::max(1, cfg.occNz());
    const size_t totalCells = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

    if (cellIndexMap_.capacity() < totalCells) cellIndexMap_.reserve(totalCells);
    cellIndexMap_.assign(totalCells, -1);
    accumScratch_.clear();

    const Mat4 worldFromIso = BEVFormer::worldFromIsoEgo(ego);
    const Mat4 isoFromWorld = worldFromIso.inverseRigid();

    auto getOrCreateSlot = [&](int ix, int iy, int iz) -> int {
        const int key = packCellKey(ix, iy, iz, nx, ny);
        int& slot = cellIndexMap_[static_cast<size_t>(key)];
        if (slot < 0) {
            slot = static_cast<int>(accumScratch_.size());
            CellAccum cell{};
            cell.ix = ix;
            cell.iy = iy;
            cell.iz = iz;
            accumScratch_.push_back(cell);
        }
        return slot;
    };

    // Preserve prior non-shadow observations without double-counting into Gaussian products.
    for (const auto& v : inOutVoxels) {
        if (v.sem == SemanticClass::Empty || v.sem == SemanticClass::OccShadow) continue;
        const int ix = static_cast<int>(std::floor((v.egoX - cfg.occMinX) / cfg.occStepX));
        const int iy = static_cast<int>(std::floor((v.egoY - cfg.occMinY) / cfg.occStepY));
        const int iz = static_cast<int>(std::floor((v.wy - cfg.occMinZ) / cfg.occStepZ));
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) continue;

        const int slot = getOrCreateSlot(ix, iy, iz);
        auto& acc = accumScratch_[static_cast<size_t>(slot)];
        const float cleanConf = clampf(v.confidence, 0.0f, kMaxGaussianOpacity);
        acc.priorConf = std::max(acc.priorConf, cleanConf);
        const float priorW = cleanConf * kPriorVoxelWeight;
        const uint32_t cls = std::min(static_cast<uint32_t>(v.sem), kNumGaussianSemClasses - 1u);
        acc.semWeighted[cls] += priorW;
        acc.velXSum += v.vx * priorW;
        acc.velZSum += v.vz * priorW;
        acc.wSum += priorW;
    }
    const size_t priorCellCount = accumScratch_.size();

    // Splat live-observed anisotropic 3D Gaussians over local voxel neighborhood.
    std::array<float, kNumGaussianSemClasses> probs{};
    for (const auto& g : gaussians) {
        if (g.opacity < kMinGaussianOpacity || g.fromStaticMem) continue;
        const Vec3 centerIso = isoFromWorld.transformPoint(g.mean);
        const float sx = clampf(g.scale.x * gfCfg.voxelScaleX, kDefaultMinScaleM, kDefaultMaxScaleM);
        const float sy = clampf(g.scale.y * gfCfg.voxelScaleY, kDefaultMinScaleM, kDefaultMaxScaleM);
        const float sz = clampf(g.scale.z * gfCfg.voxelScaleZ, kDefaultMinScaleM, kDefaultMaxScaleM);
        const float rIso = gfCfg.splatRadiusSigmaMul * std::max({sx, sy, sz});
        const int capXY = (!g.isDynamic || rIso <= 1.25f) ? 1 : kMaxSplatSpanXY;
        const int capZ = (!g.isDynamic && g.argmaxSemantic() != SemanticClass::SignalSign)
            ? 1 : kMaxSplatSpanZ;
        const int spanX = std::min(capXY, static_cast<int>(std::ceil(rIso / cfg.occStepX)));
        const int spanY = std::min(capXY, static_cast<int>(std::ceil(rIso / cfg.occStepY)));
        const int spanZ = std::min(capZ, static_cast<int>(std::ceil(rIso / cfg.occStepZ)));

        const int cx = static_cast<int>(std::floor((centerIso.x - cfg.occMinX) / cfg.occStepX));
        const int cy = static_cast<int>(std::floor((centerIso.y - cfg.occMinY) / cfg.occStepY));
        const int cz = static_cast<int>(std::floor((centerIso.z - cfg.occMinZ) / cfg.occStepZ));

        const int ixMin = std::max(0, cx - spanX), ixMax = std::min(nx - 1, cx + spanX);
        const int iyMin = std::max(0, cy - spanY), iyMax = std::min(ny - 1, cy + spanY);
        const int izMin = std::max(0, cz - spanZ), izMax = std::min(nz - 1, cz + spanZ);
        if (ixMin > ixMax || iyMin > iyMax || izMin > izMax) continue;

        g.softmaxProbs(probs);

        for (int iz = izMin; iz <= izMax; ++iz) {
            const float zIso = cfg.occMinZ + (static_cast<float>(iz) + 0.5f) * cfg.occStepZ;
            for (int iy = iyMin; iy <= iyMax; ++iy) {
                const float yIso = cfg.occMinY + (static_cast<float>(iy) + 0.5f) * cfg.occStepY;
                for (int ix = ixMin; ix <= ixMax; ++ix) {
                    const float xIso = cfg.occMinX + (static_cast<float>(ix) + 0.5f) * cfg.occStepX;
                    const Vec3 wp = worldFromIso.transformPoint({xIso, yIso, zIso});
                    const Vec3 local = g.rot.invRotateVector(wp - g.mean);
                    const float mSq = (local.x * local.x) / (sx * sx) +
                                      (local.y * local.y) / (sy * sy) +
                                      (local.z * local.z) / (sz * sz);
                    if (mSq > gfCfg.splatSigmaCutoffSq) continue;
                    const int key = packCellKey(ix, iy, iz, nx, ny);
                    const int existingSlot = cellIndexMap_[static_cast<size_t>(key)];
                    const bool hasPriorCell =
                        (existingSlot >= 0 && static_cast<size_t>(existingSlot) < priorCellCount);
                    if (!hasPriorCell && !g.fromTrack) continue;

                    const float w = clampf(g.opacity * kSplatVoxelWeight * std::exp(kGaussianExpHalf * mSq),
                                           0.0f, kMaxGaussianOpacity);
                    const int slot = getOrCreateSlot(ix, iy, iz);
                    auto& acc = accumScratch_[static_cast<size_t>(slot)];
                    if (g.fromTrack) {
                        acc.emptyProd *= (1.0 - static_cast<double>(w));
                        ++acc.dynSplatCount;
                    } else {
                        acc.maxStaticOcc = std::max(acc.maxStaticOcc, w);
                        ++acc.staticSplatCount;
                    }
                    for (uint32_t c = 0u; c < kNumGaussianSemClasses; ++c) {
                        acc.semWeighted[c] += w * probs[c];
                    }
                    acc.velXSum += g.velWorld.x * w;
                    acc.velZSum += g.velWorld.z * w;
                    acc.wSum += w;
                    acc.maxCorrob = std::max(acc.maxCorrob, g.camCorrobCount);
                }
            }
        }
    }

    // Rebuild refined activeVoxels without dropping below grid capacity reservation.
    inOutVoxels.clear();
    uint32_t splattedCount = 0u;
    const float occThresh = std::max(gfCfg.minVoxelOccProb, cfg.occThreshold * 0.85f);
    for (const auto& acc : accumScratch_) {
        const float dynOcc = (acc.dynSplatCount > 0u)
            ? clampf(static_cast<float>(1.0 - acc.emptyProd), 0.0f, 1.0f) : 0.0f;
        const float staticOcc = (acc.staticSplatCount > 0u)
            ? std::max(acc.priorConf, lerpf(acc.priorConf, acc.maxStaticOcc, 0.45f)) : acc.priorConf;
        const float occProb = std::max(dynOcc, staticOcc);

        uint32_t bestCls = 0u;
        float bestScore = -1.0f;
        for (uint32_t c = 1u; c < kNumGaussianSemClasses; ++c) {
            if (acc.semWeighted[c] > bestScore) {
                bestScore = acc.semWeighted[c];
                bestCls = c;
            }
        }
        if (bestScore <= kMinWeightEps) continue;

        const float cellThresh = (bestCls == static_cast<uint32_t>(SemanticClass::SignalSign))
            ? std::min(occThresh, 0.25f) : occThresh;
        if (occProb < cellThresh) continue;

        const float xIso = cfg.occMinX + (static_cast<float>(acc.ix) + 0.5f) * cfg.occStepX;
        const float yIso = cfg.occMinY + (static_cast<float>(acc.iy) + 0.5f) * cfg.occStepY;
        const float zIso = cfg.occMinZ + (static_cast<float>(acc.iz) + 0.5f) * cfg.occStepZ;
        const Vec3 wp = worldFromIso.transformPoint({xIso, yIso, zIso});
        const float invW = (acc.wSum > kMinWeightEps) ? (1.0f / acc.wSum) : 0.0f;
        const SemanticClass outSem = static_cast<SemanticClass>(bestCls);
        const bool isDynCls = (outSem == SemanticClass::Vehicle || outSem == SemanticClass::Pedestrian);
        const float minSigma = isDynCls ? kMinDynEpistemicSigmaM : kMinStaticEpistemicSigmaM;
        const float sigma = clampf(
            kBaseEpistemicSigmaM / (1.0f + kCorrobSigmaDecay * static_cast<float>(acc.maxCorrob)),
            minSigma, kMaxEpistemicSigmaM);

        inOutVoxels.push_back(ActiveVoxel{
            xIso, yIso, zIso, wp.x, wp.z,
            outSem,
            occProb,
            acc.velXSum * invW,
            acc.velZSum * invW,
            sigma
        });
        ++splattedCount;
    }

    int staticMemFused = 0;
    for (const auto& g : gaussians) {
        if (!g.fromStaticMem || !g.staticMemLocked || g.confidence < 0.55f || staticMemFused >= 32) continue;
        const Vec3 iso = isoFromWorld.transformPoint(g.mean);
        if (iso.x <= 0.0f || std::abs(iso.y) <= 2.2f) continue;
        if (iso.x < cfg.occMinX || iso.x > cfg.occMaxX ||
            iso.y < cfg.occMinY || iso.y > cfg.occMaxY ||
            g.mean.y < 0.0f || g.mean.y > cfg.occMaxZ) continue;
        inOutVoxels.push_back(ActiveVoxel{
            iso.x, iso.y, g.mean.y, g.mean.x, g.mean.z,
            g.argmaxSemantic(),
            g.confidence,
            0.0f, 0.0f,
            kMinDynEpistemicSigmaM
        });
        ++staticMemFused;
        ++splattedCount;
    }
    return splattedCount;
}

} // namespace cybercab::sota
