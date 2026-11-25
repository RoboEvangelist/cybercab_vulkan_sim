#include "perception/v15/GaussianWorldModel.hpp"
#include "perception/BEVFormer.hpp"

#include <algorithm>
#include <cmath>

namespace cybercab::v15 {
namespace {

constexpr size_t kStaticCellScratchCap = 60u * 40u;
constexpr float kDynScaleGrowthPerSec = 0.15f;
constexpr float kDynamicFlagThresh = 0.5f;
constexpr float kMinRoadGaussianOpacity = 0.05f;
constexpr float kMinRoadGaussianScaleYM = 0.10f;
constexpr float kMaxRoadGaussianCenterYM = 2.40f;
constexpr float kMinGaussianAxisM = 0.15f;
constexpr float kMahalanobisSqCutoff = 18.0f;
constexpr float kGaussianExpHalf = -0.5f;
constexpr float kMaxGaussianOccProb = 0.999f;

bool isValidSplatGaussian(const Gaussian3DGpu& g) {
    return std::isfinite(g.meanOpacity.x) && std::isfinite(g.meanOpacity.y) &&
           std::isfinite(g.meanOpacity.z) && std::isfinite(g.meanOpacity.w) &&
           std::isfinite(g.scaleSem.x) && std::isfinite(g.scaleSem.y) &&
           std::isfinite(g.scaleSem.z) && std::isfinite(g.velYaw.z) &&
           std::isfinite(g.colorConf.x) && std::isfinite(g.colorConf.y) &&
           std::isfinite(g.colorConf.z) &&
           g.meanOpacity.w > 0.0f &&
           g.scaleSem.x > 1e-4f && g.scaleSem.y > 1e-4f && g.scaleSem.z > 1e-4f;
}

}  // namespace

GaussianWorldModel::~GaussianWorldModel() { destroy(); }

void GaussianWorldModel::initSplatGpu(const VulkanContext& ctx) {
    destroy();
    ctx_ = &ctx;
    splatPipe_.create(ctx, "shaders/v15_gaussian_splat.comp.spv",
                      {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
                       {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1}},
                      sizeof(SplatPC));
    splatSet_ = splatPipe_.allocateDescriptorSet();
    const VkMemoryPropertyFlags mem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    gaussSSBO_.create(ctx, static_cast<VkDeviceSize>(kMaxGaussians) * sizeof(Gaussian3DGpu),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    rgbaSSBO_.create(ctx, static_cast<VkDeviceSize>(kRealityW) * kRealityH * sizeof(uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, mem);
    const VkDescriptorBufferInfo infos[2] = {gaussSSBO_.descriptorInfo(), rgbaSSBO_.descriptorInfo()};
    VkWriteDescriptorSet w[2]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet_, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[0], nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet_, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[1], nullptr},
    };
    vkUpdateDescriptorSets(ctx.device(), 2, w, 0, nullptr);
  gaussians_.reserve(kMaxGaussians);
  sortedScratch_.reserve(kMaxGaussians);
  cellScratch_.reserve(kStaticCellScratchCap);
  vruScratch_.reserve(static_cast<size_t>(PatchTokenizer::kGridW) *
                      PatchTokenizer::kGridH);
}

void GaussianWorldModel::reset() { gaussians_.clear(); }

void GaussianWorldModel::destroy() {
    if (!ctx_) return;
    rgbaSSBO_.destroy();
    gaussSSBO_.destroy();
    splatPipe_.destroy();
    splatSet_ = VK_NULL_HANDLE;
    ctx_ = nullptr;
}

void GaussianWorldModel::dispatchRealitySplat(VkCommandBuffer cmd, const SimConfig& cfg, const EgoState& ego) {
    const CameraSpec cam = cfg.cameras.empty() ? CameraSpec{} : cfg.cameras[0];
    const Mat4 camFromWorld = (BEVFormer::worldFromIsoEgo(ego) * cam.extrinsicEgoFromCamISO()).inverseRigid();
    dispatchRealitySplat(cmd, camFromWorld, cam.intrinsics(kRealityW, kRealityH));
}

void GaussianWorldModel::dispatchRealitySplat(VkCommandBuffer cmd, const Mat4& camFromWorld,
                                                const CameraIntrinsics& intr) {
    if (!ctx_) return;
    const auto isStaticBg = [](const Gaussian3DGpu& g) {
        return g.velYaw.w < 0.5f &&
               static_cast<int>(std::lround(g.scaleSem.w)) == static_cast<int>(SemanticClass::Static);
    };
    sortedScratch_.clear();
    for (const auto& g : gaussians_) {
        if (!isValidSplatGaussian(g)) continue;
        const float z = camFromWorld.transformPoint({g.meanOpacity.x, g.meanOpacity.y, g.meanOpacity.z}).z;
        if (!std::isfinite(z) || z <= 0.35f) continue;
        Gaussian3DGpu gCopy = g;
        if (isStaticBg(gCopy)) {
            gCopy.meanOpacity.w = std::min(gCopy.meanOpacity.w * 0.22f, 0.20f);
        }
        sortedScratch_.push_back(gCopy);
    }
    std::sort(sortedScratch_.begin(), sortedScratch_.end(), [&](const Gaussian3DGpu& a, const Gaussian3DGpu& b) {
        const bool aBg = isStaticBg(a);
        const bool bBg = isStaticBg(b);
        if (aBg != bBg) return !aBg;
        return camFromWorld.transformPoint({a.meanOpacity.x, a.meanOpacity.y, a.meanOpacity.z}).z <
               camFromWorld.transformPoint({b.meanOpacity.x, b.meanOpacity.y, b.meanOpacity.z}).z;
    });
    const uint32_t n = static_cast<uint32_t>(std::min<size_t>(sortedScratch_.size(), kMaxGaussians));
    if (n > 0u) gaussSSBO_.upload(sortedScratch_.data(), n * sizeof(Gaussian3DGpu));
    SplatPC pc{camFromWorld, {intr.fx, intr.fy, intr.cx, intr.cy},
               static_cast<uint32_t>(kRealityW), static_cast<uint32_t>(kRealityH), n, 0u};
    splatPipe_.bindAndDispatch(cmd, splatSet_, (kRealityW + 15) / 16, (kRealityH + 15) / 16, 1, &pc, sizeof(pc));
    cmdComputeBarrier(cmd);
}

void GaussianWorldModel::downloadRealityViewRGBA(std::vector<uint8_t>& outRgba) const {
    if (!ctx_) { outRgba.clear(); return; }
    outRgba.resize(static_cast<size_t>(kRealityW) * kRealityH * 4);
    rgbaSSBO_.download(outRgba.data(), outRgba.size());
}

void GaussianWorldModel::projectEwa2D(const Gaussian3DGpu& g, const Mat4& camFromWorld,
                                      const CameraIntrinsics& intr, Vec2& uvOut,
                                      float cov2dOut[3], float& depthOut) {
    const Vec3 pw{g.meanOpacity.x, g.meanOpacity.y, g.meanOpacity.z};
    const Vec3 pc = camFromWorld.transformPoint(pw);
    depthOut = pc.z;
    const float z = std::max(0.25f, pc.z);
    const float invZ = 1.0f / z;
    const float invZ2 = invZ * invZ;
    uvOut = {intr.cx + intr.fx * pc.x * invZ, intr.cy + intr.fy * pc.y * invZ};

    const float syaw = std::sin(g.velYaw.z), cyaw = std::cos(g.velYaw.z);
    const Vec3 axes[3] = {
        {cyaw * g.scaleSem.x, 0.0f, -syaw * g.scaleSem.x},
        {0.0f, g.scaleSem.y, 0.0f},
        {syaw * g.scaleSem.z, 0.0f, cyaw * g.scaleSem.z},
    };
    float cxx = 0.3f, cxy = 0.0f, cyy = 0.3f;
    for (int k = 0; k < 3; ++k) {
        const Vec3 ac = camFromWorld.transformVector(axes[k]);
        const float jx = intr.fx * (ac.x * invZ - pc.x * ac.z * invZ2);
        const float jy = intr.fy * (ac.y * invZ - pc.y * ac.z * invZ2);
        cxx += jx * jx;
        cxy += jx * jy;
        cyy += jy * jy;
    }
    cov2dOut[0] = cxx;
    cov2dOut[1] = cxy;
    cov2dOut[2] = cyy;
}

float GaussianWorldModel::queryOccupancyProb(
  const std::vector<V15Gaussian3D>& gaussians, const Vec3& worldPt,
  float horizonSec, bool roadBlockingOnly) {
  const float t = std::max(0.0f, horizonSec);
  const float grow = 1.0f + kDynScaleGrowthPerSec * t;
  double emptyProb = 1.0;
  for (const auto& g : gaussians) {
    if (roadBlockingOnly) {
      const int sem = static_cast<int>(std::lround(g.scaleSem.w));
      const bool roadSem =
        (sem == static_cast<int>(SemanticClass::Vehicle) ||
         sem == static_cast<int>(SemanticClass::Pedestrian));
      if (!roadSem || g.meanOpacity.w < kMinRoadGaussianOpacity ||
          g.scaleSem.y <= kMinRoadGaussianScaleYM ||
          g.meanOpacity.y > kMaxRoadGaussianCenterYM) {
        continue;
      }
    }
    const bool isDyn = (g.velYaw.w > kDynamicFlagThresh);
    const float dyn = isDyn ? t : 0.0f;
    const float dx = worldPt.x - (g.meanOpacity.x + g.velYaw.x * dyn);
    const float dy = worldPt.y - g.meanOpacity.y;
    const float dz = worldPt.z - (g.meanOpacity.z + g.velYaw.y * dyn);
    const float syaw = std::sin(g.velYaw.z);
    const float cyaw = std::cos(g.velYaw.z);
    const float lx = dx * cyaw - dz * syaw;
    const float lz = dx * syaw + dz * cyaw;
    const float dynGrow = isDyn ? grow : 1.0f;
    const float sx = std::max(kMinGaussianAxisM, g.scaleSem.x * dynGrow);
    const float sy = std::max(kMinGaussianAxisM, g.scaleSem.y);
    const float sz = std::max(kMinGaussianAxisM, g.scaleSem.z * dynGrow);
    const float mSq =
      (lx * lx) / (sx * sx) + (dy * dy) / (sy * sy) + (lz * lz) / (sz * sz);
    if (mSq > kMahalanobisSqCutoff) continue;
    const double occ = clampf(
      g.meanOpacity.w * std::exp(kGaussianExpHalf * mSq), 0.0f,
      kMaxGaussianOccProb);
    emptyProb *= (1.0 - occ);
  }
  return clampf(static_cast<float>(1.0 - emptyProb), 0.0f, 1.0f);
}

float GaussianWorldModel::queryOccupancyProb(const Vec3& worldPt,
                                             float horizonSec) const {
  return queryOccupancyProb(gaussians_, worldPt, horizonSec, false);
}

void GaussianWorldModel::setExternalGaussians(const std::vector<Gaussian3DGpu>& src) {
    if (gaussians_.capacity() < static_cast<size_t>(kMaxGaussians)) {
        gaussians_.reserve(static_cast<size_t>(kMaxGaussians));
    }
    const size_t n = std::min(src.size(), static_cast<size_t>(kMaxGaussians));
    gaussians_.assign(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(n));
}

void GaussianWorldModel::rollout4D(float horizonSec, std::vector<Gaussian3DGpu>& outPred) const {
    const float t = std::max(0.0f, horizonSec);
    const float grow = 1.0f + 0.15f * t;
    outPred = gaussians_;
    for (auto& g : outPred) {
        if (g.velYaw.w > 0.5f) {
            g.meanOpacity.x += g.velYaw.x * t;
            g.meanOpacity.z += g.velYaw.y * t;
            g.scaleSem.x *= grow;
            g.scaleSem.z *= grow;
        }
    }
}

void GaussianWorldModel::updateFromPerception(const SimConfig& cfg, const EgoState& ego,
                                                const StaticSpatialMemory& staticMem,
                                                const PatchTokenizer& patches,
                                                PerceptionFrameOutput& out) {
    gaussians_.clear();
    float vruBoost = 0.0f;
    if (patches.validSlots() > 0u && patches.ringSSBO().handle() != VK_NULL_HANDLE) {
        constexpr size_t kViewN = static_cast<size_t>(PatchTokenizer::kGridW) * PatchTokenizer::kGridH;
        const size_t offsetIdx = (static_cast<size_t>(patches.latestWrittenSlot()) * PatchTokenizer::kNumViews + 8u) * kViewN;
        vruScratch_.resize(kViewN);
        patches.ringSSBO().download(vruScratch_.data(), kViewN * sizeof(PatchTokenGpu),
                                    static_cast<VkDeviceSize>(offsetIdx) * sizeof(PatchTokenGpu));
        float edgeSum = 0.0f;
        int cnt = 0;
        for (int gy = PatchTokenizer::kGridH / 2; gy < PatchTokenizer::kGridH - 2; ++gy) {
            for (int gx = 8; gx < PatchTokenizer::kGridW - 8; ++gx) {
                const auto& pt = vruScratch_[static_cast<size_t>(gy) * PatchTokenizer::kGridW + gx];
                if (pt.app0.w > 0.5f) { edgeSum += std::hypot(pt.app1.z, pt.app1.w) + pt.app1.y; ++cnt; }
            }
        }
        if (cnt > 0) vruBoost = clampf((edgeSum / static_cast<float>(cnt)) * 0.25f, 0.0f, 0.15f);
    }

    for (const auto& tr : out.tracks) {
        if (static_cast<int>(gaussians_.size()) >= kMaxGaussians) break;
        const bool isPed = (tr.semantic == SemanticClass::Pedestrian);
        const float nearB = (isPed && std::hypot(tr.worldPos.x - ego.pos.x, tr.worldPos.z - ego.pos.z) < 16.0f) ? vruBoost : 0.0f;
        gaussians_.push_back({
            {tr.worldPos.x, std::max(0.55f, tr.halfExtents.y), tr.worldPos.z, clampf(tr.score * 0.92f + nearB, 0.35f, 0.98f)},
            {std::max(0.35f, tr.halfExtents.x), std::max(0.50f, tr.halfExtents.y),
             std::max(0.35f, tr.halfExtents.z), static_cast<float>(static_cast<int>(tr.semantic))},
            {tr.worldVel.x, tr.worldVel.z, tr.yaw, 1.0f},
            isPed ? Vec4{0.18f, 0.92f, 0.42f, tr.score} : Vec4{0.95f, 0.28f, 0.16f, tr.score}});
    }

    staticMem.readbackCells(cellScratch_);
    const int nx = staticMem.nx(), ny = staticMem.ny();
    if (static_cast<int>(cellScratch_.size()) != nx * ny || nx <= 0 || ny <= 0) return;
    const Mat4 wFromIso = BEVFormer::worldFromIsoEgo(ego);
    const float dx = (cfg.bevMaxX - cfg.bevMinX) / static_cast<float>(nx);
    const float dy = (cfg.bevMaxY - cfg.bevMinY) / static_cast<float>(ny);
    int fusedVoxels = 0;
    for (int iy = 0; iy < ny && static_cast<int>(gaussians_.size()) < kMaxGaussians; ++iy) {
        for (int ix = 0; ix < nx && static_cast<int>(gaussians_.size()) < kMaxGaussians; ++ix) {
            const auto& c = cellScratch_[static_cast<size_t>(iy) * nx + ix];
            if (c.featConf.x < 0.35f || c.featConf.w < 0.55f) continue;
            const float xIso = cfg.bevMinX + (ix + 0.5f) * dx, yIso = cfg.bevMinY + (iy + 0.5f) * dy;
            const Vec3 wp = wFromIso.transformPoint({xIso, yIso, 0.0f});
            const float h = std::max(0.7f, c.featConf.y);
            const bool isSign = (static_cast<int>(std::lround(c.featConf.z)) == static_cast<int>(SemanticClass::SignalSign));
            gaussians_.push_back({
                {wp.x, 0.5f * h, wp.z, clampf(0.45f + 0.5f * c.featConf.w, 0.0f, 0.96f)},
                {0.52f, 0.5f * h, 0.52f, c.featConf.z}, {0.0f, 0.0f, ego.yaw, 0.0f},
                isSign ? Vec4{0.96f, 0.78f, 0.18f, c.featConf.w} : Vec4{0.42f, 0.48f, 0.56f, c.featConf.w}});
            if (c.meta.z > 0.5f && fusedVoxels < 32 && xIso > 0.0f && std::abs(yIso) > 2.2f) {
                out.activeVoxels.push_back({xIso, yIso, 0.5f * h, wp.x, wp.z,
                    isSign ? SemanticClass::SignalSign : SemanticClass::Static, c.featConf.w, 0.0f, 0.0f, 0.05f});
                ++fusedVoxels;
            }
        }
    }
}

} // namespace cybercab::v15
