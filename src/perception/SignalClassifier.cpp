#include "perception/SignalPerception.hpp"
#include <algorithm>
#include <cmath>

namespace cybercab::signal {
namespace {

constexpr int kNumStates = SignalConfusionMatrix::kN;
constexpr int kBulbCount = 3;
constexpr int kCamMain = 0;
constexpr int kCamTele = 1;
constexpr int kBorderPx = 2;
constexpr int kMinImgDim = 4;
constexpr int kMinPatchRad = 1;
constexpr int kMaxPatchRad = 6;
constexpr int kIdxUnknown = static_cast<int>(SignalState::Unknown);
constexpr size_t kRgbaCh = 4u;

constexpr float kBulbOffYM = 0.38f, kBulbRadM = 0.14f, kMinDepthM = 1.0f;
constexpr float kMinLitEnergy = 0.14f, kSoftmaxBeta = 6.5f;
constexpr float kTeleMinConf = 0.55f;
constexpr float kInv255 = 1.0f / 255.0f, kEps = 1e-6f, kHalf = 0.5f;
constexpr float kRedGain = 0.35f, kRedFloor = 0.50f;
constexpr float kYelMinR = 0.45f, kYelMinG = 0.25f;
constexpr float kYelBias = 0.25f, kYelGain = 0.30f;
constexpr float kGrnGain = 0.30f, kGrnBlueMix = 0.35f;
constexpr std::array<float, kNumStates> kUnlitProbs{0.06f, 0.04f, 0.02f, 0.88f};

struct BulbPeak {
  float red = 0.0f;
  float yellow = 0.0f;
  float green = 0.0f;
};

float redBulbScore(float r, float g, float b) {
  const float base = std::max(0.0f, r - std::max(g, b));
  const float bloomR = std::max(0.0f, r - kRedFloor);
  const float redOverB = std::max(0.0f, r - b);
  return base + kRedGain * bloomR * redOverB;
}

float yellowBulbScore(float r, float g, float b) {
  const float base = std::max(0.0f, std::min(r, g) - b);
  float satBoost = 0.0f;
  if (r > kYelMinR && g > kYelMinG) {
    satBoost = std::max(0.0f, kHalf * (r + g) - b - kYelBias);
  }
  return base + kYelGain * satBoost;
}

float greenBulbScore(float r, float g, float b) {
  const float base = std::max(0.0f, g - r);
  const float cyanBoost = std::max(0.0f, g + kGrnBlueMix * b - r);
  return base + kGrnGain * cyanBoost;
}

bool sampleBulbPatch(const std::vector<uint8_t>& rgba, int w, int h,
                     const CameraRigRenderer& rig, int cam, const Vec3& pt,
                     int rad, BulbPeak& outPeak) {
  Vec2 px{};
  float depth = 0.0f;
  if (!rig.projectWorldToPixel(cam, pt, px, depth)) {
    return false;
  }
  const int cx = static_cast<int>(std::lround(px.x));
  const int cy = static_cast<int>(std::lround(px.y));
  if (cx < kBorderPx || cx >= w - kBorderPx ||
      cy < kBorderPx || cy >= h - kBorderPx) {
    return false;
  }
  outPeak = {};
  for (int dy = -rad; dy <= rad; ++dy) {
    const int y = std::clamp(cy + dy, 0, h - 1);
    for (int dx = -rad; dx <= rad; ++dx) {
      const int x = std::clamp(cx + dx, 0, w - 1);
      const size_t idx = (static_cast<size_t>(y) * w + x) * kRgbaCh;
      const float r = static_cast<float>(rgba[idx]) * kInv255;
      const float g = static_cast<float>(rgba[idx + 1u]) * kInv255;
      const float b = static_cast<float>(rgba[idx + 2u]) * kInv255;
      outPeak.red = std::max(outPeak.red, redBulbScore(r, g, b));
      outPeak.yellow = std::max(outPeak.yellow, yellowBulbScore(r, g, b));
      outPeak.green = std::max(outPeak.green, greenBulbScore(r, g, b));
    }
  }
  return true;
}

void assignSoftmaxProbs(const float energies[kBulbCount], SignalObs& obs) {
  const float peakE = std::max({energies[0], energies[1], energies[2]});
  if (peakE < kMinLitEnergy) {
    obs.rawState = SignalState::Unknown;
    obs.probs = kUnlitProbs;
    obs.confidence = obs.probs[kIdxUnknown];
    return;
  }
  const float logits[kNumStates] = {
    energies[0], energies[1], energies[2], kMinLitEnergy};
  const float maxLogit =
    std::max({logits[0], logits[1], logits[2], logits[3]});
  float expVals[kNumStates]{};
  float sumExp = 0.0f;
  for (int i = 0; i < kNumStates; ++i) {
    expVals[i] = std::exp(kSoftmaxBeta * (logits[i] - maxLogit));
    sumExp += expVals[i];
  }
  int bestIdx = kIdxUnknown;
  for (int i = 0; i < kNumStates; ++i) {
    obs.probs[static_cast<size_t>(i)] = expVals[i] / std::max(kEps, sumExp);
    if (obs.probs[static_cast<size_t>(i)] > obs.confidence) {
      obs.confidence = obs.probs[static_cast<size_t>(i)];
      bestIdx = i;
    }
  }
  obs.rawState = static_cast<SignalState>(bestIdx);
}

}  // namespace

SignalObs VisionSignalClassifier::classifyHeadFromRgba(
    const std::vector<uint8_t>& rgba, int w, int h,
    const CameraRigRenderer& rig, int cam, const SignalHeadSpec& head, int id) {
  SignalObs obs{};
  obs.headIndex = id;
  obs.targetLane = head.targetLane;
  obs.role = head.role;
  const size_t need = static_cast<size_t>(w) * h * kRgbaCh;
  if (rgba.size() < need || w <= kMinImgDim || h <= kMinImgDim) {
    return obs;
  }
  Vec2 centerPx{};
  float depth = 0.0f;
  if (!rig.projectWorldToPixel(cam, head.worldPos, centerPx, depth)) {
    return obs;
  }
  const float fx = rig.cameraMatrices(cam).intr.fx;
  const int rad = std::clamp(
    static_cast<int>(std::lround(kBulbRadM * fx / std::max(kMinDepthM, depth))),
    kMinPatchRad, kMaxPatchRad);
  const float bulbOffsetY[kBulbCount] = {kBulbOffYM, 0.0f, -kBulbOffYM};
  float energies[kBulbCount]{};
  for (int k = 0; k < kBulbCount; ++k) {
    BulbPeak peak{};
    const Vec3 bulbPos = head.worldPos + Vec3{0.0f, bulbOffsetY[k], 0.0f};
    if (!sampleBulbPatch(rgba, w, h, rig, cam, bulbPos, rad, peak)) {
      return obs;
    }
    energies[k] = (k == 0) ? peak.red : ((k == 1) ? peak.yellow : peak.green);
  }
  obs.visible = true;
  assignSoftmaxProbs(energies, obs);
  return obs;
}

std::vector<SignalObs> VisionSignalClassifier::classifyWorldHeadsFromRig(
    const CameraRigRenderer& rig, const std::vector<SignalHeadSpec>& heads) {
  const int w = rig.camWidth();
  const int h = rig.camHeight();
  const auto rgbaTele = rig.downloadCameraRGBA(kCamTele);
  const auto rgbaMain = rig.downloadCameraRGBA(kCamMain);
  std::vector<SignalObs> out;
  out.reserve(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    const int headId = static_cast<int>(i);
    const SignalObs oTele = classifyHeadFromRgba(rgbaTele, w, h, rig, kCamTele,
                                                 heads[i], headId);
    if (oTele.visible && oTele.confidence >= kTeleMinConf) {
      out.push_back(oTele);
      continue;
    }
    const SignalObs oMain = classifyHeadFromRgba(rgbaMain, w, h, rig, kCamMain,
                                                 heads[i], headId);
    const bool pickMain =
      oMain.visible && (!oTele.visible || oMain.confidence >= oTele.confidence);
    out.push_back(pickMain ? oMain : oTele);
  }
  return out;
}

}  // namespace cybercab::signal
