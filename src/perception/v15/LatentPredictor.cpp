#include "perception/v15/LatentPredictor.hpp"

#include <algorithm>
#include <cmath>

namespace cybercab::v15 {
namespace {

constexpr size_t kFrameN = static_cast<size_t>(PatchTokenizer::kTokensPerFrame);
constexpr int kViewW = PatchTokenizer::kGridW;
constexpr int kViewH = PatchTokenizer::kGridH;

float tokenDiff(const PatchTokenGpu& a, const PatchTokenGpu& b) {
    const float dr = std::abs(a.app0.x - b.app0.x) + std::abs(a.app0.y - b.app0.y) + std::abs(a.app0.z - b.app0.z);
    const float de = std::abs(a.app1.y - b.app1.y) + std::abs(a.app1.z - b.app1.z) + std::abs(a.app1.w - b.app1.w);
    return 0.45f * dr + 0.85f * de;
}

} // namespace

void LatentPredictor::reset() { last_ = {}; }

LatentEvalResult LatentPredictor::evaluateTokens(const PatchTokenGpu* prevSlot, const PatchTokenGpu* currSlot,
                                                 const EgoState& prevEgo, const EgoState& currEgo,
                                                 uint32_t validSlots, float staticSkipRatio) {
    LatentEvalResult res{};
    if (!currSlot) return res;

    int activeCnt = 0;
    for (size_t i = 0; i < kFrameN; ++i) {
        const auto& t = currSlot[i];
        const float activity = t.app1.y + std::hypot(t.app1.z, t.app1.w);
        if (t.app0.w > 0.5f && activity > 0.035f) ++activeCnt;
    }
    const float activeFrac = static_cast<float>(activeCnt) / static_cast<float>(kFrameN);
    res.telemetry.tokenSparsityRatio = 1.0f - activeFrac;
    res.telemetry.staticSkipRatio = clampf(staticSkipRatio, 0.0f, 1.0f);
    res.telemetry.kvCacheReuseRatio = (validSlots >= 2u)
        ? clampf(static_cast<float>(std::min<uint32_t>(validSlots, PatchTokenizer::kHistoryFrames) - 1u) /
                 static_cast<float>(PatchTokenizer::kHistoryFrames), 0.0f, 0.75f)
        : 0.0f;
    const float workFrac = (1.0f - 0.45f * res.telemetry.staticSkipRatio) *
                           (1.0f - 0.35f * res.telemetry.kvCacheReuseRatio) *
                           (0.45f + 0.55f * activeFrac);
    res.telemetry.effectiveSpeedupX = 2.0f / std::max(0.18f, workFrac); // 2x FP16 tensor speedup

    if (!prevSlot || validSlots < 2u) return res;

    const int yawShift = static_cast<int>(std::lround(wrapAngle(currEgo.yaw - prevEgo.yaw) * 18.0f));
    constexpr int kFwdViews[4] = {0, 1, 2, 8};
    std::vector<float> residuals;
    residuals.reserve(512);
    for (int v : kFwdViews) {
        const size_t base = static_cast<size_t>(v) * static_cast<size_t>(kViewW * kViewH);
        for (int gy = 7; gy <= 20; ++gy) {
            for (int gx = 7; gx <= 24; ++gx) {
                const PatchTokenGpu& c = currSlot[base + static_cast<size_t>(gy) * kViewW + gx];
                if (c.app0.w < 0.5f) continue;
                float best = 10.0f;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int py = std::clamp(gy + dy, 0, kViewH - 1);
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int px = std::clamp(gx + yawShift + dx, 0, kViewW - 1);
                        const PatchTokenGpu& p = prevSlot[base + static_cast<size_t>(py) * kViewW + px];
                        if (p.app0.w > 0.5f) best = std::min(best, tokenDiff(c, p));
                    }
                }
                if (best < 9.0f) residuals.push_back(best);
            }
        }
    }
    if (!residuals.empty()) {
        const size_t topK = std::min<size_t>(16u, residuals.size());
        std::nth_element(residuals.begin(), residuals.end() - topK, residuals.end());
        double topSum = 0.0;
        for (size_t i = residuals.size() - topK; i < residuals.size(); ++i) topSum += residuals[i];
        const float peakSurprise = static_cast<float>(topSum / static_cast<double>(topK));
        res.latentNovelty = clampf((peakSurprise - 0.14f) / 0.45f, 0.0f, 1.0f);
        res.surpriseSigmaDelta = 0.48f * res.latentNovelty;
    }
    return res;
}

LatentEvalResult LatentPredictor::evaluate(const PatchTokenizer& patches, const StaticSpatialMemory& staticMem) {
    const float skipRatio = staticMem.readbackStats().selectiveRefreshSkipRatio;
    const uint32_t valid = patches.validSlots();
    if (patches.ringSSBO().handle() == VK_NULL_HANDLE || valid == 0u) {
        last_ = {};
        last_.telemetry.staticSkipRatio = clampf(skipRatio, 0.0f, 1.0f);
        return last_;
    }
    const uint32_t currSlot = patches.latestWrittenSlot();
    currScratch_.resize(kFrameN);
    patches.ringSSBO().download(currScratch_.data(), kFrameN * sizeof(PatchTokenGpu),
                                static_cast<VkDeviceSize>(currSlot) * kFrameN * sizeof(PatchTokenGpu));
    const PatchTokenGpu* prevPtr = nullptr;
    uint32_t prevSlot = currSlot;
    if (valid >= 2u) {
        prevSlot = (currSlot + PatchTokenizer::kHistoryFrames - 1u) % PatchTokenizer::kHistoryFrames;
        prevScratch_.resize(kFrameN);
        patches.ringSSBO().download(prevScratch_.data(), kFrameN * sizeof(PatchTokenGpu),
                                    static_cast<VkDeviceSize>(prevSlot) * kFrameN * sizeof(PatchTokenGpu));
        prevPtr = prevScratch_.data();
    }
    const auto& egos = patches.slotEgoStates();
    last_ = evaluateTokens(prevPtr, currScratch_.data(), egos[prevSlot], egos[currSlot], valid, skipRatio);
    return last_;
}

} // namespace cybercab::v15
