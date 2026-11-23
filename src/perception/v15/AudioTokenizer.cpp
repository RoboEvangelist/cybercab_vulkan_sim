#include "perception/v15/AudioTokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace cybercab::v15 {
namespace {

const Vec3 kMicPosVeh[AudioTokenizer::kNumMics] = {
    { 0.85f, 0.80f,  1.80f}, // front-left (+X left, +Z fwd)
    {-0.85f, 0.80f,  1.80f}, // front-right
    { 0.85f, 0.80f, -1.50f}, // rear-left
    {-0.85f, 0.80f, -1.50f}, // rear-right
};

float hashNoise(uint32_t s) {
    s ^= s >> 16;
    s *= 0x7feb352du;
    s ^= s >> 15;
    s *= 0x846ca68bu;
    s ^= s >> 16;
    return (static_cast<float>(s & 0xFFFFu) / 32767.5f) - 1.0f;
}

float goertzelPower(const float* samples, int count, float targetHz, float fs) {
    const float w = 2.0f * kPi * targetHz / fs;
    const float coeff = 2.0f * std::cos(w);
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float win = 0.5f - 0.5f * std::cos((2.0f * kPi * i) / (count - 1));
        s0 = samples[i] * win + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    return (s1 * s1 + s2 * s2 - coeff * s1 * s2) / static_cast<float>(count * count);
}

float sampleLinear(const float* buf, float idx) {
    const int k0 = static_cast<int>(std::floor(idx));
    const float f = idx - static_cast<float>(k0);
    return buf[k0] * (1.0f - f) + buf[k0 + 1] * f;
}

float beamPower(const std::array<std::array<float, AudioTokenizer::kFrameSamples>, AudioTokenizer::kNumMics>& pcm,
                float theta) {
    const Vec3 u{std::sin(theta), 0.0f, std::cos(theta)};
    float shifts[AudioTokenizer::kNumMics]{};
    for (int m = 0; m < AudioTokenizer::kNumMics; ++m) {
        shifts[m] = (kMicPosVeh[m].dot(u) / AudioTokenizer::kSpeedOfSound) *
                    static_cast<float>(AudioTokenizer::kSampleRateHz);
    }
    double pSum = 0.0;
    for (int n = 260; n < AudioTokenizer::kFrameSamples - 260; n += 2) {
        float acc = 0.0f;
        for (int m = 0; m < AudioTokenizer::kNumMics; ++m) {
            acc += sampleLinear(pcm[m].data(), static_cast<float>(n) - shifts[m]);
        }
        pSum += static_cast<double>(acc * acc);
    }
    return static_cast<float>(pSum);
}

} // namespace

void AudioTokenizer::synthesize48kHzFrame(const WorldEngine& world, uint32_t frameSeed,
                                          std::array<std::array<float, kFrameSamples>, kNumMics>& outPcm,
                                          float ambientNoiseRms) {
    const EgoState& ego = world.ego();
    const EmergencySirenState& siren = world.siren();
    const Mat4 worldFromEgo = world.getEgoWorldMatrix();
    const Vec3 egoVelW{std::sin(ego.yaw) * ego.speed, 0.0f, std::cos(ego.yaw) * ego.speed};

    Vec3 micW0[kNumMics]{};
    for (int m = 0; m < kNumMics; ++m) micW0[m] = worldFromEgo.transformPoint(kMicPosVeh[m]);

    for (int m = 0; m < kNumMics; ++m) {
        for (int n = 0; n < kFrameSamples; ++n) {
            float val = ambientNoiseRms * hashNoise(frameSeed * 9781u + static_cast<uint32_t>(m * kFrameSamples + n));
            if (siren.active) {
                const float t = static_cast<float>(n) / static_cast<float>(kSampleRateHz);
                const Vec3 src = siren.pos + siren.vel * t;
                const Vec3 rx = micW0[m] + egoVelW * t;
                const float dist = std::max(1.0f, (rx - src).norm());
                const float tRet = siren.phase + t - dist / kSpeedOfSound;
                const float phiWail = 2.0f * kPi * (siren.baseFreqHz * tRet -
                    (180.0f / (2.0f * kPi * 1.3f)) * std::cos(2.0f * kPi * 1.3f * tRet));
                const float phiPilot = 2.0f * kPi * kPilotCarrierHz * tRet;
                const float amp = 8.0f / std::max(dist, 2.0f);
                val += amp * (0.55f * std::sin(phiPilot) + 0.45f * std::sin(phiWail));
            }
            outPcm[m][n] = val;
        }
    }
}

AudioToken AudioTokenizer::process48kHzFrame(const std::array<std::array<float, kFrameSamples>, kNumMics>& pcm) {
    AudioToken tok{};
    const float fs = static_cast<float>(kSampleRateHz);
    float sirenPower = 0.0f, ambPower = 0.0f;
    for (int m = 0; m < kNumMics; ++m) {
        for (float f : {720.0f, 820.0f, 920.0f, 1040.0f, 1180.0f}) {
            sirenPower = std::max(sirenPower, goertzelPower(pcm[m].data(), kFrameSamples, f, fs));
        }
        float localAmb = 0.0f;
        for (float f : {180.0f, 280.0f, 380.0f}) localAmb += goertzelPower(pcm[m].data(), kFrameSamples, f, fs);
        ambPower += localAmb / 3.0f;
    }
    ambPower = std::max(1e-7f, ambPower / static_cast<float>(kNumMics));
    const float snr = sirenPower / ambPower;
    tok.confidence = clampf((snr - 4.0f) / 16.0f, 0.0f, 1.0f);
    tok.sirenDetected = (tok.confidence > 0.35f);
    if (!tok.sirenDetected) return tok;

    float bestTheta = 0.0f, bestP = -1.0f;
    for (int k = 0; k < 36; ++k) {
        const float th = -kPi + (2.0f * kPi * k) / 36.0f;
        const float p = beamPower(pcm, th);
        if (p > bestP) { bestP = p; bestTheta = th; }
    }
    const float coarseTh = bestTheta;
    for (int k = -8; k <= 8; ++k) {
        const float th = wrapAngle(coarseTh + degToRad(1.25f * static_cast<float>(k)));
        const float p = beamPower(pcm, th);
        if (p > bestP) { bestP = p; bestTheta = th; }
    }
    tok.bearingVehRad = wrapAngle(bestTheta);

    float bestF = kPilotCarrierHz, bestFp = -1.0f;
    for (int i = 0; i <= 80; ++i) {
        const float f = 840.0f + 2.0f * static_cast<float>(i);
        const float p = goertzelPower(pcm[0].data(), kFrameSamples, f, fs);
        if (p > bestFp) { bestFp = p; bestF = f; }
    }
    tok.peakFreqHz = bestF;
    tok.dopplerShiftHz = bestF - kPilotCarrierHz;
    tok.embed = {tok.confidence, std::sin(tok.bearingVehRad), std::cos(tok.bearingVehRad),
                 clampf(tok.dopplerShiftHz / 60.0f, -1.0f, 1.0f),
                 clampf(std::log10(std::max(snr, 1.0f)) / 3.0f, 0.0f, 1.0f),
                 bestF / 1500.0f, 0.0f, 1.0f};
    return tok;
}

AudioToken AudioTokenizer::stepFromWorld(const WorldEngine& world, uint32_t frameIdx) {
    std::array<std::array<float, kFrameSamples>, kNumMics> pcm{};
    synthesize48kHzFrame(world, frameIdx, pcm);
    return process48kHzFrame(pcm);
}

} // namespace cybercab::v15
