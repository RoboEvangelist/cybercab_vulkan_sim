#pragma once
// FSD v15 Module 3a: Simulated 4-channel 48 kHz exterior acoustic array & siren tokenizer.
// Synthesizes retarded-time propagation (ITD + Doppler + 1/r attenuation) and recovers
// siren presence, 360-deg bearing (delay-and-sum beamforming), and Goertzel Doppler shift.
#include "world/WorldEngine.hpp"

#include <array>
#include <cstdint>

namespace cybercab::v15 {

struct AudioToken {
    bool sirenDetected = false;
    float confidence = 0.0f;      // [0, 1] normalized siren-band excess over noise floor
    float bearingVehRad = 0.0f;   // atan2(x_left, z_fwd) in vehicle frame (0 = ahead, +pi/2 = left)
    float dopplerShiftHz = 0.0f;  // positive = closing relative velocity toward ego
    float peakFreqHz = 0.0f;
    std::array<float, 8> embed{}; // compact audio token for multimodal fusion / planner
};

class AudioTokenizer {
public:
    static constexpr int kNumMics = 4;
    static constexpr int kSampleRateHz = 48000;
    static constexpr int kFrameSamples = 1333; // ~27.78 ms (one 36 Hz control tick)
    static constexpr float kSpeedOfSound = 343.0f;
    static constexpr float kPilotCarrierHz = 920.0f;

    // Synthesize 48 kHz exterior microphone PCM for one tick from WorldEngine state + road noise.
    static void synthesize48kHzFrame(const WorldEngine& world, uint32_t frameSeed,
                                     std::array<std::array<float, kFrameSamples>, kNumMics>& outPcm,
                                     float ambientNoiseRms = 0.015f);

    // Goertzel siren-band detection + 4-mic delay-and-sum beamforming bearing + Doppler estimate.
    static AudioToken process48kHzFrame(const std::array<std::array<float, kFrameSamples>, kNumMics>& pcm);

    // Convenience helper combining synthesis & tokenization for hot-path runtime.
    static AudioToken stepFromWorld(const WorldEngine& world, uint32_t frameIdx);
};

} // namespace cybercab::v15
