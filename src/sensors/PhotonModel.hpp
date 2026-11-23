#pragma once
// FSD v15 Module 1 (ISP bypass / "photon counting") sensor model shared by C++ and tests.
// GLSL mirror: shaders/include/photon.glsl (keep constants in sync).
//
// Radiance units: 1.0 == v14 display white (the v14 8-bit path clips there).
// The v15 sensor keeps linear headroom up to kFullWellRadiance and digitizes with a
// 14-bit linear ADC, so photon count stays proportional to scene radiance.
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cybercab::photon {

inline constexpr float kFullWellRadiance = 16.0f;  // 16x (~24 dB) highlight headroom over display white
inline constexpr int kAdcBits = 14;
inline constexpr float kAdcMaxCode = 16383.0f;     // 2^14 - 1

// Scene radiance -> integer 14-bit linear ADC code (round-to-nearest, saturating).
inline uint32_t linearCode(float radiance, float fullWell = kFullWellRadiance) {
    const float n = std::clamp(radiance / fullWell, 0.0f, 1.0f);
    return static_cast<uint32_t>(std::floor(n * kAdcMaxCode + 0.5f));
}

inline float linearCodeToRadiance(uint32_t code, float fullWell = kFullWellRadiance) {
    return static_cast<float>(code) / kAdcMaxCode * fullWell;
}

// Human-display ISP reference (tutorial Module 1.3): 8-bit gamma-2.2 encode over the same full scale.
inline uint32_t gamma8Code(float radiance, float fullScale) {
    const float n = std::clamp(radiance / fullScale, 0.0f, 1.0f);
    return static_cast<uint32_t>(std::lround(255.0f * std::pow(n, 1.0f / 2.2f)));
}

inline float gamma8CodeToRadiance(uint32_t code, float fullScale) {
    return std::pow(static_cast<float>(code) / 255.0f, 2.2f) * fullScale;
}

// Poisson-Gaussian sensor noise (Foi et al. 2008, "Practical Poissonian-Gaussian noise modeling"):
//   var_DN(mu) = a * mu + b,  a = 1/gain [DN],  b = (read_noise_e / gain)^2 [DN^2]
struct NoiseParams {
    float a = 0.0f;
    float b = 0.0f;
};

inline NoiseParams noiseParamsFromPhysical(float fullWellElectrons, float readNoiseElectrons) {
    const float gain = fullWellElectrons / kAdcMaxCode; // e- per DN
    return {1.0f / gain, (readNoiseElectrons / gain) * (readNoiseElectrons / gain)};
}

inline constexpr float kRccbLumaR = 0.299f;
inline constexpr float kRccbLumaG = 0.587f;
inline constexpr float kRccbLumaB = 0.114f;
inline constexpr float kRccbClearGain = 1.65f;
inline constexpr float kRccbLumaEps = 1e-4f;

inline float noiseVarianceDn(float meanDn, const NoiseParams& p) {
    return p.a * std::max(meanDn, 0.0f) + p.b;
}

inline float rccbClearRadiance(float r, float g, float b) {
  return kRccbClearGain * (kRccbLumaR * r + kRccbLumaG * g + kRccbLumaB * b);
}

inline float rccbEffectiveNoiseVarianceDn(float meanLumaDn,
                                          const NoiseParams& p) {
  const float clearMeanDn = std::max(meanLumaDn, 0.0f) * kRccbClearGain;
  const float rawVar = p.a * clearMeanDn + p.b;
  return rawVar / (kRccbClearGain * kRccbClearGain);
}

} // namespace cybercab::photon
