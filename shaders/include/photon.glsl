// FSD v15 Module 1 sensor model (mirror of src/sensors/PhotonModel.hpp; keep constants in sync).
#ifndef CYBERCAB_PHOTON_GLSL
#define CYBERCAB_PHOTON_GLSL

const float PHOTON_ADC_MAX_CODE = 16383.0; // 14-bit linear ADC

// Scene radiance -> normalized 14-bit linear code in [0, 1] (code / 16383). Multiply by fullWell for radiance.
vec3 photonLinearNorm(vec3 radiance, float fullWell) {
    return floor(clamp(radiance / fullWell, 0.0, 1.0) * PHOTON_ADC_MAX_CODE + 0.5) / PHOTON_ADC_MAX_CODE;
}

#endif
