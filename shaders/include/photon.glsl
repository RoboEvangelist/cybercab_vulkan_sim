// FSD v15 Module 1 sensor model (mirror of src/sensors/PhotonModel.hpp; keep constants in sync).
#ifndef CYBERCAB_PHOTON_GLSL
#define CYBERCAB_PHOTON_GLSL

const float PHOTON_ADC_MAX_CODE = 16383.0; // 14-bit linear ADC

// Scene radiance -> normalized 14-bit linear code in [0, 1] (code / 16383). Multiply by fullWell for radiance.
vec3 photonLinearNorm(vec3 radiance, float fullWell) {
    return floor(clamp(radiance / fullWell, 0.0, 1.0) * PHOTON_ADC_MAX_CODE + 0.5) / PHOTON_ADC_MAX_CODE;
}

// PCG3D hash (Jarzynski & Olano 2020): decorrelated per-pixel/channel/frame random bits.
uvec3 photonPcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

// Three independent standard normals from two hash rounds (Box-Muller).
vec3 photonGaussian3(uvec3 key) {
    const uvec3 h1 = photonPcg3d(key);
    const uvec3 h2 = photonPcg3d(h1 ^ uvec3(0x9E3779B9u));
    const vec3 u1 = (vec3(h1 >> 8u) + 0.5) / 16777216.0; // (0, 1)
    const vec3 u2 = vec3(h2 >> 8u) / 16777216.0;         // [0, 1)
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

// Poisson-Gaussian noisy 14-bit ADC (var_DN = a*mu + b), normalized to [0, 1].
vec3 photonNoisyNorm(vec3 radiance, float fullWell, float a, float b, uvec3 key) {
    const vec3 mu = clamp(radiance / fullWell, 0.0, 1.0) * PHOTON_ADC_MAX_CODE;
    const vec3 dn = mu + sqrt(a * mu + b) * photonGaussian3(key);
    return floor(clamp(dn, 0.0, PHOTON_ADC_MAX_CODE) + 0.5) / PHOTON_ADC_MAX_CODE;
}

#endif
