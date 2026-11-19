// Shared GLSL library for every Cybercab compute/graphics shader.
// Mirrors the C++ structs in src/perception/PerceptionTypes.hpp and src/sensors/CameraModel.hpp.
#ifndef CYBERCAB_COMMON_GLSL
#define CYBERCAB_COMMON_GLSL

const uint SEM_EMPTY       = 0u;
const uint SEM_VEHICLE     = 1u;
const uint SEM_PEDESTRIAN  = 2u;
const uint SEM_SIGNAL_SIGN = 3u;
const uint SEM_STATIC      = 4u;
const uint SEM_OVERHANG    = 5u;
const uint SEM_CURB        = 6u;
const uint SEM_OCC_SHADOW  = 7u;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float relu(float v) { return max(v, 0.0); }
float sigmoid(float v) { return 1.0 / (1.0 + exp(-clamp(v, -12.0, 12.0))); }
float silu(float v) { return v * sigmoid(v); }

// Brown-Conrady lens model, d = (k1, k2, p1, p2), operating on normalized image coordinates.
vec2 brownConradyTangential(vec2 xu, vec4 d) {
    float r2 = dot(xu, xu);
    return vec2(2.0 * d.z * xu.x * xu.y + d.w * (r2 + 2.0 * xu.x * xu.x),
                d.z * (r2 + 2.0 * xu.y * xu.y) + 2.0 * d.w * xu.x * xu.y);
}

vec2 brownConradyDistort(vec2 xu, vec4 d) {
    float r2 = dot(xu, xu);
    float radial = 1.0 + d.x * r2 + d.y * r2 * r2;
    return xu * radial + brownConradyTangential(xu, d);
}

// Fixed-point inversion of the forward model (4 iterations is sub-pixel for |k1| < 0.3).
vec2 brownConradyUndistort(vec2 xd, vec4 d) {
    vec2 xu = xd;
    for (int it = 0; it < 4; ++it) {
        float r2 = dot(xu, xu);
        float radial = 1.0 + d.x * r2 + d.y * r2 * r2;
        xu = (xd - brownConradyTangential(xu, d)) / max(radial, 0.35);
    }
    return xu;
}

struct TrackPriorGpu {
    vec4 posVel;   // egoX, egoY, vx_world, vz_world
    vec4 sizeMeta; // hx, hz, classId, sigmaPos
    vec4 yawValid; // yaw_world, confidence, trackId, activeFlag
};

struct WorldGhostGpu {
    vec4 worldPosScale; // wx, wy, wz, scale
    vec4 colorAlpha;    // r, g, b, alpha
    vec4 yawMeta;       // yaw, horizonSec, sigma, trackId
};

struct ActiveVoxelGpu {
    vec4 world; // wx, wy, wz, semantic
    vec4 ego;   // egoX, egoY, confidence, epistemicSigma
    vec4 flow;  // vx_world, vz_world, speed, iz
};

#endif
