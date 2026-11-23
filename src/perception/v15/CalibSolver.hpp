#pragma once
// FSD v15 Module 2a (CPU core): online pitch/roll self-calibration from Manhattan vertical structure.
// Every world-vertical edge lies in a plane through the camera centre with normal n, and n . up = 0.
// Stacking these planes across 8 cameras and a sliding time window yields up = (a, 1, b)/|.| by IRLS.
#include "config/SimConfig.hpp"
#include <vector>

namespace cybercab::v15 {

struct CalibRayObs {        // one gradient tile (GPU reduction output)
    int camIdx = 0;
    float cxPx = 0.0f;      // magnitude-weighted centroid (distorted pixels)
    float cyPx = 0.0f;
    float sumGx = 0.0f;     // summed d/dx, d/dy of log-radiance
    float sumGy = 0.0f;
    float sumMag = 0.0f;
    float coherence = 0.0f; // |sum g| / sum |g|  (1 = single straight edge)
};

struct CalibEstimate {
    float pitchRad = 0.0f, rollRad = 0.0f;
    float pitchDeg = 0.0f, rollDeg = 0.0f;
    float residualDeg = 0.0f;
    float confidence = 0.0f;
    int inlierCount = 0;
    bool converged = false;
};

namespace calib {
// World-up expressed in the sprung vehicle-local frame (+X left, +Y up, +Z fwd).
Vec3 uprightVec(float pitchRad, float rollRad);
// Rejects image-border and clamped-overscan pixels (camera-fixed edges would bias toward 0).
bool insideValidOptics(const CameraSpec& spec, const CameraIntrinsics& intr, float pxX, float pxY, const Vec2& xu);
// Edge-plane normals in vehicle-local frame: xyz = unit n, w = weight. Vertical chains are linked
// per camera; allowTileFallback adds per-tile gradient planes when < 4 chains exist (cold start).
void extractEdgeNormals(const SimConfig& cfg, int camW, int camH, const std::vector<CalibRayObs>& obs,
                        float priorPitchRad, float priorRollRad, std::vector<Vec4>& out,
                        bool allowTileFallback = true);
// LS -> Huber -> Tukey IRLS on n . (a, 1, b) = 0.
CalibEstimate solveUpright(const std::vector<Vec4>& eqs, float priorPitchRad, float priorRollRad);
CalibEstimate solveFromObservations(const SimConfig& cfg, int camW, int camH,
                                    const std::vector<CalibRayObs>& obs,
                                    float priorPitchRad = 0.0f, float priorRollRad = 0.0f);
} // namespace calib

} // namespace cybercab::v15
