#pragma once

#include "core/Math.hpp"
#include "world/WorldTypes.hpp"
#include <array>
#include <cstdint>
#include <map>
#include <vector>

namespace cybercab {

struct SoilPatchStats {
    float soilProb = 0.0f;
    float patchConfidence = 1.0f;
    float epistemicVar = 0.04f;
    float meanEdge = 0.0f;
};

struct SoilCameraSummary {
    int camIndex = 0;
    float meanSoilProb = 0.0f;
    float tokenWeight = 1.0f;
    float meanEpistemicVar = 0.04f;
    float meanEdge = 0.0f;
};

struct Detection2D {
    int camIndex = 0;
    float cxNorm = 0.0f;
    float cyNorm = 0.0f;
    float wNorm = 0.0f;
    float hNorm = 0.0f;
    float score = 0.0f;
    SemanticClass semantic = SemanticClass::Vehicle;
    float camPatchWeight = 1.0f;
    float boxSigma = 0.1f;
};

struct Detection3D {
    int camIndex = 0;
    SemanticClass semantic = SemanticClass::Vehicle;
    float score = 0.0f;
    Vec3 worldPos{};
    Vec3 egoPos{}; // ISO Ego: X fwd, Y left, Z up
    Vec3 halfExtents{0.96f, 0.72f, 2.18f};
    float yaw = 0.0f;
    float rangeM = 0.0f;
    float camWeight = 1.0f;
    float epistemicSigma = 0.2f; // Combined geometry + soil uncertainty (m)
    Detection2D det2d{};
};

enum class TrackState : uint8_t {
    Tentative = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

struct Track3D {
    int trackId = 0;
    SemanticClass semantic = SemanticClass::Vehicle;
    TrackState state = TrackState::Tentative;
    Vec3 worldPos{};
    Vec3 worldVel{};
    Vec3 egoPos{};
    Vec3 halfExtents{0.96f, 0.72f, 2.18f};
    float yaw = 0.0f;
    float score = 0.0f;
    float posSigmaM = 0.5f;   // Propagated Kalman std dev sqrt(trace(P_pos))
    int hits = 0;
    float timeSinceUpdateSec = 0.0f;
    bool matchedLowConf = false; // True when kept alive by ByteTrack Stage-2!
};

struct alignas(16) TrackPriorGpu {
    Vec4 posVel;   // egoX, egoY, vx_world, vz_world
    Vec4 sizeMeta; // hx, hz, float(classId), sigmaPos
    Vec4 yawValid; // yaw_world, confidence, float(trackId), activeFlag
};

struct alignas(16) WorldGhost {
    Vec4 worldPosScale; // wx, wy, wz, scale
    Vec4 colorAlpha;    // r, g, b, alpha
    Vec4 yawMeta;       // yaw, horizonSec, uncertaintySigma, trackId
};

struct ActiveVoxel {
    float egoX = 0.0f;
    float egoY = 0.0f;
    float wy = 0.0f;
    float wx = 0.0f;
    float wz = 0.0f;
    SemanticClass sem = SemanticClass::Empty;
    float confidence = 0.0f;
    float vx = 0.0f;
    float vz = 0.0f;
    float epistemicSigma = 0.0f;
};

struct BevSummary {
    float meanObservability = 0.0f;
    float meanEpistemicVar = 0.0f;
    int observedCells = 0;
};

struct EvalMetrics {
    float detRecall = 0.0f;
    float detPrecision = 0.0f;
    float trackPosRmseM = 0.0f;
    int matchedGtObjects = 0;
    int totalGtDynamic = 0;
    int ghostPedDetections = 0;
    int ghostVehDetections = 0;
    int misclassifiedTrucks = 0;
    int truckVehVoxels = 0;
    int truckPedVoxels = 0;
    int signPoleVoxels = 0;
    double totalPipelineMs = 0.0;
};

struct alignas(16) V15Gaussian3D {
  Vec4 meanOpacity;  // wx, wy, wz, alpha (Y up, world space)
  Vec4 scaleSem;     // sx, sy, sz, semF
  Vec4 velYaw;       // vx, vz, yaw, isDynamic
  Vec4 colorConf;    // r, g, b, confidence
};

struct V15PerceptionSummary {
    bool active = false;
    bool sirenDetected = false;
    float sirenConfidence = 0.0f;
    float sirenBearingRad = 0.0f;
    float sirenDopplerHz = 0.0f;
    float latentNovelty = 0.0f;
    float surpriseSigmaDelta = 0.0f;
    float staticSkipRatio = 0.0f;
    float tokenSparsityRatio = 0.0f;
    float kvCacheReuseRatio = 0.0f;
    float effectiveSpeedupX = 1.0f;
    float preferredLaneOffsetM = -1.95f;
    float rearClearanceM = 40.0f;
  float calibPitchDeg = 0.0f;
  float calibRollDeg = 0.0f;
};

struct PerceptionFrameOutput {
    std::array<SoilCameraSummary, 8> soilCameras{};
    std::vector<SoilPatchStats> soilPatches; // 8 * patchCols * patchRows
    float forwardEpistemicSigma = 0.0f;

    std::vector<Detection2D> detections2D;
    std::vector<Detection3D> detections3D;
    std::vector<Track3D> tracks;

    BevSummary bevSummary{};
    std::vector<ActiveVoxel> activeVoxels;
    std::vector<ActiveVoxel> shadowVoxels;
    std::vector<WorldGhost> worldGhosts;

    std::map<std::string, double> gpuStageMs;
    EvalMetrics eval{};
    V15PerceptionSummary v15{};
  std::vector<V15Gaussian3D> v15Gaussians;
  std::vector<uint8_t> v15RealityRgba;

  void reserveV15Scratch(size_t maxGaussians, size_t realityBytes) {
    v15Gaussians.reserve(maxGaussians);
    v15RealityRgba.reserve(realityBytes);
  }
};

} // namespace cybercab
