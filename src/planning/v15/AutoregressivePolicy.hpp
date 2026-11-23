#pragma once
// FSD v15 Module 7: Autoregressive trajectory policy (x, y, v, kappa) enforcing middle-of-lane
// driving (+/-1.95 m), bounded truck shy-away, comfort curvature limits, YIELDING_SIREN & Autoshift D/R.
#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldEngine.hpp"
#include <vector>

namespace cybercab::v15 {

struct AutoregConfig {
    static constexpr float kLaneCenterAbsM = 1.95f;
    static constexpr float kMinLaneAbsM = 1.65f;
    static constexpr float kMaxLaneAbsM = 2.25f;
    static constexpr float kMaxTruckShyM = 0.22f;
    float maxLatAccelMs2 = 3.0f;
    float lookaheadMinM = 14.0f;
};

class AutoregressivePolicy {
public:
    static float clampToMiddleOfLane(float lat);
    static float computeTruckShyOffset(float baseLaneLat, const EgoState& ego,
                                       const std::vector<Track3D>& tracks);
    static void rolloutMiddleOfLane(const EgoState& ego, const ClosedCatmullRomCurve3& centerline,
                                    float loopU, float currentLat, float targetLat,
                                    int steps, float lookaheadDist,
                                    std::vector<Vec3>& outTraj, float& outMaxKappa);
    static void plan(const SimConfig& cfg, WorldEngine& world,
                     const PerceptionFrameOutput& perc, PlannerCommand& cmd,
                     const AutoregConfig& acfg = {});
};

} // namespace cybercab::v15
