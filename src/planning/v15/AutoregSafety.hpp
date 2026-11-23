#pragma once
// FSD v15 Module 7: FMEA soil MRM, RSS safety shield, signal/stop-sign hold, siren yield & Autoshift D/R.
#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldEngine.hpp"
#include <string>
#include <vector>

namespace cybercab::v15 {

struct RotObbExtents {
    float longProj = 0.0f;
    float latProj = 0.0f;
};

struct SafetyEvalInput {
    const SimConfig& cfg;
    const EgoState& ego;
    const PerceptionFrameOutput& perc;
    const std::vector<Vec3>& corrPts;
    const std::vector<float>& corrArc;
    float rssDist;
    float bestLeadGap;
    bool sirenYield;
    bool bothLanesStaticBlocked;
    const std::vector<Vec3>* plannedTraj = nullptr;
};

struct SafetyEvalOutput {
    float desiredSpeed = 0.0f;
    float minForwardObsDist = 45.0f;
    bool aebTriggered = false;
    bool forwardRedundancyLost = false;
    int gear = 1;
    std::string behavior = "CRUISING";
};

RotObbExtents projectVehicleExtents(const Track3D& tr, float dirX, float dirZ);
SafetyEvalOutput evaluateAutoregSafety(WorldEngine& world, const SafetyEvalInput& in,
                                       std::string initialBehavior, float initialSpeed);

} // namespace cybercab::v15
