#pragma once

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldEngine.hpp"

namespace cybercab {

class TrajectoryPlanner {
public:
    explicit TrajectoryPlanner(const SimConfig& cfg);

    // Zero-allocation hot-path planner writing into pre-allocated cmd
    void plan(WorldEngine& world, const PerceptionFrameOutput& perc, PlannerCommand& cmd) const;
    PlannerCommand plan(WorldEngine& world, const PerceptionFrameOutput& perc) const;

private:
    const SimConfig& cfg_;
    mutable std::vector<Vec3> corridorScratch_;
    mutable std::vector<float> corridorArcScratch_;

    static float quinticLateralOffset(float d0, float d1, float tNorm);
    void buildRssCorridor(const EgoState& ego, const ClosedCatmullRomCurve3& centerline,
                          float loopLen, float loopU, float lookaheadDist,
                          float currentLat, float targetLat, float maxLen) const;
};

} // namespace cybercab
