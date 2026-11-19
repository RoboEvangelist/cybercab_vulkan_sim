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
    static float quinticLateralOffset(float d0, float d1, float tNorm);
};

} // namespace cybercab
