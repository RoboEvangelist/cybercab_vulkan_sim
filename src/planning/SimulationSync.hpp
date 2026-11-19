#pragma once

#include <algorithm>

namespace cybercab {

// Synchronized fixed-timestep simulation clock ensuring world agents, ego dynamics,
// camera rendering, and neural/Kalman delta-t remain aligned under variable display frame rates.
struct SimulationSyncClock {
    float accumSec = 0.0f;
    float simTimeSec = 0.0f;
    float lastPerceptionTimeSec = 0.0f;
    bool hasPerceivedOnce = false;
    int stepIndex = 0;

    void reset() {
        accumSec = 0.0f;
        simTimeSec = 0.0f;
        lastPerceptionTimeSec = 0.0f;
        hasPerceivedOnce = false;
        stepIndex = 0;
    }

    float perceptionDt(float fallbackDt) const {
        if (!hasPerceivedOnce) return fallbackDt;
        const float elapsed = simTimeSec - lastPerceptionTimeSec;
        return (elapsed > 1e-5f) ? elapsed : fallbackDt;
    }

    void markPerceptionExecuted() {
        lastPerceptionTimeSec = simTimeSec;
        hasPerceivedOnce = true;
    }

    // Steps a single display frame delta `rawDt` in fixed `dtControl` increments.
    // `onPerceive(simTime, percDt)` runs at most once per display frame before advancing the first physics step.
    // `onAdvancePhysics(stepIdx, simTime, dtControl)` runs for every fixed physics tick, advancing both agents and ego together.
    template <typename PerceiveFn, typename PhysicsStepFn>
    int stepFrame(float rawDt, float dtControl, float maxFrameDt,
                  PerceiveFn&& onPerceive, PhysicsStepFn&& onAdvancePhysics) {
        accumSec += std::min(std::max(0.0f, rawDt), maxFrameDt);
        int executedSteps = 0;
        bool ranPerception = false;
        while (accumSec + 1e-6f >= dtControl) {
            if (!ranPerception) {
                const float pDt = perceptionDt(dtControl);
                onPerceive(simTimeSec, pDt);
                markPerceptionExecuted();
                ranPerception = true;
            }
            onAdvancePhysics(stepIndex, simTimeSec, dtControl);
            accumSec = std::max(0.0f, accumSec - dtControl);
            simTimeSec += dtControl;
            ++stepIndex;
            ++executedSteps;
        }
        return executedSteps;
    }
};

} // namespace cybercab
