#pragma once
// FSD v15 Module 3b: 100 Hz inertial/kinematic ring buffer and vectorized lane-center navigation tokens.
// Explicitly targets the middle of the active lane (+/-1.95 m) away from the 0.0 m divider and +/-4.23 m limits.
#include "world/WorldEngine.hpp"

#include <array>
#include <deque>

namespace cybercab::v15 {

struct ImuSample {
    float speed = 0.0f;
    float accel = 0.0f;
    float yawRate = 0.0f;
    float steerAngle = 0.0f;
    float pitchRate = 0.0f;
    float rollRate = 0.0f;
    float dt = 0.01f;
};

struct NavWaypointToken {
    Vec2 posVeh{};              // (+X left, +Z fwd) in vehicle frame at middle of target lane
    Vec2 tanVeh{};              // unit road tangent in vehicle XZ
    float laneCenterOffsetM = -1.95f;
    float dividerClearanceM = 0.99f;    // inner body edge clearance to 0.0 m road divider
    float outerLimitClearanceM = 1.32f; // outer body edge clearance to +/-4.23 m curb edge
};

class KinematicNavEncoder {
public:
    static constexpr int kImuCapacity = 32; // 0.32 s history at 100 Hz
    static constexpr int kNavWaypoints = 8;
    static constexpr float kLaneCenterAbsM = 1.95f;
    static constexpr float kRoadEdgeAbsM = 4.23f;

    void reset();
    void pushImuSample(const ImuSample& s);
    void pushControlStep(const EgoState& prevEgo, const EgoState& currEgo, float dt);
    void encodeNavigation(const WorldEngine& world, float preferredLaneSign = -1.0f, float spacingM = 5.0f);

    int sampleCount() const { return static_cast<int>(imuRing_.size()); }
    const ImuSample& latestSample() const;
    ImuSample meanImu() const;
    const std::array<NavWaypointToken, kNavWaypoints>& navTokens() const { return navTokens_; }
    const std::deque<ImuSample>& imuRing() const { return imuRing_; }

private:
    std::deque<ImuSample> imuRing_;
    std::array<NavWaypointToken, kNavWaypoints> navTokens_{};
};

} // namespace cybercab::v15
