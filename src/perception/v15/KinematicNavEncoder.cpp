#include "perception/v15/KinematicNavEncoder.hpp"

#include <algorithm>
#include <cmath>

namespace cybercab::v15 {

void KinematicNavEncoder::reset() {
    imuRing_.clear();
    navTokens_ = {};
}

void KinematicNavEncoder::pushImuSample(const ImuSample& s) {
    if (static_cast<int>(imuRing_.size()) >= kImuCapacity) {
        imuRing_.pop_front();
    }
    imuRing_.push_back(s);
}

void KinematicNavEncoder::pushControlStep(const EgoState& prevEgo, const EgoState& currEgo, float dt) {
    const float safeDt = std::max(1e-4f, dt);
    const int subSteps = std::max(1, static_cast<int>(std::round(safeDt * 100.0f))); // 100 Hz sub-steps
    const float subDt = safeDt / static_cast<float>(subSteps);
    for (int i = 1; i <= subSteps; ++i) {
        const float f = static_cast<float>(i) / static_cast<float>(subSteps);
        pushImuSample({
            lerpf(prevEgo.speed, currEgo.speed, f),
            lerpf(prevEgo.accel, currEgo.accel, f),
            lerpf(prevEgo.yawRate, currEgo.yawRate, f),
            lerpf(prevEgo.steerAngle, currEgo.steerAngle, f),
            lerpf(prevEgo.pitchRate, currEgo.pitchRate, f),
            lerpf(prevEgo.rollRate, currEgo.rollRate, f),
            subDt,
        });
    }
}

const ImuSample& KinematicNavEncoder::latestSample() const {
    static const ImuSample kZero{};
    return imuRing_.empty() ? kZero : imuRing_.back();
}

ImuSample KinematicNavEncoder::meanImu() const {
    if (imuRing_.empty()) return {};
    ImuSample sum{};
    for (const auto& s : imuRing_) {
        sum.speed += s.speed;
        sum.accel += s.accel;
        sum.yawRate += s.yawRate;
        sum.steerAngle += s.steerAngle;
        sum.pitchRate += s.pitchRate;
        sum.rollRate += s.rollRate;
        sum.dt += s.dt;
    }
    const float inv = 1.0f / static_cast<float>(imuRing_.size());
    sum.speed *= inv;
    sum.accel *= inv;
    sum.yawRate *= inv;
    sum.steerAngle *= inv;
    sum.pitchRate *= inv;
    sum.rollRate *= inv;
    sum.dt *= inv;
    return sum;
}

void KinematicNavEncoder::encodeNavigation(const WorldEngine& world, float preferredLaneSign, float spacingM) {
    const auto& cl = world.centerline();
    const float loopLen = std::max(1.0f, cl.getLength());
    const float baseU = world.findClosestLoopU(world.ego().pos);
    const Mat4 vehFromWorld = world.getEgoWorldMatrix().inverseRigid();
    const float laneOffset = (preferredLaneSign >= 0.0f) ? kLaneCenterAbsM : -kLaneCenterAbsM;
    const float halfW = world.ego().hx;

    for (int i = 0; i < kNavWaypoints; ++i) {
        float u = std::fmod(baseU + (static_cast<float>(i + 1) * spacingM) / loopLen, 1.0f);
        if (u < 0.0f) u += 1.0f;
        const Vec3 ctr = cl.getPointAt(u);
        const Vec3 tanW = cl.getTangentAt(u);
        const Vec3 leftW{tanW.z, 0.0f, -tanW.x};
        const Vec3 wpW = ctr + leftW * laneOffset;
        const Vec3 pVeh = vehFromWorld.transformPoint(wpW);
        const Vec3 tVeh = vehFromWorld.transformVector(tanW);
        navTokens_[i] = {
            {pVeh.x, pVeh.z},
            {tVeh.x, tVeh.z},
            laneOffset,
            std::abs(laneOffset) - halfW,
            kRoadEdgeAbsM - (std::abs(laneOffset) + halfW),
        };
    }
}

} // namespace cybercab::v15
