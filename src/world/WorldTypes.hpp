#pragma once

#include "core/Math.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace cybercab {

enum class SemanticClass : uint8_t {
    Empty = 0,
    Vehicle = 1,
    Pedestrian = 2,
    SignalSign = 3,
    Static = 4,
    Overhang = 5,
    Curb = 6,
    OccShadow = 7
};

enum class SignalPhase : uint8_t {
    Green = 0,
    Yellow = 1,
    Red = 2
};

inline const char* signalPhaseName(SignalPhase p) {
    switch (p) {
        case SignalPhase::Green: return "GREEN";
        case SignalPhase::Yellow: return "YELLOW";
        case SignalPhase::Red: return "RED";
    }
    return "UNKNOWN";
}

struct OBB {
    std::string id;
    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
    float hx = 0.5f;
    float hy = 0.5f;
    float hz = 0.5f;
    float yaw = 0.0f;
    float vx = 0.0f;
    float vz = 0.0f;
    SemanticClass semantic = SemanticClass::Static;
    std::string label;
};

struct AIVehicle {
    std::string id;
    std::string type; // "truck" | "sedan" | "suv"
    int lane = 0;
    float u = 0.0f;
    float baseSpeed = 8.0f;
    float speed = 8.0f;
    float accel = 0.0f;
    Vec3 pos{};
    float yaw = 0.0f;
    float vx = 0.0f;
    float vz = 0.0f;
    Vec3 halfExtents{0.96f, 0.72f, 2.18f};
    Vec3 color{0.23f, 0.51f, 0.96f};
    float hardBrakeTimer = 0.0f;
};

struct Pedestrian {
    std::string id;
    float x = 0.0f;
    float z = 0.0f;
    float vx = 0.0f;
    float vz = 0.0f;
    float ax = 0.0f;
    float az = 0.0f;
    float bx = 0.0f;
    float bz = 0.0f;
    float targetX = 0.0f;
    float targetZ = 0.0f;
    float baseSpeed = 1.3f;
    float radius = 0.42f;
    float phase = 0.0f;
    float yaw = 0.0f;
    float legAngle = 0.0f;
    float armAngle = 0.0f;
    Vec3 shirtColor{0.97f, 0.45f, 0.09f};
    Vec3 pantsColor{0.12f, 0.16f, 0.23f};
    bool obeySignal = true;
    bool jaywalking = false;
    bool yieldingToCar = false;
};

struct EgoState {
    Vec3 pos{-38.0f, 0.0f, 45.9f};
    float yaw = kPi * 0.5f;
    float speed = 8.5f;
    float accel = 0.0f;
    float steerAngle = 0.0f;
    float yawRate = 0.0f;
    float wheelSpin = 0.0f;
    float pitch = 0.0f;
    float pitchRate = 0.0f;
    float roll = 0.0f;
    float rollRate = 0.0f;
    float currentLateralOffset = -1.9f;
    float stopSignHoldTimer = 0.0f;
    float passedStopSignCooldown = 0.0f;
    std::string behaviorState = "CRUISING";
    float rssSafeDist = 8.5f;
    float minObstacleDist = 40.0f;
    bool aebActive = false;
    float collisionHoldTimer = 0.0f;
    float loopProgressU = 0.03f;
    float hx = 0.96f;
    float hy = 0.70f;
    float hz = 2.19f;
    float centerOffsetForward = 1.25f;
};

struct ManualInput {
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
};

struct PlannerCommand {
    float targetSpeed = 10.5f;
    float rssSafeDist = 8.5f;
    float minObstacleDist = 40.0f;
    bool aebActive = false;
    std::string behaviorState = "CRUISING";
    int bestCandIdx = 1;
    std::vector<std::vector<Vec3>> candidateTrajectories;
    std::vector<bool> candidateBlocked;
    std::vector<Vec3> optimalTrajectory;
    std::vector<Vec3> rssCarpetPoints;
    float rssCarpetHalfWidth = 1.08f;
};

struct ContactEvent {
    bool occurred = false;
    Vec3 point{};
    std::string otherId;
    float overlap = 0.0f;
};

struct SceneVertex {
    float px, py, pz;
    float nx, ny, nz;
    float r, g, b, a;
    float emissive; // > 0 for glowing signal lights, headlights, taillights, signs
    float semId;    // SemanticClass as float for optional debug shaders
    float pad0, pad1;
};

} // namespace cybercab
