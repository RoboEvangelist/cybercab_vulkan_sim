#pragma once

#include "config/SimConfig.hpp"
#include "world/RoadCurve.hpp"
#include "world/WorldTypes.hpp"
#include <array>
#include <vector>

namespace cybercab {

struct SatHit {
    bool hit = false;
    float overlap = 0.0f;
    Vec2 normal{}; // Points from B toward A to push A out of B
};

struct EmergencySirenState {
    bool active = false;
    Vec3 pos{};
    Vec3 vel{};
    float phase = 0.0f;
    float baseFreqHz = 780.0f;
};

struct UnknownObstacleSpec {
  static constexpr float kDefaultHalfX = 0.75f;
  static constexpr float kDefaultHalfY = 0.55f;
  static constexpr float kDefaultHalfZ = 0.75f;
  static constexpr float kDefaultColR = 0.84f;
  static constexpr float kDefaultColG = 0.48f;
  static constexpr float kDefaultColB = 0.12f;

  std::string id;
  float cx = 0.0f, cy = kDefaultHalfY, cz = 0.0f;
  float hx = kDefaultHalfX, hy = kDefaultHalfY, hz = kDefaultHalfZ;
  float yaw = 0.0f;
  Vec3 color{kDefaultColR, kDefaultColG, kDefaultColB};
  bool hazardStripe = true;
};

class WorldEngine {
public:
    static constexpr float kLoopHalfX = 56.0f;
    static constexpr float kLoopHalfZ = 44.0f;
    static constexpr float kStopLineX = -10.5f;

    explicit WorldEngine(const SimConfig& cfg);

    void reset();
    void stepAgents(float dt);
    void stepEgoAndPhysics(float dt, const PlannerCommand& cmd, const ManualInput& manual);

    // Scenarios matching HTML simulation
    void triggerJaywalkerCutIn();
    void triggerOccludedDartOut();
    void triggerLeadHardBrake();
    void triggerEmergencySiren();
    void setForcedRedSignal(bool forcedRed);
  void setSplitSignalPhase(SignalPhase lane0Phase, SignalPhase lane1Phase);
  void clearSplitSignalPhase();
  bool hasSplitSignalPhase() const { return splitSignalActive_; }
  SignalPhase signalPhaseForLane(int lane) const;
  void addUnknownRoadObstacle(const UnknownObstacleSpec& obs);
  void clearUnknownRoadObstacles();
  const std::vector<UnknownObstacleSpec>& unknownObstacles() const {
    return unknownObstacles_;
  }

    // Deterministic SAT & Capsule geometry queries (also tested directly)
    static SatHit checkSAT_OBB(const OBB& a, const OBB& b);
    static bool resolvePedestrianVsVehicleOBB(Pedestrian& ped, const OBB& vehObb, EgoState* egoIfVehicle);

    OBB getEgoWorldOBB() const;
    Mat4 getEgoWorldMatrix() const;
    Mat4 getEgoSprungWorldMatrix() const;
    std::vector<OBB> allGroundTruthOBBs() const;
    std::vector<OBB> dynamicGroundTruthOBBs() const;
    void collectDynamicGroundTruthOBBs(std::vector<OBB>& out) const;

    const ClosedCatmullRomCurve3& centerline() const { return centerline_; }
    const ClosedCatmullRomCurve3& laneCurve(int lane) const { return laneCurves_[lane & 1]; }

    const EgoState& ego() const { return ego_; }
    EgoState& mutableEgo() { return ego_; }

    const std::vector<AIVehicle>& vehicles() const { return vehicles_; }
    std::vector<AIVehicle>& mutableVehicles() { return vehicles_; }

    const std::vector<Pedestrian>& pedestrians() const { return pedestrians_; }
    std::vector<Pedestrian>& mutablePedestrians() { return pedestrians_; }

    const std::vector<OBB>& staticOBBs() const { return staticOBBs_; }
    SignalPhase signalPhase() const { return signalPhase_; }
    bool forcedRedSignal() const { return forcedRed_; }
    const ContactEvent& lastContact() const { return lastContact_; }
    const EmergencySirenState& siren() const { return siren_; }
    EmergencySirenState& mutableSiren() { return siren_; }
    float contactFlashAlpha() const { return contactFlashAlpha_; }
    Vec3 contactFlashPos() const { return contactFlashPos_; }
    float findClosestLoopU(const Vec3& pos) const;

    // Geometry mesh generators for Vulkan rendering
    void buildStaticWorldGeometry(std::vector<SceneVertex>& verts, std::vector<uint32_t>& indices) const;
    void appendDynamicGeometry(std::vector<SceneVertex>& verts, std::vector<uint32_t>& indices,
                               bool fsdWireMode, bool includeEgo = true) const;

private:
    const SimConfig& cfg_;
    ClosedCatmullRomCurve3 centerline_;
    std::array<ClosedCatmullRomCurve3, 2> laneCurves_;

    EgoState ego_{};
    std::vector<AIVehicle> vehicles_;
    std::vector<Pedestrian> pedestrians_;
    std::vector<OBB> staticOBBs_;

    SignalPhase signalPhase_ = SignalPhase::Green;
    float signalTimer_ = 0.0f;
    bool forcedRed_ = false;
  bool splitSignalActive_ = false;
  std::array<SignalPhase, 2> splitSignalPhases_{SignalPhase::Green,
                                                SignalPhase::Green};
  std::vector<UnknownObstacleSpec> unknownObstacles_;

    ContactEvent lastContact_{};
    float contactFlashAlpha_ = 0.0f;
    Vec3 contactFlashPos_{};
    EmergencySirenState siren_{};
    std::vector<OBB> allCarsScratch_;

    void initStaticOBBs();
    void resetTrafficAndPedestrians();
    void resetEgo();
    void stepMultiBodyCollisionSolver();
};

} // namespace cybercab
