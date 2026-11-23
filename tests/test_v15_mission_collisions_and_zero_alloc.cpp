#include "MetricsEvalHelpers.hpp"
#include "app/V15HudOverlay.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/v15/GaussianWorldModel.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "planning/v15/AutoregressivePolicy.hpp"
#include "planning/v15/GaussianRolloutScorer.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

static thread_local bool gTrackHotAllocs = false;
static thread_local size_t gHotAllocCount = 0u;
struct HotAllocScope {
  HotAllocScope() { gHotAllocCount = 0u; gTrackHotAllocs = true; }
  ~HotAllocScope() { gTrackHotAllocs = false; }
};
void* operator new(std::size_t sz) {
  if (gTrackHotAllocs) ++gHotAllocCount;
  void* p = std::malloc(sz == 0u ? 1u : sz);
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t sz) { return ::operator new(sz); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace cybercab {
namespace {
using V15MissionAndZeroAllocTest = testutil::SharedGpuMetricsTest;
constexpr int kNomSteps = 180, kHazSteps = 60, kCutSteps = 90, kHotIters = 50;
constexpr int kVehTrackId = 1, kVehHits = 4, kDriveGear = 1;
constexpr float kMinNomDistM = 30.0f, kMaxLatDevM = 0.35f, kOffMap = -350.0f;
constexpr float kMaxComfortAcc = 12.0f, kLane1ZOff = -3.90f, kCutFwdM = 18.0f;
constexpr float kCutLeadVx = 7.0f, kHotLeadX = 24.0f, kHotLeadVx = 9.0f;
constexpr float kGaussAlpha = 0.94f, kVehScore = 0.95f, kOneF = 1.0f;
constexpr float kEvalInitDistM = 40.0f, kEvalInitSpd = 8.0f;
constexpr size_t kBehavReserve = 64u, kZeroAllocs = 0u;
constexpr float kCutEgoV[] = {8.5f, 10.5f}, kCutLatV[] = {-1.8f, -3.0f};
const Vec3 kCarHe{0.96f, 0.72f, 2.18f};
const Vec4 kVehCol{0.9f, 0.3f, 0.2f, 0.95f};

SimConfig loadV15Cfg() {
  SimConfig cfg =
    SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
  cfg.pipelineVersion = "v15"; return cfg;
}

void setSingleVeh(PerceptionFrameOutput& out, const Vec3& p, const Vec3& v,
                  float yaw) {
  const auto vs = static_cast<float>(static_cast<int>(SemanticClass::Vehicle));
  Track3D t{}; t.trackId = kVehTrackId; t.semantic = SemanticClass::Vehicle;
  t.score = kVehScore; t.hits = kVehHits; t.yaw = yaw; t.halfExtents = kCarHe;
  t.worldPos = p; t.worldVel = v; out.tracks.assign(1u, t);
  out.v15Gaussians.assign(1u, {{p.x, p.y, p.z, kGaussAlpha},
    {kCarHe.x, kCarHe.y, kCarHe.z, vs}, {v.x, v.z, yaw, kOneF}, kVehCol});
}

struct MissionOut {
  bool passed = false, collision = false;
  float distM = 0.0f, maxLatDevM = 0.0f, maxAbsAcc = 0.0f; };

MissionOut runMission(VulkanContext& ctx, const std::string& mode, int steps) {
  SimConfig cfg = loadV15Cfg();
  if (mode == "night_glare_jaywalk") {
    cfg.lightingMode = "night_glare"; cfg.photonRccbMosaic = true;
  }
  WorldEngine world(cfg);
  if (mode == "jaywalk" || mode == "night_glare_jaywalk")
    world.triggerJaywalkerCutIn();
  else if (mode == "occluded") world.triggerOccludedDartOut();
  else if (mode == "hardbrake") world.triggerLeadHardBrake();
  else if (mode == "siren") world.triggerEmergencySiren();
  CameraRigRenderer rig; rig.init(ctx, cfg, world);
  PerceptionPipeline pipe(cfg); pipe.init(ctx, rig);
  TrajectoryPlanner planner(cfg); PerceptionFrameOutput out{};
  PlannerCommand cmd{}; const float dt = 1.0f / cfg.controlHz;
  bool seenSiren = false; MissionOut r{}; Vec3 prev = world.ego().pos;
  for (int s = 0; s < steps; ++s) {
    rig.renderAllCameras(world, cfg, static_cast<float>(s) * dt);
    pipe.runFrame(ctx, rig, world, dt, true, out);
    planner.plan(world, out, cmd);
    world.stepAgents(dt); world.stepEgoAndPhysics(dt, cmd, {});
    if (world.lastContact().occurred) r.collision = true;
    if (cmd.behaviorState == "YIELDING_SIREN") seenSiren = true;
    const Vec3 p = world.ego().pos;
    r.distM += std::hypot(p.x - prev.x, p.z - prev.z); prev = p;
    r.maxAbsAcc = std::max(r.maxAbsAcc, std::abs(world.ego().accel));
    if (cmd.behaviorState == "CRUISING" && cmd.gear == kDriveGear) {
      const float d = std::abs(std::abs(world.ego().currentLateralOffset) -
                               cfg.v15Policy.laneCenterAbsM);
      r.maxLatDevM = std::max(r.maxLatDevM, d);
    }
  }
  const bool nomOk = (r.distM > kMinNomDistM) && (r.maxLatDevM < kMaxLatDevM);
  r.passed = !r.collision && (r.maxAbsAcc <= kMaxComfortAcc) &&
             (mode != "nominal" || nomOk) && (mode != "siren" || seenSiren);
  pipe.destroy(); rig.destroy(); return r;
}

TEST_F(V15MissionAndZeroAllocTest, MultiMissionCompletionRateZeroCollisions) {
  const struct { const char* name; int steps; } kMissions[] = {
    {"nominal", kNomSteps}, {"jaywalk", kHazSteps}, {"occluded", kHazSteps},
    {"hardbrake", kHazSteps}, {"siren", kHazSteps},
    {"night_glare_jaywalk", kHazSteps}};
  int completed = 0, collisions = 0, total = 0; MissionOut nomOut{};
  for (const auto& ms : kMissions) {
    const MissionOut m = runMission(s_ctx, ms.name, ms.steps);
    ++total; if (m.passed) ++completed; if (m.collision) ++collisions;
    if (std::string(ms.name) == "nominal") nomOut = m;
    EXPECT_TRUE(m.passed) << "Mission failed: " << ms.name;
  }
  RecordProperty("mission_completion_rate",
                 testutil::fmtProp(static_cast<double>(completed) / total));
  RecordProperty("total_collisions", std::to_string(collisions));
  RecordProperty("nominal_distance_m", testutil::fmtProp(nomOut.distM));
  RecordProperty("nominal_max_lat_dev_m", testutil::fmtProp(nomOut.maxLatDevM));
  EXPECT_EQ(completed, total); EXPECT_EQ(collisions, 0);
}

TEST_F(V15MissionAndZeroAllocTest, Iso34502ParametricCutInAndBitDeterminism) {
  const SimConfig cfg = loadV15Cfg(); const float dt = 1.0f / cfg.controlHz;
  for (float vEgo : kCutEgoV) for (float vCutZ : kCutLatV) {
    WorldEngine world(cfg);
    for (auto& v : world.mutableVehicles()) {
      v.pos = {kOffMap, 0.0f, kOffMap}; v.speed = v.vx = v.vz = 0.0f;
    }
    for (auto& p : world.mutablePedestrians()) {
      p.x = p.ax = p.bx = p.targetX = p.z = p.az = p.bz = p.targetZ = kOffMap;
    }
    world.mutableEgo().speed = vEgo;
    const Vec3 ePos = world.ego().pos;
    float cutX = ePos.x + kCutFwdM, cutZ = ePos.z + kLane1ZOff;
    const float tgtZ = ePos.z, yaw = world.ego().yaw;
    PerceptionFrameOutput perc{}; perc.v15.active = true; PlannerCommand cmd{};
    for (int s = 0; s < kCutSteps; ++s) {
      cutX += kCutLeadVx * dt;
      const float vz = (cutZ < tgtZ) ? -vCutZ : 0.0f;
      cutZ = std::min(tgtZ, cutZ + vz * dt);
      setSingleVeh(perc, {cutX, kCarHe.y, cutZ}, {kCutLeadVx, 0.0f, vz}, yaw);
      v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
      world.stepEgoAndPhysics(dt, cmd, {});
      EXPECT_FALSE(world.lastContact().occurred);
    }
  }
  WorldEngine wDet(cfg); PerceptionFrameOutput pDet{}; pDet.v15.active = true;
  PlannerCommand a{}, b{};
  v15::AutoregressivePolicy::plan(cfg, wDet, pDet, a);
  v15::AutoregressivePolicy::plan(cfg, wDet, pDet, b);
  EXPECT_FLOAT_EQ(a.targetSpeed, b.targetSpeed);
  EXPECT_FLOAT_EQ(a.rssSafeDist, b.rssSafeDist);
  EXPECT_EQ(a.bestCandIdx, b.bestCandIdx);
}

TEST_F(V15MissionAndZeroAllocTest, SteadyStateHotPathZeroHeapAllocations) {
  const SimConfig cfg = loadV15Cfg(); WorldEngine world(cfg);
  PerceptionFrameOutput perc{}; perc.v15.active = true;
  const auto& ego = world.ego(); const auto& pol = cfg.v15Policy;
  const Vec3 lPos{ego.pos.x + kHotLeadX, kCarHe.y, ego.pos.z};
  setSingleVeh(perc, lPos, {kHotLeadVx, 0.0f, 0.0f}, ego.yaw);
  PlannerCommand cmd{}; cmd.behaviorState.reserve(kBehavReserve);
  std::string beh; beh.reserve(kBehavReserve);
  v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
  v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
  const auto corrPts = cmd.rssCarpetPoints;
  const std::vector<float> corrArc(corrPts.size(), kOneF);
  const float u0 = world.findClosestLoopU(ego.pos);
  const float lat = ego.currentLateralOffset, la = pol.minLookaheadM;
  const auto& gs = perc.v15Gaussians;
  {
    HotAllocScope guard;
    for (int i = 0; i < kHotIters; ++i) {
      v15::AutoregressivePolicy::plan(cfg, world, perc, cmd);
      (void)v15::selectComfortSubRollout(lat, lat, lat, ego,
        world.centerline(), u0, la, la, gs, pol);
      (void)v15::GaussianWorldModel::queryOccupancyProb(gs, lPos, 0.0f, true);
      float md = kEvalInitDistM, sp = kEvalInitSpd; bool ab = false;
      beh = "CRUISING";
      v15::evaluateCorridorGaussians(gs, corrPts, corrArc, cmd.rssSafeDist,
        ego, pol, md, sp, ab, beh);
      (void)v15_hud::formatV15HudTelemetry(ego, perc, cmd);
    }
  }
  RecordProperty("hot_path_alloc_count", std::to_string(gHotAllocCount));
  EXPECT_EQ(gHotAllocCount, kZeroAllocs);
}
}  // namespace
}  // namespace cybercab
