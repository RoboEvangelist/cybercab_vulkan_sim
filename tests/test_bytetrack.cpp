#include "config/SimConfig.hpp"
#include "core/Math.hpp"
#include "perception/ByteTrack.hpp"
#include "TestPerceptionHelpers.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace cybercab;
using namespace cybercab::testutil;

TEST(ByteTrackExtendedTest, RectangularHungarianRejectsCostsAboveGate) {
    // 3 tracks x 2 detections: track 2 is far from both dets and must stay -1.
    const std::vector<std::vector<float>> tallCost = {
        {1.2f, 6.0f},
        {5.5f, 1.8f},
        {9.0f, 8.5f}
    };
    const auto tallAssign = ByteTracker::solveHungarian(tallCost, 3.5f);
    ASSERT_EQ(tallAssign.size(), 3u);
    EXPECT_EQ(tallAssign[0], 0);
    EXPECT_EQ(tallAssign[1], 1);
    EXPECT_EQ(tallAssign[2], -1);

    // 2 tracks x 3 detections: track 1 prefers col 2 at cost 0.8 even though col 0 is closer to track 0.
    const std::vector<std::vector<float>> wideCost = {
        {1.5f, 4.2f, 1.4f},
        {2.0f, 9.0f, 0.8f}
    };
    const auto wideAssign = ByteTracker::solveHungarian(wideCost, 3.5f);
    ASSERT_EQ(wideAssign.size(), 2u);
    EXPECT_EQ(wideAssign[0], 0);
    EXPECT_EQ(wideAssign[1], 2);
}

TEST(ByteTrackExtendedTest, SingleFrameGhostFlickerNeverEntersActiveTracks) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    EgoState ego{};
    ego.yaw = kPi * 0.5f;

    std::vector<Detection3D> dets(1);
    std::vector<Track3D> active;

    // Frame 0: one-frame phantom detection at high confidence.
    dets[0] = makeDet3D(SemanticClass::Vehicle, {15.0f, 0.72f, 2.0f}, 0.90f);
    tracker.update(1.0f / 36.0f, ego, dets);
    tracker.collectActiveTracks(active);
    EXPECT_TRUE(active.empty()) << "Single-hit Tentative track must not appear in activeTracks()";

    // Frame 1: ghost disappears -> Tentative must be pruned and never promoted.
    dets.clear();
    tracker.update(1.0f / 36.0f, ego, dets);
    tracker.collectActiveTracks(active);
    EXPECT_TRUE(active.empty());

    // Frame 2: even if a re-appearance happens later, a new trackId is created (never kept from step 0).
    dets.push_back(makeDet3D(SemanticClass::Vehicle, {15.1f, 0.72f, 2.0f}, 0.90f));
    tracker.update(1.0f / 36.0f, ego, dets);
    tracker.collectActiveTracks(active);
    EXPECT_TRUE(active.empty());
}

TEST(ByteTrackExtendedTest, OverlappingCrossCameraDetectionsDoNotSpawnDuplicateTracks) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = kPi * 0.5f;

    std::vector<Detection3D> dets(2);
    std::vector<Track3D> active;
    const float dt = 1.0f / 36.0f;

    // Two cameras (e.g., CAM_FRONT_MAIN & CAM_FRONT_WIDE) report the SAME vehicle 1.4m apart.
    for (int f = 0; f < 10; ++f) {
        const float x = 18.0f + static_cast<float>(f) * 6.0f * dt;
        dets[0] = makeDet3D(SemanticClass::Vehicle, {x, 0.72f, 0.0f}, 0.91f, 1.0f, 0.25f, 0);
        dets[1] = makeDet3D(SemanticClass::Vehicle, {x + 1.35f, 0.72f, 0.25f}, 0.86f, 1.0f, 0.35f, 2);
        tracker.update(dt, ego, dets);
    }

    tracker.collectActiveTracks(active);
    EXPECT_EQ(active.size(), 1u)
        << "Overlapping multi-camera detections of one car must not spawn " << active.size() << " tracks";
}

TEST(ByteTrackExtendedTest, StationaryJitterDoesNotExplodeVelocityOrSpawnMarchingClones) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    EgoState ego{};
    ego.pos = {-38.0f, 0.0f, 45.9f};
    ego.yaw = kPi * 0.5f;

    std::vector<Detection3D> dets(1);
    std::vector<Track3D> active;
    const float dt = 1.0f / 36.0f;

    // Stationary obstacle with +/-0.35m 3D range lifting jitter at 36 Hz.
    for (int f = 0; f < 30; ++f) {
        const float jitterX = (f % 2 == 0) ? 0.35f : -0.35f;
        const float jitterZ = (f % 3 == 0) ? 0.15f : -0.15f;
        dets[0] = makeDet3D(SemanticClass::Vehicle, {-20.0f + jitterX, 0.72f, 49.0f + jitterZ}, 0.78f);
        tracker.update(dt, ego, dets);
    }

    tracker.collectActiveTracks(active);
    ASSERT_EQ(active.size(), 1u) << "Stationary jitter spawned " << active.size() << " duplicate tracks";
    const float estSpeed = std::hypot(active[0].worldVel.x, active[0].worldVel.z);
    EXPECT_LT(estSpeed, 1.2f) << "Kalman velocity exploded on stationary jitter: speed=" << estSpeed << " m/s";
    EXPECT_LT(std::hypot(active[0].worldPos.x - (-20.0f), active[0].worldPos.z - 49.0f), 0.75f);
}

TEST(ByteTrackExtendedTest, Stage2LowConfidenceMatchEnforcesSemanticConsistency) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    EgoState ego{};
    ego.yaw = kPi * 0.5f;

    std::vector<Detection3D> dets(1);
    std::vector<Track3D> active;
    const float dt = 1.0f / 36.0f;

    // Establish confirmed Vehicle track at (20, 0).
    dets[0] = makeDet3D(SemanticClass::Vehicle, {20.0f, 0.72f, 0.0f}, 0.90f);
    tracker.update(dt, ego, dets);
    tracker.update(dt, ego, dets);
    tracker.collectActiveTracks(active);
    ASSERT_EQ(active.size(), 1u);

    // Frame 2: Vehicle misses detection, while a low-confidence Pedestrian appears 2.6m away
    // (inside btLowMatchDistM = 4.5m). Stage-2 must not match Pedestrian det to a Vehicle track.
    dets[0] = makeDet3D(SemanticClass::Pedestrian, {22.6f, 0.90f, 0.0f}, 0.30f, 0.85f, 0.40f);
    tracker.update(dt, ego, dets);
    tracker.collectActiveTracks(active);
    ASSERT_EQ(active.size(), 1u);
    EXPECT_FALSE(active[0].matchedLowConf)
        << "Stage-2 matched low-conf Pedestrian detection onto a Vehicle track without class gating";
    EXPECT_EQ(active[0].state, TrackState::Lost);
}

TEST(ByteTrackExtendedTest, SoiledLowWeightUpdatePreservesTrackClassAndExtents) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    const EgoState ego{};
    std::vector<Detection3D> dets(1);
    std::vector<Track3D> active;
    const float dt = 1.0f / 36.0f;

    dets[0] = makeDet3D(SemanticClass::Vehicle, {0.0f, 0.72f, 15.0f}, 0.90f, 1.0f, 0.2f);
    tracker.update(dt, ego, dets);
    tracker.update(dt, ego, dets);

    // Soiled camera (camWeight <= 0.45f) misclassifies same track as Pedestrian.
    dets[0] = makeDet3D(SemanticClass::Pedestrian, {0.1f, 0.90f, 15.1f}, 0.35f, 0.15f, 1.2f);
    tracker.update(dt, ego, dets);

    tracker.collectActiveTracks(active);
    ASSERT_EQ(active.size(), 1u);
    EXPECT_EQ(active[0].semantic, SemanticClass::Vehicle);
    EXPECT_GT(active[0].halfExtents.z, 1.5f);
}

TEST(ByteTrackExtendedTest, ConstantVelocityTargetConvergesSpeedHeadingAndUncertainty) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    const EgoState ego{};
    std::vector<Detection3D> dets(1);
    std::vector<Track3D> active;
    const float dt = 1.0f / 36.0f;
    const float vxTrue = 8.0f;
    const float vzTrue = 2.0f;

    for (int f = 0; f < 40; ++f) {
        const float t = static_cast<float>(f) * dt;
        dets[0] = makeDet3D(SemanticClass::Vehicle, {vxTrue * t, 0.72f, 12.0f + vzTrue * t}, 0.92f, 1.0f, 0.25f);
        tracker.update(dt, ego, dets);
    }

    tracker.collectActiveTracks(active);
    ASSERT_EQ(active.size(), 1u);
    EXPECT_EQ(active[0].state, TrackState::Tracked);
    EXPECT_NEAR(active[0].worldVel.x, vxTrue, 0.65f);
    EXPECT_NEAR(active[0].worldVel.z, vzTrue, 0.65f);
    const float expectedYaw = std::atan2(vxTrue, vzTrue);
    EXPECT_NEAR(wrapAngle(active[0].yaw - expectedYaw), 0.0f, 0.10f);
    EXPECT_LT(active[0].posSigmaM, 0.75f);
}

TEST(ByteTrackExtendedTest, SignalSignIgnoredAndCapacityBoundRespected) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    const EgoState ego{};
    const float dt = 1.0f / 36.0f;

    // 1. SignalSign detections must never spawn tracks.
    std::vector<Detection3D> dets;
    dets.push_back(makeDet3D(SemanticClass::SignalSign, {3.0f, 5.0f, 20.0f}, 0.95f));
    tracker.update(dt, ego, dets);
    tracker.update(dt, ego, dets);
    EXPECT_TRUE(tracker.activeTracks().empty());

    // 2. Fill tracker up to btMaxTracks (64), confirm them, then supply det #65.
    dets.clear();
    for (int i = 0; i < cfg.btMaxTracks; ++i) {
        dets.push_back(makeDet3D(SemanticClass::Vehicle, {static_cast<float>(i) * 8.0f, 0.72f, 20.0f}, 0.90f));
    }
    tracker.update(dt, ego, dets);
    dets.push_back(makeDet3D(SemanticClass::Vehicle, {600.0f, 0.72f, 20.0f}, 0.95f));
    tracker.update(dt, ego, dets);

    std::vector<Track3D> active;
    tracker.collectActiveTracks(active);
    EXPECT_EQ(static_cast<int>(active.size()), cfg.btMaxTracks);

    std::vector<TrackPriorGpu> priors;
    tracker.exportTrackPriorsGpu(ego, priors);
    EXPECT_EQ(static_cast<int>(priors.size()), cfg.btMaxTracks);
}

TEST(ByteTrackExtendedTest, EgoFrameAndGpuPriorAgreeAcrossArbitraryYaws) {
    const SimConfig cfg = SimConfig::defaultConfig();
    const float testYaws[] = {0.0f, kPi * 0.25f, kPi * 0.5f, -kPi * 0.75f, kPi};
    std::vector<Detection3D> dets(1);
    std::vector<Track3D> active;
    std::vector<TrackPriorGpu> priors;
    const float dt = 1.0f / 36.0f;

    for (const float yaw : testYaws) {
        ByteTracker tracker(cfg);
        EgoState ego{};
        ego.pos = {-12.0f, 0.0f, 28.0f};
        ego.yaw = yaw;

        // Place target strictly 14m ahead and 3.5m left of ego in world space.
        const float fwdX = std::sin(yaw), fwdZ = std::cos(yaw);
        const float leftX = std::cos(yaw), leftZ = -std::sin(yaw);
        const Vec3 targetW{ego.pos.x + 14.0f * fwdX + 3.5f * leftX, 0.72f, ego.pos.z + 14.0f * fwdZ + 3.5f * leftZ};

        dets[0] = makeDet3D(SemanticClass::Vehicle, targetW, 0.92f);
        tracker.update(dt, ego, dets);
        tracker.update(dt, ego, dets);

        tracker.collectActiveTracks(active);
        tracker.exportTrackPriorsGpu(ego, priors);
        ASSERT_EQ(active.size(), 1u);
        EXPECT_NEAR(active[0].egoPos.x, 14.0f, 0.1f);
        EXPECT_NEAR(active[0].egoPos.y, 3.5f, 0.1f);
        EXPECT_NEAR(priors[0].posVel.x, active[0].egoPos.x, 1e-4f);
        EXPECT_NEAR(priors[0].posVel.y, active[0].egoPos.y, 1e-4f);
    }
}

TEST(ByteTrackExtendedTest, CrossClassMonocularRayJumpSuppressesCollinearClone) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = 0.0f;
    const float dt = 1.0f / 36.0f;

    // 1. Track confirmed pedestrian along viewing ray (0.6, 0.8) at 5.7m range.
    const Vec3 ray{0.6f, 0.0f, 0.8f};
    const Detection3D pedDet = makeDet3D(SemanticClass::Pedestrian, ray * 5.7f + Vec3{0.0f, 0.90f, 0.0f}, 0.95f);
    for (int i = 0; i < 3; ++i) {
        tracker.update(dt, ego, {pedDet});
    }
    ASSERT_EQ(tracker.activeTracks().size(), 1u);

    // 2. Same target misclassified as Vehicle for two consecutive ticks lifts further out along exact same ray (9.8m, dRad=4.1m, dLat=0).
    const Detection3D rayJumpVeh = makeDet3D(SemanticClass::Vehicle, ray * 9.8f + Vec3{0.0f, 0.72f, 0.0f}, 0.95f);
    tracker.update(dt, ego, {pedDet, rayJumpVeh});
    tracker.update(dt, ego, {pedDet, rayJumpVeh});
    EXPECT_EQ(tracker.activeTracks().size(), 1u)
        << "Monocular class/planeY jump along same ray spawned duplicate collinear track!";

    // 3. A genuinely separate vehicle far behind along the same ray (18.0m, dRad=12.3m) MUST still spawn & confirm its own track.
    const Detection3D farVeh = makeDet3D(SemanticClass::Vehicle, ray * 18.0f + Vec3{0.0f, 0.72f, 0.0f}, 0.95f);
    tracker.update(dt, ego, {pedDet, farVeh});
    tracker.update(dt, ego, {pedDet, farVeh});
    EXPECT_EQ(tracker.activeTracks().size(), 2u);
}

TEST(ByteTrackExtendedTest, UnmatchedLostPedestrianCannotRocketBeyondPhysicalSpeedLimit) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    EgoState ego{};
    ego.pos = {0.0f, 0.0f, 0.0f};
    ego.yaw = 0.0f;
    const float dt = 1.0f / 36.0f;

    // 12 frames where radial monocular depth creep shifts +0.25m/tick (9.0 m/s spurious velocity)
    float lastX = 0.0f;
    for (int i = 0; i < 12; ++i) {
        lastX = static_cast<float>(i) * 0.25f;
        const auto d = makeDet3D(SemanticClass::Pedestrian, {lastX, 0.90f, 12.0f}, 0.85f);
        tracker.update(dt, ego, {d});
    }
    const auto initialTracks = tracker.activeTracks();
    ASSERT_EQ(initialTracks.size(), 1u);
    EXPECT_LE(std::hypot(initialTracks[0].worldVel.x, initialTracks[0].worldVel.z), 3.0f)
        << "Pedestrian velocity spiked to unphysical " << std::hypot(initialTracks[0].worldVel.x, initialTracks[0].worldVel.z) << " m/s";

    // Coast Lost for 1.2s (within maxTimeLostSec = 1.5s)
    for (int i = 0; i < 43; ++i) {
        tracker.update(dt, ego, {});
    }
    const auto lostTracks = tracker.activeTracks();
    ASSERT_EQ(lostTracks.size(), 1u);
    const float coastDrift = std::hypot(lostTracks[0].worldPos.x - lastX, lostTracks[0].worldPos.z - 12.0f);
    EXPECT_LT(coastDrift, 2.5f) << "Lost pedestrian rocketed " << coastDrift << "m away from last observation";
}

TEST(ByteTrackExtendedTest, TransientLowHitTrackExpiresFasterThanLongLivedTrack) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    const EgoState ego{};
    const float dt = 1.0f / 36.0f;

    const auto longDet = makeDet3D(SemanticClass::Vehicle, {0.0f, 0.72f, 20.0f}, 0.95f);
    const auto transientDet = makeDet3D(SemanticClass::Vehicle, {15.0f, 0.72f, 25.0f}, 0.85f);

    // Build established Track A (8 hits) and brief transient Track B (2 hits)
    for (int i = 0; i < 6; ++i) {
        tracker.update(dt, ego, {longDet});
    }
    tracker.update(dt, ego, {longDet, transientDet});
    tracker.update(dt, ego, {longDet, transientDet});
    ASSERT_EQ(tracker.activeTracks().size(), 2u);

    // At 0.20s unobserved, both remain alive for short-occlusion Stage-2 tolerance
    for (int i = 0; i < 7; ++i) tracker.update(dt, ego, {});
    ASSERT_EQ(tracker.activeTracks().size(), 2u);

    // By 0.55s unobserved, the 2-hit transient glitch expires while 8-hit established track persists
    for (int i = 0; i < 13; ++i) tracker.update(dt, ego, {});
    const auto remaining = tracker.activeTracks();
    ASSERT_EQ(remaining.size(), 1u) << "Transient 2-hit track lingered too long alongside established track";
    EXPECT_LT(std::abs(remaining[0].worldPos.x - 0.0f), 1.0f);

    // After full 1.5s, established track also expires cleanly
    for (int i = 0; i < 45; ++i) tracker.update(dt, ego, {});
    EXPECT_TRUE(tracker.activeTracks().empty());
}
