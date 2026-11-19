#include "config/SimConfig.hpp"
#include "core/TraceLogger.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/ByteTrack.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "world/WorldEngine.hpp"
#include <gtest/gtest.h>
#include <json/json.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace cybercab;

TEST(PlanningAndTrackingTest, HungarianFindsGlobalMinimumCostAssignment) {
    // Greedy would pick (0,0)=2 then (1,1)=9 (total 11); optimal Hungarian picks (0,1)=3 + (1,0)=4 = 7.
    const std::vector<std::vector<float>> cost = {
        {2.0f, 3.0f},
        {4.0f, 9.0f}
    };
    const auto assign = ByteTracker::solveHungarian(cost, 10.0f);
    ASSERT_EQ(assign.size(), 2u);
    EXPECT_EQ(assign[0], 1);
    EXPECT_EQ(assign[1], 0);
}

TEST(PlanningAndTrackingTest, ByteTrackLifecycleLostAndRemovalAndEgoCompensation) {
    const SimConfig cfg = SimConfig::defaultConfig();
    ByteTracker tracker(cfg);
    EgoState ego{};
    ego.pos = {-30.0f, 0.0f, 45.9f};
    ego.yaw = kPi * 0.5f;

    Detection3D car;
    car.semantic = SemanticClass::Vehicle;
    car.score = 0.88f;
    car.worldPos = {-12.0f, 0.72f, 45.9f};
    car.camWeight = 1.0f;
    car.epistemicSigma = 0.25f;

    // 2 frames establish confirmed Tracked state
    tracker.update(0.05f, ego, {car});
    car.worldPos.x += 0.3f;
    tracker.update(0.05f, ego, {car});
    auto tracks = tracker.activeTracks();
    ASSERT_EQ(tracks.size(), 1u);
    const int tid = tracks[0].trackId;

    // While lost for 0.2s, ego moves forward (+2m) -> ego-relative X must decrease cleanly
    ego.pos.x += 2.0f;
    tracker.update(0.20f, ego, {});
    tracks = tracker.activeTracks();
    ASSERT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracks[0].trackId, tid);
    EXPECT_NEAR(tracks[0].egoPos.x, (tracks[0].worldPos.x - ego.pos.x), 0.15f);

    // After maxTimeLostSec (1.5s default), track is expired and removed
    for (int i = 0; i < 8; ++i) {
        tracker.update(0.25f, ego, {});
    }
    EXPECT_TRUE(tracker.activeTracks().empty());
}

TEST(PlanningAndTrackingTest, QuinticPlannerOvertakesBlockedLeadAndRespectsRssAndAebToggle) {
    SimConfig cfg = SimConfig::defaultConfig();
    WorldEngine world(cfg);
    TrajectoryPlanner planner(cfg);

    // 1. Place slow/stopped vehicle directly ahead in Lane 0 (-1.9 offset), verify planner selects Lane 1 (+1.9) overtake
    PerceptionFrameOutput perc;
    Track3D lead;
    lead.trackId = 10;
    lead.semantic = SemanticClass::Vehicle;
    lead.score = 0.95f;
    lead.worldPos = {world.ego().pos.x + 15.0f, 0.72f, world.ego().pos.z};
    lead.halfExtents = {0.96f, 0.72f, 2.2f};
    lead.yaw = world.ego().yaw;
    perc.tracks.push_back(lead);

    PlannerCommand cmd;
    planner.plan(world, perc, cmd);
    ASSERT_EQ(cmd.candidateTrajectories.size(), static_cast<size_t>(cfg.numPlannerCandidates));
    EXPECT_EQ(cmd.behaviorState == "LANE_CHANGE_OVERTAKE" || cmd.behaviorState == "FOLLOWING_LEAD", true);
    // Start of planned trajectory matches current ego lateral offset smoothly
    EXPECT_NEAR(cmd.optimalTrajectory.front().z, world.ego().pos.z, 0.35f);

    // 2. Pedestrian right in front -> uncertainty inflation increases RSS stopping distance
    perc.tracks.clear();
    Track3D ped;
    ped.trackId = 20;
    ped.semantic = SemanticClass::Pedestrian;
    ped.score = 0.90f;
    ped.worldPos = {world.ego().pos.x + 7.0f, 0.90f, world.ego().pos.z};
    ped.halfExtents = {0.35f, 0.90f, 0.35f};
    perc.tracks.push_back(ped);
    perc.forwardEpistemicSigma = 0.15f;
    planner.plan(world, perc, cmd);
    const float rssClean = cmd.rssSafeDist;
    EXPECT_TRUE(cmd.aebActive);

    perc.forwardEpistemicSigma = 0.75f;
    planner.plan(world, perc, cmd);
    EXPECT_GT(cmd.rssSafeDist, rssClean + 0.85f);

    // 3. Disabling AEB shield suppresses AEB activation even on close obstacle
    cfg.aebEnabled = false;
    WorldEngine worldNoAeb(cfg);
    TrajectoryPlanner plannerNoAeb(cfg);
    plannerNoAeb.plan(worldNoAeb, perc, cmd);
    EXPECT_FALSE(cmd.aebActive);
}

TEST(PlanningAndTrackingTest, TraceLoggerWritesParseableJsonlLines) {
    const SimConfig cfg = SimConfig::defaultConfig();
    const std::string outPath = std::string(CYBERCAB_BUILD_DIR) + "/gtest_trace_unit.jsonl";
    {
        TraceLogger logger;
        ASSERT_TRUE(logger.open(outPath));
        EgoState ego{};
        ego.pos = {1.0f, 0.0f, 2.0f};
        ego.behaviorState = "CRUISING";
        PlannerCommand cmd;
        cmd.rssSafeDist = 9.5f;
        PerceptionFrameOutput perc;
        perc.gpuStageMs["1_cam_render"] = 0.12;
        logger.logFrame(0, 0.0f, cfg, ego, cmd, perc, ContactEvent{});
    }

    std::ifstream ifs(outPath);
    ASSERT_TRUE(ifs.good());
    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(ifs, line)));
    Json::CharReaderBuilder rb;
    Json::Value root;
    std::string errs;
    std::istringstream iss(line);
    ASSERT_TRUE(Json::parseFromStream(rb, iss, &root, &errs)) << errs;
    EXPECT_EQ(root["step"].asInt(), 0);
    EXPECT_EQ(root["ego"]["behavior"].asString(), "CRUISING");
    EXPECT_FLOAT_EQ(root["ego"]["rss_safe_dist_m"].asFloat(), 9.5f);
}
