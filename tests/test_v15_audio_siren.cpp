#include "core/VulkanPipeline.hpp"
#include "perception/v15/AudioTokenizer.hpp"
#include "world/WorldEngine.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace cybercab {
namespace {
using namespace v15;

std::string fmtDeg(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << v;
    return oss.str();
}

TEST(V15AudioSirenTest, InactiveSirenProducesZeroFalseAlarms) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    WorldEngine world(cfg);
    ASSERT_FALSE(world.siren().active);
    for (uint32_t f = 0; f < 5; ++f) {
        const AudioToken tok = AudioTokenizer::stepFromWorld(world, f);
        EXPECT_FALSE(tok.sirenDetected);
        EXPECT_LT(tok.confidence, 0.25f);
    }
}

TEST(V15AudioSirenTest, FourQuadrantBearingsAndDopplerShiftAt48kHz) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    WorldEngine world(cfg);
    world.mutableEgo().speed = 0.0f;
    const Mat4 egoFromWorld = world.getEgoWorldMatrix().inverseRigid();

    // Directions in vehicle frame (+X left, +Z fwd): ahead-left, ahead-right, rear-left, rear-right
    const float bearingsDeg[4] = {35.0f, -45.0f, 135.0f, -140.0f};
    double maxErrDeg = 0.0;
    for (int i = 0; i < 4; ++i) {
        const float th = degToRad(bearingsDeg[i]);
        const Vec3 dirVeh{std::sin(th), 0.0f, std::cos(th)};
        world.mutableSiren() = {true, world.getEgoWorldMatrix().transformPoint(dirVeh * 28.0f),
                                Vec3{}, 0.25f * i, 780.0f};
        const AudioToken tok = AudioTokenizer::stepFromWorld(world, static_cast<uint32_t>(i + 10));
        ASSERT_TRUE(tok.sirenDetected);
        const Vec3 relVeh = egoFromWorld.transformPoint(world.siren().pos);
        const float trueBrg = std::atan2(relVeh.x, relVeh.z);
        const double errDeg = std::abs(radToDeg(wrapAngle(tok.bearingVehRad - trueBrg)));
        maxErrDeg = std::max(maxErrDeg, errDeg);
        EXPECT_LT(errDeg, 8.0);
    }
    RecordProperty("quadrant_max_err_deg", fmtDeg(maxErrDeg));

    // Approach vs recession Doppler: emitter at +30m forward moving toward (+v) vs away (-v).
    const Vec3 fwdW = world.getEgoWorldMatrix().transformVector({0.0f, 0.0f, 1.0f});
    world.mutableSiren() = {true, world.ego().pos + fwdW * 30.0f + Vec3{0.0f, 1.2f, 0.0f},
                            fwdW * -15.0f, 0.2f, 780.0f};
    const AudioToken approaching = AudioTokenizer::stepFromWorld(world, 30);
    world.mutableSiren().vel = fwdW * 15.0f;
    const AudioToken receding = AudioTokenizer::stepFromWorld(world, 31);
    RecordProperty("doppler_approach_hz", fmtDeg(approaching.dopplerShiftHz));
    RecordProperty("doppler_recede_hz", fmtDeg(receding.dopplerShiftHz));
    EXPECT_GT(approaching.dopplerShiftHz, 20.0f);
    EXPECT_LT(receding.dopplerShiftHz, -20.0f);
    EXPECT_GT(approaching.peakFreqHz, receding.peakFreqHz + 40.0f);
}

TEST(V15AudioSirenTest, BlindBuildingCornerSirenScenarioTrackedAndClearedOnReset) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    WorldEngine world(cfg);
    world.triggerEmergencySiren();
    ASSERT_TRUE(world.siren().active);

    AudioToken tok{};
    for (int s = 0; s < 10; ++s) {
        tok = AudioTokenizer::stepFromWorld(world, static_cast<uint32_t>(s));
        world.stepAgents(1.0f / cfg.controlHz);
    }
    ASSERT_TRUE(tok.sirenDetected);
    EXPECT_GT(tok.confidence, 0.60f);
    EXPECT_GT(tok.dopplerShiftHz, 10.0f);

    const Vec3 relVeh = world.getEgoWorldMatrix().inverseRigid().transformPoint(world.siren().pos);
    const double errDeg = std::abs(radToDeg(wrapAngle(tok.bearingVehRad - std::atan2(relVeh.x, relVeh.z))));
    RecordProperty("blind_corner_err_deg", fmtDeg(errDeg));
    EXPECT_LT(errDeg, 8.0);

    world.reset();
    EXPECT_FALSE(world.siren().active);
}

} // namespace
} // namespace cybercab
