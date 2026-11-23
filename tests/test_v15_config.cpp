// FSD v15 step 1: pipeline_version config switch (v14 default, v15 opt-in, invalid rejected).
#include "config/SimConfig.hpp"
#include "core/VulkanPipeline.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cybercab {
namespace {

// Copies the shipped YAML into /tmp with the pipeline_version line replaced.
std::string writeYamlWithVersion(const std::string& version, const std::string& tag) {
    std::ifstream in(resolveAssetPath("config/cybercab_sim.yaml"));
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    const std::string key = "pipeline_version:";
    const size_t pos = text.find(key);
    EXPECT_NE(pos, std::string::npos) << "Shipped YAML must declare simulation.pipeline_version";
    const size_t eol = text.find('\n', pos);
    text.replace(pos, eol - pos, key + " \"" + version + "\"");
    const auto path = std::filesystem::temp_directory_path() / ("cybercab_v15_cfg_" + tag + ".yaml");
    std::ofstream(path) << text;
    return path.string();
}

TEST(V15ConfigTest, ShippedYamlDefaultsToV14) {
    const SimConfig cfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    EXPECT_EQ(cfg.pipelineVersion, "v14");
    EXPECT_FALSE(cfg.isV15());
    EXPECT_FALSE(SimConfig::defaultConfig().isV15()) << "Code default must stay v14 for regression safety";
}

TEST(V15ConfigTest, YamlCanSelectV15) {
    const std::string path = writeYamlWithVersion("v15", "ok");
    const SimConfig cfg = SimConfig::loadFromYaml(path);
    EXPECT_EQ(cfg.pipelineVersion, "v15");
    EXPECT_TRUE(cfg.isV15());
    EXPECT_EQ(cfg.cameras.size(), 8u) << "Switching version must not drop other config sections";
    std::filesystem::remove(path);
}

TEST(V15ConfigTest, UnknownVersionIsRejected) {
    const std::string path = writeYamlWithVersion("v99", "bad");
    EXPECT_THROW(SimConfig::loadFromYaml(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(V15ConfigTest, PolicyYamlLoadsOverridesAndRejectsInvalidBounds) {
    const SimConfig simCfg = SimConfig::loadFromYaml(resolveAssetPath("config/cybercab_sim.yaml"));
    EXPECT_FLOAT_EQ(simCfg.v15Policy.laneCenterAbsM, 1.95f);
    EXPECT_FLOAT_EQ(simCfg.v15Policy.crossingDevThreshM, 0.28f);
    EXPECT_FLOAT_EQ(simCfg.v15Policy.blindSpotBlockCost, 2600.0f);
    EXPECT_FLOAT_EQ(simCfg.v15Policy.intersectionLaneLockCost, 8500.0f);

    auto writePol = [&](const std::string& yaml, const std::string& tag) {
        const auto p = std::filesystem::temp_directory_path() / ("v15_pol_" + tag + ".yaml");
        std::ofstream(p) << yaml;
        return p.string();
    };
    const std::string customP = writePol(
        "lane_centering:\n  lane_center_abs_m: 2.05\nsafety_shield:\n  min_hard_aeb_dist_m: 2.25\n", "ok");
    const V15PolicyConfig ov = V15PolicyConfig::loadFromYaml(customP);
    EXPECT_FLOAT_EQ(ov.laneCenterAbsM, 2.05f);
    EXPECT_FLOAT_EQ(ov.minHardAebDistM, 2.25f);
    std::filesystem::remove(customP);

    for (const auto& [badYaml, tag] : std::vector<std::pair<std::string, std::string>>{
             {"lane_centering:\n  min_lane_abs_m: 2.50\n  max_lane_abs_m: 2.25\n", "lane"},
             {"latent_predictor:\n  surprise_scale: 0.0\n", "scale"},
             {"safety_shield:\n  aeb_rss_frac: 1.50\n", "aeb"}}) {
        const std::string badP = writePol(badYaml, tag);
        EXPECT_THROW(V15PolicyConfig::loadFromYaml(badP), std::runtime_error);
        std::filesystem::remove(badP);
    }
}

} // namespace
} // namespace cybercab
