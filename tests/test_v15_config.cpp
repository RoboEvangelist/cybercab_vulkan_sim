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

} // namespace
} // namespace cybercab
