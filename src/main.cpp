#include "app/VulkanViewer.hpp"
#include "config/SimConfig.hpp"
#include "core/GpuProfiler.hpp"
#include "core/TraceLogger.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/TrajectoryPlanner.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "world/WorldEngine.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

struct CliArgs {
    std::string configPath = "config/cybercab_sim.yaml";
    bool headless = false;
    int steps = 120;
    std::string scenario = "nominal";
    std::string tracePath;
    std::string dumpDir;
    std::string carDumpPath;
    std::string resMode;
};

CliArgs parseArgs(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) args.configPath = argv[++i];
        else if (a == "--headless") args.headless = true;
        else if (a == "--steps" && i + 1 < argc) args.steps = std::atoi(argv[++i]);
        else if (a == "--scenario" && i + 1 < argc) args.scenario = argv[++i];
        else if (a == "--trace" && i + 1 < argc) args.tracePath = argv[++i];
        else if (a == "--dump-dir" && i + 1 < argc) args.dumpDir = argv[++i];
        else if (a == "--car-dump" && i + 1 < argc) args.carDumpPath = argv[++i];
        else if (a == "--res" && i + 1 < argc) args.resMode = argv[++i];
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: cybercab_sim [--config path.yaml] [--headless] [--steps N] "
                      << "[--scenario nominal|jaywalk|occluded|hardbrake|soil_front] "
                      << "[--trace out.jsonl] [--dump-dir dir] [--car-dump out.bmp] [--res half|full]\n";
            std::exit(0);
        }
    }
    return args;
}

void applyScenario(const std::string& scenario, cybercab::SimConfig& cfg, cybercab::WorldEngine& world) {
    if (scenario == "jaywalk") {
        world.triggerJaywalkerCutIn();
    } else if (scenario == "occluded") {
        world.triggerOccludedDartOut();
    } else if (scenario == "hardbrake") {
        world.triggerLeadHardBrake();
    } else if (scenario == "soil_front") {
        cfg.cameras[0].soiled = true;
        cfg.cameras[2].soiled = true;
        cfg.cameras[3].soiled = true;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const CliArgs args = parseArgs(argc, argv);
        const std::string resolvedCfg = cybercab::resolveAssetPath(args.configPath);
        cybercab::SimConfig cfg = std::filesystem::exists(resolvedCfg)
                                ? cybercab::SimConfig::loadFromYaml(resolvedCfg)
                                : cybercab::SimConfig::defaultConfig();
        if (!args.resMode.empty()) {
            cfg.initialResMode = args.resMode;
        }

        cybercab::WorldEngine world(cfg);
        cybercab::VulkanContext ctx;
        cybercab::CameraRigRenderer rig;
        cybercab::PerceptionPipeline perception(cfg);
        cybercab::TrajectoryPlanner planner(cfg);
        cybercab::GpuProfiler profiler;
        cybercab::TraceLogger tracer;

        if (!args.tracePath.empty()) {
            if (!tracer.open(args.tracePath)) {
                std::cerr << "Warning: failed to open trace file " << args.tracePath << "\n";
            }
        }
        if (!args.dumpDir.empty()) {
            std::filesystem::create_directories(args.dumpDir);
        }

        cybercab::PerceptionFrameOutput percOut;
        cybercab::PlannerCommand planCmd;
        const float dtControl = 1.0f / cfg.controlHz;

        if (args.headless) {
            ctx.init(cfg.useVulkanValidation, nullptr);
            std::cout << "Vulkan Headless GPU: " << ctx.deviceName() << "\n";
            rig.init(ctx, cfg, world);
            perception.init(ctx, rig);
            profiler.init(ctx, 64);

            applyScenario(args.scenario, cfg, world);
            if (!args.carDumpPath.empty()) {
                rig.dumpEgoShowcaseBMP(world, args.carDumpPath);
            }

            float simTime = 0.0f;
            for (int step = 0; step < args.steps; ++step) {
