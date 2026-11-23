#include "app/VulkanViewer.hpp"
#include "config/SimConfig.hpp"
#include "core/GpuProfiler.hpp"
#include "core/TraceLogger.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "planning/SimulationSync.hpp"
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
                      << "[--scenario nominal|jaywalk|occluded|hardbrake|soil_front|siren] "
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
    } else if (scenario == "siren") {
        world.triggerEmergencySiren();
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

            cybercab::SimulationSyncClock clock;
            for (int step = 0; step < args.steps; ++step) {
                clock.stepFrame(dtControl, dtControl, dtControl * 2.0f,
                    [&](float simTime, float percDt) {
                        rig.renderAllCameras(world, cfg, simTime, &profiler);
                        perception.runFrame(ctx, rig, world, percDt, cfg.occEnableShadows, percOut, &profiler);
                    },
                    [&](int tickIdx, float simTime, float tickDt) {
                        planner.plan(world, percOut, planCmd);
                        if (tracer.isOpen()) {
                            tracer.logFrame(tickIdx, simTime, cfg, world.ego(), planCmd, percOut, world.lastContact());
                        }
                        if (!args.dumpDir.empty() && (tickIdx == 0 || tickIdx == args.steps / 2 || tickIdx == args.steps - 1)) {
                            char ppmName[256];
                            std::snprintf(ppmName, sizeof(ppmName), "%s/cam_grid_step_%04d.ppm", args.dumpDir.c_str(), tickIdx);
                            rig.dumpCameraGridPPM(ppmName);
                        }
                        world.stepAgents(tickDt);
                        world.stepEgoAndPhysics(tickDt, planCmd, cybercab::ManualInput{});
                    });

                if (step % 20 == 0 || step == args.steps - 1) {
                    double totGpuMs = 0.0;
                    for (const auto& kv : percOut.gpuStageMs) totGpuMs += kv.second;
                    std::cout << std::fixed << std::setprecision(2)
                              << "[step " << std::setw(3) << step << "] state=" << planCmd.behaviorState
                              << " v=" << world.ego().speed << "m/s"
                              << " rss=" << planCmd.rssSafeDist << "m"
                              << " fwdSig=" << percOut.forwardEpistemicSigma
                              << " det3D=" << percOut.detections3D.size()
                              << " trk=" << percOut.tracks.size()
                              << " vox=" << percOut.activeVoxels.size()
                              << " shad=" << percOut.shadowVoxels.size()
                              << " gpu=" << totGpuMs << "ms\n";
                }
            }

            profiler.destroy();
            perception.destroy();
            rig.destroy();
            ctx.cleanup();
            return 0;
        }

        // Interactive SDL2 + Vulkan Window Mode
        cybercab::VulkanViewer viewer;
        viewer.init(cfg, world, rig, perception, ctx);
        profiler.init(ctx, 64);
        std::cout << "Vulkan Interactive GPU: " << ctx.deviceName() << "\n";

        applyScenario(args.scenario, cfg, world);

        cybercab::ViewerToggleState toggles;
        toggles.showShadows = cfg.occEnableShadows;
        if (cfg.initialViewMode == "fsd") toggles.viewMode = cybercab::ViewMode::FsdVoxel;
        else if (cfg.initialViewMode == "bev") toggles.viewMode = cybercab::ViewMode::BevTactical;

        cybercab::ManualInput manual;
        std::string toastMsg = "Vulkan 8-Cam Perception Ready (" + ctx.deviceName() + ")";
        auto lastTime = std::chrono::steady_clock::now();
        cybercab::SimulationSyncClock clock;

        while (viewer.pollEvents(cfg, world, rig, perception, manual, toggles, toastMsg)) {
            const auto now = std::chrono::steady_clock::now();
            const float rawDt = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;

            clock.stepFrame(rawDt, dtControl, 0.05f,
                [&](float simTime, float percDt) {
                    rig.renderAllCameras(world, cfg, simTime, &profiler);
                    perception.runFrame(ctx, rig, world, percDt, toggles.showShadows, percOut, &profiler);
                },
                [&](int tickIdx, float simTime, float tickDt) {
                    planner.plan(world, percOut, planCmd);
                    if (tracer.isOpen()) {
                        tracer.logFrame(tickIdx, simTime, cfg, world.ego(), planCmd, percOut, world.lastContact());
                    }
                    world.stepAgents(tickDt);
                    world.stepEgoAndPhysics(tickDt, planCmd, manual);
                });

            viewer.renderFrame(cfg, world, rig, percOut, planCmd, toggles, toastMsg);
        }

        profiler.destroy();
        perception.destroy();
        rig.destroy();
        viewer.destroy();
        ctx.cleanup();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
