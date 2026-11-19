# cybercab_vulkan_sim

A modular, zero-hot-path-allocation **C++17 + Vulkan** simulation of the **Tesla Cybercab** 8-camera surround vision, GPU deep perception, 4D occupancy & flow, predictive world modeling, Uncertainty-Aware RSS/AEB planning, and SAT multi-body physics.

## Key Capabilities
- **100% C++17 + GLSL Compute/Graphics (Vulkan):** No Python runtime or external heavy model downloads required; shaders compile directly to SPIR-V via `glslangValidator`.
- **8-Camera Calibrated Surround Matrix:** Concurrent offscreen rendering for `CAM_FRONT_MAIN`, `CAM_FRONT_NARROW`, `CAM_FRONT_WIDE`, `CAM_PILLAR_{LEFT,RIGHT}`, `CAM_REPEATER_{LEFT,RIGHT}`, and `CAM_REAR`, with Brown-Conrady distortion/glare/mud injection in [`cam_isp_soil.comp`](shaders/cam_isp_soil.comp).
- **GPU Deep Perception Pipeline (`shaders/*.comp`):**
  1. `01_soil_detect.comp` — Patch-wise gradient/glare/soiling confidence weights `w_c(u, v)` and epistemic variance `sigma^2`.
  2. `02_backbone_fpn.comp` — Spatial-temporal S4/S5 FPN feature extraction with soil weight modulation.
  3. `03_rtdetr_encoder.comp` & `03_rtdetr_decoder.comp` — AIFI/CCFM hybrid encoder + Top-K query self/cross-attention decoder + 3D lifting.
  4. `ByteTracker` (`src/perception/ByteTrack.*`) — Two-stage Kuhn-Munkres Hungarian assignment + 4D Kalman tracking with uncertainty propagation.
  5. `04_bevformer_temporal.comp` & `04_bevformer_spatial.comp` — Ego-motion BEV feature warping + multi-camera 3D pillar spatial cross-attention.
  6. `05_occupancy_flow.comp` & `05_occlusion_shadow.comp` — 60x40x8 4D occupancy/velocity volume & ray-marched occlusion shadow casting.
  7. `06_world_model.comp` — Kinematic rollout ghosts with horizon-growing uncertainty (`t = 0.5..2.0 s`).
- **Trajectory Planning & SAT Physics:** Quintic polynomial lateral lane-change planner, uncertainty-scaled RSS stopping envelope + AEB shield, IDM vehicles, Helbing pedestrians, spring-damper pitch/roll suspension, and OBB SAT collision solver.
- **Interactive HUD & Telemetry:** Clickable mode/layer/scenario controls, 8-camera live matrix overlays (`Neural`, `RGB`, `Calib`, `SoilHeatmap`, `VoxelReproj`), per-stage Vulkan GPU query profiling, and JSONL trace logging.

## Project Structure
```text
cybercab_vulkan_sim/
├── CMakeLists.txt                  # Modular libs: cybercab_core/world/perception/planning
├── config/cybercab_sim.yaml        # 8-camera calibration, perception, RSS & vehicle parameters
├── shaders/                        # GLSL graphics & compute shaders + common.glsl
├── src/
│   ├── core/                       # VulkanContext, Buffer, Image, Pipeline, Profiler, TraceLogger, Math
│   ├── config/                     # YAML config loader (SimConfig)
│   ├── world/                      # Road splines, SAT/IDM/Helbing physics, Cybercab mesh (WorldEngine)
│   ├── sensors/                    # 8-camera Vulkan renderer & Brown-Conrady ISP (CameraRigRenderer)
│   ├── perception/                 # GPU perception stages & ByteTrack
│   ├── planning/                   # Quintic lattice + RSS/AEB safety planner (TrajectoryPlanner)
│   ├── app/                        # Interactive SDL2 + Vulkan HUD viewer & bitmap font
│   └── main.cpp                    # Interactive / headless CLI entry point
└── tests/                          # GoogleTest unit & Vulkan GPU integration tests
```

## Dependencies (Ubuntu / Debian)
```bash
sudo apt update && sudo apt install -y \
  build-essential cmake glslang-tools \
  libvulkan-dev libsdl2-dev libyaml-cpp-dev \
  libjsoncpp-dev libeigen3-dev libgtest-dev
```

## Build & Test
```bash
cmake -S . -B build
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

## Running the Simulation

### Interactive Vulkan GUI (requires display)
```bash
./build/cybercab_sim
```
- **Mouse:** Drag to orbit 3D view, scroll to zoom; click HUD buttons/checkboxes or camera cards directly.
- **Keys:** `1`–`5` (camera overlays), `V` (3D Twin / FSD Voxel / BEV Tactical), `R` (320x240 / 640x480), `Space` (Auto / `WASD` manual drive), `B` (RSS/AEB shield), `Tab` / `C` (select/soil camera), `O/H/F/M/E/K/G/P` (toggle overlays), `F1`–`F6` (Jaywalk, Occluded, HardBrake, SoilFront, Reset, ForceRed).

### Headless GPU Mode & Tracing
```bash
./build/cybercab_sim --headless --scenario nominal --steps 120 --trace build/trace_nominal.jsonl
```
Options:
- `--scenario nominal|jaywalk|occluded|hardbrake|soil_front`
- `--steps N` (control ticks at 36 Hz)
- `--res half|full` (`320x240` or `640x480` per camera)
- `--trace <out.jsonl>` (per-frame telemetry, tracks, perception metrics & GPU stage timings)
- `--dump-dir <dir>` (save 8-camera PPM mosaic frames)
- `--car-dump <out.bmp>` (save offscreen side + 3/4 Cybercab render snapshot)
