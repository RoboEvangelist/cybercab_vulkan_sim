# cybercab_vulkan_sim

A modular, zero-hot-path-allocation **C++ + Vulkan** simulation of the **Tesla Cybercab** 8-camera surround vision, GPU deep perception (v14 & opt-in FSD v15), 4D occupancy/Gaussian world modeling, uncertainty-aware planning, and SAT multi-body physics.

## Key Capabilities
- **100% C++ + GLSL Compute/Graphics (Vulkan):** No Python runtime or external heavy model downloads required; shaders compile directly to SPIR-V via `glslangValidator`.
- **Configurable Architecture Switch (`simulation.pipeline_version` in `config/cybercab_sim.yaml` or `--pipeline v14|v15`):**
  - **`v14` (default):** RT-DETR + ByteTrack + BEVFormer + 4D Occupancy/Flow + Occlusion Shadows + World Model + quintic RSS planner (`01_soil_detect`..`06_world_model`).
  - **`v15` (opt-in):** All 7 FSD v15 modules gated behind `cfg.isV15()`:
    1. **ISP-Bypass Photon Front-End:** 14-bit linear photon tensor (`photon.glsl`), Poisson-Gaussian sensor noise, log illumination/detail frequency split (`v15_freq_split.comp`).
    2. **Online Self-Calibration & Dual-Branch Virtual Rectifier:** 36-frame sliding-window Manhattan IRLS pitch/roll solver (`v15_self_calib.comp`) + 10 virtual pinhole views incl. VRU `-12 deg` and Elevated `+5 deg` (`v15_virtual_rectify.comp`).
    3. **Multimodal Tokenization:** 48 kHz quad-mic `AudioTokenizer` (Doppler + 360-deg siren bearing), 100 Hz `KinematicNavEncoder`, and 5x5 Plucker `PatchTokenizer` ring (`7,680` tokens/frame, `v15_patch_tokenize.comp`).
    4. **Self-Supervised Latent Predictor (`LatentPredictor`):** Forward-corridor novelty inflation on `forwardEpistemicSigma` plus sparse/KV-cache/FP16 compute telemetry.
    5. **Persistent SE(3) Static Spatial Memory:** Ego-warped `60x40` static map with dynamic occlusion masking & selective refresh skipping (`v15_static_memory.comp`).
    6. **3D/4D Gaussian World Model & EWA Reality View:** GaussianFormer-2 occupancy, 4D rollout, and Zwicker/Kerbl EWA `320x180` splatting (`v15_gaussian_splat.comp`).
    7. **Autoregressive Middle-of-Lane Planner (`AutoregressivePolicy` + `AutoregSafety`):** Sequential `(x, y, v, kappa)` lane-center rollout (`+/-1.95 m`), bounded truck shy-away, `YIELDING_SIREN`, and `Autoshift D/R`.
- **8-Camera Calibrated Surround Matrix:** Concurrent offscreen rendering for `CAM_FRONT_{MAIN,NARROW,WIDE}`, `CAM_PILLAR_{LEFT,RIGHT}`, `CAM_REPEATER_{LEFT,RIGHT}`, and `CAM_REAR` via [`cam_isp_soil.comp`](shaders/cam_isp_soil.comp).
- **Interactive HUD & Telemetry:** Clickable mode/layer/scenario controls, 8-camera live matrix overlays (`Neural`, `RGB`, `Calib`, `SoilHeatmap`, `VoxelReproj`), per-stage Vulkan GPU query profiling, and JSONL trace logging.

## Project Structure
```text
cybercab_vulkan_sim/
├── CMakeLists.txt                  # Modular libs: cybercab_core/world/perception/planning
├── config/cybercab_sim.yaml        # 8-camera calibration, pipeline_version (v14/v15), RSS & vehicle params
├── config/v15_policy.yaml          # Tunable v15 autoregressive rollout, OBB safety & latent predictor params
├── docs/v15/                       # Semi-formal change certificates (Steps 1–12) & index
├── shaders/                        # GLSL graphics & compute shaders (v14 + v15_*.comp + include/*.glsl)
├── src/
│   ├── core/                       # VulkanContext, Buffer, Image, Pipeline, Profiler, TraceLogger, Math
│   ├── config/                     # YAML config loaders (SimConfig, V15PolicyConfig)
│   ├── world/                      # Road splines, SAT/IDM/Helbing physics, siren state, CybercabMesh/BodyLoft
│   ├── sensors/                    # 8-camera Vulkan renderer & Brown-Conrady ISP (CameraRigRenderer)
│   ├── perception/                 # v14 stages, ByteTrack, and v15/ modules (M1–M6)
│   ├── planning/                   # v14 quintic RSS planner & v15/ autoregressive policy & OBB safety (M7)
│   ├── app/                        # Interactive SDL2 + Vulkan HUD viewer & bitmap font
│   └── main.cpp                    # Interactive / headless CLI entry point
└── tests/                          # 140 GoogleTest tests (96 v14 regression + 44 v15 unit/GPU/E2E)
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
# Run default v14
./build/cybercab_sim --config config/cybercab_sim.yaml

# Or run FSD v15 via CLI flag without altering config/cybercab_sim.yaml
./build/cybercab_sim --config config/cybercab_sim.yaml --pipeline v15
```
- **Mouse:** Drag to orbit 3D view, scroll to zoom; click HUD buttons/checkboxes or camera cards directly.
- **Keys:** `1`–`5` (camera overlays), `V` (3D Twin / FSD Voxel / BEV Tactical), `R` (320x240 / 640x480), `Space` (Auto / `WASD` manual drive), `B` (RSS/AEB shield), `Tab` / `C` (select/soil camera), `O/H/F/M/E/K/G/P` (toggle overlays), `F1`–`F6` (Jaywalk, Occluded, HardBrake, SoilFront, Reset, ForceRed).

### Headless GPU Mode & Tracing
```bash
./build/cybercab_sim --config config/cybercab_sim.yaml --pipeline v15 --headless --scenario nominal --steps 120 --trace /tmp/trace_v15.jsonl
```
Options:
- `--config <path.yaml>` (defaults to `config/cybercab_sim.yaml`; loads `config/v15_policy.yaml`)
- `--pipeline v14|v15` (overrides `simulation.pipeline_version`)
- `--scenario nominal|jaywalk|occluded|hardbrake|soil_front|siren`
- `--steps N` (control ticks at 36 Hz)
- `--res half|full` (`320x240` or `640x480` per camera)
- `--trace <out.jsonl>` (per-frame telemetry, tracks, perception metrics & GPU stage timings)
- `--dump-dir <dir>` (save 8-camera PPM mosaic frames)
- `--car-dump <out.bmp>` (save offscreen side + 3/4 Cybercab render snapshot)
