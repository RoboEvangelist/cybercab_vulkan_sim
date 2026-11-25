# cybercab_vulkan_sim

A modular, zero-hot-path-allocation **C++ + Vulkan** simulation of the **Tesla Cybercab** 8-camera surround vision, GPU deep perception (v14 & opt-in FSD v15), 4D occupancy/Gaussian world modeling, uncertainty-aware planning, and SAT multi-body physics.

## Key Capabilities
- **100% C++ + GLSL Compute/Graphics (Vulkan):** No Python runtime or external heavy model downloads required; shaders compile directly to SPIR-V via `glslangValidator`.
- **Configurable Architecture Switch (`simulation.pipeline_version` in `config/cybercab_sim.yaml` or `--pipeline v14|v15`):**
  - **`v14` (default):** RT-DETR + ByteTrack + BEVFormer + 4D Occupancy/Flow + Occlusion Shadows + World Model + quintic RSS planner (`01_soil_detect`..`06_world_model`).
  - **`v15` (opt-in):** All 7 FSD v15 modules gated behind `cfg.isV15()`:
    1. **ISP-Bypass Photon Front-End & Calibration:** 14-bit linear photon tensor (`photon.glsl`), Poisson-Gaussian noise, log illumination/detail frequency split (`v15_freq_split.comp`), plus condition-aware logit-temperature calibration ([`ConfidenceCalibrator.hpp`](src/perception/ConfidenceCalibrator.hpp)).
    2. **Online Self-Calibration & Dual-Branch Virtual Rectifier:** 36-frame sliding-window Manhattan IRLS pitch/roll solver (`v15_self_calib.comp`) + 10 virtual pinhole views incl. VRU `-12 deg` and Elevated `+5 deg` (`v15_virtual_rectify.comp`).
    3. **Multimodal Tokenization & Pure-Vision Signal FSM:** 48 kHz quad-mic `AudioTokenizer` (Doppler + 360-deg siren bearing), 100 Hz `KinematicNavEncoder`, 5x5 Plucker `PatchTokenizer` ring (`7,680` tokens/frame, `v15_patch_tokenize.comp`), and pure-vision traffic light classifier/lane associator/HMM-FSM ([`SignalPerception.hpp`](src/perception/SignalPerception.hpp)).
    4. **Self-Supervised Latent Predictor (`LatentPredictor`):** Forward-corridor novelty inflation on `forwardEpistemicSigma` plus sparse/KV-cache/FP16 compute telemetry.
    5. **Persistent SE(3) Static Spatial Memory:** Ego-warped `60x40` static map with dynamic occlusion masking & selective refresh skipping (`v15_static_memory.comp`).
    6. **3D/4D Gaussian World Model & Open-Set Protection:** GaussianFormer-2 occupancy, 4D rollout, Zwicker/Kerbl EWA `320x180` splatting (`v15_gaussian_splat.comp`), and `v15` raised-cargo + grounded on-road static voxel protection ([`StaticVoxelGuard.hpp`](src/planning/v15/StaticVoxelGuard.hpp)).
    7. **Autoregressive Middle-of-Lane Planner (`AutoregressivePolicy` + `AutoregSafety`):** Sequential `(x, y, v, kappa)` lane-center rollout (`+/-1.95 m`), bounded truck shy-away, `YIELDING_SIREN`, and `Autoshift D/R`.
- **Stage 1 Spatial Representation Switch (`simulation.spatial_rep_mode` or `--spatial-rep legacy|gaussianformer3d`):**
  - **`legacy` (default):** Existing BEVFormer/OccupancyFlowNet voxel path (`v14`) and GPU `GaussianWorldModel::updateFromPerception` (`v15`).
  - **`gaussianformer3d` (opt-in, ICRA 2026 / arXiv:2505.10685):** Replaces Stage-1 spatial construction with [`GaussianFormer3DPipeline`](src/perception/sota/GaussianFormer3DPipeline.hpp) (`VoxelToGaussianInit` -> `DeformableAttention3D` -> `GaussianToVoxelSplatter`), with YAML voxel/volume tuning (`occupancy_flow.step_x/y/z` + `gaussianformer3d.splat_voxel_step_x/y/z`, `voxel_scale_x/y/z`, `splat_sigma_cutoff_sq`, `max_gaussians`, `min_voxel_occ_prob`).
  - **A/B Summary ([report](docs/v15/stage1_gaussianformer3d_ab_report.md)):** Improves dynamic voxel precision (`0.881 -> 0.919`, `0.943` at `0.5m` splat step), Gaussian GT occupancy (`0.301 -> 0.457`), open-set `BARRIER` efficiency (`707 -> 323` voxels, `0.29 -> 1.02 m` clearance), and `v15` hardbrake/glare comfort at `0.32-0.40 ms` CPU stage time (`10.20-20.41 ms` mean wall time vs `13.57-19.69 ms` legacy under `RelWithDebInfo`) and `0` steady-state heap allocs.
- **8-Camera Calibrated Surround Matrix & HDR Night Lighting:** Concurrent offscreen rendering across 4 camera presets (`half` `320x240`, `full` `640x480`, `cybercab_binned` `1448x938`, `cybercab` `2896x1876`) with `day` / `night_glare` HDR + RCCB mosaic support via [`cam_isp_soil.comp`](shaders/cam_isp_soil.comp).
- **Interactive HUD & Telemetry:** Clickable mode/layer/scenario controls, 8-camera live matrix overlays (`Neural`, `RGB`, `Calib`, `SoilHeatmap`, `VoxelReproj`), per-stage Vulkan GPU query profiling, and JSONL trace logging.

## Project Structure
```text
cybercab_vulkan_sim/
├── CMakeLists.txt                  # Modular libs: cybercab_core/world/perception/planning
├── config/cybercab_sim.yaml        # 8-camera calibration, pipeline_version (v14/v15), RSS & vehicle params
├── config/v15_policy.yaml          # Tunable v15 autoregressive rollout, OBB safety & latent predictor params
├── docs/v15/                       # Architecture docs, A/B evaluation reports & paper references
├── shaders/                        # GLSL graphics & compute shaders (v14 + v15_*.comp + include/*.glsl)
├── src/
│   ├── core/                       # VulkanContext, Buffer, Image, Pipeline, Profiler, TraceLogger, Math
│   ├── config/                     # YAML config loaders (SimConfig, V15PolicyConfig)
│   ├── world/                      # Road splines, SAT/IDM/Helbing physics, unknown obstacles, CybercabMesh
│   ├── sensors/                    # 8-camera Vulkan renderer, HDR/RCCB & Brown-Conrady ISP (CameraRigRenderer)
│   ├── perception/                 # v14 stages, ByteTrack, ConfidenceCalibrator, SignalClassifier/Filter, v15/
│   ├── planning/                   # v14 quintic RSS planner & v15/ (StaticVoxelGuard, AutoregSafety, Policy)
│   ├── app/                        # Interactive SDL2 + Vulkan HUD viewer & bitmap font
│   └── main.cpp                    # Interactive / headless CLI entry point
└── tests/                          # 202 GoogleTest tests (v14/v15/Stage-1/validation/FMEA/benchmark)
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
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
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
- **Keys:** `1`–`5` (camera overlays), `V` (3D Twin / FSD Voxel / BEV Tactical), `R` (cycle `320x240` / `640x480` / `1448x938` / `2896x1876`), `Space` (Auto / `WASD` manual drive), `B` (RSS/AEB shield), `Tab` / `C` (select/soil camera), `O/H/F/M/E/K/G/P` (toggle overlays), `F1`–`F6` (Jaywalk, Occluded, HardBrake, SoilFront, Reset, ForceRed).

### Headless GPU Mode & Tracing
```bash
./build/cybercab_sim --config config/cybercab_sim.yaml --pipeline v15 --headless --scenario nominal --steps 120 --trace /tmp/trace_v15.jsonl
```
Options:
- `--config <path.yaml>` (defaults to `config/cybercab_sim.yaml`; loads `config/v15_policy.yaml`)
- `--pipeline v14|v15` (overrides `simulation.pipeline_version`)
- `--spatial-rep legacy|gaussianformer3d` (overrides `simulation.spatial_rep_mode`)
- `--lighting day|night_glare` (selects daytime or HDR night-glare rendering)
- `--scenario nominal|jaywalk|occluded|hardbrake|soil_front|siren`
- `--steps N` (control ticks at 36 Hz)
- `--res half|full|cybercab_binned|cybercab` (`320x240`, `640x480`, `1448x938`, or `2896x1876` per camera; default `half`)
- `--trace <out.jsonl>` (per-frame telemetry, tracks, perception metrics & GPU stage timings)
- `--dump-dir <dir>` (save 8-camera PPM mosaic frames)
- `--car-dump <out.bmp>` (save offscreen side + 3/4 Cybercab render snapshot)

## References
- **GaussianFormer3D** (ICRA 2026, [arXiv:2505.10685](https://arxiv.org/abs/2505.10685)) — Multi-modal 3D Gaussian semantic occupancy with 3D deformable attention.
- **GaussianFormer / GaussianFormer-2** (Huang et al., ECCV 2024 [arXiv:2405.17429](https://arxiv.org/abs/2405.17429); CVPR 2025 [arXiv:2412.04384](https://arxiv.org/abs/2412.04384)) — 3D semantic Gaussians & probabilistic occupancy superposition.
- **3D Gaussian Splatting & EWA Splatting** (Kerbl et al., SIGGRAPH 2023 [arXiv:2308.04079](https://arxiv.org/abs/2308.04079); Zwicker et al., IEEE TVCG 2002) — Reality View EWA projection & alpha compositing.
- **BEVFormer** (Li et al., ECCV 2022 [arXiv:2203.17270](https://arxiv.org/abs/2203.17270)) — Spatial-temporal BEV lifting.
- **RT-DETR** (Zhao et al., CVPR 2024 [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)) — Real-time multi-scale object detection.
- **ByteTrack** (Zhang et al., ECCV 2022 [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)) — Multi-object association & tracking.
- **Occ3D / SurroundOcc** (Tian et al., NeurIPS 2023 [arXiv:2304.14365](https://arxiv.org/abs/2304.14365); Wei et al., ICCV 2023 [arXiv:2303.09551](https://arxiv.org/abs/2303.09551)) — `0.4 m` / `0.5 m` 3D occupancy voxel benchmarks.
- **Responsibility-Sensitive Safety (RSS)** (Shalev-Shwartz et al., [arXiv:1708.06374](https://arxiv.org/abs/1708.06374)) — Stopping-distance & corridor safety envelope.
- **Agentic Code Reasoning** (Ugare & Chandra, Meta, [arXiv:2603.01896](https://arxiv.org/abs/2603.01896)) — Structured verification methodology.
