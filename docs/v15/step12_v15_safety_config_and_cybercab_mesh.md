# Step 12 — Configurable FSD v15 Policy, Collision Hardening & Cybercab Mesh

## 1. Scope
- **Added**: `config/v15_policy.yaml`, `src/config/V15PolicyConfig.{hpp,cpp}`, `src/world/CybercabMesh.{hpp,cpp}`, `src/world/CybercabBodyLoft.hpp`, `.clang-format`.
- **Modified**: `SimConfig.{hpp,cpp}`, `LatentPredictor.{hpp,cpp}`, `V15Pipeline.cpp`, `AutoregSafety.{hpp,cpp}`, `AutoregressivePolicy.{hpp,cpp}`, `WorldEngine.cpp`, `CameraRigRenderer.cpp`, `shaders/scene.frag`, `main.cpp`, `CMakeLists.txt`, `tests/test_v15_{config,autoreg_planner,e2e_closed_loop}.cpp`, `README.md`, `docs/v15/README.md`.
- **Out of scope**: Shipped `config/cybercab_sim.yaml` default (`pipeline_version: "v14"` unchanged).

## 2. Premises
- **P1** (`SimConfig.cpp:80,226-232`, `V15PolicyConfig.cpp:12-88`): `SimConfig::loadFromYaml` loads `config/v15_policy.yaml` into `cfg.v15Policy` and throws `std::runtime_error` on invalid bounds (`minLaneAbsM >= maxLaneAbsM`, `aebRssFrac <= 0` or `> 1`, etc.); `V15Pipeline.cpp:126` forwards `cfg_->v15Policy` to `LatentPredictor::evaluate`.
- **P2** (`AutoregressivePolicy.cpp:50-83,146-225,244-273`): Replaces quintic lateral blending with bounded autoregressive state rollout (`d_{k+1} = d_k + clamp(k_p*(d_tgt - d_k), +-v_lat_max) * dtStep`), road-secant curvature speed cap, red-light/stop-sign intersection lane lock (`intersectionLaneLockCost = 8500.0`), target-lane blind-spot vehicle/voxel gates, 3-point rotated OBB hull checks, and front-bumper-anchored RSS corridor.
- **P3** (`AutoregSafety.cpp:35-42,63-118`): Computes `projectVehicleExtents` rotated OBB projections (`longProj`, `latProj`), checks curved `corrPts`/`plannedTraj` + `3.2 m` direct bumper ray + `1.2 s` cut-in approach, suppresses only out-of-hull adjacent-lane vehicle voxels, and enforces `< 1.85 m` close-range AEB hold (`kAebHoldMinSpdMs = 0.15f`).
- **P4** (`CybercabBodyLoft.hpp:35-286`, `CybercabMesh.cpp:105-263`, `shaders/scene.frag:27`): Renders the 37-station champagne-gold Cybercab coupe, staggered aero wheels, and tinted front windshield using `kCybercabSemTag = 1.25f` (`abs(inSemId - 1.25) < 0.08`) so AI traffic (`semId == 1.0`) perception shading stays bit-identical.
- **P5** (`main.cpp:46,85-90`): Adds `--pipeline v14|v15` CLI override without altering shipped YAML defaults.

## 3. Trace
- **T1 (`v14`)**: `TrajectoryPlanner::plan` (`TrajectoryPlanner.cpp:31-34`) skips `v15::AutoregressivePolicy` when `cfg_.isV15()` is false; `shaders/scene.frag:27` skips clearcoat on `semId == 1.0` AI vehicles; all 96 pre-v15 tests pass unchanged.
- **T2 (`v15`)**: `cybercab_sim --config config/cybercab_sim.yaml --pipeline v15 --headless --steps 2400` executes all 7 FSD v15 modules, `V15PolicyConfig`, and collision-hardened autoregressive planning around curved tracks, pedestrians, stop signs, and signals.

## 4. Claims
- **C-EQ**: With `pipeline_version: "v14"`, all `96` pre-v15 regression tests pass (`0` failures), preserving `FMEAPerceptionTest` and `TrackingAndOcclusionTest` bit-parity.
- **C-V15**: Closed-loop `v15` achieves zero collisions (`sat_contact == false`) across nominal (`2400` steps), all 6 hazard modes (`60` steps each), and `360`-step red-light queue hold/green-release while staying middle-of-lane (`v15_nom_cruise_dev_m = 0.2277`).

## 5. Evidence
- Full CTest gate (`140/140` PASS, `84.74 sec`): `96` v14 tests + `44` v15 tests (`4/4` `V15ConfigTest`, `4/4` `V15AutoregPlannerTest`, `3/3` `V15EndToEndClosedLoopTest` incl. red-light queue & `CybercabMesh` normal/gear/brake checks).
- `V15EndToEndClosedLoopTest.*` (`3/3` PASS): `v15_nom_recall=0.5811`, `v15_nom_precision=0.8077`, `v15_nom_cruise_dev_m=0.2277`, `v15_half_gpu_ms=4.8787`, `v15_full_gpu_ms=7.3841`, `v15_red_queue_advance_m=2.3017`.
- `2400`-step headless `v15` `nominal` trace: `frames=2400`, `sat_hits=0`, `mean_gpu_ms=4.9248`.

## 6. Known limits
- Interactive window display FPS remains governed by swapchain vsync; headless Vulkan GPU stages run under `7.4 ms` (`< 27.8 ms`) at both half and full resolution.
