# Step 11 — End-to-End Closed-Loop FSD v15 Validation

## 1. Scope
- **Added**: [test_v15_e2e_closed_loop.cpp](../../tests/test_v15_e2e_closed_loop.cpp), [step11_e2e_v15_validation.md](step11_e2e_v15_validation.md).
- **Modified**: [AutoregressivePolicy.cpp](../../src/planning/v15/AutoregressivePolicy.cpp), [CMakeLists.txt](../../CMakeLists.txt), [README.md](README.md).
- **Out of scope**: Changes to `config/cybercab_sim.yaml` default (`v14` preserved).

## 2. Premises
- **P1** (`test_v15_e2e_closed_loop.cpp:25-83`): Executes full GPU `CameraRigRenderer` + `PerceptionPipeline` (`M1..M6`) + `AutoregressivePolicy` (`M7`) + `WorldEngine::stepAgents`/`stepEgoAndPhysics` closed loop across `nominal` (half-res 90 steps & full-res `640x480`), `jaywalk`, `occluded`, `hardbrake`, `siren`, `soil_partial`, and `soil_all_front`.
- **P2** (`AutoregressivePolicy.cpp:95-103,148`): Lateral lane-commitment hysteresis (`+55` directional barrier across `|lat| < 1.50 m`) prevents center-divider dithering and classifies active crossings until settled within `0.30 m` of `+/-1.95 m`.

## 3. Trace
- **T1 (`v14`)**: Default `cybercab_sim --headless --steps 120` runs `v14` (`gpu=1.74..1.93 ms`); all pre-v15 regression tests in `ctest` execute unmodified.
- **T2 (`v15`)**: `cybercab_sim --config /tmp/cybercab_v15.yaml --headless --steps 120` runs full 7-module `v15` (`gpu=4.64..5.56 ms`), and `--scenario siren --steps 60` transitions to `YIELDING_SIREN` and slows to `0.56 m/s`.

## 4. Claims
- **C-EQ**: `100%` of pre-v15 tests pass under default `pipeline_version: v14` (`136/136` CTest pass).
- **C-V15**: Closed-loop `v15` achieves zero collisions across all scenarios, settled middle-of-lane cruise deviation `< 0.35 m`, closes `step02 R2` (`recall=0.5681`, `precision=0.7529`), and stays under `27.8 ms` at half-res and full-res.

## 5. Evidence
- `V15EndToEndClosedLoopTest.*` (`2/2 PASS`): `v15_nom_recall=0.5681`, `v15_nom_precision=0.7529`, `v15_nom_cruise_dev_m=0.3160`, `v15_half_gpu_ms=4.8114`, `v15_full_gpu_ms=7.0049`.
- Full CTest suite: `136/136` passed (`100% tests passed, 0 tests failed out of 136`, real time `76.72 sec`).

## 6. Known limits
- Interactive windowed display timing depends on swapchain vsync; headless compute/render timing is verified well below `27.8 ms`.
