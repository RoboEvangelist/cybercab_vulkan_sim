# Step 10b — Module 7: Autoregressive Policy, Middle-of-Lane Control & Autoshift D/R

## 1. Scope
- **Added**: [AutoregressivePolicy.hpp](../../src/planning/v15/AutoregressivePolicy.hpp), [AutoregressivePolicy.cpp](../../src/planning/v15/AutoregressivePolicy.cpp), [AutoregSafety.hpp](../../src/planning/v15/AutoregSafety.hpp), [AutoregSafety.cpp](../../src/planning/v15/AutoregSafety.cpp), [test_v15_autoreg_planner.cpp](../../tests/test_v15_autoreg_planner.cpp).
- **Modified**: [WorldTypes.hpp](../../src/world/WorldTypes.hpp), [WorldEngine.cpp](../../src/world/WorldEngine.cpp), [TrajectoryPlanner.cpp](../../src/planning/TrajectoryPlanner.cpp), [CMakeLists.txt](../../CMakeLists.txt), [README.md](README.md).
- **Out of scope**: Full multi-scenario GPU closed-loop benchmark (Step 11).

## 2. Premises
- **P1** (`AutoregressivePolicy.cpp:17-65`): Generates two middle-of-lane candidate manifolds (`-1.95 m` / `+1.95 m`) clamped strictly to `|lat| in [1.65, 2.25]` with bounded `<= 0.22 m` adjacent-truck shy-away and discrete curvature speed profiling (`maxLatAccelMs2 = 3.0`).
- **P2** (`WorldEngine.cpp:959-1019`): Under `cfg_.isV15()`, `stepEgoAndPhysics` uses `cmd.gear` for `Autoshift` reverse (`[-2.2, 0.0] m/s`) and short-preview (`3.2..5.8 m`) pure-pursuit curve tracking; `v14` keeps original indexing and gear behavior.
- **P3** (`AutoregSafety.cpp:8-99`): Enforces FMEA 3-camera `SENSOR_SOILED_MRM`, RSS track/voxel braking, red-light/stop-sign holds, `YIELDING_SIREN`, and `AUTOSHIFT_REVERSE` when stopped at two-lane static dead-ends with `rearClearanceM > 6.0`.

## 3. Trace
- **T1 (`v14`)**: `TrajectoryPlanner::plan` skips `AutoregressivePolicy` and produces 9 legacy candidates; `WorldEngine::stepEgoAndPhysics` uses the unchanged v14 lookahead branch.
- **T2 (`v15`)**: `AutoregressivePolicy::plan` enforces middle-of-lane waypoints and multimodal safety states; short-preview pure pursuit tracks corners without divider cutting.

## 4. Claims
- **C-EQ**: Under `pipeline_version: v14`, planner outputs (`9` candidates, `gear == 1`) and closed-loop tests match pre-Step-10b behavior.
- **C-V15**: Across 520 closed-loop straight + curved steps, `ego.currentLateralOffset` stays in `[-2.35, -1.55]` (`max dev = 0.0518 m`); truck shy-away, `YIELDING_SIREN`, and `AUTOSHIFT_REVERSE` operate within safety bounds.

## 5. Evidence
- `V15AutoregPlannerTest.*` (`2/2 PASS`): `m7_curve_min_lat_m = -1.9464`, `m7_curve_max_lat_m = -1.8982`, `m7_max_center_dev_m = 0.0518`, `m7_truck_shy_lat_m = -2.1211`.

## 6. Known limits
- Full GPU perception + v15 planner closed-loop interactions across all six hazard scenarios are validated in Step 11.
