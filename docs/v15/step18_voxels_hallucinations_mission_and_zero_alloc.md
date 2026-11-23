# Step 18 — Voxels, Detections, Hallucinations, Missions & Zero-Alloc Tests

## 1. Scope
- **Added**: `tests/test_v15_voxels_detections_and_hallucinations.cpp` (`186`
  lines), `tests/test_v15_mission_collisions_and_zero_alloc.cpp` (`194` lines).
- **Modified**: `CMakeLists.txt:172-173`, `docs/v15/README.md`, plus Step 17 &
  Step 18 certificates.
- **Out of scope**: Shipped `config/cybercab_sim.yaml` default
  (`pipeline_version: "v14"`, `lighting_mode: "day"`, `enable_rccb_mosaic:
  false` unchanged).

## 2. Premises
- **P1** (`tests/test_v15_voxels_detections_and_hallucinations.cpp:29,89,130`):
  Tests occupancy voxel bounds (`occMinX..occMaxX`, `occMinY..occMaxY`,
  `0..occMaxZ`), `BEVFormer::worldFromIsoEgo` world transform accuracy,
  dynamic voxel-to-OBB precision, truck/signpole classification, empty-road
  zero hallucinations (`ghostVeh`, `ghostPed`, dynamic voxels/Gaussians, false
  AEB), and closed-loop precision/recall/AP/RMSE.
- **P2** (`tests/test_v15_mission_collisions_and_zero_alloc.cpp:107,127,161`):
  Tests 6 closed-loop missions (`nominal`, `jaywalk`, `occluded`, `hardbrake`,
  `siren`, `night_glare_jaywalk`), `2x2` ISO 34502 parametric cut-in sweeps plus
  bit-exact planner determinism, and a global `operator new` `HotAllocScope`
  heap-allocation gate across `50` steady-state planning + corridor + HUD
  iterations.
- **P3** (`CMakeLists.txt:172-173`): Registers both suites into
  `cybercab_tests` built under `-Wall -Wextra -Wpedantic` with `0` warnings.

## 3. Trace
- **T1 (Voxel & Hallucination Gate)**: Closed-loop `V15VoxelDetHallucTest`
  verifies `0.0000 m` world-transform max error, `88.08%` dynamic voxel OBB
  precision, `0` empty-road ghost detections/voxels/Gaussians/AEB, and `0`
  truck misclassifications.
- **T2 (Mission & Zero-Alloc Gate)**: All `6/6` missions complete with `0`
  collisions (`100%`), `2x2` ISO 34502 cut-in sweeps have `0` collisions, and
  `HotAllocScope` records `0` heap allocations after warmup.

## 4. Claims
- **C-PER**: Voxel geometry, truck/sign semantics, precision/recall/AP/RMSE, and
  empty-road hallucination immunity hold end-to-end (`3/3` tests).
- **C-MIS**: Multi-mission safety, parametric cut-in robustness, determinism,
  and zero steady-state hot-path allocations hold (`3/3` tests).
- **C-EQ**: Full repository regression suite passes `100%` (`163/163`).

## 5. Evidence
- Filtered Step 18 XML run (`V15VoxelDetHallucTest.*:V15MissionAndZeroAllocTest.*`,
  **6/6 PASS**, `16.37 s`):
  - `VoxelBoundsTransformAndSemanticLocationAccuracy` (`1.195 s`):
    `v15_voxel_loc_max_err_m=0.0000`, `v15_voxel_dyn_precision=0.8808`,
    `v15_truck_veh_voxels=294`, `v15_sign_pole_voxels=46`,
    `sumTruckPed=0`.
  - `EmptyRoadProducesZeroHallucinationsAndFalseAeb` (`1.023 s`):
    `ghostVeh=0`, `ghostPed=0`, `dynVox=0`, `dynGauss=0`, `falseAeb=0`.
  - `ClosedLoopPrecisionRecallApRmseAndZeroTruckMisclass` (`1.456 s`):
    `v15_mean_recall=0.6667`, `v15_mean_prec=0.9481`, `v15_veh_ap=0.5398`,
    `v15_pos_rmse_m=1.6177`, `misclassifiedTrucks=0`.
  - `MultiMissionCompletionRateZeroCollisions` (`12.102 s`):
    `mission_completion_rate=1.0000` (`6/6`), `total_collisions=0`,
    `nominal_distance_m=42.2529`, `nominal_max_lat_dev_m=0.0518`.
  - `Iso34502ParametricCutInAndBitDeterminism` (`0.035 s`): PASS.
  - `SteadyStateHotPathZeroHeapAllocations` (`0.008 s`):
    `hot_path_alloc_count=0`.
- Full CTest gate (`ctest --test-dir /home/pototo/VulkanTest/build --output-on-failure`):
  **163/163 PASS (100%, 105.02 sec)** (`96` v14 oracle + `67` v15 tests).

## 6. Known limits
- Declaration-only SOTA interfaces added in Step 16 (`GaussianSurfel2DWorld`,
  `TruncatedDiffusionPolicy`) remain unimplemented/untested by request.
