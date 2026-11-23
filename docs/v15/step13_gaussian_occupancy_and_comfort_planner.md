# Step 13 — 4D Gaussian Occupancy & Comfort Multi-Rollout Planner

## 1. Scope
- **Added**: `src/planning/v15/GaussianRolloutScorer.{hpp,cpp}`,
  `tests/test_v15_gaussian_planner.cpp`.
- **Modified**: `src/perception/PerceptionTypes.hpp`,
  `src/perception/PerceptionPipeline.cpp`,
  `src/perception/v15/{GaussianWorldModel.{hpp,cpp},V15Pipeline.cpp,LatentPredictor.cpp}`,
  `src/config/V15PolicyConfig.{hpp,cpp}`, `config/v15_policy.yaml`,
  `src/planning/v15/{AutoregSafety.cpp,AutoregressivePolicy.cpp}`, `CMakeLists.txt`.
- **Out of scope**: Shipped `config/cybercab_sim.yaml` default
  (`pipeline_version: "v14"` unchanged).

## 2. Premises
- **P1** (`PerceptionTypes.hpp:122-170`, `PerceptionPipeline.cpp:64-68`,
  `V15Pipeline.cpp:135-138`): `PerceptionFrameOutput::reserveV15Scratch`
  pre-reserves `v15Gaussians`, `v15RealityRgba`, and `GaussianWorldModel`
  scratch buffers; `LatentPredictor.cpp` uses stack `std::array` (`0` hot-path
  heap allocations).
- **P2** (`V15PolicyConfig.hpp:74-83`, `config/v15_policy.yaml:68-76`): Adds
  validated `gaussian_comfort` weights/thresholds (`gaussian_occ_thresh = 0.35`,
  `gaussian_safety_stop_prob = 0.62`, `nudge_offset_m = 0.16`,
  `speed_backoff_ratio = 0.84`).
- **P3** (`GaussianWorldModel.cpp:125-170`, `GaussianRolloutScorer.cpp:54-140`):
  `GaussianWorldModel::queryOccupancyProb` evaluates 4D GaussianFormer-2
  occupancy on `Vehicle`/`Pedestrian` Gaussians (skipping `Static` road-edge/
  stop-bar paint); `selectComfortSubRollout` scores `3 nudges x 2 speed scales`
  across 18 rollout steps for 4D occupancy, lateral jerk, and lane drift.
- **P4** (`AutoregressivePolicy.cpp:162-176,272-306`,
  `GaussianRolloutScorer.cpp:142-199`, `AutoregSafety.cpp:159-161`): Integrates
  `selectComfortSubRollout` and `evaluateCorridorGaussians` (which projects
  dynamic lead velocity so accelerating lead vehicles are tracked rather than
  pinned to `0 m/s`), using pre-reserved `thread_local` scratch vectors.

## 3. Trace
- **T1 (`v14`)**: `PerceptionPipeline::runFrame` skips `v15_` (`v14` leaves
  `v15Gaussians` empty) and `TrajectoryPlanner::plan` uses `v14` quintic logic;
  all 96 `v14` tests are untouched.
- **T2 (`v15`)**: `V15Pipeline::finishBackEnd` populates `out.v15Gaussians`;
  `AutoregressivePolicy::plan` scores comfort/4D-occupancy sub-rollouts and
  `AutoregSafety` checks front-bumper corridor Gaussians with named constants.

## 4. Claims
- **C-EQ**: With `pipeline_version: "v14"`, all `96` pre-v15 regression tests
  pass (`0` failures), bit-identical in observable outputs.
- **C-V15**: Unboxed 4D Gaussians trigger lane-switch or close-range AEB stop,
  dynamic future-crossing Gaussians incur >5x higher blocked-lane cost than clear
  lanes, nudges stay inside `[minLaneAbsM, maxLaneAbsM]`, and closed-loop
  driving achieves `0` collisions with improved cruising centering and green-light
  queue release.

## 5. Evidence
- Full CTest gate (`145/145` PASS, `93.40 sec`): `96` v14 tests + `49` v15
  tests, including `5/5` `V15GaussianPlannerTest`.
- `V15GaussianPlannerTest` (`18/18` targeted filter PASS):
  `UnboxedGaussianObstacleTriggersSwitchOrStop`,
  `Dynamic4DGaussianFutureCrossingPenalized`,
  `MultiRolloutNudgesAndJerkPenaltiesBehaveCleanly`,
  `DynamicLeadGaussianPullingAwayDoesNotPinZero`,
  `GaussianComfortConfigBoundsValidated`.
- `V15EndToEndClosedLoopTest`: `v15_nom_recall=0.5745`,
  `v15_nom_precision=0.8169`, `v15_nom_cruise_dev_m=0.0518`,
  `v15_red_queue_advance_m=2.3021`, `v15_half_gpu_ms=5.1547`,
  `v15_full_gpu_ms=7.4089`, `0` collisions across all scenarios.

## 6. Known limits
- Interactive "Tesla Reality View" HUD rendering and `v15.*` JSONL trace
  serialization follow in Step 14.
