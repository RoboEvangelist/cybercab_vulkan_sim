# Step 17 — Parameter Effects, 11 Driving Actions, Autoshift & Camera Knock

## 1. Scope
- **Added**: `tests/test_v15_params_and_actions.cpp`,
  `tests/test_v15_autoshift_and_calib_closed_loop.cpp`.
- **Modified**: `src/config/SimConfig.{hpp,cpp}`,
  `src/sensors/CameraRigRenderer.cpp`,
  `src/perception/v15/CameraSelfCalib.cpp`, `CMakeLists.txt`.
- **Out of scope**: Shipped `config/cybercab_sim.yaml` default
  (`pipeline_version: "v14"` unchanged).

## 2. Premises
- **P1** (`SimConfig.hpp:108`, `SimConfig.cpp:81-82`,
  `CameraRigRenderer.cpp:435-441`): Parses `camera_knock_rpy_deg` into
  `cfg.cameraKnockRpyDeg` and applies extrinsic `rotY * rotX * rotZ` knock
  rotation in `CameraRigRenderer` when nonzero.
- **P2** (`CameraSelfCalib.cpp:41,88-90`): Pre-reserves `frameEqScratch_`
  (`kMaxFrameEqReserve`) in `rebind` and reuses `eqRing_` / `eqFrameCounts_`
  for live camera-knock pitch/roll recovery.
- **P3** (`tests/test_v15_params_and_actions.cpp:60,99,137,179`): Verifies
  exact numeric effects of `v15_policy.yaml` / `SimConfig` parameters, bounds
  rejection on invalid YAML values, and exact trigger conditions across all 11
  behavior action states (`CRUISING`, `FOLLOWING_LEAD`, `SLOWING_DOWN`,
  `STOPPING_RED_LIGHT`, `STOPPED_RED_LIGHT`, `YIELDING_PEDESTRIAN`,
  `YIELDING_SIREN`, `AVOIDING_STATIC`, `EMERGENCY_AEB`, `AUTOSHIFT_REVERSE`,
  `YIELD_PASSIVE_OBS`).
- **P4** (`tests/test_v15_autoshift_and_calib_closed_loop.cpp:47,101,106`):
  Verifies closed-loop `Autoshift` `D->R->D` reverse displacement plus rear
  pedestrian abort, and closed-loop camera-knock recovery via `CameraSelfCalib`.

## 3. Trace
- **T1 (`Autoshift`)**: Blocked forward envelope triggers
  `AUTOSHIFT_REVERSE` with negative displacement until rear clearance gate
  aborts on rear pedestrian entry.
- **T2 (`Camera Knock`)**: Injecting `cfg.cameraKnockRpyDeg =
  {kKnockRollDeg, kKnockPitchDeg, 0.0f}` during closed-loop driving rotates all
  rendered cameras; `CameraSelfCalib` converges pitch/roll estimates back to the
  perturbed angles.

## 4. Claims
- **C-PAR**: All `V15PolicyConfig` bounds and driving action transitions match
  their YAML specifications (`4/4` unit tests).
- **C-CL**: Closed-loop `Autoshift` reverse (`D->R->D`) and live rendered
  camera-knock self-calibration succeed (`2/2` integration tests).

## 5. Evidence
- Commits `3efb471`..`d2431a4` (`e910731` params/actions `200` lines; `d2431a4`
  autoshift/calib `141` lines).
- `V15ParamsAndActionsTest.*` (**4/4 PASS**; CTest #149–#152):
  `V15PolicyAndSimParamsProduceExactEffects`,
  `SafetyParamsControlSirenAebAutoshiftGaussian`,
  `AllDrivingActionStatesTriggerOnExactConditions`,
  `PolicyYamlLoadsAllSectionsAndRejectsBadBounds`.
- `V15AutoshiftAndCalibTest.*` (**2/2 PASS**; CTest #153–#154):
  `AutoshiftReverseAndRearPedestrianAbort`
  (`autoshift_reverse_delta_x_m=-0.4977`),
  `LiveCameraKnockSelfCalibRecoversPitchRoll` (`knock_recov_pitch_deg=1.7545`,
  `knock_recov_roll_deg=-1.4989`).
