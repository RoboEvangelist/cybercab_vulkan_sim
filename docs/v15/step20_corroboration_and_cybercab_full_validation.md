# Step 20 — Multi-Camera Corroboration & Cybercab Full Validation

## 1. Scope
- **Added**: `src/perception/MultiCamCorroboration.hpp` (`78` lines),
  `tests/test_camera_corrob_and_bounds.cpp` (`193` lines),
  `tests/test_cybercab_res_full_validation.cpp` (`199` lines).
- **Modified**: `src/perception/PerceptionTypes.hpp`,
  `src/perception/BackboneRTDETR.{hpp,cpp}`, `src/perception/ByteTrack.cpp`,
  `src/core/TraceLogger.cpp`, `shaders/{02_backbone_fpn,03_rtdetr_decoder}.comp`,
  `CMakeLists.txt:175-176`, `docs/v15/README.md`.
- **Out of scope**: Default `config/cybercab_sim.yaml` keeps
  `pipeline_version: "v14"`, `initial_res_mode: "half"`, and
  `rtdetr.max_det_range_m: 65.0`.

## 2. Premises
- **P1** (`MultiCamCorroboration.hpp:10-76`): Defines bitmask and correlated
  multi-camera fusion helpers (`kCorrobScoreBoost=0.04f`,
  `kCorrobSigmaFactor=0.92f`, `kMinCorrobSigmaFloor=0.80f`,
  `kMinCorrobCamWeight=0.45f`), excluding low-weight soiled cameras.
- **P2** (`PerceptionTypes.hpp:51-52,76-77`, `BackboneRTDETR.cpp:567-647`,
  `ByteTrack.cpp:199-514`, `TraceLogger.cpp:74-77`): Propagates
  `corroboratingCamMask` and `corroboratingCamCount` through 3D NMS, ByteTrack
  noise scaling, and `tracks[]` JSONL telemetry (`hits`, `age_sec`,
  `corrob_cams`, `corrob_mask`), plus `candTeleCloseSliver` suppression.
- **P3** (`02_backbone_fpn.comp:92-95,295-302`,
  `03_rtdetr_decoder.comp:78-82,304-311,391-413`,
  `BackboneRTDETR.cpp:287-299`): Centers FPN sub-step sampling with symmetric
  `kHalfFrac=0.35` bounds and passes `aspectScale` (`invAspect=1.0` at `4:3`)
  for aspect-invariant `1.5437:1` (`1448x938`, `2896x1876`) vs `4:3` lifting.
- **P4** (`test_camera_corrob_and_bounds.cpp`,
  `test_cybercab_res_full_validation.cpp`): Tests YAML bounds/CLI rejection,
  corroboration/age decay, HW4 `36 Hz` range specs
  (`{150,250,60,80,80,100,100,50}` m), stratified `15/55/200 m` tracking, and
  closed-loop `v14`/`v15` I/O, PR/AP, voxels, calib, and Plucker tokens.

## 3. Trace
- **T1 (`v14` equivalence & FMEA)**: At `"half"` (`320x240`), `stepX=1`
  (`spx=px`), `isRefAspect=true`, calibrated `kCorrobSigmaFactor=0.92f` keeps
  Kalman covariance unbiased, and `candTeleCloseSliver` prevents braking splits;
  all `96` `v14` tests pass (`wm_cov25_2_0s=0.7137`, `vox_sig_prec=0.4299`).
- **T2 (`v15` Cybercab presets)**: At `1448x938` and `2896x1876`, centered
  sub-steps and `aspectScale` align detection lifting, voxels, `PatchTokenizer`,
  `VirtualCameraRectifier`, and `CameraSelfCalib`.

## 4. Claims
- **C-COR**: Corroboration raises score, tightens sigma, filters soiled
  cameras/class mismatches, and logs bitmasks/counts to JSONL.
- **C-CYB**: Both `v14` and `v15` pass closed-loop latency, PR/AP, voxel, age,
  and corroboration gates at `1448x938` and `2896x1876`.
- **C-EQ**: Full suite passes **100% (`171/171`)**.

## 5. Evidence
- Targeted 27-test XML suite (**27/27 PASS**):
  - `FMEAPerceptionTest.SingleTruckUnderBrakingPitchAndObliquityNeverSplitsBoxes`,
    `OccupancyNetExtendedTest.FullSceneVoxelAuditCountsTrueVsOrphanGhostVoxels`,
    `ModelQualityAndPrTest.PerClassPRCurvesAPAndBevRoadPrecision` (`vox_sig_prec=0.4299`),
    `WorldModelAndLatencyTest.WorldModel4DRolloutAccuracyAndUncertaintyCalibration` (`wm_cov25_2_0s=0.7137`): PASS.
  - `ByteTrackExtendedTest` (`12/12`), `V15TelemetryAndHudTest` (`3/3`),
    `CameraCorrobAndBoundsTest` (`3/3`): PASS.
  - `V15VoxelDetHallucTest` (`3/3`): `v15_voxel_loc_max_err_m=0.0000`,
    `v15_voxel_dyn_precision=0.8809`, `v15_mean_recall=0.6667`,
    `v15_mean_prec=0.9481`, `v15_veh_ap=0.5398`, `v15_pos_rmse_m=1.6177`.
  - `CybercabResFullValidationTest.V14AndV15AllModulesAndCalibPresets`:
    `v14_cybercab_binned_wall_ms=5.0291`, `v14_cybercab_wall_ms=11.6060`,
    `v15_binned_gpu_ms=13.1730`, `v15_binned_wall_ms=27.4664`,
    `binned_calib_pitch_deg=1.3078`, `binned_calib_roll_deg=-1.2937`.
  - `CybercabResFullValidationTest.CybercabResPrVoxelAgeAndCorroboration`:
    `cybercab_binned_recall=0.7063`, `cybercab_binned_prec=0.9417`,
    `cybercab_binned_ap=0.5529`, `cybercab_binned_rmse=1.5405`,
    `cybercab_binned_max_range_m=54.2261`, `cybercab_binned_max_corrob=2`;
    `cybercab_recall=0.7143`, `cybercab_prec=0.9833`, `cybercab_ap=0.4545`,
    `cybercab_rmse=1.2540`, `cybercab_max_range_m=44.0068`,
    `cybercab_max_corrob=2`.
- Full CTest gate (`ctest --test-dir /home/pototo/VulkanTest/build --output-on-failure`):
  **171/171 PASS (100%, 124.44 sec)** (`96` v14 oracle + `75` v15/Step 20 tests).

## 6. Known limits
- Default `config/cybercab_sim.yaml` retains `"v14"`, `"half"`, and `65.0 m`
  urban-loop detector cap; `250 m` HW4 specs are verified via unit harness.
