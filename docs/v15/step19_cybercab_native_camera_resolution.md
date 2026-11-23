# Step 19 — Real Cybercab Camera Resolutions (`2896x1876` / `1448x938`) & Presets

## 1. Scope
- **Added**: `tests/test_v15_cybercab_camera_res.cpp` (`195` lines).
- **Modified**: `config/cybercab_sim.yaml`, `src/config/SimConfig.{hpp,cpp}`,
  `src/sensors/CameraRigRenderer.{hpp,cpp}`,
  `src/perception/BackboneRTDETR.cpp`, `src/main.cpp`,
  `src/app/VulkanViewer.cpp`, `src/perception/v15/{CameraSelfCalib,CalibSolver}.cpp`,
  `shaders/{cam_isp_soil,01_soil_detect,02_backbone_fpn,v15_freq_split}.comp`,
  `CMakeLists.txt:174`, `docs/v15/README.md`.
- **Out of scope**: Default `config/cybercab_sim.yaml` keeps
  `pipeline_version: "v14"` and `initial_res_mode: "half"` (`320x240`).

## 2. Premises
- **P1** (`SimConfig.hpp:15-20,113-116,226-229`,
  `SimConfig.cpp:32-61,100-112,266`): Preserves `"half"` (`320x240`) and `"full"`
  (`640x480`) and adds Cybercab Sony IMX963 presets `"cybercab_binned"`
  (`1448x938`) and `"cybercab"` (`2896x1876`), with `isValidResMode()` bounds
  enforcement.
- **P2** (`CameraRigRenderer.cpp:67`, `BackboneRTDETR.cpp:143,155`,
  `main.cpp:89`, `VulkanViewer.cpp:642-648,1305`): Resolves all 4 modes at init,
  cycles all 4 in the viewer via `nextResolutionPreset()`, validates `--res`,
  and zeroes `prevCamTensorSSBO_` via GPU `vkCmdFillBuffer` (avoiding 1.4 GB
  host staging spikes at `2896x1876`).
- **P3** (`shaders/cam_isp_soil.comp:54-59`, `shaders/01_soil_detect.comp:49-55`,
  `shaders/02_backbone_fpn.comp:70-85`, `shaders/v15_freq_split.comp:50-55`):
  Normalizes soil blur/noise, strides soil and FPN cell sampling, scales pixel
  neighborhoods (`sx`/`sy`), and bounds separable blur taps (`kMaxHalfTaps=12`)
  around zero phase.
- **P4** (`CameraSelfCalib.cpp:34-39`, `CalibSolver.cpp:25-39`): Scales calib
  tile width as exact multiples of `8u` and chain pixel thresholds by `camH /
  240.0f`.

## 3. Trace
- **T1 (Backward compatibility)**: At `"half"` (`320x240`) and `"full"`
  (`640x480`), shaders reduce to `1x`/`2x` scaling and preserve all `163`
  baseline tests.
- **T2 (Native & binned Cybercab)**: At `"cybercab"` (`2896x1876`) and
  `"cybercab_binned"` (`1448x938`), rendering, frequency split, self-calib,
  soil classification, detections, voxels, and Gaussians run end-to-end.

## 4. Claims
- **C-OPT**: All 4 resolution presets load, validate, cycle, and resize
  cleanly across config, CLI, viewer, and pipeline (`3/3` Step 19 tests).
- **C-EQ**: Full repository suite passes **100% (`166/166`)**.

## 5. Evidence
- Filtered XML check (`V15CybercabCameraResConfigTest.*:V15CybercabCameraResTest.*:V15CalibTest.*:V15PipelineTest.*`,
  **7/7 PASS**, `12.566 s`):
  - `AllFourResolutionPresetsLoadValidateAndCycle`: PASS (`2 ms`).
  - `CybercabNativeAndBinnedResolutionsRenderDetectCalibrateAndResize` (`3536 ms`):
    `native_2896x1876_gpu_ms=33.5196`, `native_2896x1876_wall_ms=57.2350`,
    `native_calib_pitch_deg=1.4529`, `native_calib_roll_deg=-1.2832`,
    `binned_1448x938_gpu_ms=11.8370` (`< 27.8 ms`).
  - `ScaleInvariantSoilAndEmptyRoadAcrossPresets`: PASS (`5995 ms`).
  - `V15PipelineTest`: `3/3 PASS` (`v15_gpu_sum_ms=5.3634`,
    `v15_wall_ms=11.3085`).
  - `V15CalibTest`: PASS (`fullres_pitch_deg=-1.9125`,
    `fullres_roll_deg=1.8903`).
- Full CTest gate (`ctest --test-dir /home/pototo/VulkanTest/build --output-on-failure`):
  **166/166 PASS (100%, 111.43 sec)** (`96` v14 oracle + `70` v15 tests).

## 6. Known limits
- Uncompressed 8-camera `2896x1876` rendering + full pipeline takes `~33.52 ms`
  GPU / `~57.24 ms` wall on RTX 2080 Ti (slower than `36 Hz`), whereas
  `"cybercab_binned"` (`1448x938`, `11.84 ms` GPU), `"full"`, and default `"half"`
  (`320x240`, `5.36 ms` GPU) run real-time.
