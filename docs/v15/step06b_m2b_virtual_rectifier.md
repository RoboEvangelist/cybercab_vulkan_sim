# Step 6b: M2b Virtual Pinhole Rectifier, Dual-Branch & `V15Pipeline` Wiring

## Scope
Warp 8 distorted physical cameras into 10 canonical `160x120` pinhole views compensated by online self-calibration (`VirtualCameraRectifier`), wire `CameraSelfCalib` + `VirtualCameraRectifier` into `V15Pipeline`, and verify GPU accuracy, latency, and v14 equivalence.

## Premises
- **P1** `shaders/v15_virtual_rectify.comp:1-82`: back-projects `rayVirt = K_v^-1 (u,v,1)` through `physCamFromVirtCam = R_phys^T * R_calib^T * R_virt`, applies `brownConradyDistort`, rejects behind-camera / overscan-border pixels (`validMask = 0`), and bilinearly samples `cameraLinearSSBO()` (`rgb, 1.0`) and `FrequencySplit::bandsSSBO()`.
- **P2** `VirtualCameraRectifier.cpp:69-103`: generates 8 canonical surround pinhole views (`0..7`), view `8` (`kViewVru`: sourced from `CAM_FRONT_WIDE`, `hfov=70°`, `pitch=-12°`), and view `9` (`kViewElevated`: sourced from `CAM_FRONT_MAIN`, `hfov=36°`, `pitch=+5°`).
- **P3** `V15Pipeline.cpp:54-68` & `PerceptionPipeline.cpp:89-91`: `v15_->recordFrontEnd` dispatches `1c_v15_freq_split`, `2a_v15_self_calib`, `2b_v15_virt_rectify` inside Batch 1; `finishBatch1Cpu()` runs `selfCalib_.solveAndUpdate` post-fence.

## Trace
- **T-v14:** `v15_ == nullptr` -> `V14NeverAllocatesOrRunsV15FrontEnd`: `gpuStageMs` has zero `1c`/`2a`/`2b` stages.
- **T-v15:** `FrequencySplit` -> `CameraSelfCalib` -> `VirtualCameraRectifier` uploads `specsSSBO_` and warps linear+band buffers into `virtLinearSSBO()`/`virtBandsSSBO()` (`10 * 120 * 160` texels).

## Claims & Evidence (`ctest` **121/121** PASS)
- **C-EQ (`V14NeverAllocatesOrRunsV15FrontEnd`):** v14 pipeline untouched (`ctest` 96 legacy + 25 v15 = `121/121` PASS).
- **C-V15-1 (`V15PipelineTest`):** `v15_freq_split_ms = 0.1309`, `v15_calib_ms = 0.1065`, `v15_rectify_ms = 0.9386`, `v15_gpu_sum_ms = 3.0412`, `v15_wall_ms = 5.0827` (< `27.8 ms` 36 Hz target).
- **C-V15-2 (`V15VirtRectifyUnitTest`):** `vru_span_px = 22.6522` vs `wide_span_px = 9.2957` (`2.44x` denser ground coverage); elevated overhead signal `y = 5.4 m` at `44.5 m` projects inside valid view `9`.
- **C-V15-3 (`V15VirtRectifyTest`):** synthetic 10-view round-trip max error `0.0359 px` (`< 0.5 px`); rendered `+2.0°` pitch / `-1.5°` roll MAE drops `0.0058 -> 0.0013` (`4.5x`) with `R_calib`; zero NaN/Inf across all `192,000` texels.

## Known Limits
- Downstream M3 `5x5` patch tokenizer will consume `virtLinearSSBO()` + `virtBandsSSBO()` next.
