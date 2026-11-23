# Step 5b: `V15Pipeline` Front-End Wiring & Budget Certificate (Module 1 Complete)

## Scope
Wire `FrequencySplit` into `PerceptionPipeline` via `V15Pipeline` (`src/perception/v15/V15Pipeline.{hpp,cpp}`) strictly behind `cfg.isV15()`. Out of scope: Module 2+ consumers of `bandsSSBO()`.

## Premises
- **P1** `PerceptionPipeline.hpp:50,59`: `v15()` returns `v15_.get()` (`std::unique_ptr<v15::V15Pipeline>`).
- **P2** `PerceptionPipeline.cpp:20-35`: `v15_` is allocated/initialized ONLY when `cfg_.isV15() && rig.linearHdrEnabled()`; `onCameraResized`, `reset`, and `destroy` forward cleanly to `v15_` when non-null.
- **P3** `PerceptionPipeline.cpp:64-67`: `v15_->recordFrontEnd(cmd, profiler)` runs inside GPU Batch 1 ahead of `soil_` / `detr_` / `bev_`, profiling stage `"1c_v15_freq_split"`.
- **P4** `V15Pipeline.cpp:11-26`: throws `std::runtime_error` on `!cfg.isV15()` or `!rig.linearHdrEnabled()`; binds `rig.cameraLinearSSBO()` with `FreqSplitParams::forConfig(rig.camWidth(), cfg)`.

## Trace
- **T-v14:** `cfg_.isV15() == false` -> `v15_ == nullptr` across `init`/`onCameraResized`/`runFrame`/`destroy`; zero v15 allocations or dispatches.
- **T-v15:** `cfg_.isV15() && rig.linearHdrEnabled()` -> `v15_` executes two-pass `FrequencySplit` in Batch 1 (`1c_v15_freq_split`) producing `8 * H * W * vec4(low, high, gx, gy)`; `onCameraResized` rebinds `bands_` and updates `sigmaPx` (`4px -> 8px`).

## Claims & Evidence (`tests/test_v15_pipeline.cpp`)
- **C-EQ (`V14NeverAllocatesOrRunsV15FrontEnd`):** `v15() == nullptr`, `"1c_v15_freq_split"` absent from `gpuStageMs`, `v14` `ctest` **114/114** pass.
- **C-V15-1 (`V15RunsFreqSplitInBatch1RespectsResizeAndBudget`):** `v15_freq_split_ms = 0.1302 ms`, `v15_gpu_sum_ms = 2.0403 ms`, `v15_wall_ms = 3.3334 ms` (< `27.8 ms` 36 Hz target); detail band finite & zero-mean (`|mean(high)| < 0.05`); `resize` updates `sigmaPx = 8.0f` cleanly.
- **C-V15-2 (`GuardsRejectNonV15OrNonHdrRigAndUninitializedDispatchIsSafe`):** invalid init configs throw `std::runtime_error`; uninitialized `recordFrontEnd`/`onCameraResized` are safe no-ops.

## Known Limits
- `bandsSSBO()` is produced on GPU in Batch 1 and ready for Module 2 (`VirtualCameraRectifier`) / Module 3 (`PatchTokenizer`) consumption; v14 perception branches still read `camTensorSSBO` (`step02 R2`).
