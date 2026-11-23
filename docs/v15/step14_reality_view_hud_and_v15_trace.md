# Step 14 — Live "Tesla Reality View" HUD & v15 Multimodal Telemetry

## 1. Scope
- **Added**: `src/app/V15HudOverlay.{hpp,cpp}`,
  `tests/test_v15_telemetry_and_reality_hud.cpp`.
- **Modified**: `src/core/TraceLogger.cpp`, `src/app/VulkanViewer.{hpp,cpp}`,
  `CMakeLists.txt`, `docs/v15/README.md`.
- **Out of scope**: Shipped `config/cybercab_sim.yaml` default
  (`pipeline_version: "v14"` unchanged).

## 2. Premises
- **P1** (`TraceLogger.cpp:111-134`): `TraceLogger::logFrame` gates
  `root["v15"]` strictly on `if (perc.v15.active)`, serializing `gear`,
  `lat_offset_m`, `preferred_lane_offset_m`, `calib_pitch_deg`,
  `calib_roll_deg`, `siren_*`, `rear_clearance_m`, `latent_novelty`,
  `surprise_sigma_delta`, `static_skip_ratio`, `token_sparsity_ratio`,
  `kv_cache_reuse_ratio`, `effective_speedup_x`, `gaussian_count`, and
  `reality_view_bytes`.
- **P2** (`V15HudOverlay.hpp:10-33`, `V15HudOverlay.cpp:10-38`): Defines named
  layout/UI descriptor constants (`kRealityTexW = 320`, `kRealityTexH = 180`,
  `kRealityUiSetIdx = 8`, `kTotalUiSetCount = 9`) and trivially-copyable
  `formatV15HudTelemetry` (4 fixed `std::array` buffers, `0` heap allocs).
- **P3** (`VulkanViewer.hpp:20-25,144`, `VulkanViewer.cpp:204-296,636,1419-1485`):
  Pre-allocates `realityImg_` (`320x180 RGBA8`) + 9th UI descriptor set
  `uiSets_[8]`, adds `ViewMode::RealityView` (`V` key + `"v15 Real"` button),
  uploads `v15RealityRgba` after `vkResetFences`, and renders a bottom-center
  PIP card or full Reality View canvas plus 4-line telemetry strip ONLY when
  `perc.v15.active` is true.

## 3. Trace
- **T1 (`v14`)**: `perc.v15.active` is `false`; `TraceLogger` omits
  `root["v15"]`, `VulkanViewer::renderFrame` skips `realityImg_.uploadRGBA8`,
  and `buildUiOverlay` skips the Reality View card/PIP completely.
- **T2 (`v15`)**: `V15Pipeline::finishBackEnd` exports `v15Gaussians` and
  `320x180` EWA `v15RealityRgba`; `TraceLogger` logs `v15.*` JSONL;
  `VulkanViewer` displays live splatted Gaussian world + multimodal metrics.

## 4. Claims
- **C-EQ**: With `pipeline_version: "v14"`, `TraceLogger` emits identical
  JSONL schema (`!root.isMember("v15")`), and all `96` pre-v15 tests pass.
- **C-V15**: `TraceLogger` accurately serializes all 19 `v15` keys,
  `V15HudTelemetryLines` is trivially copyable (`0` heap allocs) and formats
  `GEAR:D`/`GEAR:R` and `SIREN:CLEAR`/`SIREN:YIELD`, and `V15Pipeline` exports
  non-empty `320x180x4` RGBA splat pixels in Vulkan.

## 5. Evidence
- `V15TelemetryAndHudTest.*:PlanningAndTrackingTest.*`: **7/7 PASS**
  (`TraceLoggerV15SchemaAndV14Equivalence`,
  `V15HudOverlayFormatsDriveReverseAndSiren`,
  `V15PipelineExportsRealityViewAndGaussians`).
- Full CTest gate (`ctest --test-dir build --output-on-failure`):
  **148/148 PASS (100%, 90.78 sec)** (`96` v14 + `52` v15 tests).

## 6. Known limits
- Interactive window visuals were verified via compilation, zero-warning linking,
  descriptor binding audit, and headless GPU Reality View RGBA readback.
