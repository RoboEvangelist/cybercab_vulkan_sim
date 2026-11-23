# Step 21 — All-Resolution Validation, FMEA & Descriptor Leak Fix

## 1. Scope
- **Modified (core)**: `src/core/VulkanPipeline.hpp:46`,
  `src/core/VulkanPipeline.cpp:122-129,154` (track allocated compute
  descriptor sets at setup and call `vkFreeDescriptorSets` in `destroy()`).
- **Modified (tests/build)**:
  `tests/test_v15_cybercab_camera_res.cpp:189` (added `"full"` `640x480`),
  `CMakeLists.txt:177-179`.
- **Added**: `tests/test_all_res_v14_v15_validation.cpp` (`175` lines),
  `tests/test_all_res_fmea_safety.cpp` (`144` lines),
  `tests/test_all_res_fmea_env_and_sync.cpp` (`133` lines).
- **Out of scope**: Default `config/cybercab_sim.yaml` keeps
  `pipeline_version: "v14"`, `initial_res_mode: "half"`, and
  `rtdetr.max_det_range_m: 65.0`.

## 2. Premises
- **P1** (`VulkanContext.cpp:250-256`, `VulkanPipeline.hpp:46`,
  `VulkanPipeline.cpp:120-156`): Shared descriptor pool uses
  `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` and `maxSets=512`.
  Freeing `allocatedSets_` in `destroy()` prevents `VK_ERROR_OUT_OF_POOL_MEMORY`
  across 30+ `Rig` lifecycles with zero hot-path allocations.
- **P2** (`test_all_res_v14_v15_validation.cpp:60-173`,
  `test_v15_cybercab_camera_res.cpp:188-193`): Validates `"half"` (`320x240`),
  `"full"` (`640x480`), `"cybercab_binned"` (`1448x938`), and `"cybercab"`
  (`2896x1876`) in `v14` (PR/AP, RMSE, dynamic voxel precision, corroboration)
  and `v15` `"full"`, plus zero-collision hazards (`jaywalk`, `occluded`,
  `hardbrake`, `siren`).
- **P3** (`test_all_res_fmea_safety.cpp:74-142`): Tests single-cam soiling and
  all-front `SENSOR_SOILED_MRM` (`v14`/`v15`), empty-road zero hallucinations
  (detections/voxels/Gaussians/AEB) for `v14`/`v15`, and curved braking-pitch
  single-truck no-split across all 4 presets.
- **P4** (`test_all_res_fmea_env_and_sync.cpp:51-129`): Tests `night_glare` +
  RCCB finite bounds (via direct `VulkanBuffer::download` first/last `Vec4`),
  online knock pitch/roll self-calib, 4-way resize sync (`mMot<0.01`,
  `v14`/`v15`), and `ByteTracker` + `TrajectoryPlanner` dropout expiry.

## 3. Trace
- **T1 (`v14`)**: `allocatedSets_` only mutates during pipeline init/destroy;
  hot-path dispatch and all `96` pre-v15 oracle tests are unchanged, and all 4
  resolution presets pass `v14` PR/AP, voxel, hazard, soiling, empty-road,
  braking-pitch, and resize checks.
- **T2 (`v15`)**: All 4 resolution presets pass `1c_v15_freq_split`,
  `5_v15_static_mem`, `6_v15_gaussian_splat`, self-calib, HDR RCCB glare,
  `SENSOR_SOILED_MRM`, zero-collision hazards, and resize/dropout safety.

## 4. Claims
- **C-LEAK**: Repeated `Rig` init/destroy cycles (`30+` in one context) never
  exhaust `maxSets=512` descriptor sets.
- **C-ALLRES-FMEA**: Both `v14` and `v15` satisfy PR/AP, voxel, corroboration,
  hazard, and FMEA gates across `"half"`, `"full"`, `"cybercab_binned"`, and
  `"cybercab"`.
- **C-EQ**: Full suite passes **100% (`176/176`)**.

## 5. Evidence
- Targeted 7-test XML suite (**7/7 PASS, `96.666 s`**):
  - `V15CybercabCameraResTest` (`2/2`): `native_2896x1876_gpu_ms=33.7521`,
    `native_2896x1876_wall_ms=51.6613`, `native_calib_pitch_deg=1.4529`,
    `native_calib_roll_deg=-1.2832`, `binned_1448x938_gpu_ms=11.7866`.
  - `AllResValidationTest.V14AllFourResolutionsAndV15FullPrVoxelCorrob`
    (`4.939 s`): `v14_half_ap=0.5241` (`vox_p=0.8236`, `rmse=1.3493`),
    `v14_full_ap=0.3961` (`vox_p=0.8254`, `rmse=1.3744`),
    `v14_cybercab_binned_ap=0.4441` (`vox_p=0.8513`, `rmse=1.4246`),
    `v14_cybercab_ap=0.5000` (`vox_p=0.8244`, `rmse=1.1997`),
    `v15_full_ap=0.5580` (`vox_p=0.7350`, `rmse=1.5338`,
    `gpu_ms=6.0935`, `wall_ms=13.0054`).
  - `AllResValidationTest.AllFourResolutionsClosedLoopHazardsZeroCollisions`
    (`34.339 s`): PASS across all 4 presets.
  - `AllResFmeaTest` (`2/2`, `33.701 s`): soiling/MRM, empty-road zero
    hallucinations, and curved braking-pitch single-truck no-split PASS.
  - `AllResFmeaEnvTest.FmeaNightGlareSelfCalibResizeSyncAndDropoutAllRes`
    (`11.238 s`): `half` (`1.2763`/`-1.1560`), `full` (`1.3473`/`-1.1540`),
    `cybercab_binned` (`1.3013`/`-1.2855`), `cybercab` (`1.3226`/`-1.1353`)
    calib deg PASS.
- Full CTest gate (`ctest --test-dir /home/pototo/VulkanTest/build --output-on-failure`):
  **176/176 PASS (100%, `207.35 sec`)** (`96` v14 oracle + `80` v15/all-res/FMEA tests).

## 6. Known limits
- Native `2896x1876` (`gpu=33.75 ms`, `wall=51.66 ms`) exceeds interactive
  `27.8 ms` (`36 Hz`) budget (`"half"` remains default); synthetic Vulkan optics.
