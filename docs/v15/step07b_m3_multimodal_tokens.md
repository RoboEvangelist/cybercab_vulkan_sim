# Step 7b: M3b 100 Hz Kinematics, Lane-Center Nav & 5x5 Plucker `PatchTokenizer`

## Scope
Implement 100 Hz IMU ring + lane-center vectorized navigation (`KinematicNavEncoder`), partition 10 `160x120` virtual views into `32x24` `5x5` Plucker patches in a 4-frame `deltaT` GPU ring buffer (`PatchTokenizer`), and wire M3 into `V15Pipeline`.

## Premises
- **P1** `KinematicNavEncoder.cpp:20-87`: substeps control ticks to `100 Hz` in a `32`-sample ring and encodes `8` centerline lookahead waypoints centered at `+/-1.95 m` (`dividerClearanceM = 0.99 m`, `outerLimitClearanceM = 1.32 m`).
- **P2** `shaders/v15_patch_tokenize.comp:34-84`: averages valid virtual pixels (`meanR, meanG, meanB, validFrac`) and bands (`meanLow, rmsHigh, meanGx, meanGy`), projects upright Plucker `(d_veh, cross(o_veh, d_veh))`, and increments active history slot ages (`pluckerM.w += dt`).
- **P3** `V15Pipeline.cpp:69-87`: `beginFrameCpu` updates 48 kHz audio + 100 Hz kinematics/nav; `recordFrontEnd` dispatches `"3_v15_patch_tokens"`; `finishBatch1Cpu` records ego pose per ring slot.

## Trace
- **T-v14:** `v15_ == nullptr` -> zero M3 allocations or dispatches (`V14NeverAllocatesOrRunsV15FrontEnd` PASS).
- **T-v15:** `VirtualCameraRectifier` -> `PatchTokenizer::recordDispatch` writes `7,680` `64B` tokens to `headSlot` and ages older slots in `ringSSBO()`.

## Claims & Evidence (`V15KinematicNavTest`, `V15PatchTokenTest`, `V15PipelineTest` 5/5 PASS)
- **C-EQ:** v14 pipeline runs zero M3 stages (`121` prior `ctest` suite untouched).
- **C-V15-1 (`PluckerInvariantsCrossViewEpipolarAndRingAges`):** across `7,680` tokens, `plucker_max_ortho_err = 0.0000` (`|d.m| < 1e-5`), `plucker_epipolar_recip = 0.0008` (`< 2.5e-3`), `> 5500` interior valid patches, exact `deltaT` progression `{2*dt, dt, 0}`.
- **C-V15-2 (`Substeps100HzRingAndTargetsLaneCenter`):** 100 Hz sub-stepping + lane-center waypoints keep `dividerClearanceM > 0.90 m`, `outerLimitClearanceM > 1.20 m`.
- **C-V15-3 (`V15PipelineTest`):** `v15_gpu_sum_ms = 4.6458 ms`, `v15_wall_ms = 7.6355 ms` (< `27.8 ms` 36 Hz target).

## Known Limits
- M3 tokens and nav/audio outputs are ready for M5 (`StaticSpatialMemory`), M4 (`LatentPredictor`), and M7 (`AutoregressivePolicy`).
