# Stage 1b — GaussianFormer3D A/B Evaluation, YAML Voxel/Volume Tuning & NFR Audit

Semi-formal change report following *Agentic Code Reasoning* (Ugare & Chandra, Meta, arXiv:2603.01896).

## 1. Scope
- **Subject:** Quantitative A/B comparison of `--spatial-rep legacy` vs. `--spatial-rep gaussianformer3d` across perception, occupancy/Gaussians, open-set obstacles, 8 closed-loop missions, latency, zero-allocation memory, determinism, and degraded-sensor robustness.
- **Configuration & Optimization:** Expose voxel grid (`perception.occupancy_flow`) and Gaussian splat volume (`perception.gaussianformer3d`) parameters in [`config/cybercab_sim.yaml`](../../config/cybercab_sim.yaml), add O(1) spatial hashing ([`SpatialHashIndex3D.hpp`](../../src/perception/sota/SpatialHashIndex3D.hpp)), fix camera-ray depth scaling, protect locked `SignalSign` / static priors, and tighten non-track cluster XZ extents.
- **Out of scope:** Switching `simulation.spatial_rep_mode` default (remains `legacy`); changing Stage-0 2D RT-DETR weights.

## 2. Premises (`file:line`)
- **P1 (`config/cybercab_sim.yaml:70-101`, `src/config/SimConfig.hpp:128-130`):** YAML controls `perception.occupancy_flow` (`step_x/y=1.0`, `step_z=0.5`, `occ_threshold=0.38`) and `perception.gaussianformer3d` (`max_gaussians=512`, `voxel_scale_x/y/z=0.72/0.80/0.72`, `splat_radius_sigma_mul=1.25`, `splat_sigma_cutoff_sq=4.50`, `min_voxel_occ_prob=0.38`).
- **P2 (`src/perception/PerceptionPipeline.cpp:145-156`):** `occ_.readbackVoxels` populates `out.activeVoxels`; when `gf3d_` is active, `gf3d_->runStage` refines `out.activeVoxels` and `v15_->finishBackEnd` consumes `gf3d_->v15Gaussians()` via `setExternalGaussians`.
- **P3 (`src/perception/sota/Gaussian3DTypes.hpp:77-89`):** `Gaussian3DPrimitive` adds `fromStaticMem`, `fromTrack`, `staticMemLocked` inside existing padding (`sizeof` unchanged, `prim_mem_bytes=53248`).
- **P4 (`src/perception/sota/VoxelToGaussianInit.cpp:157-297`, `SpatialHashIndex3D.hpp:13-73`):** Initializes `fromTrack` actors first, voxel clusters via `O(1)` spatial hash (`kMaxVoxelClusterScaleXZ=0.90f`), then 2-pass static memory (`isLocked` cells first without duplicate suppression).
- **P5 (`src/perception/sota/DeformableAttention3D.cpp:149-261`):** Refines non-static-memory Gaussians using multi-camera 2D projections (`pcMean.z` ray depth) and `voxelHash_` 3D geometry lookup.
- **P6 (`src/perception/sota/GaussianToVoxelSplatter.cpp:91-279`):** Preserves priors; splats `fromTrack` Gaussians across free cells while confining non-track clusters to prior cells (`hasPriorCell`), relaxes `SignalSign` threshold (`min(occThresh, 0.25f)`), and fuses up to 32 `staticMemLocked` voxels.
- **P7 (`src/planning/v15/AutoregSafety.cpp:125-173`, `AutoregressivePolicy.cpp:159-253`):** Consumes `out.activeVoxels` and `out.v15Gaussians` for rollout cost and RSS/AEB shielding.

## 3. Execution Trace
- **T1 (`legacy`):** `OccupancyFlowNet` readback -> `v15_` `GaussianWorldModel::updateFromPerception` (`kMaxGaussians=1024`, direct locked static-cell append) -> `v15_gaussian_splat.comp` -> planner.
- **T2 (`gaussianformer3d`):** `OccupancyFlowNet` readback -> `VoxelToGaussianInit` (`max_gaussians=512`, 2-pass `staticMem`) -> `DeformableAttention3D` (`num_refinement_layers=2`) -> `GaussianToVoxelSplatter` (`voxel_scale_x/y/z`, `splat_radius_sigma_mul`) -> `v15_->setExternalGaussians` (or refined `activeVoxels` in `v14`) -> planner.

## 4. Claims & Observed A/B Evidence (`cybercab_tests --gtest_filter=GaussianFormer3D*`, 12/12 PASS)
All metrics below were captured directly under `RelWithDebInfo` (`-O2 -g`, `task-11365`, 12/12 PASS: `GaussianFormer3DUnitTest` 5/5, `GaussianFormer3DGpuTest` 3/3, `GaussianFormer3DAbMetricsTest` 1/1, `GaussianFormer3DAbMissionsNfrTest` 2/2, `GaussianFormer3DAbRobustTest` 1/1).

### E1 — Perception, Voxels, Gaussians & Open-Set Obstacles (`ABPerceptionVoxelGaussianAndOpenSetQuality`)
| Metric | `legacy` | `gaussianformer3d` | Delta / Verdict |
|---|---:|---:|---|
| 3D Detection Recall / Prec / Veh AP | `0.6667 / 0.9481 / 0.5392` | `0.6667 / 0.9481 / 0.5392` | **Same** (Stage 1 downstream of RT-DETR) |
| Track Position RMSE (m) | `1.6177` | `1.5879` | **Slightly better** (`-0.0298 m`) |
| Dynamic Voxel Prec / GT Voxel Cov | `0.8809 / 0.5714` | `0.9071 / 0.5714` | **Better prec** (`+2.62 pp`), same cov |
| Active Voxels / frame & Gaussians / frame | `239.9 / 1024.0` | `190.0 / 512.0` | **Fewer primitives** (`-20.8%` vox, `-50%` Gauss) |
| Gaussian GT Mean Occ / GT Hit Rate / Free Occ | `0.3012 / 0.4048 / 0.0000` | `0.5421 / 0.5714 / 0.0000` | **Much better** (`+80.0%` occ, `+16.66 pp` hit, `0` free occ) |
| Truck Veh Voxels / Signal Sign Voxels | `294 / 46` | `286 / 8` | **Worse SignVox** (`46 -> 8`, non-zero); Truck `~same` |
| Open-Set `CARGO` (vox / min clear / col / avoided) | `6421 / 0.65 m / 0 / true` | `6606 / 6.76 m / 0 / true` | **Safer clearance** (`+6.11 m`) |
| Open-Set `BARRIER` (vox / min clear / endXY / col / avoided) | `707 / 0.29 m / (-5.1, 2.02, 5.39)` | `3444 / 0.93 m / (-5.1, 2.15, 8.27)` | **Safer clearance** (`+0.64 m`), more dense barrier vox |

### E2 — Closed-Loop Safety & Decisions Across 8 Missions (`ABSixMissionSafetyAndDecisionComparison`, `0/0` collisions, `8/8` pass)
| Mission | `pass` | `dist (m)` | `maxAcc (m/s^2)` | `jerk (m/s^3)` | `minClr (m)` | `brkStep` | `aeb` | `minV (m/s)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v15:nominal` | `1/1` | `43.3 / 41.9` | `5.80 / 5.77` | `9.56 / 6.30` | `1.41 / 1.53` | `40 / 40` | `0 / 0` | `2.83 / 5.50` |
| `v15:jaywalk` | `1/1` | `8.8 / 8.8` | `8.50 / 8.50` | `12.11 / 12.11` | `1.06 / 1.06` | `15 / 15` | `39 / 39` | `0.00 / 0.00` |
| `v15:occluded` | `1/1` | `12.3 / 9.5` | `8.50 / 5.78` | `27.80 / 17.19` | `2.10 / 2.10` | `6 / 4` | `13 / 0` | `4.65 / 2.46` |
| `v15:hardbrake` | `1/1` | `15.5 / 13.6` | `8.50 / 5.74` | `17.42 / 11.82` | `2.07 / 2.07` | `3 / 3` | `13 / 0` | `6.63 / 6.95` |
| `v15:siren` | `1/1` | `7.1 / 7.1` | `5.80 / 5.80` | `5.20 / 5.20` | `1.45 / 1.45` | `0 / 0` | `0 / 0` | `0.56 / 0.56` |
| `v15:night_glare_jaywalk` | `1/1` | `9.0 / 8.8` | `8.50 / 8.50` | `17.48 / 12.07` | `0.18 / 1.08` | `15 / 15` | `37 / 39` | `0.00 / 0.00` |
| `v14:nominal` | `1/1` | `51.4 / 50.0` | `2.69 / 5.56` | `1.06 / 4.26` | `1.56 / 1.56` | `-1 / 37` | `0 / 0` | `8.52 / 8.52` |
| `v14:jaywalk` | `1/1` | `7.8 / 7.8` | `8.50 / 8.50` | `12.82 / 12.82` | `1.56 / 1.56` | `12 / 12` | `38 / 38` | `0.00 / 0.00` |

### E3 — Non-Functional Latency, Memory, Determinism & Robustness (`RelWithDebInfo`)
| Resolution | `wallMean (leg / gf3d)` | `p50 (leg / gf3d)` | `p95 / max (leg / gf3d)` | `gpu (leg / gf3d)` | `gf3dCpu` |
|---|---:|---:|---:|---:|---:|
| `half` (`320x240`) | `14.07 / 10.58 ms` | `16.01 / 10.66 ms` | `18.20 / 12.38 ms` | `8.01 / 4.84 ms` | `0.343 ms` |
| `full` (`640x480`) | `13.01 / 11.76 ms` | `11.80 / 11.38 ms` | `22.95 / 14.36 ms` | `6.00 / 5.68 ms` | `0.328 ms` |
| `cybercab_binned` (`1448x938`) | `19.93 / 18.48 ms` | `19.90 / 19.06 ms` | `21.32 / 19.66 ms` | `11.83 / 11.68 ms` | `0.540 ms` |
- **Memory & Determinism:** `stage1_heap_delta_bytes=0`, `prim_mem_bytes=53248`; `det_exact=1`, `behavior_diff_frames=0/12`, `max_spd_delta=0.0000m/s`, `max_steer_delta=0.0000rad`, `max_rss_delta=0.0000m`.
- **Empty-Road & Soiled Sensors:** Both modes produce `ghostVeh=0 ghostPed=0 dynVox=0 dynGauss=0 falseAeb=0`; 3-camera front soiling under `gaussianformer3d` triggers `SENSOR_SOILED_MRM` (`seen_mrm=1`, `col=0`, `max_gauss=512`).
- **YAML Config Check (`YamlVoxelAndVolumeParamsControlSplatResolution`):** Halving `occStepX/Y` + scaling `voxel_scale_*` increases splatted `activeVoxels` (`PASS`).

## 5. Overall Verdict & Known Limits
- **Better on `v15` spatial representation, safety comfort & latency:** Higher Gaussian GT occupancy (`0.301 -> 0.542`) and hit rate (`40.5% -> 57.1%`) with `0` free-space occupancy, higher dynamic voxel precision (`88.1% -> 90.7%`), larger `CARGO`/`BARRIER` clearances (`0.65 -> 6.76 m`, `0.29 -> 0.93 m`), zero AEB spikes in `occluded`/`hardbrake` (`13 -> 0`), lower jerk across 4 `v15` missions, `0.18 -> 1.08 m` pedestrian clearance in `night_glare_jaywalk`, and faster end-to-end wall/p95 frame times across `half`/`full`/`cybercab_binned` (`gf3dCpu` only `0.328–0.540 ms` under `RelWithDebInfo`).
- **Same:** Stage-0 detection precision/recall/AP, zero collisions (`8/8` missions), empty-road `0` ghosts, zero steady-state allocations, and bitwise determinism.
- **Worse / Trade-offs:** `SignVox` is lower (`46 -> 8`), open-set `Barrier` produces more static voxels (`707 -> 3444`), and `v14:nominal` brakes earlier (`brkStep -1 -> 37`) with higher peak accel/jerk (`2.69/1.06 -> 5.56/4.26`).
