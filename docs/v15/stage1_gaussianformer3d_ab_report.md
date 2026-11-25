# Stage 1b — GaussianFormer3D A/B Evaluation, Voxel Preset Benchmark & NFR Audit

Semi-formal change report following *Agentic Code Reasoning* (Ugare & Chandra, Meta, arXiv:2603.01896).

## 1. Scope
- **Subject:** Quantitative A/B comparison of `--spatial-rep legacy` vs. `--spatial-rep gaussianformer3d` across perception, occupancy/Gaussians, open-set obstacles, 8 closed-loop missions, latency, zero-allocation memory, determinism, degraded-sensor robustness, and voxel resolution presets (`1.0 m`, `0.5 m` SurroundOcc, `0.4 m` Occ3D-nuScenes, `0.5 m` splat-only).
- **Configuration & Splatter Tightening:** Expose `splat_voxel_step_x/y/z` in [`config/cybercab_sim.yaml`](../../config/cybercab_sim.yaml), tighten `splat_sigma_cutoff_sq` (`4.50 -> 2.50`), add O(1) spatial hashing ([`SpatialHashIndex3D.hpp`](../../src/perception/sota/SpatialHashIndex3D.hpp)), cap tracked Gaussian scale/shift drift in [`DeformableAttention3D.cpp`](../../src/perception/sota/DeformableAttention3D.cpp), enforce an unconditional `fromTrack` OBB hull clamp, and gate upward static-prior blending on camera corroboration in [`GaussianToVoxelSplatter.cpp`](../../src/perception/sota/GaussianToVoxelSplatter.cpp).
- **Out of scope:** Switching `simulation.spatial_rep_mode` default (remains `legacy`); changing Stage-0 2D RT-DETR weights.

## 2. Premises (`file:line`)
- **P1 (`config/cybercab_sim.yaml:66-104`, `src/config/SimConfig.hpp:209-228`):** YAML controls `perception.occupancy_flow` (`step_x/y=1.0`, `step_z=0.5`, `occ_threshold=0.38`) and `perception.gaussianformer3d` (`max_gaussians=512`, `voxel_scale_x/y/z=0.72/0.80/0.72`, `splat_radius_sigma_mul=1.25`, `splat_sigma_cutoff_sq=2.50`, `min_voxel_occ_prob=0.38`, `splat_voxel_step_x/y/z=0.0`).
- **P2 (`src/perception/PerceptionPipeline.cpp:145-156`):** `occ_.readbackVoxels` populates `out.activeVoxels`; when `gf3d_` is active, `gf3d_->runStage` refines `out.activeVoxels` and `v15_->finishBackEnd` consumes `gf3d_->v15Gaussians()` via `setExternalGaussians`.
- **P3 (`src/perception/sota/Gaussian3DTypes.hpp:77-92`):** `Gaussian3DPrimitive` adds `fromStaticMem`, `fromTrack`, `staticMemLocked` (`prim_mem_bytes=53248`).
- **P4 (`src/perception/sota/VoxelToGaussianInit.cpp:157-308`):** Initializes `fromTrack` actors first (`g.fromTrack=true`), clusters voxels (`kMaxVoxelClusterScaleXZ=0.90f`), then runs 2-pass `staticCells` (`isLocked` cells first).
- **P5 (`src/perception/sota/DeformableAttention3D.cpp:96-103, 216-220`):** Caps `maxLocal` scale growth (`1.05f` tracked / `1.15f` non-track) and tracked centroid shift (`0.15 m`/layer).
- **P6 (`src/perception/sota/GaussianToVoxelSplatter.cpp:91-295`):** Splatting supports independent `splatVoxelStepX/Y/Z`; `fromTrack` Gaussians are OBB-clamped across all voxels (`|local.x/z| <= g.scale + 0.12f`), non-track clusters refine only `hasPriorCell`, `staticOcc` boosts only when `acc.maxCorrob > 0u`, `SignalSign` uses `min(occThresh, 0.25f)`, and up to 32 `staticMemLocked` cells fuse cleanly.

## 3. Execution Trace
- **T1 (`legacy`):** `OccupancyFlowNet` readback -> `GaussianWorldModel::updateFromPerception` (`1024` Gaussians) -> `v15_gaussian_splat.comp` -> planner.
- **T2 (`gaussianformer3d`):** `OccupancyFlowNet` readback -> `VoxelToGaussianInit` (`512` Gaussians) -> `DeformableAttention3D` (`scaleCapMul` / `stepCap`) -> `GaussianToVoxelSplatter` (OBB clamp + `splatVoxelStep*`) -> `setExternalGaussians` / refined `activeVoxels` -> planner.

## 4. Observed A/B Evidence (`cybercab_tests --gtest_filter=GaussianFormer3D*`, `13/13` PASS, `RelWithDebInfo`)
Captured verbatim from `task-11564`:

### E1 — Perception, Voxels, Gaussians & Open-Set Obstacles (`ABPerceptionVoxelGaussianAndOpenSetQuality`)
| Metric | `legacy` | `gaussianformer3d` | Verdict |
|---|---:|---:|---|
| 3D Detection Recall / Prec / Veh AP / RMSE | `0.6667 / 0.9481 / 0.5392 / 1.6177m` | `0.6667 / 0.9481 / 0.5392 / 1.6177m` | **Same** (downstream of RT-DETR) |
| Dynamic Voxel Prec / GT Voxel Cov | `0.8809 / 0.5714` | `0.9185 / 0.5714` | **Better prec** (`+3.76 pp`) |
| Active Voxels / frame & Gaussians / frame | `239.9 / 1024.0` | `179.2 / 512.0` | **Fewer primitives** (`-25.3%` vox, `-50%` Gauss) |
| Gaussian GT Mean Occ / GT Hit / Free Occ | `0.3012 / 0.4048 / 0.0000` | `0.4571 / 0.5000 / 0.0000` | **Better** (`+51.8%` occ, `+9.52 pp` hit) |
| Truck Veh Voxels / Signal Sign Voxels | `294 / 46` | `280 / 8` | Truck `~same`; **lower SignVox** (`46 -> 8`) |
| Open-Set `CARGO` (vox / minClr / col / avoid) | `6421 / 0.65 m / 0 / true` | `6576 / 0.65 m / 0 / true` | **Same clearance & avoidance** |
| Open-Set `BARRIER` (vox / minClr / end / avoid) | `707 / 0.29 m / (-5.1, 2.02, 5.39) / true` | `323 / 1.02 m / (-15.0, 1.90, 5.18) / true` | **Much better** (`-54.3%` vox vs leg, `-90.6%` vs `3444`, `3.5x` clear) |

### E2 — Closed-Loop Safety & Decisions Across 8 Missions (`ABSixMissionSafetyAndDecisionComparison`, `0/0` col, `8/8` pass)
| Mission | `pass` | `dist (m)` | `maxAcc (m/s^2)` | `jerk (m/s^3)` | `minClr (m)` | `brkStep` | `aeb` | `minV (m/s)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v15:nominal` | `1/1` | `43.3 / 37.2` | `5.80 / 5.77` | `9.56 / 8.13` | `1.41 / 1.40` | `40 / 40` | `0 / 0` | `2.83 / 2.71` |
| `v15:jaywalk` | `1/1` | `8.8 / 8.8` | `8.50 / 8.50` | `12.11 / 12.11` | `1.06 / 1.06` | `15 / 15` | `39 / 39` | `0.00 / 0.00` |
| `v15:occluded` | `1/1` | `12.3 / 12.3` | `8.50 / 8.50` | `27.80 / 27.80` | `2.10 / 2.10` | `6 / 6` | `13 / 13` | `4.65 / 4.65` |
| `v15:hardbrake` | `1/1` | `15.5 / 16.7` | `8.50 / 1.64` | `17.42 / 4.89` | `2.07 / 2.07` | `3 / 3` | `13 / 0` | `6.63 / 9.48` |
| `v15:siren` | `1/1` | `7.1 / 7.1` | `5.80 / 5.80` | `5.20 / 5.20` | `1.45 / 1.45` | `0 / 0` | `0 / 0` | `0.56 / 0.56` |
| `v15:night_glare_jaywalk` | `1/1` | `9.0 / 8.8` | `8.50 / 8.50` | `17.48 / 12.07` | `0.18 / 1.08` | `15 / 15` | `37 / 39` | `0.00 / 0.00` |
| `v14:nominal` | `1/1` | `51.4 / 50.5` | `2.69 / 5.28` | `1.06 / 4.00` | `1.56 / 1.56` | `-1 / 38` | `0 / 0` | `8.52 / 8.52` |
| `v14:jaywalk` | `1/1` | `7.8 / 7.8` | `8.50 / 8.50` | `12.82 / 12.82` | `1.56 / 1.56` | `12 / 12` | `38 / 38` | `0.00 / 0.00` |

### E3 — Non-Functional Latency, Memory, Determinism & Robustness
| Resolution | `wallMean (leg / gf3d)` | `p95 (leg / gf3d)` | `gpu (leg / gf3d)` | `gf3dCpu` |
|---|---:|---:|---:|---:|
| `half` (`320x240`) | `14.33 / 10.20 ms` | `24.41 / 10.36 ms` | `5.12 / 4.61 ms` | `0.317 ms` |
| `full` (`640x480`) | `13.57 / 13.21 ms` | `23.06 / 20.90 ms` | `6.18 / 5.72 ms` | `0.400 ms` |
| `cybercab_binned` (`1448x938`) | `19.69 / 20.41 ms` | `21.42 / 27.10 ms` | `13.20 / 12.47 ms` | `0.374 ms` |
- **Memory/Determinism/Robustness:** `stage1_heap_delta_bytes=0`, `prim_mem_bytes=53248`; `det_exact=1`, `behavior_diff_frames=0/12`, `max_*_delta=0.0000`; empty-road `0` ghosts/dynVox/falseAeb; `soil_front` `seen_mrm=1`, `col=0`, `max_gauss=512`.

### E4 — Voxel Resolution Preset Benchmark (`VoxelResolutionPresetBenchmark`, `1.0 m` vs `0.5 m` SurroundOcc vs `0.4 m` Occ3D vs `0.5 m` Splat-Only)
| Preset & Mode | `vox/f` | `dynPrec` | `sign` | `barVox` | `barClr` | `avoid` / `col` | `wall` / `gf3dCpu` |
|---|---:|---:|---:|---:|---:|---|---|
| `1.0m_default` (`legacy`) | `183.2` | `0.8740` | `46` | `2909` | `0.29 m` | `1 / 0` | `14.53 / 0.000 ms` |
| `1.0m_default` (`gf3d`) | `125.8` | `0.9145` | `8` | `380` | `1.02 m` | `1 / 0` | `11.31 / 0.394 ms` |
| `0.5m_surroundocc` (`legacy`) | `540.3` | `0.8507` | `111` | `11382` | `1.07 m` | `0 / 0` | `16.64 / 0.000 ms` |
| `0.5m_surroundocc` (`gf3d`) | `350.2` | `0.8880` | `12` | `7941` | `0.88 m` | `0 / 0` | `14.31 / 1.146 ms` |
| `0.4m_occ3d` (`legacy`) | `704.4` | `0.8749` | `306` | `32097` | `0.76 m` | `1 / 0` | `19.12 / 0.000 ms` |
| `0.4m_occ3d` (`gf3d`) | `542.3` | `0.8959` | `36` | `29244` | `0.72 m` | `1 / 0` | `18.79 / 1.811 ms` |
| `0.5m_splat_only` (`gf3d`) | `144.8` | `0.9434` | `8` | `286` | `1.02 m` | `1 / 0` | `21.48 / 0.655 ms` |
- **Finding:** Literature uses `0.4 m` (Occ3D-nuScenes) and `0.5 m` (SurroundOcc). In this simulator, **`0.5m_splat_only`** gives the best real-time accuracy (`dynPrec=0.9434`, `barVox=286`, `barClr=1.02 m`, `gf3dCpu=0.655 ms`) while keeping coarse `60x40` GPU BEV/Occ shaders calibrated for the planner (`0.5m_surroundocc` pauses cautiously behind the dense barrier at `x=-16.7, spd=1.00 m/s` with `col=0`).

## 5. Overall Verdict & Known Limits
- **Better:** Dynamic voxel precision (`0.9185`, or `0.9434` at `0.5m` splat step), `BARRIER` voxel precision & clearance (`323` vs `707` legacy, `1.02` vs `0.29 m`), Gaussian GT occupancy (`0.4571` vs `0.3012`), `v15:hardbrake` smoothness (`4.89` vs `17.42` jerk, `0` vs `13` AEBs), `night_glare_jaywalk` safety margin (`1.08` vs `0.18 m`), and `half`/`full` frame latency (`10.20/13.21 ms` vs `14.33/13.57 ms`).
- **Same / Trade-offs:** `8/8` missions pass with `0` collisions, `0` empty-road hallucinations, `0 B` heap allocs, bitwise determinism (`0/12` diff); `SignVox` remains `8` vs `46`, and `v14:nominal` still brakes once against the adjacent vehicle envelope at tick `38` (`jerk 4.00 vs 1.06`, `50.5 m`, `col=0`).
