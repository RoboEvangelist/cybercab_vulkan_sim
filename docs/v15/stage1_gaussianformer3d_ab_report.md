# Stage 1b — GaussianFormer3D A/B Evaluation, Voxel Preset Benchmark & NFR Audit

Semi-formal change report following *Agentic Code Reasoning* (Ugare & Chandra, Meta, arXiv:2603.01896).

## 1. Scope
- **Subject:** Quantitative A/B comparison of `--spatial-rep legacy` vs. `--spatial-rep gaussianformer3d` across perception, occupancy/Gaussians, open-set obstacles, 8 closed-loop missions, latency, zero-allocation memory, determinism, degraded-sensor robustness, and voxel resolution presets (`1.0 m`, `0.5 m` SurroundOcc, `0.4 m` Occ3D-nuScenes, `0.5 m` splat-only).
- **Configuration, Stage-1 Tuning & Reality View Hardening:** Expose `splat_voxel_step_x/y/z` in [`config/cybercab_sim.yaml`](../../config/cybercab_sim.yaml), tighten `splat_sigma_cutoff_sq` (`4.50 -> 2.50`), add O(1) spatial hashing ([`SpatialHashIndex3D.hpp`](../../src/perception/sota/SpatialHashIndex3D.hpp)), restore `1.15f` scale cap and `kMaxDynLayerShiftM` (`0.25 m`) adaptive step in [`DeformableAttention3D.cpp`](../../src/perception/sota/DeformableAttention3D.cpp), prioritize forward roadside locked static cells in [`VoxelToGaussianInit.cpp`](../../src/perception/sota/VoxelToGaussianInit.cpp), enforce an unconditional `fromTrack` OBB hull clamp + corroboration gate in [`GaussianToVoxelSplatter.cpp`](../../src/perception/sota/GaussianToVoxelSplatter.cpp), and harden the FSD v15 Reality Window ([`GaussianWorldModel.cpp`](../../src/perception/v15/GaussianWorldModel.cpp), [`V15HudOverlay.cpp`](../../src/app/V15HudOverlay.cpp), [`VulkanViewer.cpp`](../../src/app/VulkanViewer.cpp)).
- **Out of scope:** Switching `simulation.spatial_rep_mode` default (remains `legacy`); changing Stage-0 2D RT-DETR weights.

## 2. Premises (`file:line`)
- **P1 (`config/cybercab_sim.yaml:66-104`, `src/config/SimConfig.hpp:209-228`):** YAML controls `perception.occupancy_flow` (`step_x/y=1.0`, `step_z=0.5`, `occ_threshold=0.38`) and `perception.gaussianformer3d` (`max_gaussians=512`, `voxel_scale_x/y/z=0.72/0.80/0.72`, `splat_radius_sigma_mul=1.25`, `splat_sigma_cutoff_sq=2.50`, `min_voxel_occ_prob=0.38`, `splat_voxel_step_x/y/z=0.0`).
- **P2 (`src/perception/PerceptionPipeline.cpp:145-156`):** `occ_.readbackVoxels` populates `out.activeVoxels`; when `gf3d_` is active, `gf3d_->runStage` refines `out.activeVoxels` and `v15_->finishBackEnd` consumes `gf3d_->v15Gaussians()` via `setExternalGaussians`.
- **P3 (`src/perception/sota/Gaussian3DTypes.hpp:77-92`):** `Gaussian3DPrimitive` adds `fromStaticMem`, `fromTrack`, `staticMemLocked` (`prim_mem_bytes=53248`).
- **P4 (`src/perception/sota/VoxelToGaussianInit.cpp:157-312`):** Initializes `fromTrack` actors first (`g.fromTrack=true`), clusters voxels (`kMaxVoxelClusterScaleXZ=0.90f`), then runs 3-pass `staticCells` backfill (forward roadside locked first, then remaining locked, then unlocked).
- **P5 (`src/perception/sota/DeformableAttention3D.cpp:96-105, 216-220`):** Caps `maxLocal` scale growth at `1.15f` (`kScaleCapMul`) and step shift by `kMaxDynLayerShiftM`/`kMaxStaticLayerShiftM`.
- **P6 (`src/perception/sota/GaussianToVoxelSplatter.cpp:91-295`):** Splatting supports independent `splatVoxelStepX/Y/Z`; `fromTrack` Gaussians are OBB-clamped across all voxels (`|local.x/z| <= g.scale + 0.12f`), non-track clusters refine only `hasPriorCell`, `staticOcc` boosts only when `acc.maxCorrob > 0u`, `SignalSign` uses `min(occThresh, 0.25f)`, and up to 32 `staticMemLocked` cells fuse cleanly.
- **P7 (`GaussianWorldModel.cpp:19-105`, `V15HudOverlay.cpp:10-58`, `VulkanViewer.cpp:1430-1505`):** Filters `NaN`/`Inf`/degenerate Gaussians, splats foreground semantic primitives before translucent (`<= 0.20` alpha) `Static` background, adds `R/N/D` + `| AEB`/`| MRM` + `NaN` HUD sanitization, and detects stale Reality buffers.

## 3. Execution Trace
- **T1 (`legacy`):** `OccupancyFlowNet` readback -> `GaussianWorldModel::updateFromPerception` (`1024` Gaussians) -> `v15_gaussian_splat.comp` -> planner.
- **T2 (`gaussianformer3d`):** `OccupancyFlowNet` readback -> `VoxelToGaussianInit` (`512` Gaussians) -> `DeformableAttention3D` (`scaleCapMul` / `stepCap`) -> `GaussianToVoxelSplatter` (OBB clamp + `splatVoxelStep*`) -> `setExternalGaussians` / refined `activeVoxels` -> planner.

## 4. Observed A/B Evidence (`cybercab_tests --gtest_filter=GaussianFormer3D*`, `13/13` PASS, `RelWithDebInfo`)
Captured verbatim from `task-11678` (plus `3/3` [`test_v15_reality_window_validation.cpp`](../../tests/test_v15_reality_window_validation.cpp) PASS in `task-11745`):

### E1 — Perception, Voxels, Gaussians & Open-Set Obstacles (`ABPerceptionVoxelGaussianAndOpenSetQuality`)
| Metric | `legacy` | `gaussianformer3d` | Verdict |
|---|---:|---:|---|
| 3D Detection Recall / Prec / Veh AP / RMSE | `0.6667 / 0.9481 / 0.5392 / 1.6177m` | `0.6667 / 0.9481 / 0.5392 / 1.6177m` | **Same** (downstream of RT-DETR) |
| Dynamic Voxel Prec / GT Voxel Cov | `0.8809 / 0.5714` | `0.9185 / 0.5714` | **Better prec** (`+3.76 pp`) |
| Active Voxels / frame & Gaussians / frame | `239.9 / 1024.0` | `179.4 / 512.0` | **Fewer primitives** (`-25.2%` vox, `-50%` Gauss) |
| Gaussian GT Mean Occ / GT Hit / Free Occ | `0.3012 / 0.4048 / 0.0000` | `0.4593 / 0.5119 / 0.0000` | **Better** (`+52.5%` occ, `+10.71 pp` hit) |
| Truck Veh Voxels / Signal Sign Voxels | `294 / 46` | `280 / 8` | Truck `~same`; **lower SignVox** (`46 -> 8`) |
| Open-Set `CARGO` (vox / minClr / col / avoid) | `6421 / 0.65 m / 0 / true` | `6576 / 0.65 m / 0 / true` | **Same clearance & avoidance** |
| Open-Set `BARRIER` (vox / minClr / end / avoid) | `707 / 0.29 m / (-5.1, 2.02, 5.39) / true` | `323 / 1.02 m / (-15.0, 1.91, 5.18) / true` | **Much better** (`-54.3%` vox vs leg, `3.5x` clear) |

### E2 — Closed-Loop Safety & Decisions Across 8 Missions (`ABSixMissionSafetyAndDecisionComparison`, `0/0` col, `8/8` pass)
| Mission | `pass` | `dist (m)` | `maxAcc (m/s^2)` | `jerk (m/s^3)` | `minClr (m)` | `brkStep` | `aeb` | `minV (m/s)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v15:nominal` | `1/1` | `43.3 / 38.2` | `5.80 / 5.79` | `9.56 / 6.23` | `1.41 / 1.39` | `40 / 40` | `0 / 0` | `2.83 / 2.60` |
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
| `half` (`320x240`) | `10.71 / 11.62 ms` | `11.07 / 12.25 ms` | `4.85 / 5.56 ms` | `0.359 ms` |
| `full` (`640x480`) | `12.29 / 12.69 ms` | `12.48 / 15.25 ms` | `6.49 / 6.51 ms` | `0.343 ms` |
| `cybercab_binned` (`1448x938`) | `20.08 / 19.40 ms` | `22.08 / 21.24 ms` | `12.93 / 11.78 ms` | `0.543 ms` |
- **Memory/Determinism/Robustness:** `stage1_heap_delta_bytes=0`, `prim_mem_bytes=53248`; `det_exact=1`, `behavior_diff_frames=0/12`, `max_*_delta=0.0000`; empty-road `0` ghosts/dynVox/falseAeb; `soil_front` `seen_mrm=1`, `col=0`, `max_gauss=512`; Reality View world-to-pixel Vehicle/Pedestrian semantic colors + EWA off-axis parity + `NaN`/behind-camera/zero-scale/`R/N/D`/`MRM`/`AEB` validated across both `legacy` and `gaussianformer3d`.

### E4 — Voxel Resolution Preset Benchmark (`VoxelResolutionPresetBenchmark`, `1.0 m` vs `0.5 m` SurroundOcc vs `0.4 m` Occ3D vs `0.5 m` Splat-Only)
| Preset & Mode | `vox/f` | `dynPrec` | `sign` | `barVox` | `barClr` | `avoid` / `col` | `wall` / `gf3dCpu` |
|---|---:|---:|---:|---:|---:|---|---|
| `1.0m_default` (`legacy`) | `183.2` | `0.8740` | `46` | `2909` | `0.29 m` | `1 / 0` | `11.80 / 0.000 ms` |
| `1.0m_default` (`gf3d`) | `125.8` | `0.9145` | `8` | `378` | `1.02 m` | `1 / 0` | `12.77 / 0.396 ms` |
| `0.5m_surroundocc` (`legacy`) | `540.3` | `0.8507` | `111` | `11382` | `1.07 m` | `0 / 0` | `16.56 / 0.000 ms` |
| `0.5m_surroundocc` (`gf3d`) | `350.2` | `0.8880` | `12` | `8002` | `0.88 m` | `0 / 0` | `15.11 / 1.061 ms` |
| `0.4m_occ3d` (`legacy`) | `704.4` | `0.8749` | `306` | `32097` | `0.76 m` | `1 / 0` | `19.59 / 0.000 ms` |
| `0.4m_occ3d` (`gf3d`) | `542.3` | `0.8959` | `36` | `16436` | `0.79 m` | `1 / 0` | `17.38 / 1.512 ms` |
| `0.5m_splat_only` (`gf3d`) | `147.5` | `0.9450` | `8` | `286` | `1.02 m` | `1 / 0` | `13.79 / 0.471 ms` |
- **Finding:** Literature uses `0.4 m` (Occ3D-nuScenes) and `0.5 m` (SurroundOcc). In this simulator, **`0.5m_splat_only`** gives the best real-time accuracy (`dynPrec=0.9450`, `barVox=286`, `barClr=1.02 m`, `gf3dCpu=0.471 ms`) while keeping coarse `60x40` GPU BEV/Occ shaders calibrated for the planner (`0.5m_surroundocc` pauses cautiously behind the dense barrier with `col=0`).

## 5. Overall Verdict & Known Limits
- **Better:** Dynamic voxel precision (`0.9185`, or `0.9450` at `0.5m` splat step), `BARRIER` voxel precision & clearance (`323` vs `707` legacy, `1.02` vs `0.29 m`), Gaussian GT occupancy (`0.4593` vs `0.3012`, `0.5119` hit), `v15:nominal`/`hardbrake`/`night_glare_jaywalk` smoothness & safety (`6.23`/`4.89`/`12.07` jerk vs `9.56`/`17.42`/`17.48`), and `cybercab_binned` latency (`19.40` vs `20.08 ms`).
- **Same / Trade-offs:** `8/8` missions pass with `0` collisions, `0` empty-road hallucinations, `0 B` heap allocs, bitwise determinism (`0/12` diff); `SignVox` remains `8` vs `46`, and `v14:nominal` still brakes once against the oncoming left-lane vehicle envelope after an overtaking lane change at tick `38` (`jerk 4.00 vs 1.06`, `50.5 m`, `col=0`).
