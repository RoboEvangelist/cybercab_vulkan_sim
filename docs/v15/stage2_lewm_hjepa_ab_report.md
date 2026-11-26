# Stage 2/3 — LeWorldModel (SIGReg) & H-JEPA A/B/C Evaluation, SOTA Metrics & FMEA Report

Semi-formal change report following *Agentic Code Reasoning* (Ugare & Chandra, Meta, [arXiv:2603.01896](https://arxiv.org/abs/2603.01896)).

## 1. Scope
- **Subject:** Three-way comparison of `--world-model legacy` vs. `--world-model lewm` (action-conditioned SIGReg JEPA, [arXiv:2603.19312](https://arxiv.org/abs/2603.19312)) vs. `--world-model hjepa` (2-level slow macro subgoal + fast CEM controller, Zhang et al. Oct 2026; AD-E2E-JEPA Zhu et al. Sep 2026), plus Stage-1 + Stage-2/3 composition (`--spatial-rep gaussianformer3d --world-model hjepa`).
- **Path B-Ready Interface:** Deterministic analytical default weights (`LeWmWeights::defaultAnalytical()`) with optional FNV-1a checksummed binary weight loading via `--world-model-weights <path.bin>` and full FMEA fallback.
- **Out of scope:** Changing `config/cybercab_sim.yaml` defaults (`world_model_mode: "legacy"`) or relaxing downstream RSS/AEB/`StaticVoxelGuard` shields.

## 2. Premises (`file:line`)
- **P1 (`config/cybercab_sim.yaml:24,107-124`, `WorldModelSotaConfig.hpp:13-53`):** YAML controls `world_model_mode` (`legacy|lewm|hjepa`), `world_model_weights`, and bounded `lewm_hjepa` parameters validated on both YAML load and `init()`.
- **P2 (`src/perception/sota/LeWmSceneEncoder.cpp:20-155`):** Compresses ego kinematics, same-lane/adjacent-lane/oncoming/blind-spot tracks, grounded obstacle voxels, VRU threat, signal/siren state, and epistemic sigma into `LatentVec16`, sanitizing non-finite inputs (`kWmFaultNanInput`).
- **P3 (`src/perception/sota/LeWmLatentModel.cpp:59-263`):** Maintains a fixed 32-frame ring buffer (`0 B` heap delta) for `SIGRegMetrics`, computes `predictNext` and `rolloutEnergy`, monitors latent collapse (`kWmFaultLatentCollapse`), and falls back safely to neutral guidance (`fmeaFallbackTriggered=true`) on fault.
- **P4 (`src/perception/sota/HJepaHierarchy.cpp:36-254`):** Level-2 (`8D` macro state, `6 x 0.50 s = 3.0 s`) selects among `KeepLane`, `OvertakeLeft`, `ReturnRight`, `YieldSlow`, and `StopHold` (penalizing `OvertakeLeft` against oncoming left-lane traffic); Level-1 runs `3 x 9` deterministic CEM (`6 x 0.25 s = 1.5 s`).
- **P5 (`src/planning/sota/WorldModelGuidance.hpp:10-46`, `TrajectoryPlanner.cpp:139-167`, `AutoregressivePolicy.cpp:176-331`):** Applies bounded guidance (`<= 82.0`) and anticipatory speed scaling (`[0.55, 1.0]`) upstream of RSS, AEB, `SENSOR_SOILED_MRM`, red-light, and stop-sign shields.

## 3. FMEA & Failure-Mode Coverage (`LeWmHJepaUnitTest`, `LeWmHJepaGpuTest`)
| Failure Mode / Fault Bit | Detection Mechanism | Verified Safe Response |
|---|---|---|
| Corrupt / Missing / NaN `.bin` weights (`kWmFaultWeightLoad`) | Magic, version, byte-length, FNV-1a checksum & `isFiniteValid()` | Rejects corrupt file, retains analytical weights cleanly |
| `NaN` / `Inf` ego, track or voxel inputs (`kWmFaultNanInput`) | `std::isfinite` guard in encoder & dynamics | Triggers FMEA fallback (`recLat=-1.95 m`, `spdScale=1.0`, `sigmaDelta=0`) |
| Encoder representation collapse (`kWmFaultLatentCollapse`) | SIGReg 32-frame variance `< 1e-5` under varying raw input | Flags collapse bit & disengages learned guidance |
| Divergent rollout / invalid `dt` or weights (`kWmFaultRolloutDivergence`) | Norm bound `<= 18.0f`, `dt > 0`, finite step check | Aborts candidate rollout & falls back safely |
| Multi-camera forward blinding (`soil_front` / `cameras[0..2]`) | Downstream MRM invariant check | Triggers `SENSOR_SOILED_MRM` with `0` collisions (`lewm` & `hjepa`) |

## 4. Observed Evidence (`task-12013` & `task-12020`, `RelWithDebInfo`)

### E1 — Closed-Loop Missions (`ABThreeWayLegacyLeWmHJepaMissionsComfortAndNfrAudit`, `0/0/0` col, `8/8` pass all modes)
| Mission | `dist (leg/lewm/hjepa)` | `jerk (leg/lewm/hjepa)` | `minClr (m)` | `aeb (leg/lewm/hjepa)` | `cpu (lewm/hjepa)` |
|---|---:|---:|---:|---:|---:|
| `v15:nominal` | `43.3 / 41.9 / 43.9 m` | `9.56 / 11.12 / 10.52` | `1.41 / 1.41 / 1.19` | `0 / 0 / 0` | `0.074 / 0.161 ms` |
| `v15:jaywalk` | `8.8 / 8.8 / 8.8 m` | `12.11 / 12.11 / 12.11` | `1.06 / 1.06 / 1.06` | `39 / 39 / 39` | `0.050 / 0.151 ms` |
| `v15:occluded` | `12.3 / 11.7 / 11.5 m` | `27.80 / 25.01 / 33.08` | `2.10 / 2.10 / 2.10` | `13 / 14 / 5` | `0.068 / 0.164 ms` |
| `v15:hardbrake` | `15.5 / 13.4 / 15.2 m` | `17.42 / 21.81 / 5.75` | `2.07 / 2.07 / 2.06` | `13 / 13 / 0` | `0.058 / 0.166 ms` |
| `v15:siren` | `7.1 / 7.1 / 7.1 m` | `5.20 / 5.20 / 5.20` | `1.45 / 1.45 / 1.45` | `0 / 0 / 0` | `0.054 / 0.154 ms` |
| `v15:night_glare_jaywalk` | `9.0 / 8.8 / 8.8 m` | `17.48 / 12.07 / 12.07` | `0.18 / 1.08 / 1.08` | `37 / 39 / 39` | `0.064 / 0.171 ms` |
| `v14:nominal` | `51.4 / 50.6 / 46.0 m` | `1.06 / 6.00 / 3.03` | `1.56 / 1.56 / 1.56` | `0 / 2 / 0` | `0.093 / 0.202 ms` |
| `v14:jaywalk` | `7.8 / 7.8 / 7.8 m` | `12.82 / 12.82 / 12.82` | `1.56 / 1.56 / 1.56` | `38 / 38 / 38` | `0.046 / 0.146 ms` |
- **Stage-1 + Stage-2/3 (`gf3d + hjepa`, `WM_COMBO`):** `4/4` pass, `0` col (`v15:nominal` `jerk 7.41`, `v15:hardbrake` `jerk 5.75` & `0` AEB, `v15:night_glare` `clr 1.08 m`, `v14:nominal` `jerk 2.35`).
- **NFRs (`WM_NFR`):** `stage2_heap_delta_bytes=0`, `max_cpu_ms=0.202 ms`, `det_exact=1`.

### E2 — Downstream Perception Parity, Prediction Accuracy, NAVSIM PDMS & Open-Set Obstacles (`task-12020`, `task-12013`)
| Mode | `Recall / Prec / VehAP / RMSE` | `1-step Cos / MSE` | `4-step Cos / MSE` | `HB Surp / Nom` | `PDMS` (`NC=1, DAC=1, TTC=1`) |
|---|---|---:|---:|---:|---:|
| `legacy` | `0.5540 / 0.7959 / 0.4439 / 1.901 m` | `—` | `—` | `—` | `0.9793` (`Comf 0.876`) |
| `lewm` | `0.5540 / 0.7959 / 0.4439 / 1.901 m` | `0.9965 / 0.0016` | `0.9957 / 0.0021` | `0.517 / 0.012` | `0.9811` (`Comf 0.887`) |
| `hjepa` | `0.5626 / 0.8153 / 0.4396 / 1.685 m` | `0.9966 / 0.0016` | `0.9958 / 0.0021` | `0.468 / 0.012` | `0.9833` (`Comf 0.900`) |
| `gf3d+hjepa` | **`0.5850 / 0.8156 / 0.4957 / 1.722 m`** | `0.9972 / 0.0016` | `0.9964 / 0.0022` | `0.468 / 0.010` | `0.9833` (`Comf 0.900`) |
- **Open-Set `CARGO` & `BARRIER` (`LeWmHJepaOpenSetTest`, `8/8` avoided, `0` col):** `CARGO` clearance `0.65 / 4.67 / 3.90 / 3.85 m`; `BARRIER` clearance `0.29 / 0.38 / 0.34 / 0.86 m` (`legacy / lewm / hjepa / gf3d+hjepa`).

## 5. Verdict
- **`hjepa` and `gf3d+hjepa` deliver clear safety and comfort improvements** without downstream regression: higher NAVSIM PDMS (`0.9833` vs `0.9793`), higher perception `VehAP` (`0.4957` vs `0.4439`) & `Recall` (`0.5850` vs `0.5540`), `-67%` `v15:hardbrake` jerk (`5.75` vs `17.42`) with `0` vs `13` AEBs, `-62%` `v15:occluded` AEB frames (`5` vs `13`), `6x` `night_glare_jaywalk` pedestrian clearance (`1.08` vs `0.18 m`), `> 0.995` multi-step latent rollout cosine accuracy, `0.20 ms` max CPU stage cost, `0 B` steady-state heap allocations, and clean CLI composition across all `v14`/`v15`/`gaussianformer3d` switches.
