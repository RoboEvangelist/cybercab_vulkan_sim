# FSD v15 Change Certificates

Every v15 step is committed together with a **semi-formal change certificate**, following
*Agentic Code Reasoning* (Ugare & Chandra, Meta, arXiv:2603.01896). That paper replaces free-form
reasoning with a certificate the author cannot fill in by guessing. Here the certificate has:

1. **Scope**: one feature, the files touched, and what is explicitly *out* of scope.
2. **Premises (P#)**: concrete facts with `file:line` citations, no claims from function names alone.
3. **Trace (T#)**: symbolic execution of the changed paths, done **separately for `v14` and `v15`**.
4. **Claims (C#)**:
   - **C-EQ (patch equivalence)**: with `pipeline_version: v14`, observable outputs are *identical* to the parent commit, derived only from P/T.
   - **C-V15**: the new behavior, stated as measurable properties.
5. **Evidence (E#)**: test names and the observed numbers (gtest `RecordProperty` values / ctest totals).
   A claim without a test stays marked **UNVERIFIED**.
6. **Known limits / risks**: what is approximate, analog-only, or deferred.

## Step index

| Step | Module | Certificate | Commit |
|---|---|---|---|
| 1 | Config switch `pipeline_version` | [step01_config_switch.md](step01_config_switch.md) | `84b73e4` |
| 2 | M1a ISP-bypass linear 14-bit photon tensor | [step02_m1a_linear_photon_tensor.md](step02_m1a_linear_photon_tensor.md) | `47f26e2` |
| 3 | M1b Poisson-Gaussian sensor noise | [step03_m1b_sensor_noise.md](step03_m1b_sensor_noise.md) | `fbfa1fa` |
| 4 | M1c frequency separation (illumination vs. detail) | [step04_m1c_frequency_split.md](step04_m1c_frequency_split.md) | `8d4c767` |
| 5a | M1 dark read-noise verification & adaptive FrequencySplit | [step05a_m1_dark_noise_and_scaling.md](step05a_m1_dark_noise_and_scaling.md) | `0691227` |
| 5b | M1 `V15Pipeline` wiring (`v15()` switch & budget) | [step05b_m1_pipeline_wiring.md](step05b_m1_pipeline_wiring.md) | `886f6e2`, `12ee828` |
| 6a | M2a online pitch/roll self-calibration | [step06a_m2a_self_calibration.md](step06a_m2a_self_calibration.md) | `035b655`..`191dbeb` |
| 6b | M2b virtual pinhole rectifier & `V15Pipeline` wiring | [step06b_m2b_virtual_rectifier.md](step06b_m2b_virtual_rectifier.md) | `c5e9d45`..`bcf5164` |
| 7a | M3a simulated 48 kHz emergency siren & `AudioTokenizer` | [step07a_m3_audio_siren.md](step07a_m3_audio_siren.md) | `3861f50`..`d054c8d` |
| 7b | M3b 100 Hz kinematics, lane-center nav & Plucker `PatchTokenizer` | [step07b_m3_multimodal_tokens.md](step07b_m3_multimodal_tokens.md) | `bbe0083`..`d5ebb2b` + tests |
| 8 | M5 persistent SE(3) static spatial memory & selective refresh | [step08_m5_static_memory.md](step08_m5_static_memory.md) | `4eda6b7`..`58b84ff` + tests |
| 9 | M6 3D/4D Gaussian World Model & Zwicker/Kerbl EWA Reality View | [step09_m6_gaussian_world_model.md](step09_m6_gaussian_world_model.md) | `8c07554`..`b6c0a47` + tests |
| 10a | M4 self-supervised latent predictor & sparse/KV/FP16 telemetry | [step10a_m4_latent_predictor.md](step10a_m4_latent_predictor.md) | `1e845e4`..`d05554f` |
| 10b | M7 autoregressive middle-of-lane policy, siren yield & Autoshift D/R | [step10b_m7_autoreg_policy.md](step10b_m7_autoreg_policy.md) | `969629e`..`fb40d74` |
| 11 | End-to-end closed-loop v15 & full-res validation | [step11_e2e_v15_validation.md](step11_e2e_v15_validation.md) | `2c2d620` |

## Open UNVERIFIED items
- ~~step02 R2~~ **CLOSED in Step 11**: closed-loop `v15` perception validated (`recall=0.5681`, `precision=0.7529`, 0 collisions, `4.81 ms` half-res / `7.00 ms` full-res).
- step06a R1: rendered-world self-calibration accuracy is 0.31° (half-res) / 0.11° (full-res), acceptable within `< 0.35°` margin.

## Regression gate (every step)
- `ctest --test-dir build` must show **100%** pass.
- The 96 pre-v15 tests are the v14 equivalence oracle; they run with `pipeline_version: v14`.
- New files are kept to **350 lines or fewer**.
