# FSD v15 & Stage-1 Documentation Index

Index of FSD v15 modules and Stage-1 spatial representation reports (`docs/v15/`).

## Module Index

| Step | Module | Document | Commit |
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
| 7b | M3b 100 Hz kinematics, lane-center nav & Plucker `PatchTokenizer` | [step07b_m3_multimodal_tokens.md](step07b_m3_multimodal_tokens.md) | `bbe0083`..`8823d81` |
| 8 | M5 persistent SE(3) static spatial memory & selective refresh | [step08_m5_static_memory.md](step08_m5_static_memory.md) | `4eda6b7`..`581abea` |
| 9 | M6 3D/4D Gaussian World Model & Zwicker/Kerbl EWA Reality View | [step09_m6_gaussian_world_model.md](step09_m6_gaussian_world_model.md) | `8c07554`..`1fd1a9d` |
| 10a | M4 self-supervised latent predictor & sparse/KV/FP16 telemetry | [step10a_m4_latent_predictor.md](step10a_m4_latent_predictor.md) | `1e845e4`..`d05554f` |
| 10b | M7 autoregressive middle-of-lane policy, siren yield & Autoshift D/R | [step10b_m7_autoreg_policy.md](step10b_m7_autoreg_policy.md) | `969629e`..`fb40d74` |
| 11 | End-to-end closed-loop v15 & full-res validation | [step11_e2e_v15_validation.md](step11_e2e_v15_validation.md) | `2c2d620`..`06881c1` |
| 12 | Configurable policy (`v15_policy.yaml`), OBB/blind-spot safety, `CybercabMesh` | [step12_v15_safety_config_and_cybercab_mesh.md](step12_v15_safety_config_and_cybercab_mesh.md) | `035441d`..`124a88a` |
| 13 | Box-free 4D Gaussian occupancy & comfort multi-rollout planner | [step13_gaussian_occupancy_and_comfort_planner.md](step13_gaussian_occupancy_and_comfort_planner.md) | `47dea4f`..`6bea1fb` |
| 14 | Live Reality View HUD, `VulkanViewer` PIP & `v15.*` `TraceLogger` telemetry | [step14_reality_view_hud_and_v15_trace.md](step14_reality_view_hud_and_v15_trace.md) | `6f4e845`..`0eb32bb` |
| 15 | Dynamic-range `day` / `night_glare` lighting, RCCB mosaic & HDR closed-loop | [step15_hdr_lighting_rccb_and_glare_test.md](step15_hdr_lighting_rccb_and_glare_test.md) | `d96829b`..`c92759a` |
| 16 | Post-v15 SOTA headers (`GaussianSurfel2DWorld`, `TruncatedDiffusionPolicy`) | — | Step 16 |
| 17 | Parameter effects, driving actions, `Autoshift` D->R->D & camera-knock calib | [step17_params_actions_autoshift_and_calib_tests.md](step17_params_actions_autoshift_and_calib_tests.md) | `3efb471`..`d2431a4` |
| 18 | Voxels, detections/AP, empty-road checks, 6 missions & zero-alloc | [step18_voxels_hallucinations_mission_and_zero_alloc.md](step18_voxels_hallucinations_mission_and_zero_alloc.md) | `8ecc082`..`5cc12a4` |
| 19 | Real Cybercab `2896x1876` / `1448x938` presets + `320x240` default & shader scaling | [step19_cybercab_native_camera_resolution.md](step19_cybercab_native_camera_resolution.md) | `6bda981`..`d92ac83` |
| 20 | Multi-camera corroboration, aspect-invariant RT-DETR & Cybercab validation | [step20_corroboration_and_cybercab_full_validation.md](step20_corroboration_and_cybercab_full_validation.md) | `b92a036`..`3b2844e` |
| 21 | All-4-resolution v14/v15 runs, FMEA suite & descriptor-leak fix | [step21_all_res_validation_and_fmea.md](step21_all_res_validation_and_fmea.md) | `d56b888`..`47f4596` |
| 22 | Full-vision MOT/IDSW, Lane IoU/assoc, ECE/slicing, RSS & poles/gantry | [step22_av_validation_framework.md](step22_av_validation_framework.md) | `7589de1`..`c5c8c91` |
| 23 | Per-condition calib, vision signal FSM, open-set road obstacles & shadow replay | [step23_calibration_signal_openset_and_replay.md](step23_calibration_signal_openset_and_replay.md) | Step 23 |
| S1 | Stage 1: Multi-modal GaussianFormer3D (`--spatial-rep legacy\|gaussianformer3d`) | [stage1_gaussianformer3d.md](stage1_gaussianformer3d.md) | `fe61e78` |
| S1b | Stage 1 A/B evaluation, OBB clamp, YAML voxel presets, Reality Window hardening & 205-test validation | [stage1_gaussianformer3d_ab_report.md](stage1_gaussianformer3d_ab_report.md) | `2d10b85`..`49d7736` |
| S2 | Stage 2/3: LeWorldModel (SIGReg) & H-JEPA (`--world-model legacy\|lewm\|hjepa`), FMEA & SOTA PDMS | [stage2_lewm_hjepa_ab_report.md](stage2_lewm_hjepa_ab_report.md) | `feature/stage2-lewm-hjepa` |

- **FSD v15 Reality Window & Stage-1 Highlights:** [`GaussianWorldModel`](../../src/perception/v15/GaussianWorldModel.cpp) splats foreground semantic primitives ahead of translucent (`<= 0.20` alpha) `Static` background and filters `NaN`/degenerate Gaussians; [`V15HudOverlay`](../../src/app/V15HudOverlay.cpp) adds `R/N/D` gear, `| AEB`/`| MRM` tags, and non-finite sanitization; [`VulkanViewer`](../../src/app/VulkanViewer.cpp) guards stale buffers; validated in [`test_v15_reality_window_validation.cpp`](../../tests/test_v15_reality_window_validation.cpp).

## References
- **GaussianFormer3D** (ICRA 2026, [arXiv:2505.10685](https://arxiv.org/abs/2505.10685)) — Multi-modal 3D Gaussian semantic occupancy prediction with 3D deformable attention.
- **LeWorldModel (LeWM)** (Maes et al., [arXiv:2603.19312](https://arxiv.org/abs/2603.19312)) — Action-conditioned JEPA latent dynamics with SIGReg regularization.
- **H-JEPA & AD-E2E-JEPA** (Zhang et al., Oct 2026; Zhu et al., Sep 2026) — Multi-timescale hierarchical subgoal JEPA & compressed patch-latent planning rollouts.
- **NAVSIM** (Dauner et al., NeurIPS 2024 [arXiv:2406.15349](https://arxiv.org/abs/2406.15349)) — Predictive Driver Model Score (`PDMS`) evaluation.
- **GaussianFormer / GaussianFormer-2** (Huang et al., ECCV 2024 [arXiv:2405.17429](https://arxiv.org/abs/2405.17429); CVPR 2025 [arXiv:2412.04384](https://arxiv.org/abs/2412.04384)) — Object-centric 3D semantic Gaussians & probabilistic occupancy superposition.
- **3D Gaussian Splatting & EWA Splatting** (Kerbl et al., SIGGRAPH 2023 [arXiv:2308.04079](https://arxiv.org/abs/2308.04079); Zwicker et al., IEEE TVCG 2002) — Reality View EWA covariance projection & alpha blending.
- **BEVFormer** (Li et al., ECCV 2022 [arXiv:2203.17270](https://arxiv.org/abs/2203.17270)) — Spatial-temporal BEV lifting.
- **RT-DETR** (Zhao et al., CVPR 2024 [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)) — Real-time multi-scale object detection.
- **ByteTrack** (Zhang et al., ECCV 2022 [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)) — Multi-object association & tracking.
- **Occ3D / SurroundOcc** (Tian et al., NeurIPS 2023 [arXiv:2304.14365](https://arxiv.org/abs/2304.14365); Wei et al., ICCV 2023 [arXiv:2303.09551](https://arxiv.org/abs/2303.09551)) — `0.4 m` / `0.5 m` 3D semantic occupancy benchmarks.
- **Responsibility-Sensitive Safety (RSS)** (Shalev-Shwartz et al., [arXiv:1708.06374](https://arxiv.org/abs/1708.06374)) — Stopping-distance & corridor safety envelope.
- **Agentic Code Reasoning** (Ugare & Chandra, Meta, [arXiv:2603.01896](https://arxiv.org/abs/2603.01896)) — Structured reasoning & verification methodology.
