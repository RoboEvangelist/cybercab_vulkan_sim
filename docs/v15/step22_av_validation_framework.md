# Step 22: Full-Vision AV Validation Framework, Calibration, Slicing & Night-Glare Fix

## 1. Scope
Validated the full vision pipeline beyond traffic signals — vehicles, pedestrians,
ground-projected lanes, road structure, and class-agnostic static obstacles (poles,
signs, barriers, overhead gantry underpass) — across `v14`/`v15`, all 4 resolution
presets (`half`, `full`, `cybercab_binned`, `cybercab`), `night_glare`, and lens
soiling (`cybercab_tests` Step 22 targeted run: `9/9 PASS` in `68.257 s`).

## 2. Night-Glare Perception Defect Caught & Fixed
New sliced calibration/MOT tests exposed `night_glare` perception degradation caused
by unscaled 8-bit `UNORM` display darkness, saturated night emissives (`0.54` blue
horizon / clipped windows), and low ambient shadow contrast:
- [shaders/scene.frag](../../shaders/scene.frag): scaled sub-threshold emissive
  accents (`inEmissive < 0.5`) by `0.20` when `emisGain > 1.0`, preserving bright
  light sources (`>= 0.5`) and keeping `"day"` bit-identical.
- [shaders/cam_isp_soil.comp](../../shaders/cam_isp_soil.comp): replaced `pad0` with
  `float displayGain` (`96 B` ABI preserved) and added chromaticity-preserving
  linear + soft-knee tone-mapping into `uOutDistorted`/`uCamTensor` while keeping
  14-bit linear `uCamLinear` unscaled and `"day"` unchanged.
- [src/sensors/CameraRigRenderer.cpp](../../src/sensors/CameraRigRenderer.cpp): tuned
  neutral night sky `(0.022, 0.024, 0.028)`, `kNightAmbient = 0.068f`,
  `kNightEmisGain = 3.20f`, `kNightDisplayGain = 6.5f`.
- Verified in `V15HdrRccbTest` (`3/3 PASS`, `night_max_lin_rad = 6.358668`,
  `night_mean_shadow_rad = 0.036347`, `night_glare_max_gpu_ms = 6.210368`).

## 3. Verified Validation Results (`9/9` Step 22 Targeted XML)
1. **MOT, ID Switches, Lane Assoc & Road IoU** (`AvMotLaneValidationTest`, `2/2`):
   - Zero ID switches (`idsw = 0`, `idsw_per_100 = 0.0000`), `cls_swaps = 0`, and
     `lane_assoc = 1.0000` across all 9 runs (`v14_{half,full,cybercab_binned}`,
     `v15_{half,full,cybercab_binned,cybercab,night,soiled}`):
     `mota = 0.5222..0.7222`, `motp = 0.4622..0.7194`, `frag = 1..5`,
     `lane_iou = 0.9422..1.0000`, `edge_err = 0.0000..0.1786 m`.
   - Signal stop-line & green-response (`v14`/`v15` x `half`/`full`):
     `stop_margin = 0.8084..0.8157 m` (0 overshoot), `green_resp = 0.4722 s`;
     `v15` HDR detects red bulb at `65.0000 m` vs `v14` 8-bit green at `64.17..65.00 m`.
2. **ECE Reliability, Asymmetric Safety Cost & Slicing** (`AvCalibrationSlicingTest`):
   - Nominal (`v14_half`, `v15_half`, `v15_full`, `v15_cybercab_binned`):
     `ece = 0.1549..0.2228`, `oce = 0.0191..0.1288`, `cost = 26.2721..33.1078`,
     `near = 0.5702..0.5965`, `mid = 0.6979..0.8229`, `far = 0.6190..0.6786`,
     `front_prec = 0.7892..0.9048`, `2σ_cov = 0.6370..0.7123`.
   - `v15_night`: `ece = 0.3934`, `oce = 0.2487`, `cost = 33.9683`,
     `near = 0.5965`, `mid = 0.7604`, `far = 0.5000`, `front_prec = 0.6298`,
     `2σ_cov = 0.7434`.
   - `v15_soiled`: `ece = 0.2372`, `oce = 0.1717`, `cost = 43.1573`,
     `near = 0.4286`, `mid = 0.4184`, `far = 0.5833`, `front_prec = 0.7407`.
3. **Ego Velocity vs RSS Stopping Margin** (`AvVelocityRangeTest`, `1/1`):
   - At `{5.0, 10.5, 14.0} m/s` (`v14`/`v15`): `det_m = 44.4659 m` (`trk_m =
     42.73..47.52 m`) strictly exceeds RSS `5.7731 / 13.7058 / 21.1769 m`.
4. **Class-Agnostic Poles/Barriers, Gantry & Road Structure** (`AvRoadStructureTest`, `2/2`):
   - Across `v14_half`, `v15_half`, `v15_full`, `v15_cybercab_binned`:
     `pole_vox = 1912..2254`, `class_agn_prec = 0.7007..0.7484`,
     `gantry_false_aeb = 0`, `gantry_min_spd = 8.5000 m/s`.
   - Across all 5 presets (incl. `v15_cybercab`): `lane_iou = 0.9789..1.0000`,
     `edge_err = 0.0000..0.1000 m`, `lane_center_err = 0.0322..0.3032 m`,
     `bev_obs_cells = 2390`, `road_head_rec = 1.0000`.

## 4. Regression & Known Limits
- **Full CTest**: `182/182` (`100% tests passed, 0 failed`, `267.44 s` wall clock) —
  `96` v14 oracle tests + `86` v15/all-res/validation/FMEA tests.
- **Known limits**: `v15_night` shows un-recalibrated underconfidence (`ece = 0.3934` vs
  `oce = 0.2487`); candidate follow-up is per-lighting post-hoc temperature scaling on
  heuristic confidences without altering 14-bit linear HDR outputs.
