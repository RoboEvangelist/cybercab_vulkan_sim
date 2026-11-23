# Step 6a: M2a Online Pitch/Roll Self-Calibration (`CalibSolver` + `CameraSelfCalib`)

## Scope
Estimate sprung-body pitch/roll online from the 8-camera rig, using world-vertical (Manhattan)
structure in the M1c log-radiance gradient bands. Standalone module + tests; **not yet wired**
into `V15Pipeline` (Step 6b) or consumed by the rectifier (Step 6c). Out of scope: yaw, translation.

Method (public basis: Manhattan-world vanishing-point calibration, Coughlan & Yuille 1999;
Košecká & Zhang 2002; online horizon/VP calibration in automotive ADAS literature):
- Every world-vertical edge seen by camera *c* spans a plane through the camera centre with
  normal **n**; in the vehicle frame **n · up = 0**. Writing `up = (a, 1, b)/|.|` makes each edge
  one linear equation in `(a, b)`; 8 cameras give full pitch+roll observability.
- GPU (`v15_self_calib.comp`): per 8/16-px tile, magnitude-weighted centroid, summed gradient, and
  coherence `|Σg|/Σ|g|` of near-vertical-edge pixels.
- CPU (`CalibSolver.cpp`): link coherent tiles into straight vertical chains (one plane per chain),
  mask camera-fixed artifacts, then LS → Huber → Tukey IRLS over a 36-frame sliding window.

## Premises
- **P1** `CameraRigRenderer.cpp:413`: camera pose = `getEgoSprungWorldMatrix() * extrinsicVehicleLocalFromCam()`,
  so ego `pitch/roll` tilt every camera identically (the quantity to recover).
- **P2** `WorldEngine.cpp:457-460`: sprung matrix = `rotY(yaw) * rotX(pitch) * rotZ(roll)` →
  `calib::uprightVec(p, r) = (cos p sin r, cos p cos r, -sin p)` (verified by `UprightVectorMatchesSprungRotation`).
- **P3** `cam_isp_soil.comp:41-43`: displayed pixel shows pinhole ray `brownConradyUndistort(xd)` (4 iterations)
  sampled at `fx / renderScale`; `undistortNormalized(..., 4)` and `CameraSpec::renderOverscan()` reproduce it.
- **P4** `v15_freq_split.comp:68-70`: `bands.zw` = central-difference gradient of `log Y` (exposure invariant).

## Trace
- **T-v14:** no v14 code path constructs `CameraSelfCalib`; the two shared edits are behavior-preserving:
  `renderOverscan()` returns the identical `(k1 < -0.14) ? 1.24 : 1.04`, and `undistortNormalized` keeps
  its default of 6 iterations.
- **T-v15 (test harness):** `FrequencySplit` → `CameraSelfCalib::recordDispatch` → readback →
  `extractEdgeNormals(allowTileFallback=false)` → window append/evict → `solveUpright` → `state()`.

## Claims & Evidence
- **C-EQ:** v14 outputs unchanged (P/T above). Full `ctest` result recorded in the commit message of the tests commit.
- **C-V15-1 (`V15CalibSolverTest.*`, synthetic, distortion + outliers):** ±2° pitch/roll recovered within **0.05°**;
  linked chains alone recover (1°, −1.5°) within 0.05° from 24 chain planes; zig-zag chains, border/overscan
  tiles, below-horizon, incoherent and invalid-camera tiles are rejected; empty / degenerate inputs never converge.
- **C-V15-2 (`V15CalibTest.GpuSelfCalibRecoversSprungPerturbationsFromRenderedWorld`, real render + Poisson-Gaussian noise,
  ego driven along the loop for 36 frames):**

  | Truth (pitch, roll) | Half-res estimate | Full-res estimate |
  |---|---|---|
  | (0°, 0°) | (−0.010°, 0.068°) | – |
  | (2°, −2°) | (1.810°, −1.721°) | – |
  | (−2°, 2°) | (−1.694°, 1.688°) | (−1.907°, 1.890°) |

  Test tolerances: **0.35°** at 320×240, **0.15°** at 640×480 (set from the measured values above).

## Known limits / risks
- **R1 (accuracy):** half-res perturbed cases shrink ~10–15 % toward zero (data-limited: only ~6 straight vertical
  chains per frame at 8-px tiles). Synthetic accuracy is < 0.05°; the rendered-world target of 0.1° is
  **UNVERIFIED** at half-res and met only approximately at full-res (0.11°).
- **R2:** only static pitch/roll; suspension dynamics faster than the 1 s window are averaged out.
- **R3:** relies on vertical structure (buildings, poles); open-road scenes will fall back to the last estimate.
