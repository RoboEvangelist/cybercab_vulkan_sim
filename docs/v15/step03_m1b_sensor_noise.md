# Step 3: M1b Poisson-Gaussian sensor noise on the photon tensor

Tutorial Module 1.2 ("Sensor Noise Modeling"): without ISP smoothing, the network sees the true
CMOS noise profile σ²(I) = a·I + b. The model and its parameterization follow Foi et al. 2008,
*Practical Poissonian-Gaussian noise modeling and fitting for single-image raw data*.

## Scope
- v15 only: add signal-dependent shot noise plus read noise to the 14-bit linear ADC.
- The noise is deterministic in (seed, frame time) and configured in YAML.
- **Out of scope:** row/column fixed-pattern noise, dark current, PRNU, black-level offset, and any denoising consumer.
- Files: `src/config/SimConfig.{hpp,cpp}`, `config/cybercab_sim.yaml`, `src/sensors/PhotonModel.hpp`,
  `shaders/include/photon.glsl`, `shaders/cam_isp_soil.comp`, `src/sensors/CameraRigRenderer.cpp`,
  `tests/test_v15_photon_noise.cpp` (new), `tests/test_v15_photon_frontend.cpp` (noise disabled, see P8).

## Premises
- **P1** `SimConfig.hpp:169-172`: defaults are `photonNoiseEnabled = true`, `a = 0.4096`, `b = 1.51`, `seed = 1234`.
  YAML `perception.photon_sensor` overrides them (`SimConfig.cpp:152`). Negative or NaN values throw (`SimConfig.cpp:220-221`).
- **P2** `PhotonModel.hpp` `noiseParamsFromPhysical(40000 e⁻, 3 e⁻)`: gain = 40000/16383 e⁻/DN, so a = 1/gain = 0.4096 and b = (3/gain)² = 1.51.
- **P3** `CameraRigRenderer.cpp:35`: `static_assert(sizeof(IspPushConstants) == 80)`. The four new fields (`noiseA, noiseB, noiseSeed, noiseEnable`) follow `fullWell` in the same order as the GLSL block (`cam_isp_soil.comp:27-30`). 80 B is below the Vulkan-guaranteed 128 B push-constant limit.
- **P4** `CameraRigRenderer.cpp:496` and the following line:
  - `noiseSeed = cfg.photonNoiseSeed ^ frameNoiseKey`, where `frameNoiseKey = bits(timeSec) · 2654435761` (`:473-475`);
  - `noiseEnable = (hdr_ && cfg.photonNoiseEnabled)`.
- **P5** `cam_isp_soil.comp:72-73`: the noisy branch is taken only inside `if (pc.linearOut != 0u)`, and only when `noiseEnable != 0`. Otherwise `photonLinearNorm` (the M1a function) runs unchanged.
- **P6** `photon.glsl` `photonNoisyNorm`:
  - μ = clamp(r/fw, 0, 1)·16383;
  - dn = μ + √(aμ + b)·N, where N comes from Box-Muller on two PCG3D hash rounds keyed by `(x, y, cam) ^ seed`;
  - the output is rounded and clipped to [0, 16383].
- **P7** The per-pixel key `(x⊕s, y⊕s, cam⊕s)` is injective in (x, y, cam) for fixed s, so no two pixels share a key within a frame. The three channels use the separate x/y/z output lanes of PCG3D.
- **P8** `test_v15_photon_frontend.cpp` sets `photonNoiseEnabled = false`. Its claim (noiseless ADC within half an LSB) is about M1a and its thresholds are unchanged.

## Trace
- **T1 (v14):** `hdr_ = false`, so `linearOut = 0` and `noiseEnable = 0` (P4). P5 means the binding-3 branch is skipped. The only change on the v14 path is 16 extra pushed bytes that the shader never reads.
- **T2 (v15, noise off):** P5 calls `photonLinearNorm`, the same code as at step 2. The output does not depend on `timeSec`.
- **T3 (v15, noise on):** P6. For μ ≫ 1 the clip at 0 is inactive (μ ≥ 20 is more than 6σ above 0), so E[dn] = μ and Var[dn] = aμ + b. After rounding both the clean and noisy codes, the residual variance is aμ + b + 1/6 (two independent uniform rounding errors of 1/12 each).
- **T4 (determinism):** the output is a pure function of (rgb, x, y, cam, seed, timeSec) by P4/P6, so identical inputs give bit-identical tensors.

## Claims
- **C-EQ:** by T1, v14 is unchanged: the 96 pre-v15 tests hold. In v15 with noise off, the output equals step 2 (T2).
- **C-V15-1:** empirical variance follows aμ + b + 1/6, and the slope is recovered within 8%.
- **C-V15-2:** the noise is zero-mean away from the black clip.
- **C-V15-3:** the noise is bit-reproducible per (seed, time), fresh across frame times, and uncorrelated across time and channels.

## Evidence
- **E1 (C-EQ)** ctest **106/106**: 96 pre-v15 tests, the 3 step-2 photon tests with noise off and unchanged thresholds, and `DisabledNoiseEqualsNoiselessAdc` (tensor independent of frame time with noise off).
- **E2** `PhysicalSensorParamsMapToConfigDefaults`: P2 numbers match the YAML defaults.
- **E3 (C-V15-1, C-V15-2)** `VarianceFollowsPoissonGaussianModelAndIsUnbiased`: 6 bins, μ = 49…1103, 13.5k–873k samples each.

  | μ | measured var | model var | bias |
  |---|---|---|---|
  | 49.2 | 21.74 | 21.83 | +0.003 |
  | 115.4 | 48.96 | 48.94 | +0.013 |
  | 193.3 | 80.85 | 80.86 | −0.008 |
  | 497.5 | 205.75 | 205.47 | −0.023 |
  | 768.8 | 316.74 | 316.58 | −0.060 |
  | 1102.6 | 446.96 | 453.30 | +0.055 |

  The fit gives â = **0.4050** (−1.1%) and b̂ = **2.78** (model 1.68). SNR at 1000 DN = 33.9 dB.
- **E4 (C-V15-3)** `DeterministicPerFrameAndIndependentAcrossTimeAndChannels`:
  - same (seed, t): **all** pixels bit-identical;
  - t = 0 vs 0.1 s: more than 90% of bright pixels differ;
  - ρ_time = **−0.0003**, ρ_R,G = **−0.0037**.

## Known limits / risks
- **R1 (UNVERIFIED: b precision):** b̂ = 2.78 vs 1.68 passes only the ±3 tolerance. The lowest populated bin (μ ≈ 49) has b at under 8% of the variance, so the intercept is weakly identified by a daylight scene. A dark-scene test is needed to verify b within ±0.5 DN². Deferred to the low-light scenario added with M1c.
- **R2** Gaussian approximation of Poisson: it is accurate for μ ≳ 20 DN (≈ 50 e⁻). Below that, the true distribution is skewed, and clipping at 0 biases the mean upward. Not measured here; bins start at μ = 20.
- **R3** Noise is computed on the rasterized radiance after vignetting and soiling, not on a physical photon flux through a lens/CFA. It is a model of the noise statistics, not a sensor simulation.
- **R4** `frameNoiseKey` uses the bit pattern of `timeSec`, so two different sims with identical `timeSec` sequences share noise. This is intentional, for reproducibility.
