# Step 4: M1c pre-network frequency separation (illumination vs. reflectance detail)

Tutorial Module 1.2 ("Pre-Network Frequency Separation", stated intent of US 11,215,999): separate
high-frequency spatial gradients (curbs, debris, pedestrians) from low-frequency illumination
(ambient light, wet-road reflections) **before** the deep layers. The patent text itself was not
available; the formulation below is the standard intrinsic-image / homomorphic decomposition:
I = R·L, so log I = log R + log L.

## Scope
- New standalone GPU module `v15::FrequencySplit`, which reads the photon tensor (steps 2-3) and writes per-pixel bands.
- **Out of scope:** wiring into `PerceptionPipeline` (next step), learned filters, and multi-scale pyramids.
- Files (new): `shaders/v15_freq_split.comp` (71 lines), `src/perception/v15/FrequencySplit.{hpp,cpp}` (50 + 85 lines),
  `tests/test_v15_freq_split.cpp` (194 lines). Modified: `CMakeLists.txt` (library + test lists).

## Premises
- **P1** `v15_freq_split.comp:36`: l = log(Y·fw + ε), with Y = Rec.709 luminance of the normalized code.
  `fw` defaults to `photon::kFullWellRadiance` (`FrequencySplit.hpp`, a single source of truth that reuses step 2) and ε = 1e-3.
- **P2** `:51-56` (pass 0): stores (l, horizontally Gaussian-blurred l), radius = round(3σ), σ = 4 px (`FrequencySplit.hpp:12,15`).
- **P3** `:61-70` (pass 1): low = vertical blur of the pass-0 blur; high = l − low; gx, gy = central differences of l.
- **P4** `:29`: borders are clamp-to-edge.
- **P5** `FrequencySplit.cpp:76-81`: two dispatches, each followed by `cmdComputeBarrier`, so pass 1 reads complete pass-0 data and consumers read complete bands.
- **P6** `FrequencySplit.cpp:16`: `static_assert(sizeof(FreqPushConstants) == 32)` matches the GLSL block (4 uint + 3 float + 1 int).
- **P7** `grep FrequencySplit src/` outside `src/perception/v15/` returns nothing: no production path instantiates the module at this commit.

## Trace
- **T1 (v14 and v15 runtime):** by P7 the module is never constructed, and the shader is only compiled. Runtime behavior is unchanged.
- **T2 (exposure scale k):** for I' = k·I with ε ≪ I:
  - l' = l + log k;
  - the Gaussian is normalized (weights divided by Σw), so low' = low + log k;
  - high' = l' − low' = high;
  - gx' = gx and gy' = gy (the constant cancels in the differences).
  - With ε > 0 the invariance is approximate, with error ≈ ε/I.
- **T3 (illumination ramp):** if log L is linear in x, the Gaussian of a linear function is the same function away from borders. So log L ends up entirely in low, and high = log R − G∗log R.

## Claims
- **C-EQ:** v14 and v15 runtime are unchanged (T1).
- **C-V15-1:** the GPU kernel equals an independent CPU reference within float rounding.
- **C-V15-2:** under a 4× exposure drop the high band and log-gradients are invariant (residual only from ε), and the low band shifts by ln 4 (T2).
- **C-V15-3:** on a real v15 render all bands are finite and the high band is zero-mean (it is a residual).

## Evidence
- **E1 (C-EQ)** ctest **109/109** (96 pre-v15 + 13 v15).
- **E2 (C-V15-1)** `GpuKernelMatchesCpuReference`: max abs error **9.5e-7**.
- **E3 (C-V15-2)** `ExposureChangeMovesOnlyLowBandHighAndGradientsInvariant`. Scene: checker reflectance {0.25, 0.75} under an exponential illumination ramp 1→¼, 128×96. Measured in the interior (margin 13 px), 1,820 edge pixels:

  | Measure | Value |
  |---|---|
  | mean \|high\| (detail amplitude) | **0.516** |
  | high band, bright vs 4× darker | mean diff **0.0077**, max **0.013** |
  | low band mean shift | **1.370** (ln 4 = 1.386), max deviation **0.029** |
  | log-gradient ratio dark/bright at edges | **0.985** |
  | linear-gradient ratio (reference) | **4.000** |

- **E4 (C-V15-3)** `RealV15RenderProducesFiniteZeroMeanDetailBand`: 8 cameras × 320×240 with noise on; all values finite; mean high **0.00044**; mean |∇l| **0.074**.

## Known limits / risks
- **R1** ε = 1e-3 radiance compresses dark regions. In E3's darkest squares (radiance ≈ 0.016) it causes the 0.013 max high-band drift. At radiance near or below ε (deep shadow at night) the decomposition loses invariance. ε should eventually track the noise floor √b / 16383 · fw ≈ 0.0012, a coupling with the step 3 R1 item.
- **R2** In log space the sensor noise is amplified at low radiance (σ_l ≈ σ_I / I). The high band in dark regions is noise-dominated. No denoising is applied, matching the "no ISP smoothing" intent; a downstream consumer must weigh it.
- **R3** A single scale (σ = 4 px at 320×240) is used, and it is not resolution-adaptive. At full-res 640×480 the effective world scale halves (TODO when wiring: scale σ with width).
- **R4** Analog only: Tesla's actual separation method is not public.
