# Step 5a: Dark read-noise verification (`step03 R1`) & adaptive `FreqSplitParams` (`step04 R1/R3`)

## Scope
Close open Module 1 gaps before Module 2: (1) pin Poisson-Gaussian read-noise intercept `b`
via dark-scene `cam_isp_soil.comp` sweep, and (2) add `FreqSplitParams::forConfig` + `rebind(..., params)`
scaling `sigmaPx` with width and `logEps` with `sqrt(readNoiseB)`.

## Premises
- **P1** `FrequencySplit.hpp:23-30`: `forConfig(camW, cfg)` sets `sigmaPx = 4 * max(1, camW/320)` and
  `logEps = max(1e-3, 2 * sqrt(photonReadNoiseB) / 16383 * fullWell)`.
- **P2** `FrequencySplit.cpp:42-46`: `rebind(linearIn, camW, camH, params)` updates `params_` then delegates to `rebind`.
- **P3** `test_v15_photon_noise.cpp:163-227`: clears an `R16G16B16A16_SFLOAT` image to dark codes `{8, 16, 24, 36}` DN
  and dispatches `cam_isp_soil.comp.spv` cleanly (`pass=0`) vs noisy (`pass=1`).

## Claims & Evidence
- **C-EQ:** runtime production paths untouched (only new helper overloads).
- **C-V15-1 (`step03 R1`):** `DarkScenePinsReadNoiseInterceptB`: `dark_fit_a_hat = 0.4106` (vs `0.4096`),
  `dark_fit_b_hat = 1.6697` vs `1.6767` (`-0.007 DN^2`), `ctest` **111/111**.
- **C-V15-2 (`step04 R1/R3`):** `ConfigScalingAdaptsSigmaAndNoiseFloor`: half-res `4px`, full-res `8px`, noise-scaled `logEps`.
