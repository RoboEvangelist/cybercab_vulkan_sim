# Step 10a — Module 4: Self-Supervised Latent Predictor & Compute Telemetry

## 1. Scope
- **Added**: [LatentPredictor.hpp](../../src/perception/v15/LatentPredictor.hpp), [LatentPredictor.cpp](../../src/perception/v15/LatentPredictor.cpp), [test_v15_latent_predictor.cpp](../../tests/test_v15_latent_predictor.cpp).
- **Modified**: [PerceptionTypes.hpp](../../src/perception/PerceptionTypes.hpp), [V15Pipeline.hpp](../../src/perception/v15/V15Pipeline.hpp), [V15Pipeline.cpp](../../src/perception/v15/V15Pipeline.cpp), [CMakeLists.txt](../../CMakeLists.txt), [README.md](README.md).
- **Out of scope**: Autoregressive trajectory policy and lane-centering control (Step 10b).

## 2. Premises
- **P1** (`LatentPredictor.cpp:42-70`): Evaluates yaw-compensated `3x3` local minimum L1 appearance + high-pass gradient residuals across forward corridor views (`0, 1, 2, 8`) from `PatchTokenizer` slots `t-1 -> t` and averages the top-16 peak innovations.
- **P2** (`LatentPredictor.cpp:25-39`): Computes `tokenSparsityRatio`, `staticSkipRatio`, `kvCacheReuseRatio` (`0.75` at 4 frames), and `effectiveSpeedupX` including 2x FP16 tensor speedup.
- **P3** (`V15Pipeline.cpp:118-137`): Adds `surpriseSigmaDelta` onto `forwardEpistemicSigma` and populates `out.v15` (including acoustic siren, preferred lane center, and rear clearance).

## 3. Trace
- **T1 (`v14`)**: `v15_` is null; `out.v15.active` stays `false` and `forwardEpistemicSigma` is untouched.
- **T2 (`v15`)**: `V15Pipeline::finishBackEnd` evaluates `LatentPredictor`, inflates `forwardEpistemicSigma` on sudden corridor novelty, and records compute telemetry.

## 4. Claims
- **C-EQ**: Under `pipeline_version: v14`, perception outputs and uncertainties remain bit-identical.
- **C-V15**: Under `v15`, nominal steady driving stays low novelty (`< 0.12`), synthetic & jaywalker anomalies spike `latentNovelty` and `forwardEpistemicSigma`, and compute telemetry satisfies `kvCacheReuseRatio == 0.75`, `effectiveSpeedupX > 2.0`.

## 5. Evidence
- `V15LatentPredUnitTest.*` (`1/1 PASS`) & `V15LatentPredTest.*` (`1/1 PASS`):
  - `m4_nom_novelty = 0.0897`, `m4_jaywalk_novelty = 0.1626`, `m4_kv_reuse = 0.7500`, `m4_speedup_x = 7.1535`.

## 6. Known limits
- Warped prediction uses discrete yaw shift + `3x3` patch search rather than dense metric depth reprojection.
