# Step 9 — Module 6: 3D/4D Gaussian World Model & EWA Reality View

## 1. Scope
- **Added**: [GaussianWorldModel.hpp](../../src/perception/v15/GaussianWorldModel.hpp), [GaussianWorldModel.cpp](../../src/perception/v15/GaussianWorldModel.cpp), [shaders/v15_gaussian_splat.comp](../../shaders/v15_gaussian_splat.comp), [test_v15_gaussian_world.cpp](../../tests/test_v15_gaussian_world.cpp).
- **Modified**: [V15Pipeline.hpp](../../src/perception/v15/V15Pipeline.hpp), [V15Pipeline.cpp](../../src/perception/v15/V15Pipeline.cpp), [PerceptionPipeline.cpp](../../src/perception/PerceptionPipeline.cpp), [CMakeLists.txt](../../CMakeLists.txt), [README.md](README.md).
- **Out of scope**: Latent predictor novelty (Step 10a), planner policy (Step 10b).

## 2. Premises
- **P1** (`GaussianWorldModel.cpp:9-33`, `v15_gaussian_splat.comp:30-56`): Exact Zwicker/Kerbl EWA projection `Sigma2D = J W R diag(sx^2, sy^2, sz^2) R^T W^T J^T + 0.3 I_2`; front-to-back depth sort + alpha transmittance compositing renders a `320x180` Reality View RGBA image.
- **P2** (`GaussianWorldModel.cpp:35-65`): GaussianFormer-2 occupancy `P(x,t) = 1 - prod_i (1 - alpha_i f_i(x,t))` stays in `[0, 1]` and advances dynamic Gaussians linearly with `(vx, vz)` and `1 + 0.15t` uncertainty scale.
- **P3** (`GaussianWorldModel.cpp:154-213`): Fuses `out.tracks`, view-8 (`-12 deg`) VRU edge boost, and locked `StaticSpatialMemory` cells into up to `1024` Gaussians; reinforces forward static obstacles in `out.activeVoxels`.

## 3. Trace
- **T1 (`v14`)**: `v15_` is `nullptr`; no Gaussian world updates or Reality View splats run.
- **T2 (`v15`)**: `PerceptionPipeline::runFrame` calls `v15_->finishBackEnd(...)` after `wm_.readbackGhosts`, updating Gaussians and dispatching `6_v15_gaussian_splat`.

## 4. Claims
- **C-EQ**: Under `pipeline_version: v14`, voxel/ghost outputs and timings are untouched.
- **C-V15**: Under `v15`, EWA `Sigma2D` matches finite-difference Jacobian `< 1e-3`, depth sorting enforces front-to-back occlusion, `P(x,t)` is bounded in `[0,1]`, and Reality View splat completes `< 1 ms`.

## 5. Evidence
- `V15GaussianWorldTest.*` (`2/2 PASS`) & `V15PipelineTest.*` (`3/3 PASS`):
  - `m6_ewa_fd_rel_err = 0.0006`, `m6_gaussians = 1024`, `m6_splat_ms = 0.7146`, `v15_gpu_sum_ms = 5.2616`, `v15_wall_ms = 8.2325`.

## 6. Known limits
- Gaussians use yaw-only orientation and isotropic per-axis extent uncertainty (`1 + 0.15t`); closed-loop planner consumption is deferred to Step 10b.
