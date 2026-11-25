# Stage 1: GaussianFormer3D Spatial Representation (arXiv:2505.10685)

## 1. Architectural Role & Launch Selector
- **Role in pipeline:** When `gaussianformer3d` is enabled, it **replaces** Stage-1 spatial construction/refinement:
  - **`v14 + gaussianformer3d`:** Replaces raw post-readback voxel output by initializing SO(3) Gaussians from dynamic tracks + clustered occupancy voxels, refining them via multi-modal 3D deformable cross-attention, and splatting refined semantics back into `PerceptionFrameOutput::activeVoxels` prior to `eval` and `TrajectoryPlanner`.
  - **`v15 + gaussianformer3d`:** Ingests `v15::StaticSpatialMemory` + tracks + voxels + multi-camera 2D cues, **replaces** `GaussianWorldModel::updateFromPerception(...)` via `setExternalGaussians(...)`, and feeds refined `V15Gaussian3D` primitives into Vulkan EWA Reality View splatting (`320x180`) and 4D Gaussian trajectory scoring.
  - **`legacy` (default):** Completely untouched path; all `189` pre-existing oracle & v15/FMEA tests stay bit-identical.
- **Launch-time selection:**
  - YAML: `simulation.spatial_rep_mode: "legacy"` (`"legacy"` | `"gaussianformer3d"`) in `config/cybercab_sim.yaml`.
  - CLI: `--spatial-rep legacy|gaussianformer3d` (`SimConfig::isValidSpatialRepMode` validates both CLI and YAML).

## 2. Modular Components (`src/perception/sota/`, all <= 350 lines)
| Component | Responsibility |
|---|---|
| `Gaussian3DTypes.hpp` (185 lines) | `Quat3D` SO(3) rotation helpers, `Gaussian3DPrimitive`, `GaussianFormer3DConfig`, anisotropic Gaussian evaluation, and `toV15Gaussian3D` adapter |
| `VoxelToGaussianInit.{hpp,cpp}` (40 / 262 lines) | Dynamic track seeding, confidence-weighted 3D voxel clustering, `StaticSpatialMemory` backfill, and temporal prior blending |
| `DeformableAttention3D.{hpp,cpp}` (54 / 337 lines) | 7-point 3D ellipsoid reference offsets projected to 8 camera views + 3D voxel/track geometric attention across iterative refinement layers |
| `GaussianToVoxelSplatter.{hpp,cpp}` (53 / 231 lines) | Anisotropic mixture `P_occ(x) = 1 - prod_i (1 - alpha_i exp(-0.5 d_M^2))` semantic splatting into refined `ActiveVoxel` cells |
| `GaussianFormer3DPipeline.{hpp,cpp}` (57 / 109 lines) | Zero-steady-state-allocation orchestrator and `GaussianFormer3DSummary` computation |

## 3. Observability, Traceability & Verification
- **CLI stdout:** Prints `gf3d=<count> corr=<cam_corrob> splat=<voxels>` when `--spatial-rep gaussianformer3d` is active (`step 0`: `gf3d=512 corr=180 splat=2164`, `step 4`: `gf3d=512 corr=151 splat=2086`, GPU ~`5.0–5.6 ms` on RTX 2080 Ti).
- **`TraceLogger` JSONL:** Emits `"gaussianformer3d"` object (`active`, `used_deformable_attn`, `init_seed_count`, `gaussian_count`, `cam_corroborated_count`, `splatted_voxel_count`, `mean_opacity`, `mean_semantic_conf`, `mean_scale_m`, `attn_residual_m`).
- **`V15HudOverlay`:** Appends `| GF3D:<n>` on the HUD header line only when active while keeping `V15HudTelemetryLines` trivially copyable (`96`-byte zero-allocation buffer).
- **Tests (`196/196` CTest passing, `0` failures):** `GaussianFormer3DUnitTest` (`4/4`) + `GaussianFormer3DGpuTest` (`3/3`, incl. `half`/`full` EWA splatting and collision-free `v14`/`v15` closed-loop runs) + `189/189` baseline regression suite.
