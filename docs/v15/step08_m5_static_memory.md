# Step 8 — Module 5: Persistent SE(3) Static Spatial Memory

## 1. Scope
- **Added**: [shaders/v15_static_memory.comp](../../shaders/v15_static_memory.comp), [StaticSpatialMemory.hpp](../../src/perception/v15/StaticSpatialMemory.hpp), [StaticSpatialMemory.cpp](../../src/perception/v15/StaticSpatialMemory.cpp), [test_v15_static_memory.cpp](../../tests/test_v15_static_memory.cpp).
- **Modified**: [V15Pipeline.hpp](../../src/perception/v15/V15Pipeline.hpp), [V15Pipeline.cpp](../../src/perception/v15/V15Pipeline.cpp), [PerceptionPipeline.cpp](../../src/perception/PerceptionPipeline.cpp), [CMakeLists.txt](../../CMakeLists.txt), [README.md](README.md).
- **Out of scope**: Gaussian splatting (Step 9), downstream planner consumers (Step 10b).

## 2. Premises
- **P1** (`StaticSpatialMemory.cpp:112-128`, `v15_static_memory.comp:34-60`): Pass `0` warps `prevMem_` (`60x40` BEV grid) by `prevIsoFromCurrIso = worldFromIsoEgo(prevEgo).inverseRigid() * worldFromIsoEgo(ego)` using nearest-locked sampling (`round(fx), round(fy)`) to avoid bilinear wake smearing on turns.
- **P2** (`v15_static_memory.comp:73-135`): Pass `1` masks dynamic `TrackPriorGpu` radii (`dynamicMaskedCount`), lifts `PatchTokenGpu` gradients/energy across all 10 virtual views at `zAnchors`, and sets `staticLocked` when `conf >= 0.80`. Locked cells with `dynMask || viewsHit < 2 || unchangedLocked` skip writes (`skippedCount`), retaining `conf >= 0.76`.
- **P3** (`PerceptionPipeline.cpp:105-110`, `V15Pipeline.cpp:103-108`): `v15_->recordBackEnd(...)` runs in Batch 2 (`5_v15_static_mem`) only when `v15_` is non-null (`cfg.isV15()`).

## 3. Trace
- **T1 (`v14`)**: `v15_` is `nullptr` in `PerceptionPipeline`; neither `StaticSpatialMemory` nor `5_v15_static_mem` executes.
- **T2 (`v15`)**: `V15Pipeline::init` allocates `60x40` static memory buffers; Batch 2 executes Pass 0 warp + Pass 1 selective refresh + `vkCmdCopyBuffer(curr -> prev)`.

## 4. Claims
- **C-EQ**: Under `pipeline_version: v14`, Batch 2 stages and outputs are unchanged.
- **C-V15**: Under `v15`, static cells lock at `conf >= 0.80`, selective refresh skip ratio exceeds `0.30`, cells stay locked (`conf > 0.75`) over 16 frames of synthetic truck occlusion, and SE(3) ego pose steps warp cells without wake blur.

## 5. Evidence
- `V15StaticMemoryTest.*` (`2/2 PASS`) & `V15PipelineTest.*` (`3/3 PASS`):
  - `m5_locked_cells = 2377`, `m5_skip_ratio = 1.0000`, `m5_occluded_conf = 0.8782` (after 16 masked frames + ego translation/yaw warp), `5_v15_static_mem` stage present in `v15`.
  - `v15_gpu_sum_ms = 4.6451`, `v15_wall_ms = 11.1580` (`< 27.8 ms` 36 Hz budget).

## 6. Known limits
- Static cells are axis-aligned `60x40` (`1 m x 1 m`); continuous 3D covariance and Reality View projection are deferred to Step 9 (`GaussianWorldModel`).
