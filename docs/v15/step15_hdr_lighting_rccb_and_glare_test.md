# Step 15 — Dynamic-Range `day` / `night_glare` Lighting & RCCB Mosaic

## 1. Scope
- **Added**: `tests/test_v15_hdr_rccb_closed_loop.cpp`.
- **Modified**: `src/config/SimConfig.{hpp,cpp}`, `config/cybercab_sim.yaml`,
  `src/main.cpp`, `src/sensors/PhotonModel.hpp`,
  `shaders/include/photon.glsl`, `shaders/scene.frag`,
  `shaders/cam_isp_soil.comp`, `src/sensors/CameraRigRenderer.cpp`,
  `CMakeLists.txt`, `docs/v15/README.md`.
- **Out of scope**: Shipped `config/cybercab_sim.yaml` default
  (`pipeline_version: "v14"`, `lighting_mode: "day"`,
  `enable_rccb_mosaic: false` unchanged).

## 2. Premises
- **P1** (`SimConfig.hpp:106-107,181`, `SimConfig.cpp:80,160-161,225-228`,
  `main.cpp:48,96-102`): Parses and validates `lighting_mode` (`"day"` |
  `"night_glare"`), `enable_rccb_mosaic`, and `--lighting` CLI flag, throwing
  `std::runtime_error` on invalid values.
- **P2** (`PhotonModel.hpp:54-71`, `shaders/include/photon.glsl:34-61`):
  Implements Clear-guided RCCB luminance collection (`kRccbClearGain = 1.65`)
  reducing dark read+shot noise variance while bypassing Clear saturation above
  `0.95 * fullWell`.
- **P3** (`shaders/scene.frag:24-27`, `shaders/cam_isp_soil.comp:30-35,76-84`,
  `CameraRigRenderer.cpp:32-55,445-480,528-532`): Applies `night_glare` diffuse
  (`0.14`) and emissive glare (`5.50`) push constants (`day` leaves `hdrParams.yz`
  at `0.0`, preserving `0.85`/`1.0` exact `day` shading) and dispatches RCCB
  14-bit linear reconstruction when `hdr_ && cfg.photonRccbMosaic`.

## 3. Trace
- **T1 (`v14`, `day`, `enable_rccb_mosaic: false`)**: `hdrParams.yz == 0.0`,
  `linearOut == 0u`, `rccbEnable == 0u`; shader output is bit-identical to pre-
  Step-15 `v14`.
- **T2 (`v15`, `night_glare`, `enable_rccb_mosaic: true`)**: 8-bit `cameraTensorSSBO()`
  clips glare `<= 1.0`, while 14-bit linear HDR `cameraLinearSSBO()` retains
  `> 10.8` glare peaks alongside `0.035` mean shadow radiance.

## 4. Claims
- **C-EQ**: With `pipeline_version: "v14"`, all `96` pre-v15 regression tests
  and `8/8` `V15Photon*` tests pass with zero regression.
- **C-V15**: RCCB reduces dark shadow variance `< 0.55x`, `night_glare` HDR
  linear tensor preserves highlights (`> 1.5`) and dark shadow detail (`> 0.005`),
  and closed-loop `night_glare` jaywalker driving achieves `0` collisions under
  budget (`< 27.8 ms`).

## 5. Evidence
- `V15HdrRccbTest.*:V15Photon*.*` (**11/11 PASS**):
  `night_max_lin_rad=10.882174`, `night_mean_shadow_rad=0.035160`,
  `night_glare_max_gpu_ms=6.504128`.
- Headless CLI check: `cybercab_sim --pipeline v15 --lighting night_glare --headless --steps 5`
  ran at `5.18–5.43 ms` GPU/step.
- Full CTest gate (`ctest --test-dir build --output-on-failure`):
  **151/151 PASS (100%, 94.67 sec)** (`96` v14 + `55` v15 tests).

## 6. Known limits
- Interactive `VulkanViewer` still displays daytime 3D Twin sky unless extended
  to read `cfg.isNightGlare()` in a later UI pass.
