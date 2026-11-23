# Step 2: M1a ISP-bypass linear 14-bit photon-count tensor

Tutorial Module 1.2 ("photon counting"): feed the network linear sensor values with HDR headroom
instead of an 8-bit display image that clips highlights.

## Scope
- v15 renders the camera raw scene-referred (no clip at display white) and the ISP writes an extra
  tensor of 14-bit linear ADC codes.
- **Out of scope:** sensor noise (step M1b), frequency separation (M1c), and any v15 consumer of the tensor.
- Files: `src/sensors/PhotonModel.hpp` (new), `shaders/include/photon.glsl` (new), `shaders/scene.frag`,
  `shaders/cam_isp_soil.comp`, `src/sensors/CameraRigRenderer.{hpp,cpp}`, `src/app/VulkanViewer.cpp`,
  `src/core/VulkanImage.cpp`, `CMakeLists.txt`, `tests/test_v15_photon_frontend.cpp` (new).

## Premises
- **P1** `CameraRigRenderer.cpp:42`: `hdr_ = cfg.isV15();`, so v14 gives `hdr_ == false` (step 1, P1-P3).
- **P2** `CameraRigRenderer.hpp:86`: `rawColorFormat()` is `R8G8B8A8_UNORM` when `!hdr_` and `R16G16B16A16_SFLOAT` otherwise.
  It is used for the render-pass attachment (`.cpp:123`) and the raw image (`.cpp:281`). Both were the literal `R8G8B8A8_UNORM` before.
- **P3** `scene.frag:39`: output = `clamp(x, 0, hdrParams.x)`. It was `clamp(x, 0, 1)`.
  - Camera rig sets `hdrParams.x = hdr_ ? 16 : 1` (`CameraRigRenderer.cpp:442`).
  - Showcase BMP sets 1 (`.cpp:590`).
  - Viewer sets 1 (`VulkanViewer.cpp` `ScenePC` + fill site).
- **P4** `cam_isp_soil.comp:64-66`: `display = min(rgb, 1)` is written to both the RGBA8 image and the v14 tensor.
- **P5** `cam_isp_soil.comp:67-68`: binding 3 is written **only if** `linearOut != 0`.
  `linearOut = hdr_ ? 1 : 0` (`CameraRigRenderer.cpp:484`).
- **P6** `CameraRigRenderer.cpp:307`: binding-3 buffer is the full tensor size when `hdr_`, else 16 bytes (a valid descriptor that is never accessed when P5 is false).
- **P7** `VulkanImage.cpp:234`: `downloadRGBA8` takes the new half-float path only for `R16G16B16A16_SFLOAT`; otherwise the original byte copy runs unchanged.
- **P8** `cam_isp_soil.comp` before `min`, for UNORM raw:
  - sampled `rgb` is in [0,1];
  - vignetting factor `1 - 0.11·|nrm|²` is at most 1;
  - the soiled branch mixes values at most 1 (`mud` max ≈ (103,78,44)/255) or explicitly clamps.
  - So `rgb` stays in [0,1].
- **P9** `CameraRigRenderer.cpp:31`: `static_assert(sizeof(IspPushConstants) == 64)`. The new `uint linearOut; float fullWell;` occupy previously padded offsets 56/60, matching the GLSL block order.

## Trace
**v14 (`hdr_ == false`):**
- **T1** Render: P2 gives the same UNORM8 target. P3 with `hdrParams.x = 1` gives the same clamp. So the raw image is bit-identical.
- **T2** ISP: by P8 `rgb ∈ [0,1]`, so `min(rgb,1) = rgb` (P4). The image and v14 tensor are bit-identical.
- **T3** P5 is false, so binding 3 is never written. P6 binding is valid, so there are no validation or OOB effects.
- **T4** Readback: P7 takes the original byte path.
- **T5** Viewer: `hdrParams.x = 1`, so the clamp is unchanged. Without this fix the viewer would read an undefined push-constant range, likely 0, and render black. Fixed in this step.

**v15 (`hdr_ == true`):**
- **T6** Render: float16 target, clamp at 16, so radiance above 1 survives.
- **T7** ISP: the v14 tensor gets `min(rgb,1)` (display-referred). The photon tensor gets `floor(clamp(rgb/16,0,1)·16383+0.5)/16383` (`photon.glsl`), which is the same math as `PhotonModel::linearCode`.
- **T8** Showcase BMP: rendered with `hdrParams.x = 1`, read via the half-float path (P7) and converted to 8-bit. Before this fix, `downloadRGBA8` would have copied 8 B/px into a 4 B/px buffer (overflow).

## Claims
- **C-EQ:** by T1-T5, v14 camera raw, display image, v14 tensor, readback and viewer output are identical to the parent commit.
- **C-V15-1:** below display white, photon radiance equals display radiance within half an ADC step (≤ 16/16383/2 ≈ 4.9e-4).
- **C-V15-2:** highlights above white are preserved instead of clipped.
- **C-V15-3:** 14-bit linear coding resolves dark levels that 8-bit gamma merges (tutorial exercise 1.3).

## Evidence
- **E1 (C-EQ)** All 96 pre-v15 tests pass. `V15PhotonGpuTest.V14ModeLeavesPhotonTensorUnallocatedAndRawClipped` passes: binding-3 buffer is 16 B and v14 tensor max ≤ 1.
- **E2 (C-V15-1)** `LinearTensorTracksDisplayAndKeepsHighlightHeadroom`: max abs error **0.0005**, mean **0.0002**.
- **E3 (C-V15-2)** Same test, nominal scene at t=0:

  | Measure | Value |
  |---|---|
  | max radiance | **2.41× white** |
  | pixels clipped by v14 | **7,483** |
  | pixels above white kept by v15 | **7,460** |
  | photon pixels saturated | **0** |
  | scene dynamic range | **33.5 dB** |

- **E4 (C-V15-3)** `V15PhotonModelTest.Linear14BitPreservesDarkContrastThatGamma8Crushes`: levels 12..18 give **7** distinct linear codes vs **3** gamma-8 codes. Linear error is ≤ 0.5 LSB; gamma-8 error is > 10 counts over 8..4096.
- **E5** Headless `cybercab_sim --config <v15 yaml> --car-dump` runs 60 steps; the BMP is non-black (mean byte 82).
- **E6** ctest **102/102** pass.

## Known limits / risks
- **R1** The float16 raw has relative precision 2⁻¹¹. Above radiance 2 its step (≥ 0.002) is coarser than the 14-bit LSB (0.00098), so highlight precision is limited by fp16, not the ADC. This is negligible next to the shot noise added in M1b.
- **R2** In v15 mode the v14 perception stages read `min(float raw, 1)` instead of 8-bit-quantized values, which differ by ≤ 1/510. v14 perception quality *under v15 mode* is not yet re-validated; that is tracked for the end-to-end v15 step.
- **R3** The rasterizer's `lit` is treated as scene radiance. There is no physical spectral or exposure model (analog).
