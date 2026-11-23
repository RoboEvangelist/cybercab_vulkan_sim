# Step 7a: M3a Simulated 48 kHz Siren & Quad-Mic `AudioTokenizer`

## Scope
Synthesize retarded-time 4-channel 48 kHz exterior audio (`AudioTokenizer`) for `WorldEngine::triggerEmergencySiren()` and tokenize emergency presence, Doppler shift, and 360-deg bearing around blind building corners.

## Premises
- **P1** `WorldEngine.cpp:1098-1104`: `triggerEmergencySiren()` spawns `pos = {0, 1.2, 66}`, `vel = {0, 0, -11.5}` behind building `{-16, 62}`, stepped in `stepAgents(dt)` and cleared in `reset()`.
- **P2** `AudioTokenizer.cpp:9-14,62-84`: `0.11 m` roof-pod quad mic array avoids grating lobes (`< lambda/2`); `synthesize48kHzFrame` evaluates retarded emission time `tRet = phase + t - dist / 343` combining 920 Hz pilot + FM wail sweep and deterministic road noise.
- **P3** `AudioTokenizer.cpp:86-139`: Goertzel siren-band SNR triggers `sirenDetected`; `[-pi, pi)` delay-and-sum beamforming estimates bearing; `[840..1000]` Hz fine scan recovers Doppler shift.

## Trace
- **T-v14:** `v14` config/runtime never calls `AudioTokenizer::stepFromWorld`; inactive `siren_.active == false` returns `sirenDetected = false`.
- **T-v15:** `WorldEngine` steps `siren_` at `dt` -> `AudioTokenizer::stepFromWorld` emits `AudioToken` (`confidence`, `bearingVehRad`, `dopplerShiftHz`, `embed[8]`).

## Claims & Evidence (`tests/test_v15_audio_siren.cpp` 3/3 PASS)
- **C-EQ:** `InactiveSirenProducesZeroFalseAlarms` (`confidence < 0.25`).
- **C-V15-1 (`FourQuadrantBearingsAndDopplerShiftAt48kHz`):** `quadrant_max_err_deg = 0.0000 deg` across `{35, -45, 135, -140}` deg; `doppler_approach_hz = +48.0 Hz`, `doppler_recede_hz = -34.0 Hz`.
- **C-V15-2 (`BlindBuildingCornerSirenScenarioTrackedAndClearedOnReset`):** occluded corner bearing error `0.2335 deg` (`< 8.0 deg`), `confidence > 0.60`; `world.reset()` clears siren.

## Known Limits
- Free-field propagation without building diffraction/reverb; sufficient for acoustic approach & quadrant disambiguation.
