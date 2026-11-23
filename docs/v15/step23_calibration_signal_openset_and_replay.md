# Step 23: Per-Condition Calibration, Vision Signal FSM, Open-Set Obstacles & Shadow Replay

## 1. Scope
Closes the Step 22 `night_glare` calibration gap and expands `v15` perception/validation:
- [ConfidenceCalibrator.hpp](../../src/perception/ConfidenceCalibrator.hpp),
  [PerceptionPipeline.cpp](../../src/perception/PerceptionPipeline.cpp)
- [SignalPerception.hpp](../../src/perception/SignalPerception.hpp),
  [SignalClassifier.cpp](../../src/perception/SignalClassifier.cpp),
  [SignalFilter.cpp](../../src/perception/SignalFilter.cpp)
- [WorldEngine.hpp](../../src/world/WorldEngine.hpp),
  [WorldEngine.cpp](../../src/world/WorldEngine.cpp),
  [BackboneRTDETR.hpp](../../src/perception/BackboneRTDETR.hpp),
  [BackboneRTDETR.cpp](../../src/perception/BackboneRTDETR.cpp),
  [shaders/02_backbone_fpn.comp](../../shaders/02_backbone_fpn.comp)
- [StaticVoxelGuard.hpp](../../src/planning/v15/StaticVoxelGuard.hpp),
  [AutoregSafety.cpp](../../src/planning/v15/AutoregSafety.cpp),
  [AutoregressivePolicy.cpp](../../src/planning/v15/AutoregressivePolicy.cpp)
- [AvScenarioHierarchy.hpp](../../tests/AvScenarioHierarchy.hpp) & 4 new suites
  (plus tightened `kMaxEceNight = 0.28` in
  [test_av_ece_confusion_and_slicing.cpp](../../tests/test_av_ece_confusion_and_slicing.cpp))

## 2. Premises
- **P1 (`PerceptionPipeline.cpp:141`):** `if (cfg_.isV15()) calib::calibrateFrame(cfg_, out);` runs after `v15_->finishBackEnd` so `GaussianWorldModel` inputs are unchanged.
- **P2 (`SignalClassifier.cpp`, `SignalFilter.cpp`):** `VisionSignalClassifier` classifies Red/Yellow/Green bulbs strictly from rendered `CAM_FRONT_NARROW`/`CAM_FRONT_MAIN` RGBA buffers; `SignalLaneAssociator` matches lane + maneuver; `SignalTemporalFilter` combines a 4-state HMM with asymmetric hysteresis (3-frame `Red->Green`, fast `Red`).
- **P3 (`WorldEngine.cpp:669-688,1085-1104`):** Split-phase overrides and `UnknownObstacleSpec` SAT/rendering are no-ops unless explicitly populated.
- **P4 (`BackboneRTDETR.cpp:265`, `02_backbone_fpn.comp:73-84,172-174`):** `pc.v15Mode` replaces `pad0` (96 B ABI preserved); `detectRaisedCargoBox` returns `false` when `pc.v15Mode == 0u`.
- **P5 (`StaticVoxelGuard.hpp:18-37`, `AutoregSafety.cpp:130-151`, `AutoregressivePolicy.cpp:238-241`):** `Static` voxels engage `v15` corridor/rollout avoidance only when grounded (`wy in [0.35, 1.85]`), on-road (`|lat| <= 3.1 m`), live (`epistemicSigma >= 0.10`), high-confidence (`> 0.78`), and not adjacent-vehicle fringe.
- **P6 (`AvScenarioHierarchy.hpp:41-162`):** `ScenarioHierarchyReport` enforces zero critical errors/IDSW/collisions; `MultiAxisSliceTable` prevents micro-average masking; `parseTraceJsonl` summarizes JSONL shadow logs.

## 3. Trace & Claims
- **C-EQ (`v14`):** `cfg_.isV15() == false` skips `calibrateFrame` and sets `pc.v15Mode = 0u` (`detectRaisedCargoBox == false`), preserving bit-identical `v14` outputs across all `96/96` oracle tests.
- **C-V15:** Per-condition temperature scaling reduces `v15` day/night/soiled ECE and overconfidence; pure-vision signal classification achieves zero `Red->Green` errors with split-phase lane association and glitch-blocking FSM hysteresis; `v15` detects and safely stops/steers around open-set `FALLEN_CARGO` & `JERSEY_BARRIER` with zero empty-road ghosts and intact `red_queue` release; shadow replay is deterministic across paired `v14`/`v15` traces.

## 4. Evidence (`15/15` Targeted XML in `69.87 s`; `189/189` CTest in `307.37 s`)
- **Calibration (`AvConfidenceCalibTest`, `AvCalibrationSlicingTest`):**
  - `v15_day`: ECE `0.2230 -> 0.2088`, OCE `0.1291 -> 0.1103`
  - `v15_night`: ECE `0.3934 -> 0.2527`, OCE `0.2488 -> 0.1041` (`kMaxEceNight=0.28`)
  - `v15_soiled`: ECE `0.2354 -> 0.2196`, OCE `0.1710 -> 0.1500`
  - `v15_full` ECE `0.1934`, `v15_cybercab_binned` ECE `0.1421` (`0` collisions)
- **Vision Signal & FSM (`AvVisionSignalTest`, `2/2`):** `sig_vision_acc=1.0000`, `sig_red_to_green_cnt=0`, `sig_safety_cost=0.0000`, `sig_split_assoc_acc=1.0000`, `sig_glitch_leaks=0`, `sig_red_to_green_lat_frames=3`, `sig_green_to_red_lat_frames=1`.
- **Open-Set Obstacles & Regression Guards (`AvOpenSetObstacleTest`, `EmptyRoad*`, `FmeaEmptyRoad*`, `V15EndToEndClosedLoopTest`):**
  - `cargo_half`: `vox=6421`, `fs_err=0.0024 m`, `min_clear=0.6543 m`
  - `cargo_full`: `vox=5985`, `fs_err=0.0009 m`, `min_clear=0.5958 m`
  - `barrier_half`: `vox=707`, `fs_err=0.0125 m`, `min_clear=0.2929 m`
  - `0` empty-road ghosts/AEB across all 4 presets; `v15_red_queue_advance_m=2.3024`
- **Hierarchy & Shadow Replay (`AvShadowReplayHierarchyTest`, `2/2`):** rejects `299/300` critical failure dilution and `1/3` weak-slice failure; paired replay `ab_det_pos_diff_m=0.0000`, `ab_v14_recall=0.6730`, `ab_v15_recall=0.6444`, `ab_v14_prec=0.9364`, `ab_v15_prec=0.9411`, `ab_v14_gpu_ms=1.6441`, `ab_v15_gpu_ms=5.1392`.
- **Full CTest (`ctest --test-dir build`):** `189/189 PASS` (`0 failed`, `307.37 s`).

## 5. Known Limits
- Temperature/bias calibration parameters are condition-specific offline fits (`kDayCalib`, `kNightCalib`, `kSoiledCalib`); online EM calibration and closed-loop actuator integration of `VisionSignalClassifier` into `WorldEngine` intersection legs remain deferred.
