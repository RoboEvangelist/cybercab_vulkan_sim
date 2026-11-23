#pragma once
// Configurable v15 autoregressive policy, safety shield, and latent predictor parameters.
#include <string>

namespace cybercab {

struct V15PolicyConfig {
    // Lane centering & heavy-truck psychological shy-away
    float laneCenterAbsM = 1.95f;
    float minLaneAbsM = 1.65f;
    float maxLaneAbsM = 2.25f;
    float maxTruckShyM = 0.22f;
    float truckHalfLengthMinM = 3.20f;
    float truckShyFwdMinM = -6.0f;
    float truckShyFwdMaxM = 26.0f;
    float adjLaneMinSepM = 1.80f;
    float adjLaneMaxSepM = 5.50f;

    // Autoregressive lateral & curvature speed rollout
    float minLookaheadM = 18.0f;
    float lookaheadTimeSec = 2.10f;
    float lookaheadBaseM = 7.50f;
    float latGainPerS = 2.40f;
    float maxLatSpeedMs = 2.15f;
    float rolloutMinSpeedMs = 6.0f;
    float maxLatAccelMs2 = 4.60f;
    float minCurveSpeedMs = 7.40f;
    float curveChordHalfM = 4.0f;

    // Lane commitment & divider hysteresis
    float rightLaneSwitchCost = 180.0f;
    float leftLaneReturnHoldCost = 320.0f;
    float leftLaneStayCost = 18.0f;
    float dividerBandM = 1.50f;
    float headingCommitThresh = 0.03f;
    float headingCommitCost = 260.0f;
    float crossingDevThreshM = 0.28f;
    float intersectionLaneLockCost = 8500.0f;
    float blindSpotRearGapM = 3.60f;
    float blindSpotFwdGapM = 4.80f;
    float blindSpotClosingTimeS = 2.00f;
    float blindSpotLatPadM = 0.55f;
    float blindSpotBlockCost = 2600.0f;
    float obbHullSampleScale = 0.75f;
    float vehClearancePadM = 0.35f;

    // Safety shield, siren yield & Autoshift D/R
    float trackCorridorVehM = 1.22f;
    float trackCorridorPedM = 1.65f;
    float voxelCorridorVehM = 0.92f;
    float voxelCorridorPedM = 1.45f;
    float voxelMinConf = 0.56f;
    float adjVehVoxelSuppressM = 2.60f;
    float adjVehTrackMinLatM = 1.30f;
    float vehHullSafetyPadM = 0.38f;
    float minHardAebDistM = 1.85f;
    float aebRssFrac = 0.75f;
    float aebRelSpeedMinMs = 1.10f;
    float followRampSpanM = 12.0f;
    float followSpeedGain = 0.50f;
    float sirenConfThresh = 0.35f;
    float sirenStopConfThresh = 0.55f;
    float sirenYieldSpeedMs = 1.20f;
    float sirenRightPullM = -2.05f;
    float autoshiftMinRearClearM = 6.0f;
    float autoshiftReverseSpeedMs = -1.60f;

    // Self-supervised latent predictor
    float latentSurpriseBaseline = 0.22f;
    float latentSurpriseScale = 0.45f;
    float latentSigmaGain = 0.34f;

    static V15PolicyConfig loadFromYaml(const std::string& path);
};

} // namespace cybercab
