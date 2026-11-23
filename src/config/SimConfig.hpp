#pragma once

#include "config/V15PolicyConfig.hpp"
#include "core/Math.hpp"
#include <array>
#include <string>
#include <vector>

namespace cybercab {

inline constexpr int kLegacyHalfCamW = 320;
inline constexpr int kLegacyHalfCamH = 240;
inline constexpr int kLegacyFullCamW = 640;
inline constexpr int kLegacyFullCamH = 480;
inline constexpr int kCybercabBinnedCamW = 1448;
inline constexpr int kCybercabBinnedCamH = 938;
inline constexpr int kCybercabNativeCamW = 2896;
inline constexpr int kCybercabNativeCamH = 1876;

struct CamResPreset {
  const char* mode;
  int width;
  int height;
  const char* label;
};

struct CameraIntrinsics {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float hfovRad = 0.0f;
    float vfovRad = 0.0f;
    float vfovDeg = 0.0f;
  int width = kLegacyHalfCamW;
  int height = kLegacyHalfCamH;
};

struct CameraSpec {
    std::string id;
    std::string shortName;
    std::string desc;
    float hfovDeg = 50.0f;
    float maxRangeM = 58.0f;
    Vec3 posIso{};          // ISO 8855 Ego frame: X fwd, Y left, Z up
    Vec3 rpyDeg{};          // roll (deg), pitch (deg, negative=down), yaw (deg, positive=left)
    std::array<float, 4> distortion{0.0f, 0.0f, 0.0f, 0.0f}; // k1, k2, p1, p2
    Vec3 uiColor{0.22f, 0.74f, 0.97f};
    bool soiled = false;    // Ground-truth simulation injection flag

    CameraIntrinsics intrinsics(int width, int height) const {
        CameraIntrinsics out;
        out.width = width;
        out.height = height;
        float aspect = static_cast<float>(width) / static_cast<float>(height);
        out.hfovRad = degToRad(hfovDeg);
        out.vfovRad = 2.0f * std::atan(std::tan(out.hfovRad * 0.5f) / aspect);
        out.vfovDeg = radToDeg(out.vfovRad);
        out.fx = static_cast<float>(width) / (2.0f * std::tan(out.hfovRad * 0.5f));
        out.fy = static_cast<float>(height) / (2.0f * std::tan(out.vfovRad * 0.5f));
        out.cx = static_cast<float>(width) * 0.5f;
        out.cy = static_cast<float>(height) * 0.5f;
        return out;
    }

    // Pinhole render overscan so Brown-Conrady barrel distortion fills the image (single source of truth).
    float renderOverscan() const { return (distortion[0] < -0.14f) ? 1.24f : 1.04f; }

    // Extrinsic T_{ego <- cam} in ISO 8855 Ego frame (X fwd, Y left, Z up)
    // to CV Camera frame (X right, Y down, Z fwd).
    Mat4 extrinsicEgoFromCamISO() const {
        Mat4 B0{};
        // col 0: cam +X (right) = -Y_ego
        B0.at(1, 0) = -1.0f;
        // col 1: cam +Y (down) = -Z_ego
        B0.at(2, 1) = -1.0f;
        // col 2: cam +Z (fwd) = +X_ego
        B0.at(0, 2) = 1.0f;
        B0.at(3, 3) = 1.0f;

        float r = degToRad(rpyDeg.x);
        float p = degToRad(rpyDeg.y);
        float y = degToRad(rpyDeg.z);
        // Positive yaw turns left (+Z_ego), negative pitch tilts nose down (-(-pitch) around +Y_ego)
        Mat4 R = Mat4::rotZ(y) * Mat4::rotY(-p) * Mat4::rotX(r) * B0;
        return Mat4::translation(posIso) * R;
    }

    Mat4 extrinsicCamFromEgoISO() const {
        return extrinsicEgoFromCamISO().inverseRigid();
    }

    // Extrinsic T_{vehLocal <- cam} where Vehicle Local is (+X left, +Y up, +Z fwd).
    Mat4 extrinsicVehicleLocalFromCam() const {
        Mat4 T_ego_cam = extrinsicEgoFromCamISO();
        // P_local = M_local_from_ego * P_ego, where local = (ego.y, ego.z, ego.x)
        Mat4 M{};
        M.at(0, 1) = 1.0f; // local X = ego Y
        M.at(1, 2) = 1.0f; // local Y = ego Z
        M.at(2, 0) = 1.0f; // local Z = ego X
        M.at(3, 3) = 1.0f;
        return M * T_ego_cam;
    }
};

struct SimConfig {
    // Simulation
    float controlHz = 36.0f;
  int camWidth = kLegacyHalfCamW;
  int camHeight = kLegacyHalfCamH;
  int fullCamWidth = kLegacyFullCamW;
  int fullCamHeight = kLegacyFullCamH;
  int cybercabBinnedCamWidth = kCybercabBinnedCamW;
  int cybercabBinnedCamHeight = kCybercabBinnedCamH;
  int cybercabCamWidth = kCybercabNativeCamW;
  int cybercabCamHeight = kCybercabNativeCamH;
  std::string initialResMode = "half";
    std::string initialViewMode = "chase";
    std::string initialCamOverlay = "neural";
    std::string driveMode = "auto";
    bool aebEnabled = true;
    bool useVulkanValidation = false;
    // Perception architecture selector: "v14" (RT-DETR/BEVFormer/voxel) | "v15" (photon-count/Plücker/Gaussian)
    std::string pipelineVersion = "v14";
    bool isV15() const { return pipelineVersion == "v15"; }
  std::string lightingMode = "day";
  bool isNightGlare() const { return lightingMode == "night_glare"; }
  Vec3 cameraKnockRpyDeg{0.0f, 0.0f, 0.0f};
    std::string v15PolicyPath = "config/v15_policy.yaml";
    V15PolicyConfig v15Policy{};

    // Backbone
    int featWidth = 40;
    int featHeight = 30;
    int featChannels = 16;

    // Soil Detector
    int soilPatchCols = 10;
    int soilPatchRows = 8;
    float soilEdgeThreshold = 0.045f;
    float soilGlareThreshold = 0.82f;
    float soilMinTokenWeight = 0.18f;
    float soilBaseEpistemicVar = 0.04f;
    float soilVarGain = 0.85f;

    // RT-DETR
    int rtdetrNumQueries = 16;
    int rtdetrDecoderLayers = 2;
    float rtdetrHighConfThresh = 0.46f;
    float rtdetrLowConfThresh = 0.22f;
    float rtdetrNmsIouThresh = 0.45f;
    float rtdetrMaxRangeM = 65.0f;

    // ByteTrack
    int btMaxTracks = 64;
    float btMaxTimeLostSec = 1.5f;
    float btHighMatchDistM = 4.0f;
    float btLowMatchDistM = 5.2f;
    float btProcessNoisePos = 0.18f;
    float btProcessNoiseVel = 0.85f;
    float btBaseMeasNoiseM = 0.45f;

    // BEVFormer
    int bevNx = 60;
    int bevNy = 40;
    float bevMinX = -22.0f;
    float bevMaxX = 38.0f;
    float bevMinY = -20.0f;
    float bevMaxY = 20.0f;
    std::vector<float> bevZAnchors{0.45f, 1.05f, 1.75f, 2.65f};
    int bevChannels = 16;
    float bevTemporalBlend = 0.45f;

    // Occupancy & Flow
    float occMinX = -22.0f;
    float occMaxX = 38.0f;
    float occStepX = 1.0f;
    float occMinY = -20.0f;
    float occMaxY = 20.0f;
    float occStepY = 1.0f;
    float occMinZ = 0.25f;
    float occMaxZ = 4.25f;
    float occStepZ = 0.5f;
    float occDecayRate = 1.15f;
    float occThreshold = 0.38f;
    bool occEnableShadows = true;

    int occNx() const { return static_cast<int>(std::round((occMaxX - occMinX) / occStepX)); }
    int occNy() const { return static_cast<int>(std::round((occMaxY - occMinY) / occStepY)); }
    int occNz() const { return static_cast<int>(std::round((occMaxZ - occMinZ) / occStepZ)); }
    int occTotalVoxels() const { return occNx() * occNy() * occNz(); }

    // World Model
    std::vector<float> wmHorizonsSec{0.5f, 1.0f, 1.5f, 2.0f};
    float wmUncertaintyGrowthRate = 0.90f;

    // v15 Module 1 photon sensor (Poisson-Gaussian noise: var_DN = a * mean_DN + b), used only when isV15()
    bool photonNoiseEnabled = true;
    float photonShotNoiseA = 0.4096f; // DN per DN (= 1 / gain[e-/DN]); 40000 e- full well over 14-bit ADC
    float photonReadNoiseB = 1.51f;   // DN^2 (= (read_noise_e / gain)^2); 3 e- read noise
    uint32_t photonNoiseSeed = 1234u;
  bool photonRccbMosaic = false;

    // Planning & RSS
    float targetCruiseSpeedMs = 10.5f;
    int numPlannerCandidates = 9;
    int trajSteps = 24;
    float rssTReactSec = 0.25f;
    float rssABrakeMs2 = 6.5f;
    float rssABrakeAebMs2 = 8.5f;
    float rssDBufferM = 2.6f;
    float rssUncertaintyBufferGain = 1.8f;

    // Vehicle
    float wheelbaseM = 2.75f;
    float maxSteerRad = 0.54f;
    float egoHx = 0.96f;
    float egoHy = 0.70f;
    float egoHz = 2.19f;
    float egoCenterOffsetFwd = 1.25f;
    float suspensionKSpring = 68.0f;
    float suspensionCDamper = 11.5f;

    std::vector<CameraSpec> cameras;

  std::array<CamResPreset, 4> resolutionPresets() const;
  CamResPreset resolveInitialRes() const;
  CamResPreset nextResolutionPreset(int curW, int curH) const;
  static bool isValidResMode(const std::string& mode);

    static SimConfig loadFromYaml(const std::string& path);
    static SimConfig defaultConfig();
};

} // namespace cybercab
