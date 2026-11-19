#include "config/SimConfig.hpp"

#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace cybercab {

namespace {

template <typename T>
T readOr(const YAML::Node& node, const char* key, const T& fallback) {
    if (node && node[key]) {
        return node[key].as<T>();
    }
    return fallback;
}

Vec3 readVec3Or(const YAML::Node& node, const char* key, const Vec3& fallback) {
    if (node && node[key] && node[key].IsSequence() && node[key].size() == 3) {
        return {
            node[key][0].as<float>(),
            node[key][1].as<float>(),
            node[key][2].as<float>()
        };
    }
    return fallback;
}

} // namespace

SimConfig SimConfig::defaultConfig() {
    SimConfig cfg;
    cfg.cameras = {
        {"CAM_FRONT_MAIN", "FRONT MAIN", "Windshield Header Primary Forward Imager (50 deg HFOV)",
         50.0f, 58.0f, {1.48f, 0.00f, 1.42f}, {0.0f, -1.5f, 0.0f},
         {-0.062f, 0.014f, 0.0002f, -0.0001f}, {0.220f, 0.741f, 0.973f}, false},
        {"CAM_FRONT_NARROW", "FRONT NARROW", "Long-Range Telephoto Imager for Signals & Horizon (28 deg HFOV)",
         28.0f, 85.0f, {1.49f, -0.07f, 1.42f}, {0.0f, -0.8f, 0.0f},
         {-0.018f, 0.004f, 0.0001f, 0.0001f}, {0.984f, 0.749f, 0.141f}, false},
        {"CAM_FRONT_WIDE", "FRONT WIDE", "Wide Fisheye Crosswalk, Cut-In & Intersection Imager (120 deg HFOV)",
         120.0f, 36.0f, {1.49f, 0.07f, 1.42f}, {0.0f, -3.5f, 0.0f},
         {-0.245f, 0.072f, -0.0004f, 0.0002f}, {0.204f, 0.827f, 0.600f}, false},
        {"CAM_PILLAR_LEFT", "PILLAR L", "Left B-Pillar Forward-Oblique Cross-Traffic Imager (90 deg HFOV)",
         90.0f, 38.0f, {0.65f, 0.84f, 1.30f}, {0.0f, -3.0f, 55.0f},
         {-0.112f, 0.024f, 0.0002f, 0.0001f}, {0.655f, 0.545f, 0.980f}, false},
        {"CAM_PILLAR_RIGHT", "PILLAR R", "Right B-Pillar Forward-Oblique Curb & Pedestrian Imager (90 deg HFOV)",
         90.0f, 38.0f, {0.65f, -0.84f, 1.30f}, {0.0f, -3.0f, -55.0f},
         {-0.112f, 0.024f, -0.0002f, 0.0001f}, {0.655f, 0.545f, 0.980f}, false},
        {"CAM_REPEATER_LEFT", "REPEATER L", "Left Front Fender Rearward Blind-Spot & Lane-Change Imager (85 deg HFOV)",
         85.0f, 34.0f, {1.72f, 0.88f, 0.78f}, {0.0f, -2.5f, 138.0f},
         {-0.105f, 0.022f, 0.0001f, -0.0002f}, {0.957f, 0.447f, 0.714f}, false},
        {"CAM_REPEATER_RIGHT", "REPEATER R", "Right Front Fender Rearward Blind-Spot & Curb Imager (85 deg HFOV)",
         85.0f, 34.0f, {1.72f, -0.88f, 0.78f}, {0.0f, -2.5f, -138.0f},
         {-0.105f, 0.022f, -0.0001f, -0.0002f}, {0.957f, 0.447f, 0.714f}, false},
        {"CAM_REAR", "REAR", "Rear Decklid Surround & Following-Traffic Imager (100 deg HFOV)",
         100.0f, 38.0f, {-1.15f, 0.00f, 1.12f}, {0.0f, -5.0f, 180.0f},
         {-0.165f, 0.038f, 0.0000f, 0.0003f}, {0.984f, 0.443f, 0.522f}, false}
    };
    return cfg;
}

SimConfig SimConfig::loadFromYaml(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    SimConfig cfg = defaultConfig();

    if (const auto sim = root["simulation"]) {
        cfg.controlHz = readOr(sim, "control_hz", cfg.controlHz);
        cfg.camWidth = readOr(sim, "cam_width", cfg.camWidth);
        cfg.camHeight = readOr(sim, "cam_height", cfg.camHeight);
        cfg.fullCamWidth = readOr(sim, "full_cam_width", cfg.fullCamWidth);
        cfg.fullCamHeight = readOr(sim, "full_cam_height", cfg.fullCamHeight);
        cfg.initialResMode = readOr(sim, "initial_res_mode", cfg.initialResMode);
        cfg.initialViewMode = readOr(sim, "initial_view_mode", cfg.initialViewMode);
        cfg.initialCamOverlay = readOr(sim, "initial_cam_overlay", cfg.initialCamOverlay);
        cfg.driveMode = readOr(sim, "drive_mode", cfg.driveMode);
        cfg.aebEnabled = readOr(sim, "aeb_enabled", cfg.aebEnabled);
        cfg.useVulkanValidation = readOr(sim, "use_vulkan_validation", cfg.useVulkanValidation);
    }

    if (const auto perc = root["perception"]) {
        if (const auto bb = perc["backbone"]) {
            cfg.featWidth = readOr(bb, "feat_width", cfg.featWidth);
            cfg.featHeight = readOr(bb, "feat_height", cfg.featHeight);
            cfg.featChannels = readOr(bb, "channels", cfg.featChannels);
        }
        if (const auto soil = perc["soil_detector"]) {
            cfg.soilPatchCols = readOr(soil, "patch_cols", cfg.soilPatchCols);
            cfg.soilPatchRows = readOr(soil, "patch_rows", cfg.soilPatchRows);
            cfg.soilEdgeThreshold = readOr(soil, "edge_threshold", cfg.soilEdgeThreshold);
            cfg.soilGlareThreshold = readOr(soil, "glare_threshold", cfg.soilGlareThreshold);
            cfg.soilMinTokenWeight = readOr(soil, "min_token_weight", cfg.soilMinTokenWeight);
            cfg.soilBaseEpistemicVar = readOr(soil, "base_epistemic_var", cfg.soilBaseEpistemicVar);
            cfg.soilVarGain = readOr(soil, "soil_var_gain", cfg.soilVarGain);
        }
        if (const auto det = perc["rtdetr"]) {
            cfg.rtdetrNumQueries = readOr(det, "num_queries", cfg.rtdetrNumQueries);
            cfg.rtdetrDecoderLayers = readOr(det, "num_decoder_layers", cfg.rtdetrDecoderLayers);
            cfg.rtdetrHighConfThresh = readOr(det, "high_conf_thresh", cfg.rtdetrHighConfThresh);
            cfg.rtdetrLowConfThresh = readOr(det, "low_conf_thresh", cfg.rtdetrLowConfThresh);
            cfg.rtdetrNmsIouThresh = readOr(det, "nms_iou_thresh", cfg.rtdetrNmsIouThresh);
            cfg.rtdetrMaxRangeM = readOr(det, "max_det_range_m", cfg.rtdetrMaxRangeM);
        }
        if (const auto bt = perc["bytetrack"]) {
            cfg.btMaxTracks = readOr(bt, "max_tracks", cfg.btMaxTracks);
            cfg.btMaxTimeLostSec = readOr(bt, "max_time_lost_sec", cfg.btMaxTimeLostSec);
            cfg.btHighMatchDistM = readOr(bt, "high_match_dist_m", cfg.btHighMatchDistM);
            cfg.btLowMatchDistM = readOr(bt, "low_match_dist_m", cfg.btLowMatchDistM);
            cfg.btProcessNoisePos = readOr(bt, "process_noise_pos", cfg.btProcessNoisePos);
            cfg.btProcessNoiseVel = readOr(bt, "process_noise_vel", cfg.btProcessNoiseVel);
            cfg.btBaseMeasNoiseM = readOr(bt, "base_meas_noise_m", cfg.btBaseMeasNoiseM);
