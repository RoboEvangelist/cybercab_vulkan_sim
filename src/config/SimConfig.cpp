#include "config/SimConfig.hpp"

#include <yaml-cpp/yaml.h>
#include <filesystem>
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
        cfg.pipelineVersion = readOr(sim, "pipeline_version", cfg.pipelineVersion);
    cfg.lightingMode = readOr(sim, "lighting_mode", cfg.lightingMode);
        cfg.v15PolicyPath = readOr(sim, "v15_policy_config", cfg.v15PolicyPath);
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
        }
        if (const auto bev = perc["bevformer"]) {
            cfg.bevNx = readOr(bev, "grid_nx", cfg.bevNx);
            cfg.bevNy = readOr(bev, "grid_ny", cfg.bevNy);
            cfg.bevMinX = readOr(bev, "min_x", cfg.bevMinX);
            cfg.bevMaxX = readOr(bev, "max_x", cfg.bevMaxX);
            cfg.bevMinY = readOr(bev, "min_y", cfg.bevMinY);
            cfg.bevMaxY = readOr(bev, "max_y", cfg.bevMaxY);
            cfg.bevChannels = readOr(bev, "channels", cfg.bevChannels);
            cfg.bevTemporalBlend = readOr(bev, "temporal_blend", cfg.bevTemporalBlend);
            if (bev["z_anchors"] && bev["z_anchors"].IsSequence()) {
                cfg.bevZAnchors.clear();
                for (const auto& z : bev["z_anchors"]) {
                    cfg.bevZAnchors.push_back(z.as<float>());
                }
            }
        }
        if (const auto occ = perc["occupancy_flow"]) {
            cfg.occMinX = readOr(occ, "min_x", cfg.occMinX);
            cfg.occMaxX = readOr(occ, "max_x", cfg.occMaxX);
            cfg.occStepX = readOr(occ, "step_x", cfg.occStepX);
            cfg.occMinY = readOr(occ, "min_y", cfg.occMinY);
            cfg.occMaxY = readOr(occ, "max_y", cfg.occMaxY);
            cfg.occStepY = readOr(occ, "step_y", cfg.occStepY);
            cfg.occMinZ = readOr(occ, "min_z", cfg.occMinZ);
            cfg.occMaxZ = readOr(occ, "max_z", cfg.occMaxZ);
            cfg.occStepZ = readOr(occ, "step_z", cfg.occStepZ);
            cfg.occDecayRate = readOr(occ, "confidence_decay_rate", cfg.occDecayRate);
            cfg.occThreshold = readOr(occ, "occ_threshold", cfg.occThreshold);
            cfg.occEnableShadows = readOr(occ, "enable_shadows", cfg.occEnableShadows);
        }
        if (const auto wm = perc["world_model"]) {
            cfg.wmUncertaintyGrowthRate = readOr(wm, "uncertainty_growth_rate", cfg.wmUncertaintyGrowthRate);
            if (wm["horizons_sec"] && wm["horizons_sec"].IsSequence()) {
                cfg.wmHorizonsSec.clear();
                for (const auto& h : wm["horizons_sec"]) {
                    cfg.wmHorizonsSec.push_back(h.as<float>());
                }
            }
        }
        if (const auto ps = perc["photon_sensor"]) {
            cfg.photonNoiseEnabled = readOr(ps, "enable_noise", cfg.photonNoiseEnabled);
            cfg.photonShotNoiseA = readOr(ps, "shot_noise_a_dn", cfg.photonShotNoiseA);
            cfg.photonReadNoiseB = readOr(ps, "read_noise_b_dn2", cfg.photonReadNoiseB);
            cfg.photonNoiseSeed = readOr(ps, "seed", cfg.photonNoiseSeed);
      cfg.photonRccbMosaic =
        readOr(ps, "enable_rccb_mosaic", cfg.photonRccbMosaic);
        }
    }

    if (const auto pl = root["planning"]) {
        cfg.targetCruiseSpeedMs = readOr(pl, "target_cruise_speed_ms", cfg.targetCruiseSpeedMs);
        cfg.numPlannerCandidates = readOr(pl, "num_candidates", cfg.numPlannerCandidates);
        cfg.trajSteps = readOr(pl, "traj_steps", cfg.trajSteps);
        if (const auto rss = pl["rss"]) {
            cfg.rssTReactSec = readOr(rss, "t_react_sec", cfg.rssTReactSec);
            cfg.rssABrakeMs2 = readOr(rss, "a_brake_rss_ms2", cfg.rssABrakeMs2);
            cfg.rssABrakeAebMs2 = readOr(rss, "a_brake_aeb_ms2", cfg.rssABrakeAebMs2);
            cfg.rssDBufferM = readOr(rss, "d_buffer_m", cfg.rssDBufferM);
            cfg.rssUncertaintyBufferGain = readOr(rss, "uncertainty_buffer_gain", cfg.rssUncertaintyBufferGain);
        }
    }

    if (const auto veh = root["vehicle"]) {
        cfg.wheelbaseM = readOr(veh, "wheelbase_m", cfg.wheelbaseM);
        cfg.maxSteerRad = readOr(veh, "max_steer_rad", cfg.maxSteerRad);
        cfg.egoHx = readOr(veh, "half_width_m", cfg.egoHx);
        cfg.egoHy = readOr(veh, "half_height_m", cfg.egoHy);
        cfg.egoHz = readOr(veh, "half_length_m", cfg.egoHz);
        cfg.egoCenterOffsetFwd = readOr(veh, "center_offset_fwd_m", cfg.egoCenterOffsetFwd);
        cfg.suspensionKSpring = readOr(veh, "suspension_k_spring", cfg.suspensionKSpring);
        cfg.suspensionCDamper = readOr(veh, "suspension_c_damper", cfg.suspensionCDamper);
    }

    if (root["cameras"] && root["cameras"].IsSequence()) {
        cfg.cameras.clear();
        for (const auto& c : root["cameras"]) {
            CameraSpec spec;
            spec.id = readOr<std::string>(c, "id", "CAM");
            spec.shortName = readOr<std::string>(c, "short_name", spec.id);
            spec.desc = readOr<std::string>(c, "desc", "");
            spec.hfovDeg = readOr(c, "hfov_deg", 50.0f);
            spec.maxRangeM = readOr(c, "max_range_m", 50.0f);
            spec.posIso = readVec3Or(c, "pos_iso", {0.0f, 0.0f, 1.2f});
            spec.rpyDeg = readVec3Or(c, "rpy_deg", {0.0f, 0.0f, 0.0f});
            if (c["distortion"] && c["distortion"].IsSequence() && c["distortion"].size() == 4) {
                for (int k = 0; k < 4; ++k) spec.distortion[k] = c["distortion"][k].as<float>();
            }
            spec.uiColor = readVec3Or(c, "color", {0.22f, 0.74f, 0.97f});
            spec.soiled = readOr(c, "soiled", false);
            cfg.cameras.push_back(spec);
        }
        if (cfg.cameras.size() != 8) {
            throw std::runtime_error("Expected exactly 8 cameras in config YAML");
        }
    }

    if (cfg.controlHz <= 0.0f || cfg.camWidth <= 0 || cfg.camHeight <= 0 ||
        cfg.featWidth <= 0 || cfg.featHeight <= 0 ||
        cfg.occStepX <= 0.0f || cfg.occStepY <= 0.0f || cfg.occStepZ <= 0.0f ||
        cfg.occNx() <= 0 || cfg.occNy() <= 0 || cfg.occNz() <= 0 ||
        cfg.rtdetrLowConfThresh <= 0.0f || cfg.rtdetrLowConfThresh >= cfg.rtdetrHighConfThresh ||
        cfg.soilMinTokenWeight <= 0.0f || cfg.soilMinTokenWeight >= 1.0f ||
        cfg.wheelbaseM <= 0.0f) {
        throw std::runtime_error("Invalid numeric parameter bounds in SimConfig YAML");
    }
    if (cfg.pipelineVersion != "v14" && cfg.pipelineVersion != "v15") {
        throw std::runtime_error("simulation.pipeline_version must be \"v14\" or \"v15\", got \"" +
                                 cfg.pipelineVersion + "\"");
    }
  if (cfg.lightingMode != "day" && cfg.lightingMode != "night_glare") {
    throw std::runtime_error(
      "simulation.lighting_mode must be \"day\" or \"night_glare\", got \"" +
      cfg.lightingMode + "\"");
  }
    if (!(cfg.photonShotNoiseA >= 0.0f) || !(cfg.photonReadNoiseB >= 0.0f)) {
        throw std::runtime_error("perception.photon_sensor noise parameters must be >= 0");
    }

    std::filesystem::path polP(cfg.v15PolicyPath);
    if (!std::filesystem::exists(polP)) {
        const auto relToCfg = std::filesystem::path(path).parent_path() / polP.filename();
        polP = std::filesystem::exists(relToCfg) ? relToCfg : (std::filesystem::path(CYBERCAB_SOURCE_DIR) / cfg.v15PolicyPath);
    }
    if (std::filesystem::exists(polP)) {
        cfg.v15Policy = V15PolicyConfig::loadFromYaml(polP.string());
    }

    return cfg;
}

} // namespace cybercab
