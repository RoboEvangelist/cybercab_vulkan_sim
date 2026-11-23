#include "config/V15PolicyConfig.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace cybercab {
namespace {
void rdF(const YAML::Node& n, const char* k, float& v) {
  if (n && n[k]) v = n[k].as<float>();
}
}  // namespace

V15PolicyConfig V15PolicyConfig::loadFromYaml(const std::string& path) {
  YAML::Node r = YAML::LoadFile(path);
  V15PolicyConfig c{};
  if (const auto n = r["lane_centering"]) {
    rdF(n, "lane_center_abs_m", c.laneCenterAbsM);
    rdF(n, "min_lane_abs_m", c.minLaneAbsM);
    rdF(n, "max_lane_abs_m", c.maxLaneAbsM);
    rdF(n, "max_truck_shy_m", c.maxTruckShyM);
    rdF(n, "truck_half_length_min_m", c.truckHalfLengthMinM);
    rdF(n, "truck_shy_fwd_min_m", c.truckShyFwdMinM);
    rdF(n, "truck_shy_fwd_max_m", c.truckShyFwdMaxM);
    rdF(n, "adj_lane_min_sep_m", c.adjLaneMinSepM);
    rdF(n, "adj_lane_max_sep_m", c.adjLaneMaxSepM);
  }
  if (const auto n = r["autoregressive_rollout"]) {
    rdF(n, "min_lookahead_m", c.minLookaheadM);
    rdF(n, "lookahead_time_sec", c.lookaheadTimeSec);
    rdF(n, "lookahead_base_m", c.lookaheadBaseM);
    rdF(n, "lat_gain_per_s", c.latGainPerS);
    rdF(n, "max_lat_speed_ms", c.maxLatSpeedMs);
    rdF(n, "rollout_min_speed_ms", c.rolloutMinSpeedMs);
    rdF(n, "max_lat_accel_ms2", c.maxLatAccelMs2);
    rdF(n, "min_curve_speed_ms", c.minCurveSpeedMs);
    rdF(n, "curve_chord_half_m", c.curveChordHalfM);
  }
  if (const auto n = r["lane_hysteresis"]) {
    rdF(n, "right_lane_switch_cost", c.rightLaneSwitchCost);
    rdF(n, "left_lane_return_hold_cost", c.leftLaneReturnHoldCost);
    rdF(n, "left_lane_stay_cost", c.leftLaneStayCost);
    rdF(n, "divider_band_m", c.dividerBandM);
    rdF(n, "heading_commit_thresh", c.headingCommitThresh);
    rdF(n, "heading_commit_cost", c.headingCommitCost);
    rdF(n, "crossing_dev_thresh_m", c.crossingDevThreshM);
    rdF(n, "intersection_lane_lock_cost", c.intersectionLaneLockCost);
    rdF(n, "blind_spot_rear_gap_m", c.blindSpotRearGapM);
    rdF(n, "blind_spot_fwd_gap_m", c.blindSpotFwdGapM);
    rdF(n, "blind_spot_closing_time_s", c.blindSpotClosingTimeS);
    rdF(n, "blind_spot_lat_pad_m", c.blindSpotLatPadM);
    rdF(n, "blind_spot_block_cost", c.blindSpotBlockCost);
    rdF(n, "obb_hull_sample_scale", c.obbHullSampleScale);
    rdF(n, "veh_clearance_pad_m", c.vehClearancePadM);
  }
  if (const auto n = r["safety_shield"]) {
    rdF(n, "track_corridor_veh_m", c.trackCorridorVehM);
    rdF(n, "track_corridor_ped_m", c.trackCorridorPedM);
    rdF(n, "voxel_corridor_veh_m", c.voxelCorridorVehM);
    rdF(n, "voxel_corridor_ped_m", c.voxelCorridorPedM);
    rdF(n, "voxel_min_conf", c.voxelMinConf);
    rdF(n, "adj_veh_voxel_suppress_m", c.adjVehVoxelSuppressM);
    rdF(n, "adj_veh_track_min_lat_m", c.adjVehTrackMinLatM);
    rdF(n, "veh_hull_safety_pad_m", c.vehHullSafetyPadM);
    rdF(n, "min_hard_aeb_dist_m", c.minHardAebDistM);
    rdF(n, "aeb_rss_frac", c.aebRssFrac);
    rdF(n, "aeb_rel_speed_min_ms", c.aebRelSpeedMinMs);
    rdF(n, "follow_ramp_span_m", c.followRampSpanM);
    rdF(n, "follow_speed_gain", c.followSpeedGain);
    rdF(n, "siren_conf_thresh", c.sirenConfThresh);
    rdF(n, "siren_stop_conf_thresh", c.sirenStopConfThresh);
    rdF(n, "siren_yield_speed_ms", c.sirenYieldSpeedMs);
    rdF(n, "siren_right_pull_m", c.sirenRightPullM);
    rdF(n, "autoshift_min_rear_clear_m", c.autoshiftMinRearClearM);
    rdF(n, "autoshift_reverse_speed_ms", c.autoshiftReverseSpeedMs);
  }
  if (const auto n = r["latent_predictor"]) {
    rdF(n, "surprise_baseline", c.latentSurpriseBaseline);
    rdF(n, "surprise_scale", c.latentSurpriseScale);
    rdF(n, "sigma_gain", c.latentSigmaGain);
  }
  if (const auto n = r["gaussian_comfort"]) {
    rdF(n, "gaussian_occ_thresh", c.gaussianOccThresh);
    rdF(n, "gaussian_occ_cost_gain", c.gaussianOccCostGain);
    rdF(n, "gaussian_safety_stop_prob", c.gaussianSafetyStopProb);
    rdF(n, "nudge_offset_m", c.nudgeOffsetM);
    rdF(n, "nudge_cost_weight", c.nudgeCostWeight);
    rdF(n, "jerk_cost_weight", c.jerkCostWeight);
    rdF(n, "speed_backoff_ratio", c.speedBackoffRatio);
    rdF(n, "speed_backoff_cost", c.speedBackoffCost);
  }
  if (c.minLaneAbsM <= 0.0f || c.minLaneAbsM >= c.maxLaneAbsM ||
      c.laneCenterAbsM < c.minLaneAbsM || c.laneCenterAbsM > c.maxLaneAbsM ||
      c.latentSurpriseScale <= 0.0f || c.maxLatAccelMs2 <= 0.0f ||
      c.obbHullSampleScale <= 0.0f || c.minHardAebDistM <= 0.0f ||
      c.aebRssFrac <= 0.0f || c.aebRssFrac > 1.0f ||
      c.gaussianOccThresh <= 0.0f || c.gaussianOccThresh >= 1.0f ||
      c.gaussianSafetyStopProb <= 0.0f || c.gaussianSafetyStopProb >= 1.0f ||
      c.speedBackoffRatio <= 0.0f || c.speedBackoffRatio > 1.0f) {
    throw std::runtime_error("Invalid numeric bounds in V15PolicyConfig YAML");
  }
  return c;
}

}  // namespace cybercab
