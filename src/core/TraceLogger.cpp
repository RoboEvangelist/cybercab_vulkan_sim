#include "core/TraceLogger.hpp"
#include <json/json.h>

namespace cybercab {

TraceLogger::~TraceLogger() {
    close();
}

bool TraceLogger::open(const std::string& jsonlPath) {
    close();
    file_.open(jsonlPath, std::ios::out | std::ios::trunc);
    return file_.is_open();
}

void TraceLogger::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

void TraceLogger::logFrame(int stepIdx, float simTimeSec,
                           const SimConfig& cfg,
                           const EgoState& ego,
                           const PlannerCommand& cmd,
                           const PerceptionFrameOutput& perc,
                           const ContactEvent& contact) {
    if (!file_.is_open()) return;
    file_ << formatFrameJson(stepIdx, simTimeSec, cfg, ego, cmd, perc, contact) << "\n";
}

std::string TraceLogger::formatFrameJson(int stepIdx, float simTimeSec,
                                         const SimConfig& cfg,
                                         const EgoState& ego,
                                         const PlannerCommand& cmd,
                                         const PerceptionFrameOutput& perc,
                                         const ContactEvent& contact) {
    Json::Value root;
    root["step"] = stepIdx;
    root["time_sec"] = simTimeSec;

    Json::Value jEgo;
    jEgo["x"] = ego.pos.x;
    jEgo["z"] = ego.pos.z;
    jEgo["yaw"] = ego.yaw;
    jEgo["speed_ms"] = ego.speed;
    jEgo["accel_ms2"] = ego.accel;
    jEgo["steer_rad"] = ego.steerAngle;
    jEgo["pitch_rad"] = ego.pitch;
    jEgo["roll_rad"] = ego.roll;
    jEgo["behavior"] = cmd.behaviorState;
    jEgo["rss_safe_dist_m"] = cmd.rssSafeDist;
    jEgo["min_obs_dist_m"] = cmd.minObstacleDist;
    jEgo["aeb_active"] = cmd.aebActive;
    root["ego"] = jEgo;

    Json::Value jSoil(Json::arrayValue);
    for (int c = 0; c < 8; ++c) {
        Json::Value sc;
        sc["cam"] = cfg.cameras[c].id;
        sc["gt_soiled"] = cfg.cameras[c].soiled;
        sc["soil_prob"] = perc.soilCameras[c].meanSoilProb;
        sc["token_weight"] = perc.soilCameras[c].tokenWeight;
        sc["epistemic_var"] = perc.soilCameras[c].meanEpistemicVar;
        jSoil.append(sc);
    }
    root["soil"] = jSoil;
    root["forward_epistemic_sigma"] = perc.forwardEpistemicSigma;

    Json::Value jTracks(Json::arrayValue);
    for (const auto& t : perc.tracks) {
        Json::Value jt;
        jt["id"] = t.trackId;
        jt["sem"] = static_cast<int>(t.semantic);
        jt["x"] = t.worldPos.x;
        jt["z"] = t.worldPos.z;
        jt["vx"] = t.worldVel.x;
        jt["vz"] = t.worldVel.z;
        jt["score"] = t.score;
        jt["pos_sigma_m"] = t.posSigmaM;
        jt["low_conf_stage2"] = t.matchedLowConf;
      jt["hits"] = t.hits;
      jt["age_sec"] = t.timeSinceUpdateSec;
      jt["corrob_cams"] = t.corroboratingCamCount;
      jt["corrob_mask"] = static_cast<int>(t.corroboratingCamMask);
        jTracks.append(jt);
    }
    root["tracks"] = jTracks;

    Json::Value jBev;
    jBev["observed_cells"] = perc.bevSummary.observedCells;
    jBev["mean_observability"] = perc.bevSummary.meanObservability;
    jBev["mean_epistemic_var"] = perc.bevSummary.meanEpistemicVar;
    root["bevformer"] = jBev;

    root["observed_voxels"] = static_cast<int>(perc.activeVoxels.size());
    root["shadow_voxels"] = static_cast<int>(perc.shadowVoxels.size());
    root["world_ghosts"] = static_cast<int>(perc.worldGhosts.size());
    root["detections_2d"] = static_cast<int>(perc.detections2D.size());
    root["detections_3d"] = static_cast<int>(perc.detections3D.size());

    Json::Value jEval;
    jEval["det_recall"] = perc.eval.detRecall;
    jEval["det_precision"] = perc.eval.detPrecision;
    jEval["pos_rmse_m"] = perc.eval.trackPosRmseM;
    jEval["matched_gt"] = perc.eval.matchedGtObjects;
    jEval["total_gt"] = perc.eval.totalGtDynamic;
    jEval["ghost_ped_dets"] = perc.eval.ghostPedDetections;
    jEval["ghost_veh_dets"] = perc.eval.ghostVehDetections;
    jEval["misclassified_trucks"] = perc.eval.misclassifiedTrucks;
    jEval["truck_veh_voxels"] = perc.eval.truckVehVoxels;
    jEval["truck_ped_voxels"] = perc.eval.truckPedVoxels;
    jEval["sign_pole_voxels"] = perc.eval.signPoleVoxels;
    jEval["total_pipeline_ms"] = perc.eval.totalPipelineMs;
    root["eval_metrics"] = jEval;

    Json::Value jGpu;
    for (const auto& kv : perc.gpuStageMs) {
        jGpu[kv.first] = kv.second;
    }
    root["gpu_stage_ms"] = jGpu;

  if (perc.v15.active) {
    Json::Value jV15;
    jV15["active"] = true;
    jV15["gear"] = ego.gear;
    jV15["lat_offset_m"] = ego.currentLateralOffset;
    jV15["preferred_lane_offset_m"] = perc.v15.preferredLaneOffsetM;
    jV15["calib_pitch_deg"] = perc.v15.calibPitchDeg;
    jV15["calib_roll_deg"] = perc.v15.calibRollDeg;
    jV15["siren_detected"] = perc.v15.sirenDetected;
    jV15["siren_confidence"] = perc.v15.sirenConfidence;
    jV15["siren_bearing_rad"] = perc.v15.sirenBearingRad;
    jV15["siren_doppler_hz"] = perc.v15.sirenDopplerHz;
    jV15["rear_clearance_m"] = perc.v15.rearClearanceM;
    jV15["latent_novelty"] = perc.v15.latentNovelty;
    jV15["surprise_sigma_delta"] = perc.v15.surpriseSigmaDelta;
    jV15["static_skip_ratio"] = perc.v15.staticSkipRatio;
    jV15["token_sparsity_ratio"] = perc.v15.tokenSparsityRatio;
    jV15["kv_cache_reuse_ratio"] = perc.v15.kvCacheReuseRatio;
    jV15["effective_speedup_x"] = perc.v15.effectiveSpeedupX;
    jV15["gaussian_count"] = static_cast<int>(perc.v15Gaussians.size());
    jV15["reality_view_bytes"] = static_cast<int>(perc.v15RealityRgba.size());
    root["v15"] = jV15;
  }

  if (perc.gaussianFormer3D.active) {
    Json::Value jGf;
    jGf["active"] = true;
    jGf["used_deformable_attn"] = perc.gaussianFormer3D.usedDeformableAttn;
    jGf["init_seed_count"] = perc.gaussianFormer3D.initSeedCount;
    jGf["gaussian_count"] = perc.gaussianFormer3D.gaussianCount;
    jGf["cam_corroborated_count"] = perc.gaussianFormer3D.camCorroboratedCount;
    jGf["splatted_voxel_count"] = perc.gaussianFormer3D.splattedVoxelCount;
    jGf["mean_opacity"] = perc.gaussianFormer3D.meanOpacity;
    jGf["mean_semantic_conf"] = perc.gaussianFormer3D.meanSemanticConf;
    jGf["mean_scale_m"] = perc.gaussianFormer3D.meanScaleM;
    jGf["attn_residual_m"] = perc.gaussianFormer3D.attnResidualM;
    root["gaussianformer3d"] = jGf;
  }

  if (perc.worldModelSota.active) {
    Json::Value jWm;
    jWm["mode"] = cfg.worldModelMode;
    jWm["hierarchical"] = perc.worldModelSota.hierarchical;
    jWm["weights_from_file"] = perc.worldModelSota.weightsLoadedFromFile;
    jWm["fmea_fallback"] = perc.worldModelSota.fmeaFallbackTriggered;
    jWm["fmea_fault_mask"] = static_cast<Json::UInt>(perc.worldModelSota.fmeaFaultMask);
    jWm["pred_surprise"] = perc.worldModelSota.predSurprise;
    jWm["surprise_sigma_delta"] = perc.worldModelSota.surpriseSigmaDelta;
    jWm["sigreg_loss"] = perc.worldModelSota.sigregLoss;
    jWm["latent_norm"] = perc.worldModelSota.latentNorm;
    jWm["best_rollout_energy"] = perc.worldModelSota.bestRolloutEnergy;
    jWm["macro_subgoal"] = perc.worldModelSota.macroSubgoalCode;
    jWm["macro_subgoal_lat_m"] = perc.worldModelSota.macroSubgoalLatM;
    jWm["macro_subgoal_speed_ms"] = perc.worldModelSota.macroSubgoalSpeedMs;
    jWm["rec_lat_m"] = perc.worldModelSota.recommendedLatM;
    jWm["rec_speed_scale"] = perc.worldModelSota.recommendedSpeedScale;
    jWm["stage_cpu_ms"] = perc.worldModelSota.stageCpuMs;
    root["world_model_sota"] = jWm;
  }

    root["sat_contact"] = contact.occurred;
    if (contact.occurred) root["sat_contact_other"] = contact.otherId;

    Json::StreamWriterBuilder wb;
    wb["indentation"] = "";
    return Json::writeString(wb, root);
}

} // namespace cybercab
