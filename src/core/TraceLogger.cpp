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
    root["eval_metrics"] = jEval;

    Json::Value jGpu;
    for (const auto& kv : perc.gpuStageMs) {
        jGpu[kv.first] = kv.second;
    }
    root["gpu_stage_ms"] = jGpu;

    root["sat_contact"] = contact.occurred;
    if (contact.occurred) root["sat_contact_other"] = contact.otherId;

    Json::StreamWriterBuilder wb;
    wb["indentation"] = "";
    file_ << Json::writeString(wb, root) << "\n";
}

} // namespace cybercab
