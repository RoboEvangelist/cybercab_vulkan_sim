#pragma once
// 6-level zero-tolerance AV scenario validation hierarchy, multi-axis M-slice
// accumulator, and deterministic TraceLogger JSONL replay parser.

#include <json/json.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <string>

namespace cybercab::testutil {

struct SliceKey {
  std::string scenario;
  std::string cls;
  std::string distBin;
  std::string weather;
  std::string lighting;
  std::string occlusion;
  std::string maneuver;

  std::string str() const {
    return scenario + "|" + cls + "|" + distBin + "|" + weather + "|" +
           lighting + "|" + occlusion + "|" + maneuver;
  }
};

struct MultiAxisSliceEntry {
  int total = 0;
  int correct = 0;
  int criticalErrors = 0;
  double costSum = 0.0;

  double accuracy() const {
    return (total > 0) ? static_cast<double>(correct) / total : 1.0;
  }
};

struct MultiAxisSliceTable {
  std::map<std::string, MultiAxisSliceEntry> slices;
  int totalObs = 0;
  int totalCorrect = 0;
  int totalCritical = 0;

  void record(const SliceKey& k, bool correct, bool criticalErr,
              double safetyCost) {
    auto& e = slices[k.str()];
    ++e.total;
    if (correct) ++e.correct;
    if (criticalErr) ++e.criticalErrors;
    e.costSum += safetyCost;
    ++totalObs;
    if (correct) ++totalCorrect;
    if (criticalErr) ++totalCritical;
  }
  double microAccuracy() const {
    return (totalObs > 0) ? static_cast<double>(totalCorrect) / totalObs : 0.0;
  }
  double worstSliceAccuracy() const {
    double w = 1.0;
    for (const auto& kv : slices) w = std::min(w, kv.second.accuracy());
    return w;
  }
  bool anyCriticalFailure() const { return totalCritical > 0; }
  bool allSlicesPass(double minAcc) const {
    return !anyCriticalFailure() && worstSliceAccuracy() >= minAcc;
  }
  size_t sliceCount() const { return slices.size(); }
};

struct ScenarioHierarchyReport {
  int framesEvaluated = 0;
  int frameCorrect = 0;
  int trackIdsw = 0;
  int assocTotal = 0;
  int assocCorrect = 0;
  int decisionTotal = 0;
  int decisionCorrect = 0;
  int criticalFailures = 0;
  bool behaviorZeroCollision = true;

  double frameAcc() const {
    return framesEvaluated > 0
             ? static_cast<double>(frameCorrect) / framesEvaluated
             : 0.0;
  }
  double assocAcc() const {
    return assocTotal > 0 ? static_cast<double>(assocCorrect) / assocTotal
                          : 1.0;
  }
  double decisionAcc() const {
    return decisionTotal > 0 ? static_cast<double>(decisionCorrect) /
                                 decisionTotal
                             : 1.0;
  }
  // Zero-tolerance: a single safety-critical failure fails the entire
  // encounter (rejects 299/300 dilution).
  bool scenarioPassed(double minFrame, double minAssoc, double minDec) const {
    return criticalFailures == 0 && behaviorZeroCollision && trackIdsw == 0 &&
           framesEvaluated > 0 && frameAcc() >= minFrame &&
           assocAcc() >= minAssoc && decisionAcc() >= minDec;
  }
};

struct TraceReplaySummary {
  int frames = 0;
  int collisions = 0;
  double meanRecall = 0.0;
  double meanPrec = 0.0;
  double meanRmse = 0.0;
  double meanGpuMs = 0.0;
  double finalX = 0.0;
  double finalZ = 0.0;
  int totalTracks = 0;
  int v15Frames = 0;
  double meanGaussians = 0.0;
};

inline TraceReplaySummary parseTraceJsonl(const std::string& path) {
  TraceReplaySummary s{};
  std::ifstream in(path);
  if (!in.is_open()) return s;
  Json::CharReaderBuilder rb;
  std::string line, errs;
  double recSum = 0.0, precSum = 0.0, rmseSum = 0.0, gpuSum = 0.0, gSum = 0.0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Json::Value root;
    std::unique_ptr<Json::CharReader> reader(rb.newCharReader());
    if (!reader->parse(line.data(), line.data() + line.size(), &root, &errs)) {
      continue;
    }
    ++s.frames;
    if (root["sat_contact"].asBool()) ++s.collisions;
    const auto& ego = root["ego"];
    s.finalX = ego["x"].asDouble();
    s.finalZ = ego["z"].asDouble();
    s.totalTracks += static_cast<int>(root["tracks"].size());
    const auto& ev = root["eval_metrics"];
    recSum += ev["det_recall"].asDouble();
    precSum += ev["det_precision"].asDouble();
    rmseSum += ev["pos_rmse_m"].asDouble();
    gpuSum += ev["total_pipeline_ms"].asDouble();
    if (root.isMember("v15") && root["v15"]["active"].asBool()) {
      ++s.v15Frames;
      gSum += root["v15"]["gaussian_count"].asDouble();
    }
  }
  if (s.frames > 0) {
    const double n = static_cast<double>(s.frames);
    s.meanRecall = recSum / n;
    s.meanPrec = precSum / n;
    s.meanRmse = rmseSum / n;
    s.meanGpuMs = gpuSum / n;
  }
  if (s.v15Frames > 0) {
    s.meanGaussians = gSum / static_cast<double>(s.v15Frames);
  }
  return s;
}

}  // namespace cybercab::testutil
