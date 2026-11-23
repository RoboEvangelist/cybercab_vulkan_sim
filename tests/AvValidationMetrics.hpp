#pragma once

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldEngine.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace cybercab::testutil {

struct EceAccum {
  static constexpr int kEceBins = 5;
  std::array<int, kEceBins> cnt_{};
  std::array<double, kEceBins> confSum_{};
  std::array<int, kEceBins> corSum_{};
  int tot_ = 0, allCor_ = 0;
  double allConf_ = 0.0;

  void add(float conf, bool correct) {
    const float c = clampf(conf, 0.0f, 1.0f);
    const int b = std::min(kEceBins - 1, static_cast<int>(c * kEceBins));
    ++cnt_[b]; confSum_[b] += c;
    if (correct) { ++corSum_[b]; ++allCor_; }
    ++tot_; allConf_ += c;
  }
  double ece() const {
    if (tot_ == 0) return 0.0;
    double s = 0.0;
    for (int b = 0; b < kEceBins; ++b)
      if (cnt_[b] > 0) s += std::abs(corSum_[b] - confSum_[b]);
    return s / static_cast<double>(tot_);
  }
  double overconfError() const {
    if (tot_ == 0) return 0.0;
    double s = 0.0;
    for (int b = 0; b < kEceBins; ++b)
      if (cnt_[b] > 0) s += std::max(0.0, confSum_[b] - corSum_[b]);
    return s / static_cast<double>(tot_);
  }
  double meanConf() const { return tot_ > 0 ? allConf_ / tot_ : 0.0; }
  double accuracy() const {
    return tot_ > 0 ? static_cast<double>(allCor_) / tot_ : 0.0;
  }
  int total() const { return tot_; }
};

struct SafetyConfusionMatrix {
  static constexpr int kClsN = 4;  // 0=Bg, 1=Veh, 2=Ped, 3=Sig
  static constexpr double kCost[kClsN][kClsN] = {
    {0., 15., 15., 10.}, {40., 0., 10., 20.},
    {100., 25., 0., 50.}, {50., 15., 20., 0.},
  };
  std::array<std::array<int, kClsN>, kClsN> m_{};

  static int clsIdx(SemanticClass s) {
    if (s == SemanticClass::Vehicle) return 1;
    if (s == SemanticClass::Pedestrian) return 2;
    return (s == SemanticClass::SignalSign) ? 3 : 0;
  }
  void add(int gt, int pred) {
    if (gt >= 0 && gt < kClsN && pred >= 0 && pred < kClsN) ++m_[gt][pred];
  }
  double weightedCost() const {
    double s = 0.0;
    for (int r = 0; r < kClsN; ++r)
      for (int c = 0; c < kClsN; ++c) s += kCost[r][c] * m_[r][c];
    return s;
  }
  int gtTotal() const {
    int s = 0;
    for (int r = 1; r < kClsN; ++r)
      for (int c = 0; c < kClsN; ++c) s += m_[r][c];
    return s;
  }
  double costPerGt() const {
    const int n = gtTotal();
    return n > 0 ? weightedCost() / static_cast<double>(n) : 0.0;
  }
};

struct SliceStats {
  int tp = 0, fp = 0, fn = 0, nErr = 0;
  double errSum = 0.0;
  double recall() const {
    return (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
  }
  double precision() const {
    return (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 1.0;
  }
  double meanErrM() const { return nErr > 0 ? errSum / nErr : 0.0; }
};

struct MotLaneMonitor {
  std::unordered_map<std::string, int> lastId;
  std::unordered_map<std::string, SemanticClass> lastSem;
  std::unordered_map<std::string, bool> wasTracked;
  int gtObs = 0, tp = 0, fp = 0, fn = 0, idsw = 0, frag = 0;
  int clsSwaps = 0, dupGt = 0, laneOk = 0, laneTot = 0;
  int sig2Ok = 0, sig2Tot = 0, frames = 0;
  double distSum = 0.0;

  void update(const WorldEngine& w, const std::vector<Track3D>& tracks,
              float evalRangeM = 35.0f) {
    ++frames;
    const EgoState& ego = w.ego();
    const auto dyn = w.dynamicGroundTruthOBBs();
    std::vector<OBB> gtIn;
    for (const auto& g : dyn)
      if (std::hypot(g.cx - ego.pos.x, g.cz - ego.pos.z) <= evalRangeM)
        gtIn.push_back(g);
    gtObs += static_cast<int>(gtIn.size());

    std::unordered_map<std::string, int> vehLane, gtMatch;
    for (const auto& v : w.vehicles()) vehLane[v.id] = v.lane;
    std::vector<uint8_t> trUsed(tracks.size(), 0);
    for (size_t ti = 0; ti < tracks.size(); ++ti)
      if (tracks[ti].semantic == SemanticClass::SignalSign ||
          tracks[ti].semantic == SemanticClass::Static) trUsed[ti] = 1;

    for (const auto& g : gtIn) {
      int bestTi = -1;
      float bestD = (g.semantic == SemanticClass::Pedestrian) ? 3.5f : 5.5f;
      for (size_t ti = 0; ti < tracks.size(); ++ti) {
        if (trUsed[ti]) continue;
        const float d = std::hypot(tracks[ti].worldPos.x - g.cx,
                                   tracks[ti].worldPos.z - g.cz);
        if (d < bestD) { bestD = d; bestTi = static_cast<int>(ti); }
      }
      if (bestTi < 0) { ++fn; continue; }
      trUsed[bestTi] = 1; ++tp; distSum += bestD;
      const Track3D& tr = tracks[bestTi];
      ++gtMatch[g.id];
      if (lastId.count(g.id) && lastId[g.id] != tr.trackId) ++idsw;
      if (lastSem.count(g.id) && lastSem[g.id] != tr.semantic) ++clsSwaps;
      if (wasTracked[g.id] && tr.state == TrackState::Lost) ++frag;
      lastId[g.id] = tr.trackId; lastSem[g.id] = tr.semantic;
      wasTracked[g.id] = (tr.state == TrackState::Tracked);
      ++sig2Tot;
      if (bestD <= 2.0f * tr.posSigmaM) ++sig2Ok;
      auto it = vehLane.find(g.id);
      if (it != vehLane.end()) {
        const float u = w.findClosestLoopU(tr.worldPos);
        const Vec3 ct = w.centerline().getTangentAt(u);
        const float lat =
          (tr.worldPos - w.centerline().getPointAt(u)).dot({ct.z, 0.f, -ct.x});
        ++laneTot;
        if (((lat >= 0.0f) ? 1 : 0) == it->second) ++laneOk;
      }
    }
    for (const auto& kv : gtMatch)
      if (kv.second > 1) dupGt += kv.second - 1;
    for (size_t ti = 0; ti < tracks.size(); ++ti) {
      if (trUsed[ti] ||
          std::hypot(tracks[ti].egoPos.x, tracks[ti].egoPos.y) > evalRangeM)
        continue;
      float minD = 999.0f;
      for (const auto& g : dyn)
        minD = std::min(minD, std::hypot(tracks[ti].worldPos.x - g.cx,
                                         tracks[ti].worldPos.z - g.cz));
      if (minD > 5.5f) ++fp;
    }
  }
  double mota() const {
    return gtObs > 0 ? 1.0 - static_cast<double>(fn + fp + idsw) / gtObs : 0.0;
  }
  double motp() const { return tp > 0 ? distSum / tp : 0.0; }
  double idswPer100() const {
    return gtObs > 0 ? 100.0 * static_cast<double>(idsw) / gtObs : 0.0;
  }
  double laneAssocAcc() const {
    return laneTot > 0 ? static_cast<double>(laneOk) / laneTot : 1.0;
  }
  double sigma2Cov() const {
    return sig2Tot > 0 ? static_cast<double>(sig2Ok) / sig2Tot : 0.0;
  }
};

inline float rssStopDistM(float v, const SimConfig& c) {
  return c.rssDBufferM + v * c.rssTReactSec + (v * v) / (2.0f * c.rssABrakeMs2);
}

}  // namespace cybercab::testutil
