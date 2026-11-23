#include "config/SimConfig.hpp"
#include "core/TraceLogger.hpp"
#include "core/VulkanPipeline.hpp"
#include "perception/ByteTrack.hpp"
#include "perception/MultiCamCorroboration.hpp"
#include <gtest/gtest.h>
#include <json/json.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cybercab {
namespace {
constexpr int kFbW = 320, kFbH = 240, kUnkRes = 999, kBadN = -1, kBadP = 8;
constexpr int kMask3 = 7, kTrN = 5, kTrace14 = 0, kTrace15 = 1, kCamCnt = 8;
constexpr int kMain = 0, kNar = 1, kWide = 2, kPilL = 3, kRepL = 5, kRear = 7;
constexpr int kHw4Tr = 3, kHw4Corrob = 2, kHierN = 6;
constexpr int kBinW = 1448, kBinH = 938, kNatW = 2896, kNatH = 1876;
constexpr float kSc0 = 0.72f, kScA = 0.70f, kScB = 0.65f, kSig0 = 0.40f;
constexpr float kClnW = 1.0f, kSoilW = 0.25f, kDt = 1.0f / 36.0f, kX0 = 20.0f;
constexpr float kVx = 4.0f, kVehY = 0.72f, kOvDz = 0.1f, kLatM = 12.0f;
constexpr float kEps = 1e-4f, kT1 = 0.1f, kAgePad = 0.2f, kTeslaHz = 36.0f;
constexpr float kNearM = 15.0f, kMidM = 55.0f, kFarM = 200.0f, kOverM = 260.0f;
constexpr float kWideOverM = 70.0f;
constexpr float kTeslaCamMaxM[kCamCnt] = {
  150.0f, 250.0f, 60.0f, 80.0f, 80.0f, 100.0f, 100.0f, 50.0f};
constexpr int kRangeOrd[kHierN] = {kNar, kMain, kRepL, kPilL, kWide, kRear};

struct TempGuard {
  std::filesystem::path p;
  ~TempGuard() { std::error_code ec; std::filesystem::remove(p, ec); }
};

std::string replaceKey(const std::string& src, const std::string& key,
                       const std::string& val) {
  std::istringstream iss(src); std::ostringstream oss; std::string ln;
  const std::string k = "  " + key + ":"; bool hit = false;
  while (std::getline(iss, ln)) {
    const bool m = !hit && ln.rfind(k, 0) == 0;
    oss << (m ? (k + " " + val) : ln) << "\n"; hit = hit || m;
  }
  EXPECT_TRUE(hit) << key; return oss.str();
}

Detection3D detAt(int cam, float fwdZ, float latX = 0.0f, float sc = kSc0,
                  float w = kClnW) {
  Detection3D d{}; d.camIndex = cam; d.semantic = SemanticClass::Vehicle;
  d.score = sc; d.epistemicSigma = kSig0; d.camWeight = w;
  d.worldPos = {latX, kVehY, fwdZ}; d.egoPos = {fwdZ, -latX, kVehY};
  d.rangeM = std::hypot(latX, fwdZ); return d;
}

bool inCamRange(const SimConfig& cfg, int cam, float r) {
  if (cam < 0 || static_cast<size_t>(cam) >= cfg.cameras.size()) return false;
  return r > 0.0f && r <= std::min(cfg.rtdetrMaxRangeM,
                                   cfg.cameras[cam].maxRangeM);
}

TEST(CameraCorrobAndBoundsTest, SimConfigBadCameraBoundsFallbacksAndCli) {
  std::ifstream yIf(resolveAssetPath("config/cybercab_sim.yaml"));
  std::ostringstream ySs; ySs << yIf.rdbuf();
  const std::string baseYaml = ySs.str();
  const TempGuard tmp{
    std::filesystem::temp_directory_path() / "cybercab_bad_res_bounds.yaml"};
  const struct { const char* k; const char* v; } kBad[] = {
    {"initial_res_mode", "\"invalid_mode\""},
    {"cybercab_binned_cam_width", "0"}, {"cybercab_binned_cam_height", "-1"},
    {"cybercab_cam_width", "1000"}, {"cybercab_cam_height", "500"},
    {"full_cam_width", "160"}, {"full_cam_height", "120"}};
  for (const auto& tc : kBad) {
    { std::ofstream o(tmp.p, std::ios::trunc);
      o << replaceKey(baseYaml, tc.k, tc.v); }
    EXPECT_THROW((void)SimConfig::loadFromYaml(tmp.p.string()),
                 std::runtime_error) << tc.k << "=" << tc.v;
  }
  SimConfig fb = SimConfig::defaultConfig(); fb.initialResMode = "bogus";
  EXPECT_EQ(fb.resolveInitialRes().width, kFbW);
  EXPECT_EQ(fb.resolveInitialRes().height, kFbH);
  const auto nxt = fb.nextResolutionPreset(kUnkRes, kUnkRes);
  EXPECT_EQ(nxt.width, kFbW); EXPECT_EQ(nxt.height, kFbH);
  EXPECT_NE(std::system((std::string(CYBERCAB_BUILD_DIR) +
    "/cybercab_sim --res invalid --headless --steps 1 >/dev/null 2>&1")
    .c_str()), 0);
}

TEST(CameraCorrobAndBoundsTest,
     MultiCameraCorroborationTrackerPrecisionAndTrace) {
  using namespace multicam;
  EXPECT_EQ(camBit(kBadN), 0u); EXPECT_EQ(camBit(kBadP), 0u);
  EXPECT_EQ(camBit(0), 1u); EXPECT_FLOAT_EQ(corroborationNoiseScale(1), 1.0f);
  EXPECT_LT(corroborationNoiseScale(3), 1.0f);
  auto cor = [](Detection3D& k, const Detection3D& c) {
    applyDetectionCorroboration(k, c);
  };
  Detection3D keep = detAt(0, kX0); keep.corroboratingCamMask = camBit(0);
  cor(keep, detAt(0, kX0, 0.0f, kScA));
  EXPECT_EQ(keep.corroboratingCamCount, 1);
  Detection3D ped = detAt(1, kX0, 0.0f, kScA);
  ped.semantic = SemanticClass::Pedestrian; cor(keep, ped);
  cor(keep, detAt(1, kX0, 0.0f, kScA, kSoilW));
  EXPECT_EQ(keep.corroboratingCamCount, 1);
  cor(keep, detAt(1, kX0, 0.0f, kScA)); cor(keep, detAt(2, kX0, 0.0f, kScB));
  EXPECT_EQ(keep.corroboratingCamCount, 3);
  EXPECT_EQ(static_cast<int>(keep.corroboratingCamMask), kMask3);
  EXPECT_GT(keep.score, kSc0); EXPECT_LE(keep.score, kMaxCorrobScore);
  EXPECT_LT(keep.epistemicSigma, kSig0);
  const SimConfig cfg = SimConfig::defaultConfig();
  ByteTracker tr1(cfg), trM(cfg); EgoState ego{};
  std::vector<Track3D> out1, outM;
  for (int s = 0; s < kTrN; ++s) {
    const float z = kX0 + kVx * static_cast<float>(s) * kDt;
    Detection3D d1 = detAt(0, z), d2 = d1, ov = detAt(2, z + kOvDz, 0, kScA);
    cor(d2, ov); tr1.update(kDt, ego, {d1}); trM.update(kDt, ego, {d2, ov});
  }
  tr1.collectActiveTracks(out1); trM.collectActiveTracks(outM);
  ASSERT_EQ(out1.size(), 1u); ASSERT_EQ(outM.size(), 1u);
  EXPECT_EQ(outM[0].hits, kTrN); EXPECT_GE(outM[0].corroboratingCamCount, 2);
  EXPECT_LT(outM[0].posSigmaM, out1[0].posSigmaM);
  EXPECT_FLOAT_EQ(outM[0].timeSinceUpdateSec, 0.0f);
  for (float p = 0; !outM.empty() && p <= cfg.btMaxTimeLostSec + kAgePad;) {
    trM.update(kDt, ego, {}); trM.collectActiveTracks(outM);
    if (!outM.empty()) {
      EXPECT_GT(outM[0].timeSinceUpdateSec, p); p = outM[0].timeSinceUpdateSec;
    }
  }
  EXPECT_TRUE(outM.empty());
  const TempGuard trTmp{
    std::filesystem::temp_directory_path() / "corrob_trace_unit.jsonl"};
  PerceptionFrameOutput p14{}, p15{}; p14.tracks = out1; p15.tracks = out1;
  p15.tracks[0].corroboratingCamCount = 3;
  p15.tracks[0].corroboratingCamMask = kMask3; p15.v15.active = true;
  { TraceLogger log; ASSERT_TRUE(log.open(trTmp.p.string()));
    log.logFrame(kTrace14, 0.0f, cfg, ego, {}, p14, {});
    log.logFrame(kTrace15, kT1, cfg, ego, {}, p15, {}); }
  std::ifstream ifs(trTmp.p); Json::CharReaderBuilder rb;
  for (int i = 0; i < 2; ++i) {
    std::string ln, err; Json::Value rt;
    ASSERT_TRUE(static_cast<bool>(std::getline(ifs, ln)));
    std::istringstream iss(ln);
    ASSERT_TRUE(Json::parseFromStream(rb, iss, &rt, &err)) << err;
    ASSERT_EQ(rt["tracks"].size(), 1u); const auto& jt = rt["tracks"][0];
    EXPECT_EQ(jt["hits"].asInt(), kTrN);
    EXPECT_NEAR(jt["age_sec"].asFloat(), 0.0f, kEps);
    EXPECT_EQ(jt["corrob_cams"].asInt(), i == 0 ? 1 : 3);
    EXPECT_EQ(jt["corrob_mask"].asInt(), i == 0 ? 1 : kMask3);
    EXPECT_EQ(rt.isMember("v15"), i == 1);
  }
}

TEST(CameraCorrobAndBoundsTest, TeslaHw4PublicRangeAndStratifiedTracks) {
  SimConfig hw4 = SimConfig::defaultConfig();
  ASSERT_EQ(hw4.cameras.size(), static_cast<size_t>(kCamCnt));
  EXPECT_FLOAT_EQ(hw4.controlHz, kTeslaHz);
  hw4.rtdetrMaxRangeM = kTeslaCamMaxM[kNar];
  for (int i = 0; i < kCamCnt; ++i) hw4.cameras[i].maxRangeM = kTeslaCamMaxM[i];
  for (int i = 1; i < kHierN; ++i)
    EXPECT_GT(hw4.cameras[kRangeOrd[i - 1]].maxRangeM,
              hw4.cameras[kRangeOrd[i]].maxRangeM);
  hw4.initialResMode = "cybercab_binned";
  EXPECT_EQ(hw4.resolveInitialRes().width, kBinW);
  EXPECT_EQ(hw4.resolveInitialRes().height, kBinH);
  hw4.initialResMode = "cybercab";
  EXPECT_EQ(hw4.resolveInitialRes().width, kNatW);
  EXPECT_EQ(hw4.resolveInitialRes().height, kNatH);
  const struct { int c; float r; bool ok; } kG[] = {
    {kWide, kNearM, true}, {kMain, kMidM, true}, {kWide, kMidM, true},
    {kNar, kFarM, true}, {kNar, kOverM, false}, {kWide, kWideOverM, false}};
  for (const auto& g : kG) EXPECT_EQ(inCamRange(hw4, g.c, g.r), g.ok);
  ByteTracker trHw4(hw4); EgoState ego{}; std::vector<Track3D> tracks;
  for (int s = 0; s < kTrN; ++s) {
    const float dz = kVx * static_cast<float>(s) * kDt;
    trHw4.update(kDt, ego, {detAt(kMain, kNearM + dz, -kLatM),
      detAt(kWide, kNearM + dz + kOvDz, -kLatM, kScA),
      detAt(kMain, kMidM + dz), detAt(kNar, kMidM + dz + kOvDz, 0.0f, kScA),
      detAt(kNar, kFarM + dz, kLatM)});
  }
  trHw4.collectActiveTracks(tracks);
  ASSERT_EQ(tracks.size(), static_cast<size_t>(kHw4Tr));
  int corrobCnt = 0;
  for (const auto& t : tracks) {
    EXPECT_EQ(t.hits, kTrN); EXPECT_TRUE(std::isfinite(t.posSigmaM));
    if (t.corroboratingCamCount >= 2) ++corrobCnt;
  }
  EXPECT_EQ(corrobCnt, kHw4Corrob);
}
}  // namespace
}  // namespace cybercab
