#pragma once
// Shared geometry stations, loft helpers, and constants for CybercabMesh.
#include "world/WorldTypes.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cybercab::mesh_detail {

inline const Vec3 kGoldBase{0.775f, 0.618f, 0.355f};
inline const Vec3 kGoldBright{0.865f, 0.720f, 0.440f};
inline const Vec3 kGoldDark{0.560f, 0.435f, 0.245f};
inline const Vec3 kCanopyGlass{0.024f, 0.045f, 0.095f};
inline const Vec3 kTrimDark{0.042f, 0.044f, 0.048f};
inline const Vec3 kCharcoalTrim{0.120f, 0.122f, 0.128f};
inline const Vec3 kTireDark{0.052f, 0.054f, 0.058f};
inline const Vec3 kPlateWhite{0.820f, 0.880f, 0.930f};
inline const Vec3 kAmberMarker{0.920f, 0.480f, 0.080f};
inline const Vec3 kHoloCyan{0.220f, 0.740f, 0.970f};
inline const Vec3 kDrlLed{0.930f, 0.980f, 1.000f};
inline const Vec3 kTailLed{0.960f, 0.130f, 0.130f};
inline const Vec3 kRevLed{0.980f, 0.980f, 0.960f};

inline constexpr float kRearAxleZ = 0.00f, kFrontAxleZ = 2.75f;
inline constexpr float kWheelAxleYR = 0.382f, kWheelAxleYF = 0.356f;
inline constexpr float kTireOuterRR = 0.382f, kTireOuterRF = 0.356f;
inline constexpr float kWheelArchRR = 0.416f, kWheelArchRF = 0.390f;
inline constexpr float kTrackHalfR = 0.836f, kTrackHalfF = 0.818f;
inline constexpr float kBodyRearZ = -0.88f, kBodyCrestZ = 1.02f;
inline constexpr float kBodyCowlZ = 2.08f, kBodyNoseZ = 3.35f;
inline constexpr float kWindshieldRearZ = 1.06f, kWindshieldFrontZ = 2.08f;
inline constexpr int kNumSt = 37, kSideWinSt = 12, kWheelSegs = 36;
inline constexpr float kCybercabSemTag = 1.25f;

inline constexpr std::array<float, kNumSt> kZStations = {
  {-0.88f, -0.78f, -0.64f, -0.45f, -0.41f, -0.34f, -0.22f, -0.10f, 0.00f, 0.12f,
   0.24f,  0.34f,  0.41f,  0.46f,  0.64f,  0.84f,  0.96f,  1.06f,  1.42f, 1.76f,
   2.08f,  2.24f,  2.36f,  2.39f,  2.46f,  2.56f,  2.66f,  2.75f,  2.85f, 2.96f,
   3.06f,  3.13f,  3.16f,  3.22f,  3.28f,  3.32f,  3.35f}};

struct Station {
  float z, yBot, yBelly, yShld, yUpMid, yCan, yRoof;
  float wBot, wBelly, wShld, wUpMid, wCan;
};

inline float wheelArchY(float z, float baseY) {
  float y = baseY;
  const struct {
    float az, ay, ar;
  } arches[2] = {{kRearAxleZ, kWheelAxleYR, kWheelArchRR},
                 {kFrontAxleZ, kWheelAxleYF, kWheelArchRF}};
  for (const auto& a : arches) {
    const float dz = std::abs(z - a.az);
    if (dz < a.ar) {
      y = std::max(y, a.ay + std::sqrt(std::max(0.0f, a.ar * a.ar - dz * dz)));
    }
  }
  return y;
}

inline Station evalStation(float z) {
  Station s{};
  s.z = z;
  const float baseY = (z < -0.40f)
                        ? lerpf(0.475f, 0.335f, (z - kBodyRearZ) / 0.48f)
                      : (z > 3.15f) ? lerpf(0.335f, 0.372f, (z - 3.15f) / 0.20f)
                                    : 0.330f;
  s.yBot = wheelArchY(z, baseY);
  if (z < kBodyCrestZ) {
    const float t = (kBodyCrestZ - z) / (kBodyCrestZ - kBodyRearZ);
    s.yRoof = 1.392f - 0.482f * (0.30f * t + 0.70f * t * t);
  } else if (z <= kBodyCowlZ) {
    const float t = (z - kBodyCrestZ) / (kBodyCowlZ - kBodyCrestZ);
    s.yRoof = 1.392f - 0.494f * (t * t * (1.22f - 0.22f * t));
  } else {
    const float t = (z - kBodyCowlZ) / (kBodyNoseZ - kBodyCowlZ);
    s.yRoof = 0.898f - 0.308f * (0.18f * t + 0.82f * t * t * t);
  }
  const bool cab = (z > -0.35f && z < kBodyCowlZ);
  s.yCan = s.yRoof - (cab ? 0.048f : 0.022f);
  if (z <= kBodyCowlZ) {
    const float t =
      clampf((z - kBodyRearZ) / (kBodyCowlZ - kBodyRearZ), 0.0f, 1.0f);
    s.yShld = lerpf(0.915f, 0.835f, t);
  } else if (z <= 2.80f) {
    s.yShld = lerpf(0.835f, 0.782f, (z - kBodyCowlZ) / (2.80f - kBodyCowlZ));
  } else {
    const float t = clampf((z - 2.80f) / (kBodyNoseZ - 2.80f), 0.0f, 1.0f);
    s.yShld = lerpf(0.782f, 0.560f, t * t);
  }
  s.yShld = clampf(s.yShld, s.yBot + 0.030f, s.yCan - 0.018f);
  s.yBelly = lerpf(s.yBot, s.yShld, 0.55f);
  s.yUpMid = lerpf(s.yShld, s.yCan, 0.52f);

  const float trR =
    (z < -0.40f) ? clampf((-0.40f - z) / 0.48f, 0.0f, 1.0f) : 0.0f;
  const float trF =
    (z > 2.84f) ? clampf((z - 2.84f) / 0.51f, 0.0f, 1.0f) : 0.0f;
  s.wBot = 0.902f - 0.115f * trR * trR - 0.155f * trF * trF;
  s.wBelly = 0.962f - 0.105f * trR * trR - 0.155f * trF * trF;
  s.wShld = 0.946f - 0.100f * trR * trR - 0.155f * trF * trF;
  const float cabTaper =
    (z > -0.35f && z < kBodyCowlZ)
      ? 0.215f * std::sin(kPi * ((z + 0.35f) / (kBodyCowlZ + 0.35f)))
      : 0.045f;
  s.wCan = s.wShld - cabTaper - 0.045f;
  s.wUpMid = lerpf(s.wShld, s.wCan, 0.42f);
  return s;
}

template <int MaxS, int MaxC>
inline void emitGrid(std::vector<SceneVertex>& verts,
                     std::vector<uint32_t>& idx, const Mat4& xf,
                     const Vec3 (&pos)[MaxS][MaxC], int nS, int nC, Vec3 col,
                     float a, float em, bool flip = false) {
  Vec3 nrm[MaxS][MaxC]{};
  for (int s = 0; s < nS - 1; ++s) {
    for (int c = 0; c < nC - 1; ++c) {
      const Vec3 p00 = pos[s][c], p10 = pos[s][c + 1];
      const Vec3 p11 = pos[s + 1][c + 1], p01 = pos[s + 1][c];
      Vec3 fn =
        flip ? (p01 - p00).cross(p11 - p00) : (p10 - p00).cross(p11 - p00);
      if (fn.normSq() > 1e-10f) {
        fn = fn.normalized();
        nrm[s][c] += fn;
        nrm[s][c + 1] += fn;
        nrm[s + 1][c + 1] += fn;
        nrm[s + 1][c] += fn;
      }
    }
  }
  const uint32_t base = static_cast<uint32_t>(verts.size());
  for (int s = 0; s < nS; ++s) {
    for (int c = 0; c < nC; ++c) {
      const Vec3 wp = xf.transformPoint(pos[s][c]);
      const Vec3 wn = xf.transformVector(nrm[s][c].normalized()).normalized();
      verts.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, col.x, col.y, col.z,
                       a, em, kCybercabSemTag, 0.0f, 0.0f});
    }
  }
  for (int s = 0; s < nS - 1; ++s) {
    for (int c = 0; c < nC - 1; ++c) {
      const uint32_t i00 = base + static_cast<uint32_t>(s * nC + c);
      const uint32_t i10 = i00 + 1;
      const uint32_t i01 = i00 + static_cast<uint32_t>(nC);
      const uint32_t i11 = i01 + 1;
      if (!flip) {
        idx.insert(idx.end(), {i00, i10, i11, i00, i11, i01});
      } else {
        idx.insert(idx.end(), {i00, i11, i10, i00, i01, i11});
      }
    }
  }
}

inline void appendBox(std::vector<SceneVertex>& v, std::vector<uint32_t>& idx,
                      const Mat4& xf, Vec3 sz, Vec3 col, float a = 1.0f,
                      float em = 0.0f) {
  const float hx = sz.x * 0.5f, hy = sz.y * 0.5f, hz = sz.z * 0.5f;
  const Vec3 ns[6] = {{0, 0, 1},  {0, 0, -1}, {1, 0, 0},
                      {-1, 0, 0}, {0, 1, 0},  {0, -1, 0}};
  const Vec3 ps[6][4] = {
    {{-hx, -hy, hz}, {hx, -hy, hz}, {hx, hy, hz}, {-hx, hy, hz}},
    {{hx, -hy, -hz}, {-hx, -hy, -hz}, {-hx, hy, -hz}, {hx, hy, -hz}},
    {{hx, -hy, hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {hx, hy, hz}},
    {{-hx, -hy, -hz}, {-hx, -hy, hz}, {-hx, hy, hz}, {-hx, hy, -hz}},
    {{-hx, hy, hz}, {hx, hy, hz}, {hx, hy, -hz}, {-hx, hy, -hz}},
    {{-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, -hy, hz}, {-hx, -hy, hz}}};
  for (int f = 0; f < 6; ++f) {
    const uint32_t b = static_cast<uint32_t>(v.size());
    const Vec3 wn = xf.transformVector(ns[f]).normalized();
    for (int k = 0; k < 4; ++k) {
      const Vec3 wp = xf.transformPoint(ps[f][k]);
      v.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, col.x, col.y, col.z, a,
                   em, kCybercabSemTag, 0.0f, 0.0f});
    }
    idx.insert(idx.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
  }
}

inline void emitSculptedEndCap(std::vector<SceneVertex>& verts,
                               std::vector<uint32_t>& idx, const Mat4& xf,
                               const Station& st, bool isFront) {
  constexpr int kCols = 7, kRows = 4;
  Vec3 g[kRows][kCols]{};
  const float ys[kRows] = {st.yBot, st.yShld, st.yCan, st.yRoof};
  const float ws[kRows] = {st.wBot, st.wShld, st.wCan, 0.0f};
  const float zCurve = isFront ? -0.075f : 0.055f;
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const float u = lerpf(
        -1.0f, 1.0f, static_cast<float>(c) / static_cast<float>(kCols - 1));
      const float yMid = (r == kRows - 1)
                           ? st.yRoof
                           : lerpf(ys[r], st.yRoof, 0.08f * (1.0f - u * u));
      g[r][c] = {u * ws[r], yMid, st.z + zCurve * u * u};
    }
  }
  emitGrid<kRows, kCols>(verts, idx, xf, g, kRows, kCols, kGoldDark, 1.0f, 0.0f,
                         !isFront);
}

inline void emitSideWindows(std::vector<SceneVertex>& verts,
                            std::vector<uint32_t>& idx, const Mat4& xf,
                            const Station (&st)[kNumSt]) {
  constexpr float wz[kSideWinSt] = {0.10f, 0.24f, 0.40f, 0.58f, 0.78f, 0.98f,
                                    1.18f, 1.38f, 1.56f, 1.74f, 1.90f, 2.02f};
  constexpr float wyL[kSideWinSt] = {1.110f, 1.045f, 0.995f, 0.962f,
                                     0.942f, 0.922f, 0.905f, 0.892f,
                                     0.880f, 0.870f, 0.862f, 0.858f};
  constexpr float wyH[kSideWinSt] = {1.132f, 1.200f, 1.255f, 1.295f,
                                     1.318f, 1.318f, 1.280f, 1.205f,
                                     1.115f, 1.015f, 0.925f, 0.868f};
  auto evalUpperX = [&](float zq, float yq) -> float {
    int seg = kNumSt - 2;
    for (int s = 0; s < kNumSt - 1; ++s) {
      if (zq >= st[s].z && zq <= st[s + 1].z) {
        seg = s;
        break;
      }
    }
    const float dz = std::max(1e-4f, st[seg + 1].z - st[seg].z);
    const float tz = clampf((zq - st[seg].z) / dz, 0.0f, 1.0f);
    const float yS = lerpf(st[seg].yShld, st[seg + 1].yShld, tz);
    const float yM = lerpf(st[seg].yUpMid, st[seg + 1].yUpMid, tz);
    const float yC = lerpf(st[seg].yCan, st[seg + 1].yCan, tz);
    const float wS = lerpf(st[seg].wShld, st[seg + 1].wShld, tz);
    const float wM = lerpf(st[seg].wUpMid, st[seg + 1].wUpMid, tz);
    const float wC = lerpf(st[seg].wCan, st[seg + 1].wCan, tz);
    const float wVal =
      (yq <= yM)
        ? lerpf(wS, wM,
                clampf((yq - yS) / std::max(1e-4f, yM - yS), 0.0f, 1.0f))
        : lerpf(wM, wC,
                clampf((yq - yM) / std::max(1e-4f, yC - yM), 0.0f, 1.0f));
    return wVal + 0.013f;
  };
  Vec3 wR[kSideWinSt][3]{}, wL[kSideWinSt][3]{};
  for (int i = 0; i < kSideWinSt; ++i) {
    const float yMidW = lerpf(wyL[i], wyH[i], 0.50f);
    const float x0 = evalUpperX(wz[i], wyL[i]);
    const float xm = evalUpperX(wz[i], yMidW) + 0.005f;
    const float x1 = evalUpperX(wz[i], wyH[i]);
    wR[i][0] = {x0, wyL[i], wz[i]};
    wR[i][1] = {xm, yMidW, wz[i]};
    wR[i][2] = {x1, wyH[i], wz[i]};
    wL[i][0] = {-x1, wyH[i], wz[i]};
    wL[i][1] = {-xm, yMidW, wz[i]};
    wL[i][2] = {-x0, wyL[i], wz[i]};
  }
  emitGrid<kSideWinSt, 3>(verts, idx, xf, wR, kSideWinSt, 3, kCanopyGlass, 1.0f,
                          0.0f, false);
  emitGrid<kSideWinSt, 3>(verts, idx, xf, wL, kSideWinSt, 3, kCanopyGlass, 1.0f,
                          0.0f, false);
}

inline void emitWindshieldAndPillars(std::vector<SceneVertex>& verts,
                                     std::vector<uint32_t>& idx,
                                     const Mat4& xf) {
  constexpr int kWsS = 6, kWsC = 5;
  Vec3 wsG[kWsS][kWsC]{}, railR[kWsS][2]{}, railL[kWsS][2]{};
  for (int i = 0; i < kWsS; ++i) {
    const float z = lerpf(kWindshieldRearZ, kWindshieldFrontZ,
                          static_cast<float>(i) / static_cast<float>(kWsS - 1));
    const Station s = evalStation(z);
    const float wGlass = std::max(0.24f, s.wCan - 0.058f);
    for (int c = 0; c < kWsC; ++c) {
      const float u = lerpf(
        -1.0f, 1.0f, static_cast<float>(c) / static_cast<float>(kWsC - 1));
      wsG[i][c] = {u * wGlass, lerpf(s.yRoof + 0.008f, s.yCan + 0.005f, u * u),
                   z};
    }
    railR[i][0] = {wGlass, s.yCan + 0.004f, z};
    railR[i][1] = {s.wCan, s.yCan, z};
    railL[i][0] = {-s.wCan, s.yCan, z};
    railL[i][1] = {-wGlass, s.yCan + 0.004f, z};
  }
  emitGrid<kWsS, kWsC>(verts, idx, xf, wsG, kWsS, kWsC, kCanopyGlass, 1.0f,
                       0.0f, true);
  emitGrid<kWsS, 2>(verts, idx, xf, railR, kWsS, 2, kGoldBright, 1.0f, 0.0f,
                    true);
  emitGrid<kWsS, 2>(verts, idx, xf, railL, kWsS, 2, kGoldBright, 1.0f, 0.0f,
                    true);
}

}  // namespace cybercab::mesh_detail
