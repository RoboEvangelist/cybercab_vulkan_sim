#include "world/CybercabMesh.hpp"
#include "world/CybercabBodyLoft.hpp"
#include <cmath>

namespace cybercab {
using namespace mesh_detail;
namespace {

void emitAeroWheel(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                   const Mat4& wm, float outerR, float sideSign) {
  struct RingPt {
    float rf, xOff, nxSign, nRad;
    Vec3 col;
  };
  const RingPt prof[] = {{1.000f, -0.114f, 0.00f, 1.00f, kTireDark},
                         {1.000f, 0.114f, 0.00f, 1.00f, kTireDark},
                         {0.925f, 0.125f, 0.65f, 0.76f, kTireDark},
                         {0.835f, 0.126f, 1.00f, 0.00f, kTireDark},
                         {0.825f, 0.132f, 0.85f, 0.53f, kTrimDark},
                         {0.810f, 0.134f, 1.00f, 0.00f, kGoldBase},
                         {0.685f, 0.126f, 0.93f, -0.36f, kGoldBright},
                         {0.115f, 0.121f, 0.99f, -0.12f, kGoldBright},
                         {0.055f, 0.115f, 0.92f, -0.38f, kGoldBase},
                         {0.000f, 0.115f, 1.00f, 0.00f, kGoldDark}};
  constexpr int kNR = static_cast<int>(sizeof(prof) / sizeof(prof[0]));
  const float semF = kCybercabSemTag;
  const float dTheta = (2.0f * kPi) / static_cast<float>(kWheelSegs);
  for (int p = 0; p + 1 < kNR; ++p) {
    const auto& a = prof[p];
    const auto& b = prof[p + 1];
    const uint32_t base = static_cast<uint32_t>(verts.size());
    for (int s = 0; s <= kWheelSegs; ++s) {
      const float th = static_cast<float>(s) * dTheta, cs = std::cos(th),
                  sn = std::sin(th);
      const Vec3 p0 = wm.transformPoint(
        {sideSign * a.xOff, cs * (a.rf * outerR), sn * (a.rf * outerR)});
      const Vec3 p1 = wm.transformPoint(
        {sideSign * b.xOff, cs * (b.rf * outerR), sn * (b.rf * outerR)});
      const Vec3 n0 =
        wm.transformVector({sideSign * a.nxSign, cs * a.nRad, sn * a.nRad})
          .normalized();
      const Vec3 n1 =
        wm.transformVector({sideSign * b.nxSign, cs * b.nRad, sn * b.nRad})
          .normalized();
      verts.push_back({p0.x, p0.y, p0.z, n0.x, n0.y, n0.z, b.col.x, b.col.y,
                       b.col.z, 1.0f, 0.0f, semF, 0, 0});
      verts.push_back({p1.x, p1.y, p1.z, n1.x, n1.y, n1.z, b.col.x, b.col.y,
                       b.col.z, 1.0f, 0.0f, semF, 0, 0});
    }
    for (int s = 0; s < kWheelSegs; ++s) {
      const uint32_t i0 = base + static_cast<uint32_t>(s * 2);
      if (sideSign > 0.0f) {
        idx.push_back(i0);
        idx.push_back(i0 + 3);
        idx.push_back(i0 + 1);
        idx.push_back(i0);
        idx.push_back(i0 + 2);
        idx.push_back(i0 + 3);
      } else {
        idx.push_back(i0);
        idx.push_back(i0 + 1);
        idx.push_back(i0 + 3);
        idx.push_back(i0);
        idx.push_back(i0 + 3);
        idx.push_back(i0 + 2);
      }
    }
  }
}

void emitWheelWellLiner(std::vector<SceneVertex>& verts,
                        std::vector<uint32_t>& idx, const Mat4& sprungMat,
                        float xCenter, float zCenter, float axleY,
                        float linerR) {
  constexpr int kArchSegs = 16;
  const float semF = kCybercabSemTag;
  const uint32_t base = static_cast<uint32_t>(verts.size());
  for (int i = 0; i <= kArchSegs; ++i) {
    const float a =
      lerpf(0.0f, kPi, static_cast<float>(i) / static_cast<float>(kArchSegs));
    const float ly = axleY + std::sin(a) * linerR,
                lz = zCenter - std::cos(a) * linerR;
    const Vec3 n =
      sprungMat.transformVector({0.0f, -std::sin(a), std::cos(a)}).normalized();
    const Vec3 pIn = sprungMat.transformPoint({xCenter - 0.19f, ly, lz});
    const Vec3 pOut = sprungMat.transformPoint({xCenter + 0.19f, ly, lz});
    verts.push_back({pIn.x, pIn.y, pIn.z, n.x, n.y, n.z, kTrimDark.x,
                     kTrimDark.y, kTrimDark.z, 1, 0, semF, 0, 0});
    verts.push_back({pOut.x, pOut.y, pOut.z, n.x, n.y, n.z, kTrimDark.x,
                     kTrimDark.y, kTrimDark.z, 1, 0, semF, 0, 0});
  }
  for (int i = 0; i < kArchSegs; ++i) {
    const uint32_t b = base + static_cast<uint32_t>(i * 2);
    idx.push_back(b);
    idx.push_back(b + 3);
    idx.push_back(b + 2);
    idx.push_back(b);
    idx.push_back(b + 1);
    idx.push_back(b + 3);
  }
}

}  // namespace

void appendCybercabVehicleMesh(std::vector<SceneVertex>& verts, std::vector<uint32_t>& indices,
                               const EgoState& ego, const Mat4& egoMat, const Mat4& sprungMat,
                               bool fsdWireMode) {
    Station st[kNumSt]{};
    for (int i = 0; i < kNumSt; ++i) st[i] = evalStation(kZStations[i]);

    if (fsdWireMode) {
        Vec3 hRoof[kNumSt][3]{}, hSideR[kNumSt][2]{}, hSideL[kNumSt][2]{};
        for (int i = 0; i < kNumSt; ++i) {
            hRoof[i][0] = {-st[i].wCan, st[i].yCan, st[i].z}; hRoof[i][1] = {0.0f, st[i].yRoof, st[i].z};
            hRoof[i][2] = { st[i].wCan, st[i].yCan, st[i].z};
            hSideR[i][0] = { st[i].wBot, st[i].yBot, st[i].z}; hSideR[i][1] = { st[i].wCan, st[i].yCan, st[i].z};
            hSideL[i][0] = {-st[i].wCan, st[i].yCan, st[i].z}; hSideL[i][1] = {-st[i].wBot, st[i].yBot, st[i].z};
        }
        emitGrid<kNumSt, 3>(verts, indices, sprungMat, hRoof, kNumSt, 3, kHoloCyan, 0.45f, 1.0f, true);
        emitGrid<kNumSt, 2>(verts, indices, sprungMat, hSideR, kNumSt, 2, kHoloCyan, 0.40f, 0.9f, false);
        emitGrid<kNumSt, 2>(verts, indices, sprungMat, hSideL, kNumSt, 2, kHoloCyan, 0.40f, 0.9f, false);
        return;
    }

    Vec3 loR[kNumSt][3]{}, loL[kNumSt][3]{}, upR[kNumSt][3]{}, upL[kNumSt][3]{};
    Vec3 roofGrid[18][7]{}, hoodGrid[17][7]{};
    constexpr float kRoofCols[7] = {-1.0f, -0.68f, -0.36f, 0.0f, 0.36f, 0.68f, 1.0f};
    for (int i = 0; i < kNumSt; ++i) {
        loR[i][0] = { st[i].wBot, st[i].yBot, st[i].z}; loR[i][1] = { st[i].wBelly, st[i].yBelly, st[i].z}; loR[i][2] = { st[i].wShld, st[i].yShld, st[i].z};
        loL[i][0] = {-st[i].wShld, st[i].yShld, st[i].z}; loL[i][1] = {-st[i].wBelly, st[i].yBelly, st[i].z}; loL[i][2] = {-st[i].wBot, st[i].yBot, st[i].z};
        upR[i][0] = { st[i].wShld, st[i].yShld, st[i].z}; upR[i][1] = { st[i].wUpMid, st[i].yUpMid, st[i].z}; upR[i][2] = { st[i].wCan, st[i].yCan, st[i].z};
        upL[i][0] = {-st[i].wCan, st[i].yCan, st[i].z};   upL[i][1] = {-st[i].wUpMid, st[i].yUpMid, st[i].z}; upL[i][2] = {-st[i].wShld, st[i].yShld, st[i].z};
        if (i < 18) {
            const float rFade = (st[i].z > -0.82f && st[i].z < 1.02f) ? std::sin(kPi * clampf((st[i].z + 0.82f) / 1.84f, 0.0f, 1.0f)) : 0.0f;
            for (int c = 0; c < 7; ++c) roofGrid[i][c] = {kRoofCols[c] * st[i].wCan, lerpf(st[i].yRoof, st[i].yCan, kRoofCols[c] * kRoofCols[c]) + ((c == 2 || c == 4) ? 0.012f * rFade : 0.0f), st[i].z};
        } else if (i >= 20) {
            for (int c = 0; c < 7; ++c) hoodGrid[i - 20][c] = {kRoofCols[c] * st[i].wCan, lerpf(st[i].yRoof, st[i].yCan, kRoofCols[c] * kRoofCols[c]), st[i].z};
        }
    }
    emitGrid<kNumSt, 3>(verts, indices, sprungMat, loR, kNumSt, 3, kGoldBase, 1.0f, 0.0f, false);
    emitGrid<kNumSt, 3>(verts, indices, sprungMat, loL, kNumSt, 3, kGoldBase, 1.0f, 0.0f, false);
    emitGrid<kNumSt, 3>(verts, indices, sprungMat, upR, kNumSt, 3, kGoldBright, 1.0f, 0.0f, false);
    emitGrid<kNumSt, 3>(verts, indices, sprungMat, upL, kNumSt, 3, kGoldBright, 1.0f, 0.0f, false);
    emitGrid<18, 7>(verts, indices, sprungMat, roofGrid, 18, 7, kGoldBright, 1.0f, 0.0f, true);
    emitGrid<17, 7>(verts, indices, sprungMat, hoodGrid, 17, 7, kGoldBright, 1.0f, 0.0f, true);
    emitWindshieldAndPillars(verts, indices, sprungMat); emitSculptedEndCap(verts, indices, sprungMat, st[0], false); emitSculptedEndCap(verts, indices, sprungMat, st[kNumSt - 1], true); emitSideWindows(verts, indices, sprungMat, st);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.34f, 1.24f}), {1.12f, 0.34f, 3.60f}, kTrimDark);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.305f, 1.38f}), {1.845f, 0.130f, 1.92f}, kCharcoalTrim);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.290f, 3.280f}), {1.38f, 0.045f, 0.22f}, kCharcoalTrim);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.335f, 3.295f}), {1.16f, 0.075f, 0.12f}, kTrimDark);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.380f, -0.825f}), {1.52f, 0.165f, 0.13f}, kCharcoalTrim);
    for (const float sx : {-1.0f, 1.0f}) {
        appendBox(verts, indices, sprungMat * Mat4::translation({sx * 0.76f, 0.380f, -0.60f}), {0.09f, 0.155f, 0.44f}, kCharcoalTrim);
        appendBox(verts, indices, sprungMat * Mat4::translation({sx * 0.818f, 1.13f, 0.78f}) * Mat4::rotZ(sx * 0.50f), {0.024f, 0.34f, 0.040f}, kTrimDark);
        appendBox(verts, indices, sprungMat * Mat4::translation({sx * 0.902f, 0.80f, 2.24f}), {0.046f, 0.042f, 0.13f}, kTrimDark);
        appendBox(verts, indices, sprungMat * Mat4::translation({sx * 0.835f, 0.435f, 3.09f}), {0.022f, 0.095f, 0.024f}, kAmberMarker, 1.0f, 0.35f);
        appendBox(verts, indices, sprungMat * Mat4::translation({sx * 0.620f, 0.562f, 3.25f}) * Mat4::rotY(sx > 0 ? -0.35f : 0.35f), {0.26f, 0.045f, 0.07f}, kTrimDark);
    }
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.430f, 3.356f}), {0.28f, 0.13f, 0.014f}, kPlateWhite);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.385f, -0.892f}), {0.28f, 0.13f, 0.012f}, kPlateWhite);
    const bool isBraking = ego.accel < -0.45f || ego.speed < 0.25f || ego.aebActive;
    for (const float rx : {-0.58f, 0.58f}) appendBox(verts, indices, sprungMat * Mat4::translation({rx, 0.468f, -0.890f}), {0.28f, 0.020f, 0.018f}, kTailLed, 1.0f, isBraking ? 2.4f : 0.9f);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.558f, 3.31f}), {1.40f, 0.048f, 0.05f}, kTrimDark);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.558f, 3.332f}), {1.34f, 0.024f, 0.04f}, kDrlLed, 1.0f, 2.9f);
    for (const float sx : {-0.72f, 0.72f}) appendBox(verts, indices, sprungMat * Mat4::translation({sx, 0.558f, 3.24f}) * Mat4::rotY(sx > 0 ? -0.46f : 0.46f), {0.04f, 0.022f, 0.15f}, kDrlLed, 1.0f, 2.6f);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.880f, -0.888f}), {1.50f, 0.032f, 0.055f}, kTrimDark);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.848f, -0.884f}), {1.46f, 0.030f, 0.036f}, kTailLed, 1.0f, isBraking ? 4.4f : 1.5f);
    if (ego.gear < 0) appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.818f, -0.888f}), {0.48f, 0.024f, 0.03f}, kRevLed, 1.0f, 3.2f);
    const struct { float x, y, z, r, lr; bool front; } wheels[4] = {
        { kTrackHalfF, kWheelAxleYF, kFrontAxleZ, kTireOuterRF, 0.398f, true},  {-kTrackHalfF, kWheelAxleYF, kFrontAxleZ, kTireOuterRF, 0.398f, true},
        { kTrackHalfR, kWheelAxleYR, kRearAxleZ,  kTireOuterRR, 0.424f, false}, {-kTrackHalfR, kWheelAxleYR, kRearAxleZ,  kTireOuterRR, 0.424f, false}
    };
    for (const auto& w : wheels) {
        emitWheelWellLiner(verts, indices, sprungMat, w.x * 0.90f, w.z, w.y, w.lr);
        const Mat4 wm = egoMat * Mat4::translation({w.x, w.y, w.z}) * Mat4::rotY(w.front ? ego.steerAngle : 0.0f) * Mat4::rotX(ego.wheelSpin);
        emitAeroWheel(verts, indices, wm, w.r, (w.x >= 0.0f) ? 1.0f : -1.0f);
    }
}

} // namespace cybercab
