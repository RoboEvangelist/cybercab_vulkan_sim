#pragma once

#include "config/SimConfig.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "world/WorldEngine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cybercab::testutil {

struct LaneCorridorMetrics {
  int tp = 0, fp = 0, fn = 0, edgeN = 0;
  double edgeErrSum = 0.0;
  double iou() const {
    const int d = tp + fp + fn;
    return d > 0 ? static_cast<double>(tp) / d : 0.0;
  }
  double edgeErrM() const { return edgeN > 0 ? edgeErrSum / edgeN : 0.0; }
};

inline void evalGroundLaneIoU(const WorldEngine& w, const CameraRigRenderer& r,
                              const SimConfig& cfg,
                              LaneCorridorMetrics& out) {
  const auto dyn = w.dynamicGroundTruthOBBs();
  const int cw = r.camWidth(), ch = r.camHeight();
  const std::vector<uint8_t> rgba0 =
    (!cfg.cameras[0].soiled) ? r.downloadCameraRGBA(0) : std::vector<uint8_t>{};
  const std::vector<uint8_t> rgba2 = r.downloadCameraRGBA(2);
  auto blocked = [&](float wx, float wz) {
    for (const auto& b : dyn) {
      const float c = std::cos(b.yaw), s = std::sin(b.yaw);
      const float dx = wx - b.cx, dz = wz - b.cz;
      if (std::abs(dx * c - dz * s) <= b.hx + 0.8f &&
          std::abs(dx * s + dz * c) <= b.hz + 0.8f) return true;
    }
    return false;
  };
  auto tryCam = [&](int cam, const Vec3& wp, Vec2& px) {
    float dep = 0.0f;
    return r.projectWorldToPixel(cam, wp, px, dep) && px.x >= 0.0f &&
           px.x < cw && px.y >= 0.0f && px.y < ch;
  };
  auto isRoadPx = [](const std::vector<uint8_t>& buf, int wPx, int hPx,
                     const Vec2& px) {
    const int ix = std::clamp(static_cast<int>(px.x), 0, wPx - 1);
    const int iy = std::clamp(static_cast<int>(px.y), 0, hPx - 1);
    const size_t i = (static_cast<size_t>(iy) * wPx + ix) * 4u;
    const float rf = buf[i] / 255.f, gf = buf[i + 1] / 255.f;
    const float bf = buf[i + 2] / 255.f;
    const float mx = std::max({rf, gf, bf}), mn = std::min({rf, gf, bf});
    const float sat = (mx > 1e-4f) ? ((mx - mn) / mx) : 0.0f;
    const float lum = 0.299f * rf + 0.587f * gf + 0.114f * bf;
    const bool asp = (sat < 0.36f && lum >= 0.06f && lum <= 0.42f);
    const bool dash = (sat < 0.22f && lum > 0.42f);
    const bool curb =
      (rf > gf * 1.05f && gf > bf * 1.15f && (rf - bf) > 0.10f);
    return asp || dash || curb;
  };

  const float u0 = w.ego().loopProgressU;
  const float len = w.centerline().getLength();
  for (float fwd : {8.0f, 11.0f, 14.0f, 17.0f, 20.0f}) {
    float uF = std::fmod(u0 + fwd / len, 1.0f);
    if (uF < 0.0f) uF += 1.0f;
    const Vec3 cp = w.centerline().getPointAt(uF);
    const Vec3 ct = w.centerline().getTangentAt(uF);
    const Vec3 left{ct.z, 0.0f, -ct.x};
    float gtMn = 1e9f, gtMx = -1e9f, prMn = 1e9f, prMx = -1e9f;
    bool crossHit = false;
    for (float cLat = -6.5f; cLat <= 6.51f; cLat += 0.5f) {
      const Vec3 wp{cp.x + left.x * cLat, 0.03f, cp.z + left.z * cLat};
      if (blocked(wp.x, wp.z)) continue;
      Vec2 px{};
      const bool use0 = (!rgba0.empty() && tryCam(0, wp, px));
      if (!use0 && !tryCam(2, wp, px)) continue;
      const bool prRoad = isRoadPx(use0 ? rgba0 : rgba2, cw, ch, px);
      const bool inCross =
        std::abs(wp.x) <= 5.5f && wp.z >= -40.0f && wp.z <= 90.0f;
      const bool gtRoad = inCross || (std::abs(cLat) <= 4.6f);
      if (inCross) crossHit = true;
      if (gtRoad && prRoad) ++out.tp;
      else if (!gtRoad && prRoad) ++out.fp;
      else if (gtRoad && !prRoad) ++out.fn;
      if (gtRoad) { gtMn = std::min(gtMn, cLat); gtMx = std::max(gtMx, cLat); }
      if (prRoad) { prMn = std::min(prMn, cLat); prMx = std::max(prMx, cLat); }
    }
    if (!crossHit && gtMn < gtMx && prMn < prMx) {
      out.edgeErrSum += std::abs(prMn - gtMn) + std::abs(prMx - gtMx);
      out.edgeN += 2;
    }
  }
}

}  // namespace cybercab::testutil
