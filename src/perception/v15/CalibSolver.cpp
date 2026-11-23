#include "perception/v15/CalibSolver.hpp"
#include <algorithm>
#include <cmath>

namespace cybercab::v15::calib {
namespace {
constexpr float kMinCoher = 0.75f, kMinTileMag = 0.35f, kMinElevSin = 0.05f, kMaxVertDot = 0.26f;
constexpr int kIspUndistortIters = 4; // must match brownConradyUndistort() used by cam_isp_soil.comp

struct Node { float cx, cy, gx, w; Vec3 rayCam; };

// Pixel-gradient edge plane: push g back through the distortion Jacobian, then n = (g, -g . xu).
Vec3 tileEdgeNormalCam(const CameraIntrinsics& intr, const std::array<float, 4>& d, const Vec2& xu,
                       float gx, float gy) {
    const Vec2 jx = (distortNormalized(xu.x + 1e-4f, xu.y, d) - distortNormalized(xu.x - 1e-4f, xu.y, d)) * 5000.0f;
    const Vec2 jy = (distortNormalized(xu.x, xu.y + 1e-4f, d) - distortNormalized(xu.x, xu.y - 1e-4f, d)) * 5000.0f;
    const float gxD = gx * intr.fx, gyD = gy * intr.fy;
    const float gxU = jx.x * gxD + jx.y * gyD, gyU = jy.x * gxD + jy.y * gyD;
    return Vec3{gxU, gyU, -(gxU * xu.x + gyU * xu.y)}.normalized();
}

// Greedy bottom-up linking of near-vertical tiles into straight chains; one plane per chain.
void appendChainPlanes(const std::vector<Node>& nodesIn, const CameraIntrinsics& intr, const Mat4& rVehCam,
                       const Vec3& uPrior, int camH, std::vector<Vec4>& out) {
    std::vector<Node> nodes = nodesIn;
    std::sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) { return a.cy < b.cy; });
    const float maxStepY = (camH >= 400) ? 26.0f : 14.0f, minSpanY = 0.11f * static_cast<float>(camH);
    const float maxOrthPx = (camH >= 400) ? 1.25f : 0.75f;
    std::vector<uint8_t> used(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (used[i]) continue;
        std::vector<size_t> ch{i};
        used[i] = 1;
        for (;;) {
            const Node& tail = nodes[ch.back()];
            size_t bestJ = nodes.size();
            float bestDx = 1e9f;
            for (size_t j = ch.back() + 1; j < nodes.size(); ++j) {
                const float dy = nodes[j].cy - tail.cy, dx = std::abs(nodes[j].cx - tail.cx);
                if (used[j] || tail.gx * nodes[j].gx <= 0.0f || dy < 3.0f || dy > maxStepY) continue;
                if (dx <= 0.14f * dy + 1.5f && dx < bestDx) { bestDx = dx; bestJ = j; }
            }
            if (bestJ == nodes.size()) break;
            used[bestJ] = 1;
            ch.push_back(bestJ);
        }
        const float spanY = nodes[ch.back()].cy - nodes[ch.front()].cy;
        if (ch.size() < 4 || spanY < minSpanY) continue;
        Vec3 topR{}, botR{};
        for (size_t k = 0; k < ch.size(); ++k) {
            const Node& nd = nodes[ch[k]];
            (k < ch.size() / 2 ? topR : botR) += nd.rayCam * nd.w;
        }
        const Vec3 nRaw = topR.cross(botR);
        if (nRaw.normSq() < 1e-10f) continue;
        const Vec3 nCam = nRaw.normalized();
        bool straight = true; // reject kinked chains (window jumps) via ray-plane distance in pixels
        for (size_t k : ch) straight = straight && std::abs(nodes[k].rayCam.dot(nCam)) * intr.fx <= maxOrthPx;
        const Vec3 nLoc = rVehCam.transformVector(nCam);
        if (straight && std::abs(nLoc.dot(uPrior)) <= kMaxVertDot) {
            out.push_back({nLoc.x, nLoc.y, nLoc.z, spanY / static_cast<float>(camH)});
        }
    }
}
} // namespace

Vec3 uprightVec(float pitchRad, float rollRad) {
    const float cp = std::cos(pitchRad), sp = std::sin(pitchRad);
    return {cp * std::sin(rollRad), cp * std::cos(rollRad), -sp};
}

bool insideValidOptics(const CameraSpec& spec, const CameraIntrinsics& intr, float pxX, float pxY, const Vec2& xu) {
    const float w = static_cast<float>(intr.width), h = static_cast<float>(intr.height);
    if (pxX < 0.04f * w || pxX > 0.96f * w || pxY < 0.02f * h) return false;
    const float rs = spec.renderOverscan();
    const float uPin = (xu.x * intr.fx / rs + intr.cx) / w, vPin = (xu.y * intr.fy / rs + intr.cy) / h;
    return uPin > 0.02f && uPin < 0.98f && vPin > 0.02f && vPin < 0.98f;
}

void extractEdgeNormals(const SimConfig& cfg, int camW, int camH, const std::vector<CalibRayObs>& obs,
                        float priorPitchRad, float priorRollRad, std::vector<Vec4>& out, bool allowTileFallback) {
    out.clear();
    const int nCams = static_cast<int>(cfg.cameras.size());
    if (nCams == 0 || camW <= 0 || camH <= 0) return;
    const Vec3 uPrior = uprightVec(priorPitchRad, priorRollRad);
    std::vector<CameraIntrinsics> intrs(static_cast<size_t>(nCams));
    std::vector<Mat4> rVeh(static_cast<size_t>(nCams));
    std::vector<std::vector<Node>> byCam(static_cast<size_t>(nCams));
    std::vector<Vec4> tilePlanes;
    for (int c = 0; c < nCams; ++c) {
        intrs[c] = cfg.cameras[c].intrinsics(camW, camH);
        rVeh[c] = cfg.cameras[c].extrinsicVehicleLocalFromCam();
    }
    for (const auto& o : obs) {
        if (o.camIdx < 0 || o.camIdx >= nCams || o.coherence < kMinCoher || o.sumMag < kMinTileMag) continue;
        if (o.cyPx > 0.85f * static_cast<float>(camH)) continue;
        const auto& spec = cfg.cameras[o.camIdx];
        const auto& intr = intrs[o.camIdx];
        const Vec2 xu = undistortNormalized((o.cxPx - intr.cx) / intr.fx, (o.cyPx - intr.cy) / intr.fy,
                                            spec.distortion, kIspUndistortIters);
        if (!insideValidOptics(spec, intr, o.cxPx, o.cyPx, xu)) continue;
        const Vec3 rCam = Vec3{xu.x, xu.y, 1.0f}.normalized();
        if (rVeh[o.camIdx].transformVector(rCam).dot(uPrior) < kMinElevSin) continue; // above horizon only
        if (std::abs(o.sumGx) >= 1.8f * std::abs(o.sumGy)) {
            byCam[o.camIdx].push_back({o.cxPx, o.cyPx, o.sumGx, std::min(o.sumMag, 2.0f) * o.coherence, rCam});
        }
        const Vec3 nLoc = rVeh[o.camIdx].transformVector(tileEdgeNormalCam(intr, spec.distortion, xu, o.sumGx, o.sumGy));
        if (std::abs(nLoc.dot(uPrior)) <= kMaxVertDot) {
            tilePlanes.push_back({nLoc.x, nLoc.y, nLoc.z, std::min(o.sumMag, 2.0f) * o.coherence * o.coherence * o.coherence});
        }
    }
    for (int c = 0; c < nCams; ++c) appendChainPlanes(byCam[c], intrs[c], rVeh[c], uPrior, camH, out);
    if (allowTileFallback && out.size() < 4) out = tilePlanes;
}

CalibEstimate solveUpright(const std::vector<Vec4>& eqs, float priorPitchRad, float priorRollRad) {
    if (eqs.size() < 4) return {};
    const Vec3 uPrior = uprightVec(priorPitchRad, priorRollRad);
    float a = uPrior.x / std::max(0.2f, uPrior.y), b = uPrior.z / std::max(0.2f, uPrior.y);
    for (int it = 0; it < 7; ++it) { // it 0: LS; 1-2: Huber; 3+: Tukey biweight (hard outlier rejection)
        const Vec3 u = Vec3{a, 1.0f, b}.normalized();
        double sxx = 1e-4, sxz = 0.0, szz = 1e-4, sxy = 0.0, szy = 0.0;
        for (const auto& e : eqs) {
            const float r = std::abs(e.x * u.x + e.y * u.y + e.z * u.z);
            float rw = 1.0f;
            if (it >= 1 && it <= 2) { const float hd = (it == 1) ? 0.05f : 0.025f; rw = (r <= hd) ? 1.0f : hd / r; }
            else if (it >= 3) { const float t = r / ((it <= 4) ? 0.020f : 0.012f); rw = (t < 1.0f) ? (1 - t * t) * (1 - t * t) : 0.0f; }
            const float hw = e.w * rw;
            sxx += hw * e.x * e.x; sxz += hw * e.x * e.z; szz += hw * e.z * e.z;
            sxy += hw * e.x * e.y; szy += hw * e.z * e.y;
        }
        const double det = sxx * szz - sxz * sxz;
        if (det <= 1e-6) return {};
        a = clampf(static_cast<float>((-szz * sxy + sxz * szy) / det), -0.35f, 0.35f);
        b = clampf(static_cast<float>((sxz * sxy - sxx * szy) / det), -0.35f, 0.35f);
    }
    const Vec3 u = Vec3{a, 1.0f, b}.normalized();
    double sq = 0.0, ws = 0.0;
    CalibEstimate out{};
    for (const auto& e : eqs) {
        const float r = std::abs(e.x * u.x + e.y * u.y + e.z * u.z);
        if (r < 0.012f) { ++out.inlierCount; sq += e.w * r * r; ws += e.w; }
    }
    out.rollRad = std::atan2(u.x, u.y);
    out.pitchRad = -std::asin(clampf(u.z, -0.95f, 0.95f));
    out.rollDeg = radToDeg(out.rollRad);
    out.pitchDeg = radToDeg(out.pitchRad);
    out.residualDeg = (ws > 0.0) ? radToDeg(static_cast<float>(std::sqrt(sq / ws))) : 90.0f;
    out.confidence = clampf(static_cast<float>(out.inlierCount) / 40.0f, 0.0f, 1.0f);
    out.converged = out.inlierCount >= 4 && out.residualDeg < 1.0f;
    return out;
}

CalibEstimate solveFromObservations(const SimConfig& cfg, int camW, int camH, const std::vector<CalibRayObs>& obs,
                                    float priorPitchRad, float priorRollRad) {
    std::vector<Vec4> eqs;
    extractEdgeNormals(cfg, camW, camH, obs, priorPitchRad, priorRollRad, eqs);
    return solveUpright(eqs, priorPitchRad, priorRollRad);
}

} // namespace cybercab::v15::calib
