#include "world/CybercabMesh.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace cybercab {
namespace {

constexpr Vec3 kGoldBase{0.824f, 0.706f, 0.490f};
constexpr Vec3 kGoldBright{0.886f, 0.780f, 0.576f};
constexpr Vec3 kGoldDark{0.615f, 0.514f, 0.337f};
constexpr Vec3 kCanopyGlass{0.030f, 0.055f, 0.115f};
constexpr Vec3 kTrimDark{0.055f, 0.065f, 0.090f};
constexpr Vec3 kTireDark{0.045f, 0.053f, 0.068f};
constexpr Vec3 kHoloCyan{0.220f, 0.740f, 0.970f};
constexpr Vec3 kDrlLed{0.930f, 0.980f, 1.000f};
constexpr Vec3 kTailLed{0.960f, 0.130f, 0.130f};
constexpr Vec3 kRevLed{0.980f, 0.980f, 0.960f};

constexpr float kRearAxleZ = 0.00f, kFrontAxleZ = 2.75f, kWheelAxleY = 0.365f;
constexpr float kTireOuterR = 0.365f, kTireWidthM = 0.245f, kWheelArchR = 0.402f;
constexpr float kTrackHalfR = 0.836f, kTrackHalfF = 0.818f;
constexpr float kBodyRearZ = -0.88f, kBodyCrestZ = 1.04f, kBodyCowlZ = 2.10f, kBodyNoseZ = 3.35f;
constexpr float kCanopyRearZ = -0.49f, kCanopyFrontZ = 2.05f;
constexpr int kNumSt = 37, kSideWinSt = 12, kWheelSegs = 36;

constexpr std::array<float, kNumSt> kZStations = {{
    -0.88f, -0.78f, -0.64f, -0.49f, -0.40f, -0.37f, -0.30f, -0.20f, -0.10f, 0.00f,
     0.10f,  0.20f,  0.30f,  0.37f,  0.40f,  0.62f,  0.86f,  1.12f,  1.38f, 1.64f,
     1.90f,  2.12f,  2.34f,  2.38f,  2.45f,  2.55f,  2.65f,  2.75f,  2.85f, 2.95f,
     3.05f,  3.12f,  3.15f,  3.21f,  3.27f,  3.32f,  3.35f
}};

struct Station {
    float z, yBot, yBelly, yShld, yUpMid, yCan, yRoof;
    float wBot, wBelly, wShld, wUpMid, wCan;
};

float wheelArchY(float z, float baseY) {
    float y = baseY;
    for (const float axZ : {kRearAxleZ, kFrontAxleZ}) {
        const float dz = std::abs(z - axZ);
        if (dz < kWheelArchR) y = std::max(y, kWheelAxleY + std::sqrt(std::max(0.0f, kWheelArchR * kWheelArchR - dz * dz)));
    }
    return y;
}

Station evalStation(float z) {
    Station s{};
    s.z = z;
    const float baseY = (z < -0.40f) ? lerpf(0.475f, 0.338f, (z - kBodyRearZ) / 0.48f)
                      : (z > 3.15f)  ? lerpf(0.338f, 0.375f, (z - 3.15f) / 0.20f) : 0.335f;
    s.yBot = wheelArchY(z, baseY);
    if (z < kBodyCrestZ) {
        const float t = (kBodyCrestZ - z) / (kBodyCrestZ - kBodyRearZ);
        s.yRoof = 1.390f - 0.485f * (0.32f * t + 0.68f * t * t);
    } else if (z <= kBodyCowlZ) {
        const float t = (z - kBodyCrestZ) / (kBodyCowlZ - kBodyCrestZ);
        s.yRoof = 1.390f - 0.488f * (t * t * (1.20f - 0.20f * t));
    } else {
        const float t = (z - kBodyCowlZ) / (kBodyNoseZ - kBodyCowlZ);
        s.yRoof = 0.902f - 0.312f * (0.18f * t + 0.82f * t * t * t);
    }
    const bool cab = (z > -0.35f && z < kBodyCowlZ);
    s.yCan = s.yRoof - (cab ? 0.048f : 0.022f);
    if (z <= kBodyCowlZ) s.yShld = lerpf(0.875f, 0.805f, clampf((z - kBodyRearZ) / (kBodyCowlZ - kBodyRearZ), 0.0f, 1.0f));
    else if (z <= 2.84f) s.yShld = 0.805f;
    else {
        const float t = clampf((z - 2.84f) / (kBodyNoseZ - 2.84f), 0.0f, 1.0f);
        s.yShld = lerpf(0.805f, 0.555f, t * t);
    }
    s.yShld = clampf(s.yShld, s.yBot + 0.028f, s.yCan - 0.018f);
    s.yBelly = lerpf(s.yBot, s.yShld, 0.55f);
    s.yUpMid = lerpf(s.yShld, s.yCan, 0.52f);

    const float trR = (z < -0.40f) ? clampf((-0.40f - z) / 0.48f, 0.0f, 1.0f) : 0.0f;
    const float trF = (z > 2.84f)  ? clampf((z - 2.84f) / 0.51f, 0.0f, 1.0f) : 0.0f;
    s.wBot = 0.902f - 0.115f * trR * trR - 0.155f * trF * trF;
    s.wBelly = 0.962f - 0.105f * trR * trR - 0.155f * trF * trF;
    s.wShld = 0.946f - 0.100f * trR * trR - 0.155f * trF * trF;
    const float cabTaper = (z > -0.35f && z < kBodyCowlZ) ? 0.215f * std::sin(kPi * ((z + 0.35f) / (kBodyCowlZ + 0.35f))) : 0.045f;
    s.wCan = s.wShld - cabTaper - 0.045f;
    s.wUpMid = lerpf(s.wShld, s.wCan, 0.42f);
    return s;
}

template <int MaxS, int MaxC>
void emitGrid(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx, const Mat4& xf,
              const Vec3 (&pos)[MaxS][MaxC], int nS, int nC, Vec3 col, float a, float em, bool flip = false) {
    Vec3 nrm[MaxS][MaxC]{};
    for (int s = 0; s < nS - 1; ++s) {
        for (int c = 0; c < nC - 1; ++c) {
            const Vec3 p00 = pos[s][c], p10 = pos[s][c + 1], p11 = pos[s + 1][c + 1], p01 = pos[s + 1][c];
            Vec3 fn = flip ? (p01 - p00).cross(p11 - p00) : (p10 - p00).cross(p11 - p00);
            if (fn.normSq() > 1e-10f) {
                fn = fn.normalized();
                nrm[s][c] += fn; nrm[s][c + 1] += fn; nrm[s + 1][c + 1] += fn; nrm[s + 1][c] += fn;
            }
        }
    }
    const float semF = static_cast<float>(static_cast<int>(SemanticClass::Vehicle));
    const uint32_t base = static_cast<uint32_t>(verts.size());
    for (int s = 0; s < nS; ++s) {
        for (int c = 0; c < nC; ++c) {
            const Vec3 wp = xf.transformPoint(pos[s][c]);
            const Vec3 wn = xf.transformVector(nrm[s][c].normalized()).normalized();
            verts.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, col.x, col.y, col.z, a, em, semF, 0.0f, 0.0f});
        }
    }
    for (int s = 0; s < nS - 1; ++s) {
        for (int c = 0; c < nC - 1; ++c) {
            const uint32_t i00 = base + static_cast<uint32_t>(s * nC + c), i10 = i00 + 1;
            const uint32_t i01 = i00 + static_cast<uint32_t>(nC), i11 = i01 + 1;
            if (!flip) { idx.push_back(i00); idx.push_back(i10); idx.push_back(i11); idx.push_back(i00); idx.push_back(i11); idx.push_back(i01); }
            else       { idx.push_back(i00); idx.push_back(i11); idx.push_back(i10); idx.push_back(i00); idx.push_back(i01); idx.push_back(i11); }
        }
    }
}

void appendBox(std::vector<SceneVertex>& v, std::vector<uint32_t>& idx, const Mat4& xf,
               Vec3 sz, Vec3 col, float a = 1.0f, float em = 0.0f) {
    const float hx = sz.x * 0.5f, hy = sz.y * 0.5f, hz = sz.z * 0.5f;
    const float semF = static_cast<float>(static_cast<int>(SemanticClass::Vehicle));
    const Vec3 ns[6] = {{0,0,1},{0,0,-1},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}};
    const Vec3 ps[6][4] = {
        {{-hx,-hy, hz},{ hx,-hy, hz},{ hx, hy, hz},{-hx, hy, hz}}, {{ hx,-hy,-hz},{-hx,-hy,-hz},{-hx, hy,-hz},{ hx, hy,-hz}},
        {{ hx,-hy, hz},{ hx,-hy,-hz},{ hx, hy,-hz},{ hx, hy, hz}}, {{-hx,-hy,-hz},{-hx,-hy, hz},{-hx, hy, hz},{-hx, hy,-hz}},
        {{-hx, hy, hz},{ hx, hy, hz},{ hx, hy,-hz},{-hx, hy,-hz}}, {{-hx,-hy,-hz},{ hx,-hy,-hz},{ hx,-hy, hz},{-hx,-hy, hz}}
    };
    for (int f = 0; f < 6; ++f) {
        const uint32_t b = static_cast<uint32_t>(v.size());
        const Vec3 wn = xf.transformVector(ns[f]).normalized();
        for (int k = 0; k < 4; ++k) {
            const Vec3 wp = xf.transformPoint(ps[f][k]);
            v.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, col.x, col.y, col.z, a, em, semF, 0.0f, 0.0f});
        }
        idx.push_back(b); idx.push_back(b + 1); idx.push_back(b + 2);
        idx.push_back(b); idx.push_back(b + 2); idx.push_back(b + 3);
    }
}

void emitSculptedEndCap(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                        const Mat4& xf, const Station& st, bool isFront) {
    constexpr int kCols = 7, kRows = 4;
    Vec3 g[kRows][kCols]{};
    const float ys[kRows] = {st.yBot, st.yShld, st.yCan, st.yRoof};
    const float ws[kRows] = {st.wBot, st.wShld, st.wCan, 0.0f};
    const float zCurve = isFront ? -0.075f : 0.055f;
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            const float u = lerpf(-1.0f, 1.0f, static_cast<float>(c) / static_cast<float>(kCols - 1));
            const float yMid = (r == kRows - 1) ? st.yRoof : lerpf(ys[r], st.yRoof, 0.08f * (1.0f - u * u));
            g[r][c] = {u * ws[r], yMid, st.z + zCurve * u * u};
        }
    }
    emitGrid<kRows, kCols>(verts, idx, xf, g, kRows, kCols, kGoldDark, 1.0f, 0.0f, !isFront);
}

void emitSideWindows(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                     const Mat4& xf, const Station (&st)[kNumSt]) {
    constexpr float wz[kSideWinSt] = {0.22f, 0.38f, 0.56f, 0.76f, 0.96f, 1.16f, 1.34f, 1.52f, 1.68f, 1.82f, 1.92f, 2.00f};
    constexpr float wyL[kSideWinSt] = {0.97f, 0.945f, 0.92f, 0.90f, 0.885f, 0.87f, 0.86f, 0.85f, 0.84f, 0.835f, 0.835f, 0.845f};
    constexpr float wyH[kSideWinSt] = {0.99f, 1.11f, 1.21f, 1.28f, 1.32f, 1.325f, 1.29f, 1.20f, 1.08f, 0.965f, 0.895f, 0.858f};
    auto evalUpperX = [&](float zq, float yq) -> float {
        int seg = kNumSt - 2;
        for (int s = 0; s < kNumSt - 1; ++s) { if (zq >= st[s].z && zq <= st[s + 1].z) { seg = s; break; } }
        const float tz = clampf((zq - st[seg].z) / std::max(1e-4f, st[seg + 1].z - st[seg].z), 0.0f, 1.0f);
        const float yS = lerpf(st[seg].yShld, st[seg + 1].yShld, tz), yM = lerpf(st[seg].yUpMid, st[seg + 1].yUpMid, tz);
        const float yC = lerpf(st[seg].yCan, st[seg + 1].yCan, tz);
        const float wS = lerpf(st[seg].wShld, st[seg + 1].wShld, tz), wM = lerpf(st[seg].wUpMid, st[seg + 1].wUpMid, tz);
        const float wC = lerpf(st[seg].wCan, st[seg + 1].wCan, tz);
        return ((yq <= yM) ? lerpf(wS, wM, clampf((yq - yS) / std::max(1e-4f, yM - yS), 0.0f, 1.0f))
                           : lerpf(wM, wC, clampf((yq - yM) / std::max(1e-4f, yC - yM), 0.0f, 1.0f))) + 0.006f;
    };
    Vec3 wR[kSideWinSt][2]{}, wL[kSideWinSt][2]{};
    for (int i = 0; i < kSideWinSt; ++i) {
        const float x0 = evalUpperX(wz[i], wyL[i]), x1 = evalUpperX(wz[i], wyH[i]);
        wR[i][0] = { x0, wyL[i], wz[i]}; wR[i][1] = { x1, wyH[i], wz[i]};
        wL[i][0] = {-x1, wyH[i], wz[i]}; wL[i][1] = {-x0, wyL[i], wz[i]};
    }
    emitGrid<kSideWinSt, 2>(verts, idx, xf, wR, kSideWinSt, 2, kCanopyGlass, 1.0f, 0.0f, false);
    emitGrid<kSideWinSt, 2>(verts, idx, xf, wL, kSideWinSt, 2, kCanopyGlass, 1.0f, 0.0f, false);
}
