#include "world/CybercabMesh.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace cybercab {
namespace {

const Vec3 kGoldBase{0.824f, 0.706f, 0.490f}, kGoldBright{0.886f, 0.780f, 0.576f}, kGoldDark{0.615f, 0.514f, 0.337f};
const Vec3 kCanopyGlass{0.024f, 0.045f, 0.095f}, kTrimDark{0.028f, 0.033f, 0.046f}, kCharcoalTrim{0.115f, 0.122f, 0.138f};
const Vec3 kTireDark{0.036f, 0.042f, 0.055f}, kPlateWhite{0.820f, 0.880f, 0.930f};
const Vec3 kHoloCyan{0.220f, 0.740f, 0.970f}, kDrlLed{0.930f, 0.980f, 1.000f};
const Vec3 kTailLed{0.960f, 0.130f, 0.130f}, kRevLed{0.980f, 0.980f, 0.960f};

constexpr float kRearAxleZ = 0.00f, kFrontAxleZ = 2.75f, kWheelAxleY = 0.365f;
constexpr float kTireOuterR = 0.365f, kTireWidthM = 0.245f, kWheelArchR = 0.402f;
constexpr float kTrackHalfR = 0.836f, kTrackHalfF = 0.818f;
constexpr float kBodyRearZ = -0.88f, kBodyCrestZ = 1.04f, kBodyCowlZ = 2.10f, kBodyNoseZ = 3.35f;
constexpr float kWindshieldRearZ = 1.32f, kWindshieldFrontZ = 2.05f;
constexpr int kNumSt = 37, kSideWinSt = 12, kWheelSegs = 36;

constexpr std::array<float, kNumSt> kZStations = {{
    -0.88f, -0.78f, -0.64f, -0.49f, -0.40f, -0.37f, -0.30f, -0.20f, -0.10f, 0.00f,
     0.10f,  0.20f,  0.30f,  0.37f,  0.40f,  0.62f,  0.86f,  1.08f,  1.32f, 1.70f,
     2.05f,  2.20f,  2.34f,  2.38f,  2.45f,  2.55f,  2.65f,  2.75f,  2.85f, 2.95f,
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
                           : lerpf(wM, wC, clampf((yq - yM) / std::max(1e-4f, yC - yM), 0.0f, 1.0f))) + 0.012f;
    };
    Vec3 wR[kSideWinSt][3]{}, wL[kSideWinSt][3]{};
    for (int i = 0; i < kSideWinSt; ++i) {
        const float yMidW = lerpf(wyL[i], wyH[i], 0.50f);
        const float x0 = evalUpperX(wz[i], wyL[i]), xm = evalUpperX(wz[i], yMidW) + 0.005f, x1 = evalUpperX(wz[i], wyH[i]);
        wR[i][0] = { x0, wyL[i], wz[i]}; wR[i][1] = { xm, yMidW, wz[i]}; wR[i][2] = { x1, wyH[i], wz[i]};
        wL[i][0] = {-x1, wyH[i], wz[i]}; wL[i][1] = {-xm, yMidW, wz[i]}; wL[i][2] = {-x0, wyL[i], wz[i]};
    }
    emitGrid<kSideWinSt, 3>(verts, idx, xf, wR, kSideWinSt, 3, kCanopyGlass, 1.0f, 0.0f, false);
    emitGrid<kSideWinSt, 3>(verts, idx, xf, wL, kSideWinSt, 3, kCanopyGlass, 1.0f, 0.0f, false);
}

void emitWindshieldAndPillars(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx, const Mat4& xf) {
    constexpr int kWsS = 6, kWsC = 5;
    Vec3 wsG[kWsS][kWsC]{}, railR[kWsS][2]{}, railL[kWsS][2]{};
    for (int i = 0; i < kWsS; ++i) {
        const float z = lerpf(kWindshieldRearZ, kWindshieldFrontZ, static_cast<float>(i) / static_cast<float>(kWsS - 1));
        const Station s = evalStation(z);
        const float wGlass = std::max(0.24f, s.wCan - 0.058f);
        for (int c = 0; c < kWsC; ++c) {
            const float u = lerpf(-1.0f, 1.0f, static_cast<float>(c) / static_cast<float>(kWsC - 1));
            wsG[i][c] = {u * wGlass, lerpf(s.yRoof + 0.008f, s.yCan + 0.005f, u * u), z};
        }
        railR[i][0] = { wGlass, s.yCan + 0.004f, z}; railR[i][1] = { s.wCan, s.yCan, z};
        railL[i][0] = {-s.wCan, s.yCan, z};          railL[i][1] = {-wGlass, s.yCan + 0.004f, z};
    }
    emitGrid<kWsS, kWsC>(verts, idx, xf, wsG, kWsS, kWsC, kCanopyGlass, 1.0f, 0.0f, true);
    emitGrid<kWsS, 2>(verts, idx, xf, railR, kWsS, 2, kGoldBright, 1.0f, 0.0f, true);
    emitGrid<kWsS, 2>(verts, idx, xf, railL, kWsS, 2, kGoldBright, 1.0f, 0.0f, true);
}

void emitAeroWheel(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx, const Mat4& wm, float sideSign) {
    struct RingPt { float r, xOff, nxSign, nRad; Vec3 col; };
    const RingPt prof[] = {
        {kTireOuterR, -0.112f, 0.0f, 1.0f, kTireDark},  {kTireOuterR, 0.112f, 0.0f, 1.0f, kTireDark},
        {0.336f,       0.123f, 0.7f, 0.7f, kTireDark},  {0.306f,      0.123f, 1.0f, 0.0f, kTireDark},
        {0.298f,       0.128f, 0.8f, 0.6f, kTrimDark},  {0.265f,      0.130f, 1.0f, 0.0f, kGoldBright},
        {0.026f,       0.131f, 1.0f, 0.0f, kGoldBright},{0.020f,      0.125f, 0.9f,-0.3f, kGoldDark},
        {0.000f,       0.125f, 1.0f, 0.0f, kGoldDark}
    };
    constexpr int kNR = static_cast<int>(sizeof(prof) / sizeof(prof[0]));
    const float semF = static_cast<float>(static_cast<int>(SemanticClass::Vehicle));
    const float dTheta = (2.0f * kPi) / static_cast<float>(kWheelSegs);
    for (int p = 0; p + 1 < kNR; ++p) {
        const auto& a = prof[p]; const auto& b = prof[p + 1];
        const uint32_t base = static_cast<uint32_t>(verts.size());
        for (int s = 0; s <= kWheelSegs; ++s) {
            const float th = static_cast<float>(s) * dTheta, cs = std::cos(th), sn = std::sin(th);
            const Vec3 p0 = wm.transformPoint({sideSign * a.xOff, cs * a.r, sn * a.r});
            const Vec3 p1 = wm.transformPoint({sideSign * b.xOff, cs * b.r, sn * b.r});
            const Vec3 n0 = wm.transformVector({sideSign * a.nxSign, cs * a.nRad, sn * a.nRad}).normalized();
            const Vec3 n1 = wm.transformVector({sideSign * b.nxSign, cs * b.nRad, sn * b.nRad}).normalized();
            verts.push_back({p0.x, p0.y, p0.z, n0.x, n0.y, n0.z, b.col.x, b.col.y, b.col.z, 1.0f, 0.0f, semF, 0, 0});
            verts.push_back({p1.x, p1.y, p1.z, n1.x, n1.y, n1.z, b.col.x, b.col.y, b.col.z, 1.0f, 0.0f, semF, 0, 0});
        }
        for (int s = 0; s < kWheelSegs; ++s) {
            const uint32_t i0 = base + static_cast<uint32_t>(s * 2);
            if (sideSign > 0.0f) { idx.push_back(i0); idx.push_back(i0 + 3); idx.push_back(i0 + 1); idx.push_back(i0); idx.push_back(i0 + 2); idx.push_back(i0 + 3); }
            else                 { idx.push_back(i0); idx.push_back(i0 + 1); idx.push_back(i0 + 3); idx.push_back(i0); idx.push_back(i0 + 3); idx.push_back(i0 + 2); }
        }
    }
}

void emitWheelWellLiner(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                        const Mat4& sprungMat, float xCenter, float zCenter) {
    constexpr int kArchSegs = 14;
    const float semF = static_cast<float>(static_cast<int>(SemanticClass::Vehicle));
    const uint32_t base = static_cast<uint32_t>(verts.size());
    for (int i = 0; i <= kArchSegs; ++i) {
        const float a = lerpf(0.0f, kPi, static_cast<float>(i) / static_cast<float>(kArchSegs));
        const float ly = kWheelAxleY + std::sin(a) * 0.395f, lz = zCenter - std::cos(a) * 0.395f;
        const Vec3 n = sprungMat.transformVector({0.0f, -std::sin(a), std::cos(a)}).normalized();
        const Vec3 pIn = sprungMat.transformPoint({xCenter - 0.16f, ly, lz});
        const Vec3 pOut = sprungMat.transformPoint({xCenter + 0.16f, ly, lz});
        verts.push_back({pIn.x, pIn.y, pIn.z, n.x, n.y, n.z, kTrimDark.x, kTrimDark.y, kTrimDark.z, 1, 0, semF, 0, 0});
        verts.push_back({pOut.x, pOut.y, pOut.z, n.x, n.y, n.z, kTrimDark.x, kTrimDark.y, kTrimDark.z, 1, 0, semF, 0, 0});
    }
    for (int i = 0; i < kArchSegs; ++i) {
        const uint32_t b = base + static_cast<uint32_t>(i * 2);
        idx.push_back(b); idx.push_back(b + 3); idx.push_back(b + 2);
        idx.push_back(b); idx.push_back(b + 1); idx.push_back(b + 3);
    }
}

} // namespace

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
    Vec3 roofGrid[19][7]{}, hoodGrid[17][7]{};
    constexpr float kRoofCols[7] = {-1.0f, -0.68f, -0.36f, 0.0f, 0.36f, 0.68f, 1.0f};
    for (int i = 0; i < kNumSt; ++i) {
        loR[i][0] = { st[i].wBot, st[i].yBot, st[i].z}; loR[i][1] = { st[i].wBelly, st[i].yBelly, st[i].z}; loR[i][2] = { st[i].wShld, st[i].yShld, st[i].z};
        loL[i][0] = {-st[i].wShld, st[i].yShld, st[i].z}; loL[i][1] = {-st[i].wBelly, st[i].yBelly, st[i].z}; loL[i][2] = {-st[i].wBot, st[i].yBot, st[i].z};
        upR[i][0] = { st[i].wShld, st[i].yShld, st[i].z}; upR[i][1] = { st[i].wUpMid, st[i].yUpMid, st[i].z}; upR[i][2] = { st[i].wCan, st[i].yCan, st[i].z};
        upL[i][0] = {-st[i].wCan, st[i].yCan, st[i].z};   upL[i][1] = {-st[i].wUpMid, st[i].yUpMid, st[i].z}; upL[i][2] = {-st[i].wShld, st[i].yShld, st[i].z};
        if (i < 19) {
            const float rFade = (st[i].z > -0.82f && st[i].z < 1.24f) ? std::sin(kPi * clampf((st[i].z + 0.82f) / 2.06f, 0.0f, 1.0f)) : 0.0f;
            for (int c = 0; c < 7; ++c) roofGrid[i][c] = {kRoofCols[c] * st[i].wCan, lerpf(st[i].yRoof, st[i].yCan, kRoofCols[c] * kRoofCols[c]) + ((c == 2 || c == 4) ? 0.012f * rFade : 0.0f), st[i].z};
        } else if (i >= 20) {
            for (int c = 0; c < 7; ++c) hoodGrid[i - 20][c] = {kRoofCols[c] * st[i].wCan, lerpf(st[i].yRoof, st[i].yCan, kRoofCols[c] * kRoofCols[c]), st[i].z};
        }
    }
    emitGrid<kNumSt, 3>(verts, indices, sprungMat, loR, kNumSt, 3, kGoldBase, 1.0f, 0.0f, false);
    emitGrid<kNumSt, 3>(verts, indices, sprungMat, loL, kNumSt, 3, kGoldBase, 1.0f, 0.0f, false);
    emitGrid<kNumSt, 3>(verts, indices, sprungMat, upR, kNumSt, 3, kGoldBright, 1.0f, 0.0f, false);
    emitGrid<kNumSt, 3>(verts, indices, sprungMat, upL, kNumSt, 3, kGoldBright, 1.0f, 0.0f, false);
    emitGrid<19, 7>(verts, indices, sprungMat, roofGrid, 19, 7, kGoldBright, 1.0f, 0.0f, true);
    emitGrid<17, 7>(verts, indices, sprungMat, hoodGrid, 17, 7, kGoldBright, 1.0f, 0.0f, true);
    emitWindshieldAndPillars(verts, indices, sprungMat); emitSculptedEndCap(verts, indices, sprungMat, st[0], false); emitSculptedEndCap(verts, indices, sprungMat, st[kNumSt - 1], true); emitSideWindows(verts, indices, sprungMat, st);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.34f, 1.24f}), {1.12f, 0.34f, 3.60f}, kTrimDark);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.285f, 1.375f}), {1.72f, 0.115f, 1.92f}, kCharcoalTrim);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.295f, 3.16f}), {1.26f, 0.06f, 0.18f}, kCharcoalTrim);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.380f, -0.825f}), {1.52f, 0.165f, 0.13f}, kCharcoalTrim);
    for (const float sx : {-1.0f, 1.0f}) {
        appendBox(verts, indices, sprungMat * Mat4::translation({sx * 0.76f, 0.380f, -0.60f}), {0.09f, 0.155f, 0.44f}, kCharcoalTrim);
        appendBox(verts, indices, sprungMat * Mat4::translation({sx * 0.805f, 1.07f, 0.66f}) * Mat4::rotZ(sx * 0.44f), {0.028f, 0.31f, 0.048f}, kTrimDark);
        appendBox(verts, indices, sprungMat * Mat4::translation({sx * 0.902f, 0.77f, 2.16f}), {0.042f, 0.046f, 0.13f}, kTrimDark);
    }
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.385f, -0.892f}), {0.28f, 0.13f, 0.012f}, kPlateWhite);
    const bool isBraking = ego.accel < -0.45f || ego.speed < 0.25f || ego.aebActive;
    for (const float rx : {-0.58f, 0.58f}) appendBox(verts, indices, sprungMat * Mat4::translation({rx, 0.468f, -0.890f}), {0.28f, 0.020f, 0.018f}, kTailLed, 1.0f, isBraking ? 2.4f : 0.9f);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.555f, 3.31f}), {1.38f, 0.055f, 0.05f}, kTrimDark);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.555f, 3.332f}), {1.32f, 0.028f, 0.04f}, kDrlLed, 1.0f, 2.9f);
    for (const float sx : {-0.72f, 0.72f}) appendBox(verts, indices, sprungMat * Mat4::translation({sx, 0.555f, 3.24f}) * Mat4::rotY(sx > 0 ? -0.46f : 0.46f), {0.04f, 0.026f, 0.15f}, kDrlLed, 1.0f, 2.6f);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.880f, -0.888f}), {1.50f, 0.032f, 0.055f}, kTrimDark);
    appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.848f, -0.884f}), {1.46f, 0.030f, 0.036f}, kTailLed, 1.0f, isBraking ? 4.4f : 1.5f);
    if (ego.gear < 0) appendBox(verts, indices, sprungMat * Mat4::translation({0.0f, 0.818f, -0.888f}), {0.48f, 0.024f, 0.03f}, kRevLed, 1.0f, 3.2f);
    const struct { float x, z; bool front; } wheels[4] = {{kTrackHalfF, kFrontAxleZ, true}, {-kTrackHalfF, kFrontAxleZ, true}, {kTrackHalfR, kRearAxleZ, false}, {-kTrackHalfR, kRearAxleZ, false}};
    for (const auto& w : wheels) {
        emitWheelWellLiner(verts, indices, sprungMat, w.x * 0.90f, w.z);
        const Mat4 wm = egoMat * Mat4::translation({w.x, kWheelAxleY, w.z}) * Mat4::rotY(w.front ? ego.steerAngle : 0.0f) * Mat4::rotX(ego.wheelSpin);
        emitAeroWheel(verts, indices, wm, (w.x >= 0.0f) ? 1.0f : -1.0f);
    }
}

} // namespace cybercab
