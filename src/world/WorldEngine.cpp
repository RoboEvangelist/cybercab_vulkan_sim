#include "world/WorldEngine.hpp"

#include <cmath>
#include <limits>

namespace cybercab {

namespace {

void appendBox(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
               const Mat4& xform, Vec3 size, Vec3 color, float alpha, float emissive, SemanticClass sem) {
    const float hx = size.x * 0.5f;
    const float hy = size.y * 0.5f;
    const float hz = size.z * 0.5f;
    const float semF = static_cast<float>(static_cast<int>(sem));

    struct FaceDef {
        Vec3 n;
        Vec3 p[4];
    };
    const FaceDef faces[6] = {
        {{ 0,  0,  1}, {{-hx, -hy,  hz}, { hx, -hy,  hz}, { hx,  hy,  hz}, {-hx,  hy,  hz}}}, // +Z
        {{ 0,  0, -1}, {{ hx, -hy, -hz}, {-hx, -hy, -hz}, {-hx,  hy, -hz}, { hx,  hy, -hz}}}, // -Z
        {{ 1,  0,  0}, {{ hx, -hy,  hz}, { hx, -hy, -hz}, { hx,  hy, -hz}, { hx,  hy,  hz}}}, // +X
        {{-1,  0,  0}, {{-hx, -hy, -hz}, {-hx, -hy,  hz}, {-hx,  hy,  hz}, {-hx,  hy, -hz}}}, // -X
        {{ 0,  1,  0}, {{-hx,  hy,  hz}, { hx,  hy,  hz}, { hx,  hy, -hz}, {-hx,  hy, -hz}}}, // +Y
        {{ 0, -1,  0}, {{-hx, -hy, -hz}, { hx, -hy, -hz}, { hx, -hy,  hz}, {-hx, -hy,  hz}}}, // -Y
    };

    for (const auto& f : faces) {
        const uint32_t base = static_cast<uint32_t>(verts.size());
        const Vec3 wn = xform.transformVector(f.n).normalized();
        for (int i = 0; i < 4; ++i) {
            const Vec3 wp = xform.transformPoint(f.p[i]);
            verts.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, color.x, color.y, color.z, alpha, emissive, semF, 0.0f, 0.0f});
        }
        idx.push_back(base + 0); idx.push_back(base + 1); idx.push_back(base + 2);
        idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 3);
    }
}

// Appends a tapered frustum prism (botSize -> topSize, shifted by topZShift in Z) with exact face normals.
void appendFrustumBox(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                      const Mat4& xform, Vec2 botXZ, Vec2 topXZ, float height, float topZShift,
                      Vec3 color, float alpha, float emissive, SemanticClass sem) {
    const float hbx = botXZ.x * 0.5f;
    const float hbz = botXZ.y * 0.5f;
    const float htx = topXZ.x * 0.5f;
    const float htz = topXZ.y * 0.5f;
    const float hy = height * 0.5f;
    const float semF = static_cast<float>(static_cast<int>(sem));

    const Vec3 b[4] = {
        {-hbx, -hy,  hbz},
        { hbx, -hy,  hbz},
        { hbx, -hy, -hbz},
        {-hbx, -hy, -hbz}
    };
    const Vec3 t[4] = {
        {-htx,  hy,  htz + topZShift},
        { htx,  hy,  htz + topZShift},
        { htx,  hy, -htz + topZShift},
        {-htx,  hy, -htz + topZShift}
    };

    const Vec3 quads[6][4] = {
        {b[0], b[1], t[1], t[0]}, // +Z front
        {b[2], b[3], t[3], t[2]}, // -Z rear
        {b[1], b[2], t[2], t[1]}, // +X right
        {b[3], b[0], t[0], t[3]}, // -X left
        {t[0], t[1], t[2], t[3]}, // +Y top
        {b[3], b[2], b[1], b[0]}  // -Y bottom
    };

    for (const auto& q : quads) {
        const Vec3 wp0 = xform.transformPoint(q[0]);
        const Vec3 wp1 = xform.transformPoint(q[1]);
        const Vec3 wp2 = xform.transformPoint(q[2]);
        const Vec3 wp3 = xform.transformPoint(q[3]);
        const Vec3 wn = (wp1 - wp0).cross(wp2 - wp0).normalized();
        const uint32_t base = static_cast<uint32_t>(verts.size());
        verts.push_back({wp0.x, wp0.y, wp0.z, wn.x, wn.y, wn.z, color.x, color.y, color.z, alpha, emissive, semF, 0.0f, 0.0f});
        verts.push_back({wp1.x, wp1.y, wp1.z, wn.x, wn.y, wn.z, color.x, color.y, color.z, alpha, emissive, semF, 0.0f, 0.0f});
        verts.push_back({wp2.x, wp2.y, wp2.z, wn.x, wn.y, wn.z, color.x, color.y, color.z, alpha, emissive, semF, 0.0f, 0.0f});
        verts.push_back({wp3.x, wp3.y, wp3.z, wn.x, wn.y, wn.z, color.x, color.y, color.z, alpha, emissive, semF, 0.0f, 0.0f});
        idx.push_back(base + 0); idx.push_back(base + 1); idx.push_back(base + 2);
        idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 3);
    }
}

// Appends a Y-axis cylinder (rotate by rotZ(kPi*0.5) for wheel axles along X).
void appendCylinderY(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                     const Mat4& xform, float radiusTop, float radiusBot, float height, int segments,
                     Vec3 color, float alpha, float emissive, SemanticClass sem) {
    const float hy = height * 0.5f;
    const float semF = static_cast<float>(static_cast<int>(sem));
    const float dTheta = (2.0f * kPi) / static_cast<float>(segments);

    for (int i = 0; i < segments; ++i) {
        const float a0 = static_cast<float>(i) * dTheta;
        const float a1 = static_cast<float>(i + 1) * dTheta;
        const float c0 = std::cos(a0), s0 = std::sin(a0);
        const float c1 = std::cos(a1), s1 = std::sin(a1);

        const Vec3 lb0{c0 * radiusBot, -hy, s0 * radiusBot};
        const Vec3 lb1{c1 * radiusBot, -hy, s1 * radiusBot};
        const Vec3 lt0{c0 * radiusTop,  hy, s0 * radiusTop};
        const Vec3 lt1{c1 * radiusTop,  hy, s1 * radiusTop};

        // Side quad (CCW seen from outside: b1 -> b0 -> t0 -> t1)
        const Vec3 n0 = xform.transformVector({c0, 0.0f, s0}).normalized();
        const Vec3 n1 = xform.transformVector({c1, 0.0f, s1}).normalized();
        const Vec3 wb0 = xform.transformPoint(lb0);
        const Vec3 wb1 = xform.transformPoint(lb1);
        const Vec3 wt0 = xform.transformPoint(lt0);
        const Vec3 wt1 = xform.transformPoint(lt1);
        const uint32_t sb = static_cast<uint32_t>(verts.size());
        verts.push_back({wb1.x, wb1.y, wb1.z, n1.x, n1.y, n1.z, color.x, color.y, color.z, alpha, emissive, semF, 0, 0});
        verts.push_back({wb0.x, wb0.y, wb0.z, n0.x, n0.y, n0.z, color.x, color.y, color.z, alpha, emissive, semF, 0, 0});
        verts.push_back({wt0.x, wt0.y, wt0.z, n0.x, n0.y, n0.z, color.x, color.y, color.z, alpha, emissive, semF, 0, 0});
        verts.push_back({wt1.x, wt1.y, wt1.z, n1.x, n1.y, n1.z, color.x, color.y, color.z, alpha, emissive, semF, 0, 0});
        idx.push_back(sb + 0); idx.push_back(sb + 1); idx.push_back(sb + 2);
        idx.push_back(sb + 0); idx.push_back(sb + 2); idx.push_back(sb + 3);
    }

    // Top (+Y) and Bottom (-Y) caps
    const Vec3 wtCenter = xform.transformPoint({0.0f, hy, 0.0f});
    const Vec3 wtNorm = xform.transformVector({0.0f, 1.0f, 0.0f}).normalized();
    const uint32_t tc = static_cast<uint32_t>(verts.size());
    verts.push_back({wtCenter.x, wtCenter.y, wtCenter.z, wtNorm.x, wtNorm.y, wtNorm.z, color.x, color.y, color.z, alpha, emissive, semF, 0, 0});
    for (int i = 0; i <= segments; ++i) {
        const float a = static_cast<float>(i) * dTheta;
        const Vec3 p = xform.transformPoint({std::cos(a) * radiusTop, hy, std::sin(a) * radiusTop});
        verts.push_back({p.x, p.y, p.z, wtNorm.x, wtNorm.y, wtNorm.z, color.x, color.y, color.z, alpha, emissive, semF, 0, 0});
    }
    for (int i = 0; i < segments; ++i) {
        idx.push_back(tc); idx.push_back(tc + 1 + static_cast<uint32_t>(i + 1)); idx.push_back(tc + 1 + static_cast<uint32_t>(i));
    }

    const Vec3 wbCenter = xform.transformPoint({0.0f, -hy, 0.0f});
    const Vec3 wbNorm = xform.transformVector({0.0f, -1.0f, 0.0f}).normalized();
    const uint32_t bc = static_cast<uint32_t>(verts.size());
    verts.push_back({wbCenter.x, wbCenter.y, wbCenter.z, wbNorm.x, wbNorm.y, wbNorm.z, color.x, color.y, color.z, alpha, emissive, semF, 0, 0});
    for (int i = 0; i <= segments; ++i) {
        const float a = static_cast<float>(i) * dTheta;
        const Vec3 p = xform.transformPoint({std::cos(a) * radiusBot, -hy, std::sin(a) * radiusBot});
        verts.push_back({p.x, p.y, p.z, wbNorm.x, wbNorm.y, wbNorm.z, color.x, color.y, color.z, alpha, emissive, semF, 0, 0});
    }
    for (int i = 0; i < segments; ++i) {
        idx.push_back(bc); idx.push_back(bc + 1 + static_cast<uint32_t>(i)); idx.push_back(bc + 1 + static_cast<uint32_t>(i + 1));
    }
}

void appendSphere(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                  const Mat4& xform, float radius, int stacks, int slices,
                  Vec3 color, float alpha, float emissive, SemanticClass sem) {
    const float semF = static_cast<float>(static_cast<int>(sem));
    const uint32_t base = static_cast<uint32_t>(verts.size());
    for (int i = 0; i <= stacks; ++i) {
        const float phi = kPi * static_cast<float>(i) / static_cast<float>(stacks);
        const float y = std::cos(phi);
        const float rRing = std::sin(phi);
        for (int j = 0; j <= slices; ++j) {
            const float theta = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(slices);
            const Vec3 nLocal{rRing * std::cos(theta), y, rRing * std::sin(theta)};
            const Vec3 pLocal = nLocal * radius;
            const Vec3 wp = xform.transformPoint(pLocal);
            const Vec3 wn = xform.transformVector(nLocal).normalized();
            verts.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, color.x, color.y, color.z, alpha, emissive, semF, 0, 0});
        }
    }
    const int rowStride = slices + 1;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            const uint32_t a = base + static_cast<uint32_t>(i * rowStride + j);
            const uint32_t b = a + static_cast<uint32_t>(rowStride);
            idx.push_back(a); idx.push_back(a + 1); idx.push_back(b + 1);
            idx.push_back(a); idx.push_back(b + 1); idx.push_back(b);
        }
    }
}

void appendWheelAssembly(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                         const Mat4& wheelMat, float tireRadius, float tireWidth,
                         Vec3 tireCol, Vec3 rimCol, SemanticClass sem) {
    const Mat4 axMat = wheelMat * Mat4::rotZ(kPi * 0.5f);
    appendCylinderY(verts, idx, axMat, tireRadius, tireRadius, tireWidth, 24, tireCol, 1.0f, 0.0f, sem);
    appendCylinderY(verts, idx, axMat, tireRadius * 0.78f, tireRadius * 0.78f, tireWidth + 0.016f, 24, rimCol, 1.0f, 0.0f, sem);
    appendCylinderY(verts, idx, axMat, tireRadius * 0.28f, tireRadius * 0.28f, tireWidth + 0.028f, 16, {0.10f, 0.12f, 0.15f}, 1.0f, 0.0f, sem);
}

// Solid champagne-gold shallow-dished Cybercab aero wheel cover (32-segment circular profile)
void appendCybercabAeroWheel(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                             const Mat4& wheelMat, float sideSign,
                             Vec3 tireCol, Vec3 rimGold, Vec3 centerDimpleCol, SemanticClass sem) {
    const float tireR = 0.365f;
    const float tireW = 0.245f;
    const Mat4 axMat = wheelMat * Mat4::rotZ(kPi * 0.5f);
    // Main round tire + black inner bezel ring
    appendCylinderY(verts, idx, axMat, tireR, tireR, tireW, 32, tireCol, 1.0f, 0.0f, sem);
    appendCylinderY(verts, idx, axMat, 0.334f, 0.334f, tireW + 0.008f, 32, {0.035f, 0.040f, 0.055f}, 1.0f, 0.0f, sem);
    // Giant solid champagne-gold outer collar covering 90% of wheel diameter
    appendCylinderY(verts, idx, axMat, 0.324f, 0.324f, tireW + 0.016f, 32, rimGold, 1.0f, 0.0f, sem);
    // Shallow aerodynamic dished cone on outboard wheel face + tiny recessed dimple pin
    const Mat4 dishRot = (sideSign > 0.0f) ? Mat4::rotZ(-kPi * 0.5f) : Mat4::rotZ(kPi * 0.5f);
    const Mat4 outboardMat = wheelMat * Mat4::translation({sideSign * (tireW * 0.5f + 0.014f), 0.0f, 0.0f}) * dishRot;
    appendCylinderY(verts, idx, outboardMat, 0.038f, 0.318f, 0.018f, 32, rimGold, 1.0f, 0.0f, sem);
    appendCylinderY(verts, idx, wheelMat * Mat4::translation({sideSign * (tireW * 0.5f + 0.022f), 0.0f, 0.0f}) * dishRot,
                    0.030f, 0.038f, 0.006f, 16, centerDimpleCol, 1.0f, 0.0f, sem);
}

// Emits a smooth longitudinally-lofted strip grid (`stations x cols`) without heap allocations.
template <int MaxS, int MaxC>
void emitSmoothStripGrid(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                         const Mat4& xform,
                         const Vec3 (&pos)[MaxS][MaxC], int nStations, int nCols,
                         Vec3 color, float alpha, float emissive, SemanticClass sem, bool flipWinding = false) {
    Vec3 nrm[MaxS][MaxC]{};
    for (int s = 0; s < nStations - 1; ++s) {
        for (int c = 0; c < nCols - 1; ++c) {
            const Vec3 p00 = pos[s][c];
            const Vec3 p10 = pos[s][c + 1];
            const Vec3 p11 = pos[s + 1][c + 1];
            const Vec3 p01 = pos[s + 1][c];
            Vec3 fn = flipWinding ? (p01 - p00).cross(p11 - p00) : (p10 - p00).cross(p11 - p00);
            const float l2 = fn.normSq();
            if (l2 > 1e-10f) {
                fn = fn * (1.0f / std::sqrt(l2));
                nrm[s][c] += fn;
                nrm[s][c + 1] += fn;
                nrm[s + 1][c + 1] += fn;
                nrm[s + 1][c] += fn;
            }
        }
    }

    const float semF = static_cast<float>(static_cast<int>(sem));
    const uint32_t base = static_cast<uint32_t>(verts.size());
    for (int s = 0; s < nStations; ++s) {
        for (int c = 0; c < nCols; ++c) {
            const Vec3 wp = xform.transformPoint(pos[s][c]);
            const Vec3 wn = xform.transformVector(nrm[s][c].normalized()).normalized();
            verts.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, color.x, color.y, color.z, alpha, emissive, semF, 0.0f, 0.0f});
        }
    }
    for (int s = 0; s < nStations - 1; ++s) {
        for (int c = 0; c < nCols - 1; ++c) {
            const uint32_t i00 = base + static_cast<uint32_t>(s * nCols + c);
            const uint32_t i10 = base + static_cast<uint32_t>(s * nCols + c + 1);
            const uint32_t i01 = base + static_cast<uint32_t>((s + 1) * nCols + c);
            const uint32_t i11 = base + static_cast<uint32_t>((s + 1) * nCols + c + 1);
            if (!flipWinding) {
                idx.push_back(i00); idx.push_back(i10); idx.push_back(i11);
                idx.push_back(i00); idx.push_back(i11); idx.push_back(i01);
            } else {
                idx.push_back(i00); idx.push_back(i11); idx.push_back(i10);
                idx.push_back(i00); idx.push_back(i01); idx.push_back(i11);
            }
        }
    }
}

void appendQuadXZ(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                  Vec3 center, float w, float len, float yaw, Vec3 color, float emissive, SemanticClass sem) {
    float c = std::cos(yaw), s = std::sin(yaw);
    float hw = w * 0.5f, hl = len * 0.5f;
    auto xf = [&](float lx, float lz) -> Vec3 {
        return {center.x + lx * c + lz * s, center.y, center.z - lx * s + lz * c};
    };
    Vec3 p0 = xf(-hw, -hl);
    Vec3 p1 = xf( hw, -hl);
    Vec3 p2 = xf( hw,  hl);
    Vec3 p3 = xf(-hw,  hl);
    float semF = static_cast<float>(static_cast<int>(sem));
    uint32_t base = static_cast<uint32_t>(verts.size());
    verts.push_back({p0.x, p0.y, p0.z, 0, 1, 0, color.x, color.y, color.z, 1.0f, emissive, semF, 0, 0});
    verts.push_back({p1.x, p1.y, p1.z, 0, 1, 0, color.x, color.y, color.z, 1.0f, emissive, semF, 0, 0});
    verts.push_back({p2.x, p2.y, p2.z, 0, 1, 0, color.x, color.y, color.z, 1.0f, emissive, semF, 0, 0});
    verts.push_back({p3.x, p3.y, p3.z, 0, 1, 0, color.x, color.y, color.z, 1.0f, emissive, semF, 0, 0});
    idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 1);
    idx.push_back(base + 0); idx.push_back(base + 3); idx.push_back(base + 2);
}

struct BuildingDef {
    float x, z, w, h, d;
    Vec3 baseCol, winCol;
};

const std::vector<BuildingDef>& buildingDefs() {
    static const std::vector<BuildingDef> defs = {
        {-24, -16, 24, 18, 22, {0.20f, 0.25f, 0.33f}, {0.22f, 0.74f, 0.97f}},
        { 22, -15, 26, 24, 22, {0.12f, 0.16f, 0.23f}, {0.99f, 0.90f, 0.54f}},
        {-25,  18, 22, 16, 20, {0.25f, 0.25f, 0.27f}, {0.58f, 0.77f, 0.99f}},
        { 24,  18, 24, 20, 20, {0.20f, 0.25f, 0.33f}, {0.22f, 0.74f, 0.97f}},
        {-16,  62, 18, 15, 18, {0.12f, 0.16f, 0.23f}, {0.99f, 0.90f, 0.54f}},
        { 18,  62, 20, 19, 18, {0.25f, 0.25f, 0.27f}, {0.58f, 0.77f, 0.99f}},
        {-76,  22, 18, 14, 26, {0.20f, 0.25f, 0.33f}, {0.22f, 0.74f, 0.97f}},
        {-76, -22, 18, 18, 26, {0.12f, 0.16f, 0.23f}, {0.99f, 0.90f, 0.54f}},
        { 76,  22, 18, 16, 26, {0.25f, 0.25f, 0.27f}, {0.58f, 0.77f, 0.99f}},
        { 76, -22, 18, 21, 26, {0.20f, 0.25f, 0.33f}, {0.22f, 0.74f, 0.97f}},
        {  0, -64, 46, 16, 16, {0.12f, 0.16f, 0.23f}, {0.99f, 0.90f, 0.54f}},
    };
    return defs;
}

} // namespace

WorldEngine::WorldEngine(const SimConfig& cfg)
    : cfg_(cfg),
      centerline_(buildCybercabLoopPath(0.0f)),
      laneCurves_{buildCybercabLoopPath(-1.9f), buildCybercabLoopPath(1.9f)} {
    initStaticOBBs();
    reset();
}

void WorldEngine::initStaticOBBs() {
    staticOBBs_.clear();
    staticOBBs_.push_back({"SIGNAL_MAST", 6.5f, 3.1f, kLoopHalfZ - 6.2f, 0.4f, 3.1f, 0.4f, 0.0f, 0, 0, SemanticClass::SignalSign, "SIGNAL"});
    staticOBBs_.push_back({"GANTRY_LEG_L", kLoopHalfX - 6.0f, 2.5f, 0.0f, 0.55f, 2.5f, 0.7f, 0.0f, 0, 0, SemanticClass::Static, ""});
    staticOBBs_.push_back({"GANTRY_LEG_R", kLoopHalfX + 6.0f, 2.5f, 0.0f, 0.55f, 2.5f, 0.7f, 0.0f, 0, 0, SemanticClass::Static, ""});
    staticOBBs_.push_back({"GANTRY_OVERHANG", kLoopHalfX, 4.1f, 0.0f, 6.5f, 0.45f, 0.85f, 0.0f, 0, 0, SemanticClass::Overhang, ""});
    staticOBBs_.push_back({"SIGN_STOP", 12.5f, 1.5f, -kLoopHalfZ - 5.6f, 0.45f, 1.5f, 0.45f, kPi * 0.5f, 0, 0, SemanticClass::SignalSign, "STOP"});
    staticOBBs_.push_back({"SIGN_SPEED35", -34.0f, 1.5f, kLoopHalfZ + 5.6f, 0.45f, 1.5f, 0.45f, -kPi * 0.5f, 0, 0, SemanticClass::SignalSign, "LIMIT 35"});

    for (size_t i = 0; i < buildingDefs().size(); ++i) {
        const auto& b = buildingDefs()[i];
        staticOBBs_.push_back({
            "BLDG_" + std::to_string(i),
            b.x, std::min(b.h * 0.5f, 2.2f), b.z,
            b.w * 0.5f, std::min(b.h * 0.5f, 2.2f), b.d * 0.5f,
            0.0f, 0, 0, SemanticClass::Static, ""
        });
    }

    for (float z = -22.0f; z <= 22.0f; z += 4.5f) {
        const float bx = -kLoopHalfX - 5.4f;
        staticOBBs_.push_back({"BARRIER", bx, 0.45f, z, 0.35f, 0.45f, 1.9f, 0.0f, 0, 0, SemanticClass::Curb, ""});
    }
}

void WorldEngine::resetTrafficAndPedestrians() {
    vehicles_.clear();
    pedestrians_.clear();

    struct VehInit { const char* id; const char* type; int lane; float u; float spd; Vec3 col; };
    const VehInit vSpecs[] = {
        {"TRUCK_01", "truck", 0, 0.115f, 6.2f,  {0.231f, 0.510f, 0.965f}},
        {"SEDAN_01", "sedan", 0, 0.270f, 9.2f,  {0.055f, 0.647f, 0.914f}},
        {"SUV_01",   "suv",   1, 0.440f, 10.2f, {0.580f, 0.639f, 0.722f}},
        {"SEDAN_02", "sedan", 0, 0.610f, 8.8f,  {0.882f, 0.114f, 0.282f}},
        {"SUV_02",   "suv",   1, 0.760f, 9.8f,  {0.063f, 0.725f, 0.506f}},
        {"SEDAN_03", "sedan", 0, 0.890f, 8.6f,  {0.659f, 0.333f, 0.969f}},
    };

    for (const auto& vs : vSpecs) {
        AIVehicle v;
        v.id = vs.id;
        v.type = vs.type;
        v.lane = vs.lane;
        v.u = vs.u;
        v.baseSpeed = vs.spd;
        v.speed = vs.spd;
        v.color = vs.col;
        if (v.type == "truck") {
            v.halfExtents = {1.22f, 1.55f, 2.90f};
        } else if (v.type == "suv") {
            v.halfExtents = {0.96f, 0.79f, 2.275f};
        } else {
            v.halfExtents = {0.96f, 0.69f, 2.175f};
        }
        v.pos = laneCurves_[v.lane].getPointAt(v.u);
        Vec3 tan = laneCurves_[v.lane].getTangentAt(v.u);
        v.yaw = std::atan2(tan.x, tan.z);
        v.vx = tan.x * v.speed;
        v.vz = tan.z * v.speed;
        vehicles_.push_back(v);
    }

    struct PedInit { const char* id; float x, z, tx, tz, spd; Vec3 col; bool obey; };
    const PedInit pSpecs[] = {
        {"PED_01", -7.6f, kLoopHalfZ + 5.6f, -7.6f, kLoopHalfZ - 5.6f, 1.35f, {0.976f, 0.451f, 0.086f}, true},
        {"PED_02", -7.0f, kLoopHalfZ - 5.8f, -7.0f, kLoopHalfZ + 5.8f, 1.28f, {0.220f, 0.741f, 0.973f}, true},
        {"PED_03", 17.5f, -kLoopHalfZ + 5.6f, 17.5f, -kLoopHalfZ - 5.6f, 1.32f, {0.925f, 0.282f, 0.600f}, false},
        {"PED_04", 18.2f, -kLoopHalfZ - 5.6f, 18.2f, -kLoopHalfZ + 5.6f, 1.25f, {0.659f, 0.333f, 0.969f}, false},
        {"PED_05", -32.0f, kLoopHalfZ + 6.0f, -12.0f, kLoopHalfZ + 6.0f, 1.38f, {0.063f, 0.725f, 0.506f}, false},
        {"PED_06", -15.0f, kLoopHalfZ - 6.0f, -38.0f, kLoopHalfZ - 6.0f, 1.30f, {0.918f, 0.702f, 0.031f}, false},
        {"PED_07", kLoopHalfX + 6.0f, 22.0f, kLoopHalfX + 6.0f, -22.0f, 1.40f, {0.376f, 0.647f, 0.980f}, false},
        {"PED_08", -22.0f, kLoopHalfZ + 5.9f, -22.0f, kLoopHalfZ - 5.9f, 1.45f, {0.957f, 0.247f, 0.369f}, true},
    };

    for (int i = 0; i < 8; ++i) {
        const auto& pc = pSpecs[i];
        Pedestrian p;
        p.id = pc.id;
        p.x = pc.x;
        p.z = pc.z;
        p.ax = pc.x;
        p.az = pc.z;
        p.bx = pc.tx;
        p.bz = pc.tz;
        p.targetX = pc.tx;
        p.targetZ = pc.tz;
        p.baseSpeed = pc.spd;
        p.radius = 0.42f;
        p.phase = static_cast<float>(i) * 1.1f;
        p.shirtColor = pc.col;
        p.pantsColor = (i % 2 == 0) ? Vec3{0.118f, 0.161f, 0.231f} : Vec3{0.200f, 0.255f, 0.333f};
        p.obeySignal = pc.obey;
        pedestrians_.push_back(p);
    }
}

void WorldEngine::resetEgo() {
    ego_ = EgoState{};
    ego_.pos = {-38.0f, 0.0f, kLoopHalfZ + 1.9f};
    ego_.yaw = kPi * 0.5f;
    ego_.speed = 8.5f;
    ego_.hx = cfg_.egoHx;
    ego_.hy = cfg_.egoHy;
    ego_.hz = cfg_.egoHz;
    ego_.centerOffsetForward = cfg_.egoCenterOffsetFwd;
    ego_.currentLateralOffset = -1.9f;
    ego_.loopProgressU = 0.03f;
}

void WorldEngine::reset() {
    signalPhase_ = SignalPhase::Green;
    signalTimer_ = 0.0f;
    forcedRed_ = false;
    lastContact_ = {};
    contactFlashAlpha_ = 0.0f;
    resetTrafficAndPedestrians();
    resetEgo();
}

OBB WorldEngine::getEgoWorldOBB() const {
    float sinY = std::sin(ego_.yaw);
    float cosY = std::cos(ego_.yaw);
    return {
        "EGO_CYBERCAB",
        ego_.pos.x + sinY * ego_.centerOffsetForward,
        ego_.hy,
        ego_.pos.z + cosY * ego_.centerOffsetForward,
        ego_.hx, ego_.hy, ego_.hz,
        ego_.yaw,
        sinY * ego_.speed,
        cosY * ego_.speed,
        SemanticClass::Vehicle,
        "EGO"
    };
}

Mat4 WorldEngine::getEgoWorldMatrix() const {
    return Mat4::translation(ego_.pos) * Mat4::rotY(ego_.yaw);
}

Mat4 WorldEngine::getEgoSprungWorldMatrix() const {
    // Matches HTML: yaw around world Y, then pitch around local X, roll around local Z
    return getEgoWorldMatrix() * Mat4::rotX(ego_.pitch) * Mat4::rotZ(ego_.roll);
}

void WorldEngine::collectDynamicGroundTruthOBBs(std::vector<OBB>& out) const {
    out.clear();
    for (const auto& v : vehicles_) {
        out.push_back({
            v.id, v.pos.x, v.halfExtents.y, v.pos.z,
            v.halfExtents.x, v.halfExtents.y, v.halfExtents.z,
            v.yaw, v.vx, v.vz, SemanticClass::Vehicle, v.type
        });
    }
    for (const auto& p : pedestrians_) {
        out.push_back({
            p.id, p.x, 0.9f, p.z,
            0.35f, 0.9f, 0.35f,
            p.yaw, p.vx, p.vz, SemanticClass::Pedestrian, "PED"
        });
    }
}

std::vector<OBB> WorldEngine::dynamicGroundTruthOBBs() const {
    std::vector<OBB> out;
    out.reserve(vehicles_.size() + pedestrians_.size());
    collectDynamicGroundTruthOBBs(out);
    return out;
}

std::vector<OBB> WorldEngine::allGroundTruthOBBs() const {
    std::vector<OBB> out = staticOBBs_;
    auto dyn = dynamicGroundTruthOBBs();
    out.insert(out.end(), dyn.begin(), dyn.end());
    return out;
}

float WorldEngine::findClosestLoopU(const Vec3& pos) const {
    float bestU = ego_.loopProgressU;
    float bestDistSq = std::numeric_limits<float>::infinity();
    for (int i = -25; i <= 25; ++i) {
        float u = std::fmod(ego_.loopProgressU + static_cast<float>(i) * 0.004f, 1.0f);
        if (u < 0.0f) u += 1.0f;
        Vec3 pt = centerline_.getPointAt(u);
        float dSq = (pt.x - pos.x) * (pt.x - pos.x) + (pt.z - pos.z) * (pt.z - pos.z);
        if (dSq < bestDistSq) {
            bestDistSq = dSq;
            bestU = u;
        }
    }
    if (bestDistSq > 144.0f) {
        for (int i = 0; i < 120; ++i) {
            float u = static_cast<float>(i) / 120.0f;
            Vec3 pt = centerline_.getPointAt(u);
            float dSq = (pt.x - pos.x) * (pt.x - pos.x) + (pt.z - pos.z) * (pt.z - pos.z);
            if (dSq < bestDistSq) {
                bestDistSq = dSq;
                bestU = u;
            }
        }
    }
    return bestU;
}

bool WorldEngine::resolvePedestrianVsVehicleOBB(Pedestrian& ped, const OBB& vehObb, EgoState* egoIfVehicle) {
    float dx = ped.x - vehObb.cx;
    float dz = ped.z - vehObb.cz;
    float cosY = std::cos(vehObb.yaw);
    float sinY = std::sin(vehObb.yaw);

    float localZ = dx * sinY + dz * cosY;
    float localX = dx * cosY - dz * sinY;

    float hx = vehObb.hx + 0.14f;
    float hz = vehObb.hz + 0.14f;

    float clampedX = clampf(localX, -hx, hx);
    float clampedZ = clampf(localZ, -hz, hz);

    float diffX = localX - clampedX;
    float diffZ = localZ - clampedZ;
    float distSq = diffX * diffX + diffZ * diffZ;
    float minRadius = ped.radius;

    if (distSq < minRadius * minRadius) {
        float penDepth = 0.0f;
        float normLocalX = 0.0f;
        float normLocalZ = 0.0f;

        if (distSq > 1e-6f) {
            float dist = std::sqrt(distSq);
            penDepth = minRadius - dist + 0.04f;
            normLocalX = diffX / dist;
            normLocalZ = diffZ / dist;
        } else {
            float penX = hx - std::abs(localX);
            float penZ = hz - std::abs(localZ);
            if (penX < penZ) {
                penDepth = penX + minRadius + 0.05f;
                normLocalX = (localX >= 0.0f) ? 1.0f : -1.0f;
            } else {
                penDepth = penZ + minRadius + 0.05f;
                normLocalZ = (localZ >= 0.0f) ? 1.0f : -1.0f;
            }
        }

        float worldNormX = normLocalX * cosY + normLocalZ * sinY;
        float worldNormZ = -normLocalX * sinY + normLocalZ * cosY;

        ped.x += worldNormX * penDepth;
        ped.z += worldNormZ * penDepth;

        float vDotN = ped.vx * worldNormX + ped.vz * worldNormZ;
        if (vDotN < 0.0f) {
            ped.vx -= worldNormX * vDotN;
            ped.vz -= worldNormZ * vDotN;
        }

        float tangSign = (localZ >= 0.0f) ? 1.0f : -1.0f;
        ped.vx += -worldNormZ * tangSign * 1.1f;
        ped.vz += worldNormX * tangSign * 1.1f;

        if (egoIfVehicle && egoIfVehicle->speed > 0.2f) {
            egoIfVehicle->speed = 0.0f;
            egoIfVehicle->accel = -8.5f;
            egoIfVehicle->pitchRate += 0.18f;
            egoIfVehicle->collisionHoldTimer = 1.0f;
        }
        return true;
    }
    return false;
}

SatHit WorldEngine::checkSAT_OBB(const OBB& a, const OBB& b) {
    float dx = b.cx - a.cx;
    float dz = b.cz - a.cz;
    float maxR = std::hypot(a.hx, a.hz) + std::hypot(b.hx, b.hz);
    if (dx * dx + dz * dz > maxR * maxR) return {};

    float sinA = std::sin(a.yaw), cosA = std::cos(a.yaw);
    float sinB = std::sin(b.yaw), cosB = std::cos(b.yaw);

    const Vec2 axes[4] = {
        {cosA, -sinA},
        {sinA,  cosA},
        {cosB, -sinB},
        {sinB,  cosB},
    };

    auto cornersOf = [](const OBB& obb, float s, float c) {
        return std::array<Vec2, 4>{{
            {obb.cx + c * obb.hx + s * obb.hz, obb.cz - s * obb.hx + c * obb.hz},
            {obb.cx - c * obb.hx + s * obb.hz, obb.cz + s * obb.hx + c * obb.hz},
            {obb.cx - c * obb.hx - s * obb.hz, obb.cz + s * obb.hx - c * obb.hz},
            {obb.cx + c * obb.hx - s * obb.hz, obb.cz - s * obb.hx - c * obb.hz},
        }};
    };

    auto vertsA = cornersOf(a, sinA, cosA);
    auto vertsB = cornersOf(b, sinB, cosB);

    float minOverlap = std::numeric_limits<float>::infinity();
    Vec2 mtvAxis{};

    for (int i = 0; i < 4; ++i) {
        const Vec2& ax = axes[i];
        float minA = std::numeric_limits<float>::infinity();
        float maxA = -std::numeric_limits<float>::infinity();
        float minB = std::numeric_limits<float>::infinity();
        float maxB = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < 4; ++k) {
            float pA = vertsA[k].x * ax.x + vertsA[k].y * ax.y;
            minA = std::min(minA, pA);
            maxA = std::max(maxA, pA);
            float pB = vertsB[k].x * ax.x + vertsB[k].y * ax.y;
            minB = std::min(minB, pB);
            maxB = std::max(maxB, pB);
        }
        float overlap = std::min(maxA, maxB) - std::max(minA, minB);
        if (overlap <= 0.0f) return {};
        if (overlap < minOverlap) {
            minOverlap = overlap;
            float sign = ((dx * ax.x + dz * ax.y) >= 0.0f) ? -1.0f : 1.0f;
            mtvAxis = {ax.x * sign, ax.y * sign};
        }
    }

    return {true, minOverlap, mtvAxis};
}

void WorldEngine::stepMultiBodyCollisionSolver() {
    OBB egoOBB = getEgoWorldOBB();
    lastContact_ = {};

    for (size_t i = 0; i < pedestrians_.size(); ++i) {
        Pedestrian& p = pedestrians_[i];
        if (resolvePedestrianVsVehicleOBB(p, egoOBB, &ego_)) {
            lastContact_ = {true, {p.x, 0.3f, p.z}, p.id, p.radius};
        }

        for (auto& veh : vehicles_) {
            const OBB vOBB{veh.id, veh.pos.x, veh.halfExtents.y, veh.pos.z,
                           veh.halfExtents.x, veh.halfExtents.y, veh.halfExtents.z, veh.yaw,
                           veh.vx, veh.vz, SemanticClass::Vehicle, ""};
            if (resolvePedestrianVsVehicleOBB(p, vOBB, nullptr)) {
                veh.speed = 0.0f;
            }
        }

        for (size_t j = i + 1; j < pedestrians_.size(); ++j) {
            Pedestrian& p2 = pedestrians_[j];
            float dx = p2.x - p.x;
            float dz = p2.z - p.z;
            float dSq = dx * dx + dz * dz;
            float minD = p.radius + p2.radius + 0.08f;
            if (dSq < minD * minD && dSq > 1e-6f) {
                float d = std::sqrt(dSq);
                float push = (minD - d) * 0.5f;
                float nx = dx / d, nz = dz / d;
                p.x -= nx * push; p.z -= nz * push;
                p2.x += nx * push; p2.z += nz * push;
            }
        }
    }

    for (auto& veh : vehicles_) {
        const OBB vOBB{veh.id, veh.pos.x, veh.halfExtents.y, veh.pos.z,
                       veh.halfExtents.x, veh.halfExtents.y, veh.halfExtents.z, veh.yaw,
                       veh.vx, veh.vz, SemanticClass::Vehicle, ""};
        SatHit hit = checkSAT_OBB(egoOBB, vOBB);
        if (hit.hit) {
            ego_.pos.x += hit.normal.x * (hit.overlap * 0.7f + 0.03f);
            ego_.pos.z += hit.normal.y * (hit.overlap * 0.7f + 0.03f);
            veh.pos.x -= hit.normal.x * (hit.overlap * 0.3f);
            veh.pos.z -= hit.normal.y * (hit.overlap * 0.3f);

            ego_.speed *= 0.15f;
            veh.speed *= 0.2f;
            ego_.pitchRate += 0.25f;
            ego_.collisionHoldTimer = 1.2f;

            contactFlashPos_ = {(egoOBB.cx + vOBB.cx) * 0.5f, 0.3f, (egoOBB.cz + vOBB.cz) * 0.5f};
            contactFlashAlpha_ = 1.0f;
            lastContact_ = {true, contactFlashPos_, veh.id, hit.overlap};
            egoOBB = getEgoWorldOBB();
        }
    }

    for (const auto& sOBB : staticOBBs_) {
        if (sOBB.semantic == SemanticClass::Overhang) continue;
        SatHit hit = checkSAT_OBB(egoOBB, sOBB);
        if (hit.hit) {
            ego_.pos.x += hit.normal.x * (hit.overlap + 0.04f);
            ego_.pos.z += hit.normal.y * (hit.overlap + 0.04f);
            ego_.speed *= 0.1f;
            ego_.pitchRate += 0.22f;
            ego_.collisionHoldTimer = 1.0f;
            contactFlashPos_ = {egoOBB.cx - hit.normal.x * 1.5f, 0.3f, egoOBB.cz - hit.normal.y * 1.5f};
            contactFlashAlpha_ = 1.0f;
            lastContact_ = {true, contactFlashPos_, sOBB.id, hit.overlap};
            egoOBB = getEgoWorldOBB();
        }
    }
}

void WorldEngine::stepAgents(float dt) {
    if (!forcedRed_) {
        signalTimer_ += dt;
        if (signalPhase_ == SignalPhase::Green && signalTimer_ > 9.0f) {
            signalPhase_ = SignalPhase::Yellow;
            signalTimer_ = 0.0f;
        } else if (signalPhase_ == SignalPhase::Yellow && signalTimer_ > 2.5f) {
            signalPhase_ = SignalPhase::Red;
            signalTimer_ = 0.0f;
        } else if (signalPhase_ == SignalPhase::Red && signalTimer_ > 7.5f) {
            signalPhase_ = SignalPhase::Green;
            signalTimer_ = 0.0f;
        }
    } else {
        signalPhase_ = SignalPhase::Red;
    }

    if (contactFlashAlpha_ > 0.0f) {
        contactFlashAlpha_ = std::max(0.0f, contactFlashAlpha_ - dt * 1.8f);
    }

    const float loopLen = centerline_.getLength();
    const OBB egoOBB = getEgoWorldOBB();

    // Update IDM vehicles
    for (size_t i = 0; i < vehicles_.size(); ++i) {
        AIVehicle& v = vehicles_[i];
        const auto& curve = laneCurves_[v.lane & 1];
        Vec3 pt = curve.getPointAt(v.u);
        Vec3 tan = curve.getTangentAt(v.u);

        float minGap = 60.0f;
        float leadSpeed = v.baseSpeed;

        float toEgoX = egoOBB.cx - pt.x;
        float toEgoZ = egoOBB.cz - pt.z;
        float egoFwd = toEgoX * tan.x + toEgoZ * tan.z - (v.halfExtents.z + egoOBB.hz);
        float egoLat = std::abs(toEgoX * (-tan.z) + toEgoZ * tan.x);
        if (egoFwd > -1.0f && egoFwd < minGap && egoLat < 2.2f) {
            minGap = std::max(0.1f, egoFwd);
            leadSpeed = ego_.speed;
        }

        for (size_t j = 0; j < vehicles_.size(); ++j) {
            if (i == j) continue;
            const auto& other = vehicles_[j];
            float dx = other.pos.x - pt.x;
            float dz = other.pos.z - pt.z;
            float fwd = dx * tan.x + dz * tan.z - (v.halfExtents.z + other.halfExtents.z);
            float lat = std::abs(dx * (-tan.z) + dz * tan.x);
            if (fwd > -0.5f && fwd < minGap && lat < 2.1f) {
                minGap = std::max(0.1f, fwd);
                leadSpeed = other.speed;
            }
        }

        for (const auto& ped : pedestrians_) {
            float dx = ped.x - pt.x;
            float dz = ped.z - pt.z;
            float fwd = dx * tan.x + dz * tan.z - v.halfExtents.z;
            float lat = std::abs(dx * (-tan.z) + dz * tan.x);
            if (fwd > -0.5f && fwd < 16.0f && lat < 2.8f && fwd < minGap) {
                minGap = std::max(0.1f, fwd);
                leadSpeed = 0.0f;
            }
        }

        if (pt.z > 36.0f && pt.x < kStopLineX + 0.5f && pt.x > kStopLineX - 20.0f && signalPhase_ != SignalPhase::Green) {
            float dLight = kStopLineX - pt.x - v.halfExtents.z;
            if (dLight < minGap) {
                minGap = std::max(0.1f, dLight);
                leadSpeed = 0.0f;
            }
        }

        if (v.hardBrakeTimer > 0.0f) {
            v.hardBrakeTimer -= dt;
            v.accel = -7.5f;
        } else {
            const float s0 = 2.8f, T = 1.25f, aMax = 2.4f, bComf = 3.8f;
            float dv = v.speed - leadSpeed;
            float sStar = s0 + std::max(0.0f, v.speed * T + (v.speed * dv) / (2.0f * std::sqrt(aMax * bComf)));
            v.accel = aMax * (1.0f - std::pow(v.speed / v.baseSpeed, 4.0f) - std::pow(sStar / std::max(0.4f, minGap), 2.0f));
            v.accel = clampf(v.accel, -7.0f, 2.6f);
        }

        v.speed = std::max(0.0f, v.speed + v.accel * dt);
        v.u = std::fmod(v.u + (v.speed * dt) / loopLen, 1.0f);
        if (v.u < 0.0f) v.u += 1.0f;

        Vec3 newPt = curve.getPointAt(v.u);
        Vec3 newTan = curve.getTangentAt(v.u);
        v.pos = newPt;
        v.yaw = std::atan2(newTan.x, newTan.z);
        v.vx = newTan.x * v.speed;
        v.vz = newTan.z * v.speed;
    }

    // Update Helbing pedestrians (reuse pre-allocated scratch buffer to avoid hot-path allocations)
    allCarsScratch_.clear();
    allCarsScratch_.push_back(egoOBB);
    for (const auto& v : vehicles_) {
        allCarsScratch_.push_back({v.id, v.pos.x, v.halfExtents.y, v.pos.z,
                                   v.halfExtents.x, v.halfExtents.y, v.halfExtents.z,
                                   v.yaw, v.vx, v.vz, SemanticClass::Vehicle, ""});
    }

    for (auto& p : pedestrians_) {
        float distToGoal = std::hypot(p.targetX - p.x, p.targetZ - p.z);
        if (distToGoal < 0.45f) {
            if (std::hypot(p.targetX - p.bx, p.targetZ - p.bz) < 0.5f) {
                p.targetX = p.ax; p.targetZ = p.az;
            } else {
                p.targetX = p.bx; p.targetZ = p.bz;
            }
            p.jaywalking = false;
            distToGoal = std::hypot(p.targetX - p.x, p.targetZ - p.z);
        }

        float distFromSouthRoad = std::abs(p.z - kLoopHalfZ);
        bool waitAtCurb = false;
        if (p.obeySignal && !p.jaywalking && signalPhase_ == SignalPhase::Green &&
            distFromSouthRoad > 4.2f && distFromSouthRoad < 6.5f) {
            if ((p.z - kLoopHalfZ) * (p.targetZ - kLoopHalfZ) < 0.0f) {
                waitAtCurb = true;
            }
        }

        float repulseX = 0.0f, repulseZ = 0.0f;
