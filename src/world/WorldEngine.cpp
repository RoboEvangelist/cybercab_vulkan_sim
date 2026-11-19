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

