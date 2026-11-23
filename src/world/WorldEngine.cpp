#include "world/WorldEngine.hpp"
#include "world/CybercabMesh.hpp"

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
  splitSignalActive_ = false;
  splitSignalPhases_ = {SignalPhase::Green, SignalPhase::Green};
  unknownObstacles_.clear();
    lastContact_ = {};
    contactFlashAlpha_ = 0.0f;
    siren_ = {};
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
    float step = 0.002f;
    for (int pass = 0; pass < 3; ++pass) {
        const float centerU = bestU;
        for (int i = -4; i <= 4; ++i) {
            float u = std::fmod(centerU + static_cast<float>(i) * step, 1.0f);
            if (u < 0.0f) u += 1.0f;
            const Vec3 pt = centerline_.getPointAt(u);
            const float dSq = (pt.x - pos.x) * (pt.x - pos.x) + (pt.z - pos.z) * (pt.z - pos.z);
            if (dSq < bestDistSq) {
                bestDistSq = dSq;
                bestU = u;
            }
        }
        step *= 0.25f;
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
  constexpr float kPushPadM = 0.04f, kContactSpdScale = 0.1f;
  constexpr float kContactPitch = 0.22f, kContactHoldSec = 1.0f;
  constexpr float kFlashBackM = 1.5f, kFlashHeightM = 0.3f;
  for (const auto& uo : unknownObstacles_) {
    const OBB uOBB{uo.id, uo.cx, uo.cy, uo.cz, uo.hx, uo.hy, uo.hz, uo.yaw,
                   0.0f, 0.0f, SemanticClass::Static, ""};
    const SatHit hit = checkSAT_OBB(egoOBB, uOBB);
    if (hit.hit) {
      ego_.pos.x += hit.normal.x * (hit.overlap + kPushPadM);
      ego_.pos.z += hit.normal.y * (hit.overlap + kPushPadM);
      ego_.speed *= kContactSpdScale;
      ego_.pitchRate += kContactPitch;
      ego_.collisionHoldTimer = kContactHoldSec;
      contactFlashPos_ = {egoOBB.cx - hit.normal.x * kFlashBackM,
                          kFlashHeightM,
                          egoOBB.cz - hit.normal.y * kFlashBackM};
      contactFlashAlpha_ = 1.0f;
      lastContact_ = {true, contactFlashPos_, uo.id, hit.overlap};
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
        p.yieldingToCar = false;

        for (const auto& car : allCarsScratch_) {
            float dx = p.x - car.cx;
            float dz = p.z - car.cz;
            if (std::hypot(dx, dz) > 14.0f) continue;

            float cosY = std::cos(car.yaw), sinY = std::sin(car.yaw);
            float localZ = dx * sinY + dz * cosY;
            float localX = dx * cosY - dz * sinY;
            float carSpd = std::hypot(car.vx, car.vz);

            if (carSpd > 0.8f && localZ > car.hz - 0.5f && localZ < car.hz + 9.5f &&
                std::abs(localX) < car.hx + 2.4f) {
                if (std::abs(localX) > car.hx + 0.55f && !p.jaywalking) {
                    p.yieldingToCar = true;
                }
            }

            float safeHx = car.hx + p.radius + 0.75f;
            float safeHz = car.hz + p.radius + 0.85f;
            if (std::abs(localX) < safeHx && std::abs(localZ) < safeHz) {
                float penX = safeHx - std::abs(localX);
                float penZ = safeHz - std::abs(localZ);
                float nLocX = (localX >= 0.0f) ? 1.0f : -1.0f;
                float nLocZ = (localZ >= 0.0f) ? 1.0f : -1.0f;
                float wnx = nLocX * cosY + nLocZ * sinY;
                float wnz = -nLocX * sinY + nLocZ * cosY;
                float strength = 4.2f * std::max(penX, penZ);
                repulseX += wnx * strength;
                repulseZ += wnz * strength;

                float dirToGoalX = (p.targetX - p.x) / std::max(0.1f, distToGoal);
                float dirToGoalZ = (p.targetZ - p.z) / std::max(0.1f, distToGoal);
                float tangSign = (localZ >= 0.0f) ? 1.0f : -1.0f;
                repulseX += sinY * tangSign * 2.2f + dirToGoalX * 0.5f;
                repulseZ += cosY * tangSign * 2.2f + dirToGoalZ * 0.5f;
            }
        }

        float desVx = 0.0f, desVz = 0.0f;
        if (!waitAtCurb && !p.yieldingToCar) {
            float dirX = (p.targetX - p.x) / std::max(0.01f, distToGoal);
            float dirZ = (p.targetZ - p.z) / std::max(0.01f, distToGoal);
            float spd = p.jaywalking ? p.baseSpeed * 1.45f : p.baseSpeed;
            desVx = dirX * spd + repulseX;
            desVz = dirZ * spd + repulseZ;
        } else {
            desVx = repulseX * 0.6f;
            desVz = repulseZ * 0.6f;
        }

        float maxPedSpd = p.jaywalking ? 2.4f : 1.75f;
        float curMag = std::hypot(desVx, desVz);
        if (curMag > maxPedSpd) {
            desVx = (desVx / curMag) * maxPedSpd;
            desVz = (desVz / curMag) * maxPedSpd;
        }

        float blend = clampf(7.5f * dt, 0.0f, 1.0f);
        p.vx = lerpf(p.vx, desVx, blend);
        p.vz = lerpf(p.vz, desVz, blend);
        p.x += p.vx * dt;
        p.z += p.vz * dt;

        resolvePedestrianVsVehicleOBB(p, egoOBB, &ego_);

        float actualSpd = std::hypot(p.vx, p.vz);
        if (actualSpd > 0.12f) {
            p.yaw = std::atan2(p.vx, p.vz);
            p.phase += dt * actualSpd * 5.8f;
            p.legAngle = std::sin(p.phase) * 0.58f;
            p.armAngle = -std::sin(p.phase) * 0.45f;
        } else {
            p.legAngle = lerpf(p.legAngle, 0.0f, clampf(8.0f * dt, 0.0f, 1.0f));
            p.armAngle = lerpf(p.armAngle, 0.0f, clampf(8.0f * dt, 0.0f, 1.0f));
        }
    }

    if (siren_.active) {
        siren_.pos += siren_.vel * dt;
        siren_.phase += dt;
        if (siren_.pos.z < -65.0f || siren_.pos.z > 80.0f) {
            siren_.active = false;
        }
    }
}

void WorldEngine::stepEgoAndPhysics(float dt, const PlannerCommand& cmd, const ManualInput& manual) {
    ego_.rssSafeDist = cmd.rssSafeDist;
    ego_.minObstacleDist = cmd.minObstacleDist;
    ego_.aebActive = cmd.aebActive;
    ego_.behaviorState = cmd.behaviorState;
    if (cfg_.isV15()) ego_.gear = cmd.gear;

    if (ego_.collisionHoldTimer > 0.0f) {
        ego_.collisionHoldTimer = std::max(0.0f, ego_.collisionHoldTimer - dt);
    }

    if (ego_.aebActive) {
        ego_.accel = -cfg_.rssABrakeAebMs2;
        ego_.speed = std::max(0.0f, ego_.speed + ego_.accel * dt);
    } else if (cfg_.isV15() && cmd.gear < 0 && cfg_.driveMode != "manual") {
        ego_.steerAngle = lerpf(ego_.steerAngle, 0.0f, clampf(8.0f * dt, 0.0f, 1.0f));
        if (ego_.speed > 0.2f) {
            ego_.accel = -5.5f;
            ego_.speed = std::max(0.0f, ego_.speed + ego_.accel * dt);
        } else {
            const float revTarget = clampf(cmd.targetSpeed, -2.2f, 0.0f);
            ego_.accel = clampf((revTarget - ego_.speed) * 2.5f, -3.5f, 3.0f);
            ego_.speed = clampf(ego_.speed + ego_.accel * dt, -2.2f, 0.0f);
        }
    } else if (cfg_.driveMode == "manual") {
        float throttle = 0.0f;
        if (manual.forward) throttle += 4.5f;
        if (manual.backward) throttle -= 7.5f;
        if (!manual.forward && !manual.backward) throttle = -ego_.speed * 0.45f;
        ego_.accel = throttle;
        ego_.speed = clampf(ego_.speed + ego_.accel * dt, 0.0f, 18.0f);

        float targetSteer = 0.0f;
        if (manual.left) targetSteer += 0.40f;
        if (manual.right) targetSteer -= 0.40f;
        ego_.steerAngle = lerpf(ego_.steerAngle, targetSteer, clampf(8.5f * dt, 0.0f, 1.0f));
    } else {
        if (cmd.optimalTrajectory.size() > 4) {
            Vec3 targetPt = cmd.optimalTrajectory.back();
            if (cfg_.isV15()) {
                const float wantLd = clampf(ego_.speed * 0.42f + 2.4f, 3.2f, 5.8f);
                for (size_t i = 1; i < cmd.optimalTrajectory.size(); ++i) {
                    const Vec3& p = cmd.optimalTrajectory[i];
                    if (std::hypot(p.x - ego_.pos.x, p.z - ego_.pos.z) >= wantLd) { targetPt = p; break; }
                }
            } else {
                int lookIdx = std::min(static_cast<int>(cmd.optimalTrajectory.size()) - 1,
                                       std::max(4, static_cast<int>(std::round(ego_.speed * 0.72f + 4.0f))));
                targetPt = cmd.optimalTrajectory[lookIdx];
            }
            float dx = targetPt.x - ego_.pos.x;
            float dz = targetPt.z - ego_.pos.z;
            float targetYaw = std::atan2(dx, dz);
            float alpha = wrapAngle(targetYaw - ego_.yaw);
            float ld = std::max(cfg_.isV15() ? 2.4f : 3.4f, std::hypot(dx, dz));
            float rawSteer = std::atan2(2.0f * cfg_.wheelbaseM * std::sin(alpha), ld);
            float clampedSteer = clampf(rawSteer, -cfg_.maxSteerRad, cfg_.maxSteerRad);
            ego_.steerAngle = lerpf(ego_.steerAngle, clampedSteer, clampf((cfg_.isV15() ? 11.5f : 9.5f) * dt, 0.0f, 1.0f));
        }

        float speedErr = cmd.targetSpeed - ego_.speed;
        float maxBrake = (ego_.behaviorState == "YIELDING_PEDESTRIAN") ? -7.2f : -5.8f;
        float rawAccel = clampf(speedErr * 2.1f, maxBrake, 2.8f);
        ego_.accel = lerpf(ego_.accel, rawAccel, clampf(8.0f * dt, 0.0f, 1.0f));
        if (cfg_.isV15() && ego_.speed < -0.02f) {
            ego_.accel = 4.5f;
            ego_.speed = std::min(0.0f, ego_.speed + ego_.accel * dt);
        } else {
            ego_.speed = std::max(0.0f, ego_.speed + ego_.accel * dt);
        }
    }

    ego_.yawRate = (ego_.speed / cfg_.wheelbaseM) * std::tan(ego_.steerAngle);
    ego_.yaw = wrapAngle(ego_.yaw + ego_.yawRate * dt);
    ego_.pos.x += std::sin(ego_.yaw) * ego_.speed * dt;
    ego_.pos.z += std::cos(ego_.yaw) * ego_.speed * dt;
    ego_.wheelSpin += (ego_.speed / 0.36f) * dt;

    float latAccel = ego_.speed * ego_.yawRate;
    float targetPitch = clampf(-ego_.accel * 0.0085f, -0.065f, 0.065f);
    float targetRoll = clampf(latAccel * 0.0095f, -0.07f, 0.07f);

    float pitchAcc = cfg_.suspensionKSpring * (targetPitch - ego_.pitch) - cfg_.suspensionCDamper * ego_.pitchRate;
    float rollAcc = cfg_.suspensionKSpring * (targetRoll - ego_.roll) - cfg_.suspensionCDamper * ego_.rollRate;

    ego_.pitchRate += pitchAcc * dt;
    ego_.pitch += ego_.pitchRate * dt;
    ego_.rollRate += rollAcc * dt;
    ego_.roll += ego_.rollRate * dt;

    stepMultiBodyCollisionSolver();

    ego_.loopProgressU = findClosestLoopU(ego_.pos);
    Vec3 curCenter = centerline_.getPointAt(ego_.loopProgressU);
    Vec3 curTan = centerline_.getTangentAt(ego_.loopProgressU);
    Vec3 curLeft{curTan.z, 0.0f, -curTan.x};
    ego_.currentLateralOffset = (ego_.pos - curCenter).dot(curLeft);
}

void WorldEngine::triggerJaywalkerCutIn() {
    float sinY = std::sin(ego_.yaw);
    float cosY = std::cos(ego_.yaw);
    float rightX = -cosY;
    float rightZ = sinY;

    Pedestrian& ped = pedestrians_.at(7);
    ped.x = ego_.pos.x + sinY * 11.5f + rightX * 3.8f;
    ped.z = ego_.pos.z + cosY * 11.5f + rightZ * 3.8f;
    ped.ax = ped.x;
    ped.az = ped.z;
    ped.bx = ego_.pos.x + sinY * 11.5f - rightX * 5.2f;
    ped.bz = ego_.pos.z + cosY * 11.5f - rightZ * 5.2f;
    ped.targetX = ped.bx;
    ped.targetZ = ped.bz;
    ped.jaywalking = true;
    ego_.speed = std::max(ego_.speed, 8.5f);
}

void WorldEngine::triggerOccludedDartOut() {
    AIVehicle& truck = vehicles_.at(0);
    float loopLen = centerline_.getLength();
    truck.lane = 0;
    truck.u = std::fmod(ego_.loopProgressU + 13.5f / loopLen, 1.0f);
    truck.speed = 0.0f;
    truck.hardBrakeTimer = 5.5f;

    Vec3 tPt = laneCurves_[0].getPointAt(truck.u);
    Vec3 tTan = laneCurves_[0].getTangentAt(truck.u);
    Vec3 tLeft{-tTan.z, 0.0f, tTan.x};
    truck.pos = tPt;
    truck.yaw = std::atan2(tTan.x, tTan.z);
    truck.vx = 0.0f;
    truck.vz = 0.0f;

    Pedestrian& ped = pedestrians_.at(6);
    ped.x = tPt.x + tTan.x * 3.6f - tLeft.x * 1.4f;
    ped.z = tPt.z + tTan.z * 3.6f - tLeft.z * 1.4f;
    ped.ax = ped.x;
    ped.az = ped.z;
    ped.bx = tPt.x + tTan.x * 3.6f + tLeft.x * 5.5f;
    ped.bz = tPt.z + tTan.z * 3.6f + tLeft.z * 5.5f;
    ped.targetX = ped.bx;
    ped.targetZ = ped.bz;
    ped.jaywalking = true;
    ego_.speed = std::max(ego_.speed, 8.0f);
}

void WorldEngine::triggerLeadHardBrake() {
    AIVehicle& lead = vehicles_.at(1);
    float loopLen = centerline_.getLength();
    lead.lane = (ego_.currentLateralOffset > 0.0f) ? 1 : 0;
    lead.u = std::fmod(ego_.loopProgressU + 12.0f / loopLen, 1.0f);
    lead.speed = ego_.speed * 0.8f;
    lead.hardBrakeTimer = 4.0f;
    Vec3 pt = laneCurves_[lead.lane].getPointAt(lead.u);
    Vec3 tan = laneCurves_[lead.lane].getTangentAt(lead.u);
    lead.pos = pt;
    lead.yaw = std::atan2(tan.x, tan.z);
    lead.vx = tan.x * lead.speed;
    lead.vz = tan.z * lead.speed;
    ego_.speed = std::max(ego_.speed, 9.5f);
}

void WorldEngine::triggerEmergencySiren() {
    siren_.active = true;
    siren_.pos = {0.0f, 1.2f, 66.0f};
    siren_.vel = {0.0f, 0.0f, -11.5f};
    siren_.phase = 0.0f;
    siren_.baseFreqHz = 780.0f;
}

void WorldEngine::setForcedRedSignal(bool forcedRed) {
    forcedRed_ = forcedRed;
    if (forcedRed) signalPhase_ = SignalPhase::Red;
}

void WorldEngine::setSplitSignalPhase(SignalPhase lane0Phase,
                                      SignalPhase lane1Phase) {
  splitSignalActive_ = true;
  splitSignalPhases_[0] = lane0Phase;
  splitSignalPhases_[1] = lane1Phase;
  forcedRed_ = false;
  signalPhase_ = lane0Phase;
}

void WorldEngine::clearSplitSignalPhase() { splitSignalActive_ = false; }

SignalPhase WorldEngine::signalPhaseForLane(int lane) const {
  return splitSignalActive_ ? splitSignalPhases_[lane & 1] : signalPhase_;
}

void WorldEngine::addUnknownRoadObstacle(const UnknownObstacleSpec& obs) {
  unknownObstacles_.push_back(obs);
}

void WorldEngine::clearUnknownRoadObstacles() { unknownObstacles_.clear(); }

void WorldEngine::buildStaticWorldGeometry(std::vector<SceneVertex>& verts, std::vector<uint32_t>& indices) const {
    verts.clear();
    indices.clear();

    // Ground plane
    appendQuadXZ(verts, indices, {0.0f, 0.0f, 0.0f}, 360.0f, 360.0f, 0.0f, {0.118f, 0.161f, 0.231f}, 0.0f, SemanticClass::Static);

    // Loop Asphalt Road
    const float roadWidth = 9.2f;
    const int samples = 240;
    const Vec3 roadCol{0.149f, 0.169f, 0.204f};
    uint32_t roadBase = static_cast<uint32_t>(verts.size());
    for (int i = 0; i <= samples; ++i) {
        float u = static_cast<float>(i % samples) / static_cast<float>(samples);
        Vec3 pt = centerline_.getPointAt(u);
        Vec3 tan = centerline_.getTangentAt(u);
        Vec3 left{-tan.z, 0.0f, tan.x};
        Vec3 pL = pt + left * (roadWidth * 0.5f);
        Vec3 pR = pt - left * (roadWidth * 0.5f);
        verts.push_back({pL.x, 0.03f, pL.z, 0, 1, 0, roadCol.x, roadCol.y, roadCol.z, 1, 0, 4, 0, 0});
        verts.push_back({pR.x, 0.03f, pR.z, 0, 1, 0, roadCol.x, roadCol.y, roadCol.z, 1, 0, 4, 0, 0});
        if (i < samples) {
            uint32_t b = roadBase + static_cast<uint32_t>(i * 2);
            indices.push_back(b); indices.push_back(b + 2); indices.push_back(b + 1);
            indices.push_back(b + 1); indices.push_back(b + 2); indices.push_back(b + 3);
        }
    }

    // Cross-Avenue Asphalt
    appendQuadXZ(verts, indices, {0.0f, 0.028f, 25.0f}, 11.0f, 130.0f, 0.0f, roadCol, 0.0f, SemanticClass::Static);

    // Lane dashes and edges
    const Vec3 dashCol{0.945f, 0.961f, 0.976f};
    const Vec3 curbCol{0.961f, 0.620f, 0.043f};
    for (int i = 0; i < samples; ++i) {
        float u = static_cast<float>(i) / static_cast<float>(samples);
        Vec3 pt = centerline_.getPointAt(u);
        Vec3 tan = centerline_.getTangentAt(u);
        Vec3 left{-tan.z, 0.0f, tan.x};
        float yaw = std::atan2(tan.x, tan.z);
        if (std::abs(pt.x) < 7.5f && std::abs(pt.z - kLoopHalfZ) < 7.5f) continue;

        if (i % 2 == 0) {
            appendQuadXZ(verts, indices, {pt.x, 0.045f, pt.z}, 0.16f, 1.4f, yaw, dashCol, 0.1f, SemanticClass::Static);
        }
        Vec3 edgeL = pt + left * (roadWidth * 0.46f);
        Vec3 edgeR = pt - left * (roadWidth * 0.46f);
        appendQuadXZ(verts, indices, {edgeL.x, 0.042f, edgeL.z}, 0.14f, 1.9f, yaw, curbCol, 0.1f, SemanticClass::Curb);
        appendQuadXZ(verts, indices, {edgeR.x, 0.042f, edgeR.z}, 0.14f, 1.9f, yaw, dashCol, 0.1f, SemanticClass::Static);
    }

    // Stop Bar & Crosswalks
    appendQuadXZ(verts, indices, {kStopLineX, 0.05f, kLoopHalfZ}, 0.6f, 8.6f, 0.0f, dashCol, 0.15f, SemanticClass::Static);
    for (float zOff = -3.8f; zOff <= 3.81f; zOff += 1.25f) {
        appendQuadXZ(verts, indices, {-7.6f, 0.05f, kLoopHalfZ + zOff}, 2.6f, 0.65f, 0.0f, dashCol, 0.12f, SemanticClass::Static);
        appendQuadXZ(verts, indices, {17.5f, 0.05f, -kLoopHalfZ + zOff}, 2.6f, 0.65f, 0.0f, dashCol, 0.12f, SemanticClass::Static);
    }

    // Traffic light cylindrical mast arm pole & housings
    const Vec3 poleCol{0.20f, 0.255f, 0.333f};
    const Vec3 housingCol{0.059f, 0.090f, 0.165f};
    appendCylinderY(verts, indices, Mat4::translation({6.5f, 3.1f, kLoopHalfZ - 6.2f}), 0.16f, 0.22f, 6.2f, 16, poleCol, 1.0f, 0.0f, SemanticClass::SignalSign);
    appendCylinderY(verts, indices, Mat4::translation({6.5f, 5.9f, kLoopHalfZ - 2.6f}) * Mat4::rotX(kPi * 0.5f), 0.12f, 0.15f, 7.6f, 12, poleCol, 1.0f, 0.0f, SemanticClass::SignalSign);
    for (const float zArm : {3.2f, 6.2f}) {
        appendBox(verts, indices, Mat4::translation({6.4f, 5.4f, kLoopHalfZ - 6.2f + zArm}),
                  {0.42f, 1.25f, 0.45f}, housingCol, 1.0f, 0.0f, SemanticClass::SignalSign);
    }

    // Overhanging Pedestrian Gantry
    appendBox(verts, indices, Mat4::translation({kLoopHalfX - 6.0f, 2.6f, 0.0f}), {0.9f, 5.2f, 1.2f}, poleCol, 1.0f, 0.0f, SemanticClass::Static);
    appendBox(verts, indices, Mat4::translation({kLoopHalfX + 6.0f, 2.6f, 0.0f}), {0.9f, 5.2f, 1.2f}, poleCol, 1.0f, 0.0f, SemanticClass::Static);
    appendBox(verts, indices, Mat4::translation({kLoopHalfX, 4.75f, 0.0f}), {13.2f, 0.95f, 1.6f}, {0.278f, 0.333f, 0.412f}, 1.0f, 0.0f, SemanticClass::Overhang);

    // Road signs (STOP octagon-style & SPEED LIMIT 35) with cylindrical posts
    auto addSign = [&](float x, float z, float rotY, Vec3 boardCol, bool stopOct) {
        const Mat4 base = Mat4::translation({x, 0.0f, z}) * Mat4::rotY(rotY);
        appendCylinderY(verts, indices, base * Mat4::translation({0.0f, 1.35f, 0.0f}), 0.055f, 0.065f, 2.7f, 10, poleCol, 1.0f, 0.0f, SemanticClass::SignalSign);
        if (stopOct) {
            appendCylinderY(verts, indices, base * Mat4::translation({0.0f, 2.45f, 0.0f}) * Mat4::rotX(kPi * 0.5f), 0.52f, 0.52f, 0.08f, 8, boardCol, 1.0f, 0.35f, SemanticClass::SignalSign);
        } else {
            appendBox(verts, indices, base * Mat4::translation({0.0f, 2.45f, 0.0f}), {0.90f, 1.05f, 0.08f}, boardCol, 1.0f, 0.35f, SemanticClass::SignalSign);
        }
    };
    addSign(12.5f, -kLoopHalfZ - 5.6f, kPi * 0.5f, {0.86f, 0.15f, 0.15f}, true);
    addSign(-34.0f, kLoopHalfZ + 5.6f, -kPi * 0.5f, {0.96f, 0.98f, 0.99f}, false);

    // Buildings + subtle window strips for real gradient texture in camera images
    for (const auto& b : buildingDefs()) {
        appendBox(verts, indices, Mat4::translation({b.x, b.h * 0.5f, b.z}), {b.w, b.h, b.d}, b.baseCol, 1.0f, 0.0f, SemanticClass::Static);
        for (float wy = 2.5f; wy < b.h - 1.5f; wy += 3.2f) {
            appendBox(verts, indices, Mat4::translation({b.x, wy, b.z}), {b.w + 0.08f, 1.1f, b.d + 0.08f}, b.winCol, 1.0f, 0.25f, SemanticClass::Static);
        }
    }

    // Concrete Barriers
    for (float z = -22.0f; z <= 22.0f; z += 4.5f) {
        appendBox(verts, indices, Mat4::translation({-kLoopHalfX - 5.4f, 0.45f, z}), {0.6f, 0.9f, 3.8f}, {0.39f, 0.45f, 0.55f}, 1.0f, 0.0f, SemanticClass::Curb);
    }
}

void WorldEngine::appendDynamicGeometry(std::vector<SceneVertex>& verts, std::vector<uint32_t>& indices,
                                        bool fsdWireMode, bool includeEgo) const {
    // 1. Dynamic Traffic Light Lenses (spherical emissive lenses, ALWAYS visible even in FSD mode)
  constexpr float kSplitHeadLaneZThreshM = 4.5f;
  constexpr float kStripeOffYM = 0.12f, kStripePadM = 0.04f;
  constexpr float kStripeHFrac = 0.34f, kStripeEmis = 0.15f;
  const Vec3 kStripeCol{0.96f, 0.94f, 0.90f};
  for (const float zArm : {3.2f, 6.2f}) {
        const SignalPhase ph =
          signalPhaseForLane((zArm < kSplitHeadLaneZThreshM) ? 1 : 0);
        const Vec3 rCol = (ph == SignalPhase::Red) ? Vec3{0.96f, 0.24f, 0.24f}
                                                   : Vec3{0.25f, 0.05f, 0.07f};
        const Vec3 yCol = (ph == SignalPhase::Yellow)
                            ? Vec3{0.98f, 0.64f, 0.05f}
                            : Vec3{0.26f, 0.13f, 0.02f};
        const Vec3 gCol = (ph == SignalPhase::Green)
                            ? Vec3{0.08f, 0.78f, 0.52f}
                            : Vec3{0.02f, 0.18f, 0.09f};
        const float rEm = (ph == SignalPhase::Red) ? 2.6f : 0.0f;
        const float yEm = (ph == SignalPhase::Yellow) ? 2.6f : 0.0f;
        const float gEm = (ph == SignalPhase::Green) ? 2.6f : 0.0f;
        const float zW = kLoopHalfZ - 6.2f + zArm;
        appendSphere(verts, indices, Mat4::translation({6.18f, 5.78f, zW}), 0.14f, 8, 12, rCol, 1.0f, rEm, SemanticClass::SignalSign);
        appendSphere(verts, indices, Mat4::translation({6.18f, 5.40f, zW}), 0.14f, 8, 12, yCol, 1.0f, yEm, SemanticClass::SignalSign);
        appendSphere(verts, indices, Mat4::translation({6.18f, 5.02f, zW}), 0.14f, 8, 12, gCol, 1.0f, gEm, SemanticClass::SignalSign);
    }

    // 2. Ego Tesla Cybercab (sculpted Robotaxi coupe + panoramic canopy + aero wheel covers)
    if (includeEgo) {
        appendCybercabVehicleMesh(verts, indices, ego_, getEgoWorldMatrix(), getEgoSprungWorldMatrix(), fsdWireMode);
    }

    if (fsdWireMode) return;

  for (const auto& uo : unknownObstacles_) {
    const Mat4 um =
      Mat4::translation({uo.cx, uo.cy, uo.cz}) * Mat4::rotY(uo.yaw);
    appendBox(verts, indices, um, {uo.hx * 2.0f, uo.hy * 2.0f, uo.hz * 2.0f},
              uo.color, 1.0f, 0.0f, SemanticClass::Static);
    if (uo.hazardStripe) {
      appendBox(verts, indices,
                um * Mat4::translation({0.0f, kStripeOffYM, 0.0f}),
                {uo.hx * 2.0f + kStripePadM, uo.hy * kStripeHFrac,
                 uo.hz * 2.0f + kStripePadM},
                kStripeCol, 1.0f, kStripeEmis, SemanticClass::Static);
    }
  }

    // 3. AI Traffic Vehicles (tapered cabins, glass, wheel assemblies, emissive headlights & taillights)
    for (const auto& v : vehicles_) {
        const Mat4 vm = Mat4::translation(v.pos) * Mat4::rotY(v.yaw);
        const Vec3 tireCol{0.08f, 0.10f, 0.13f};
        const Vec3 alloyCol{0.62f, 0.68f, 0.75f};
        const Vec3 glassCol{0.05f, 0.08f, 0.15f};
        if (v.type == "truck") {
            appendFrustumBox(verts, indices, vm * Mat4::translation({0.0f, 1.15f, 2.02f}),
                             {2.26f, 1.56f}, {2.05f, 1.28f}, 1.92f, -0.10f, v.color, 1.0f, 0.0f, SemanticClass::Vehicle);
            appendFrustumBox(verts, indices, vm * Mat4::translation({0.0f, 1.48f, 2.48f}),
                             {2.04f, 0.60f}, {1.92f, 0.42f}, 0.68f, -0.08f, glassCol, 1.0f, 0.0f, SemanticClass::Vehicle);
            appendBox(verts, indices, vm * Mat4::translation({0.0f, 1.72f, -0.78f}), {2.38f, 2.68f, 4.12f}, {0.945f, 0.961f, 0.976f}, 1.0f, 0.0f, SemanticClass::Vehicle);
            appendBox(verts, indices, vm * Mat4::translation({0.0f, 0.34f, -2.83f}), {2.34f, 0.22f, 0.12f}, tireCol, 1.0f, 0.0f, SemanticClass::Vehicle);
            appendBox(verts, indices, vm * Mat4::translation({0.0f, 0.72f, -2.86f}), {2.24f, 0.18f, 0.08f}, {0.94f, 0.22f, 0.22f}, 1.0f, 2.2f, SemanticClass::Vehicle);
            appendBox(verts, indices, vm * Mat4::translation({0.0f, 0.56f, 2.80f}), {2.02f, 0.14f, 0.08f}, {0.94f, 0.97f, 1.0f}, 1.0f, 1.4f, SemanticClass::Vehicle);
            for (const float zAx : {-1.95f, -0.65f, 1.95f}) {
                for (const float xAx : {-1.08f, 1.08f}) {
                    appendWheelAssembly(verts, indices, vm * Mat4::translation({xAx, 0.44f, zAx}), 0.44f, 0.30f, tireCol, alloyCol, SemanticClass::Vehicle);
                }
            }
        } else {
            const float h = (v.type == "suv") ? 1.58f : 1.38f;
            const float len = (v.type == "suv") ? 4.55f : 4.35f;
            appendFrustumBox(verts, indices, vm * Mat4::translation({0.0f, h * 0.37f + 0.12f, 0.0f}),
                             {1.88f, len}, {1.76f, len * 0.93f}, h * 0.46f, -0.05f, v.color, 1.0f, 0.0f, SemanticClass::Vehicle);
            appendFrustumBox(verts, indices, vm * Mat4::translation({0.0f, h * 0.78f + 0.06f, -0.22f}),
                             {1.66f, len * 0.56f}, {1.34f, len * 0.36f}, h * 0.44f, -0.10f, glassCol, 1.0f, 0.0f, SemanticClass::Vehicle);
            appendBox(verts, indices, vm * Mat4::translation({0.0f, h * 0.42f + 0.12f, len * 0.5f + 0.01f}),
                      {1.62f, 0.10f, 0.06f}, {0.94f, 0.97f, 1.0f}, 1.0f, 1.5f, SemanticClass::Vehicle);
            appendBox(verts, indices, vm * Mat4::translation({0.0f, h * 0.48f + 0.10f, -len * 0.5f - 0.02f}),
                      {1.66f, 0.11f, 0.06f}, {0.95f, 0.20f, 0.20f}, 1.0f, 1.9f, SemanticClass::Vehicle);
            for (const float zAx : {-len * 0.30f, len * 0.30f}) {
                for (const float xAx : {-0.86f, 0.86f}) {
                    appendWheelAssembly(verts, indices, vm * Mat4::translation({xAx, 0.34f, zAx}), 0.34f, 0.24f, tireCol, alloyCol, SemanticClass::Vehicle);
                }
            }
        }
    }

    // 4. Articulated Pedestrians (spherical heads, tapered torso, shoulder/hip-pivoted cylindrical limbs)
    for (const auto& p : pedestrians_) {
        const Mat4 pm = Mat4::translation({p.x, 0.0f, p.z}) * Mat4::rotY(p.yaw);
        const Vec3 skinCol{0.97f, 0.75f, 0.54f};
        const Vec3 shoeCol{0.08f, 0.10f, 0.14f};
        appendFrustumBox(verts, indices, pm * Mat4::translation({0.0f, 1.04f, 0.0f}),
                         {0.34f, 0.20f}, {0.42f, 0.24f}, 0.58f, 0.0f, p.shirtColor, 1.0f, 0.0f, SemanticClass::Pedestrian);
        appendCylinderY(verts, indices, pm * Mat4::translation({0.0f, 1.37f, 0.0f}), 0.07f, 0.08f, 0.10f, 10, skinCol, 1.0f, 0.0f, SemanticClass::Pedestrian);
        appendSphere(verts, indices, pm * Mat4::translation({0.0f, 1.54f, 0.0f}), 0.135f, 10, 12, skinCol, 1.0f, 0.0f, SemanticClass::Pedestrian);
        // Legs pivoting around hip (y = 0.72)
        for (const int side : {-1, 1}) {
            const float ang = static_cast<float>(side) * p.legAngle;
            const Mat4 legMat = pm * Mat4::translation({static_cast<float>(side) * 0.11f, 0.72f, 0.0f}) * Mat4::rotX(ang);
            appendCylinderY(verts, indices, legMat * Mat4::translation({0.0f, -0.34f, 0.0f}), 0.075f, 0.062f, 0.66f, 10, p.pantsColor, 1.0f, 0.0f, SemanticClass::Pedestrian);
            appendBox(verts, indices, legMat * Mat4::translation({0.0f, -0.68f, 0.04f}), {0.11f, 0.08f, 0.20f}, shoeCol, 1.0f, 0.0f, SemanticClass::Pedestrian);
        }
        // Arms pivoting around shoulder (y = 1.28)
        for (const int side : {-1, 1}) {
            const float ang = static_cast<float>(side) * p.armAngle;
            const Mat4 armMat = pm * Mat4::translation({static_cast<float>(side) * 0.25f, 1.28f, 0.0f}) * Mat4::rotX(ang);
            appendCylinderY(verts, indices, armMat * Mat4::translation({0.0f, -0.24f, 0.0f}), 0.055f, 0.048f, 0.48f, 10, p.shirtColor, 1.0f, 0.0f, SemanticClass::Pedestrian);
            appendSphere(verts, indices, armMat * Mat4::translation({0.0f, -0.50f, 0.0f}), 0.052f, 6, 8, skinCol, 1.0f, 0.0f, SemanticClass::Pedestrian);
        }
    }
}

} // namespace cybercab
