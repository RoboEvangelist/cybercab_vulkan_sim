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
