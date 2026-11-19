#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#if __has_include(<numbers>)
#include <numbers>
#endif

namespace cybercab {

#if defined(__cpp_lib_math_constants) && (__cpp_lib_math_constants >= 201907L)
inline constexpr float kPi = std::numbers::pi_v<float>;
#else
inline constexpr float kPi = static_cast<float>(M_PI);
#endif

inline float degToRad(float deg) { return deg * (kPi / 180.0f); }
inline float radToDeg(float rad) { return rad * (180.0f / kPi); }

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

inline float wrapAngle(float a) {
    while (a > kPi) a -= 2.0f * kPi;
    while (a < -kPi) a += 2.0f * kPi;
    return a;
}

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;

    Vec2() = default;
    Vec2(float x_, float y_) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
    float dot(const Vec2& o) const { return x * o.x + y * o.y; }
    float norm() const { return std::sqrt(x * x + y * y); }
};

struct alignas(16) Vec4;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }

    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    float normSq() const { return x * x + y * y + z * z; }
    float norm() const { return std::sqrt(normSq()); }
    Vec3 normalized() const {
        float n = norm();
        return (n > 1e-7f) ? (*this / n) : Vec3{0.0f, 0.0f, 0.0f};
    }
};

struct alignas(16) Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;

    Vec4() = default;
    Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    Vec4(const Vec3& v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
};

// Column-major 4x4 matrix matching GLSL mat4 / Vulkan std140 & std430 layout.
// m[col * 4 + row]
struct alignas(16) Mat4 {
    std::array<float, 16> m{};

    static Mat4 identity() {
        Mat4 r{};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    float& at(int row, int col) { return m[col * 4 + row]; }
    float at(int row, int col) const { return m[col * 4 + row]; }

    Mat4 operator*(const Mat4& b) const {
        Mat4 out{};
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                float s = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    s += at(r, k) * b.at(k, c);
                }
                out.at(r, c) = s;
            }
        }
        return out;
    }

    Vec4 operator*(const Vec4& v) const {
        return {
            at(0, 0) * v.x + at(0, 1) * v.y + at(0, 2) * v.z + at(0, 3) * v.w,
            at(1, 0) * v.x + at(1, 1) * v.y + at(1, 2) * v.z + at(1, 3) * v.w,
            at(2, 0) * v.x + at(2, 1) * v.y + at(2, 2) * v.z + at(2, 3) * v.w,
            at(3, 0) * v.x + at(3, 1) * v.y + at(3, 2) * v.z + at(3, 3) * v.w
        };
    }

    Vec3 transformPoint(const Vec3& p) const {
        Vec4 r = (*this) * Vec4{p.x, p.y, p.z, 1.0f};
        return {r.x, r.y, r.z};
    }

    Vec3 transformVector(const Vec3& v) const {
        Vec4 r = (*this) * Vec4{v.x, v.y, v.z, 0.0f};
        return {r.x, r.y, r.z};
    }

    Mat4 transposed() const {
        Mat4 t{};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                t.at(r, c) = at(c, r);
            }
        }
        return t;
    }

    // Invert rigid-body SE(3) transform [R t; 0 1].
    Mat4 inverseRigid() const {
        Mat4 inv = identity();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                inv.at(r, c) = at(c, r);
            }
        }
        Vec3 t{at(0, 3), at(1, 3), at(2, 3)};
        for (int r = 0; r < 3; ++r) {
            inv.at(r, 3) = -(inv.at(r, 0) * t.x + inv.at(r, 1) * t.y + inv.at(r, 2) * t.z);
        }
        return inv;
    }

    static Mat4 translation(const Vec3& t) {
        Mat4 r = identity();
        r.at(0, 3) = t.x;
        r.at(1, 3) = t.y;
        r.at(2, 3) = t.z;
        return r;
    }

    static Mat4 scale(const Vec3& s) {
        Mat4 r{};
        r.at(0, 0) = s.x;
        r.at(1, 1) = s.y;
        r.at(2, 2) = s.z;
        r.at(3, 3) = 1.0f;
        return r;
    }

    static Mat4 rotX(float rad) {
        Mat4 r = identity();
        float c = std::cos(rad), s = std::sin(rad);
        r.at(1, 1) = c; r.at(1, 2) = -s;
        r.at(2, 1) = s; r.at(2, 2) = c;
        return r;
    }

    static Mat4 rotY(float rad) {
        Mat4 r = identity();
        float c = std::cos(rad), s = std::sin(rad);
        r.at(0, 0) = c;  r.at(0, 2) = s;
        r.at(2, 0) = -s; r.at(2, 2) = c;
        return r;
    }

    static Mat4 rotZ(float rad) {
        Mat4 r = identity();
        float c = std::cos(rad), s = std::sin(rad);
        r.at(0, 0) = c; r.at(0, 1) = -s;
        r.at(1, 0) = s; r.at(1, 1) = c;
        return r;
    }

    // Standard Vulkan projection from CV Camera space (+X right, +Y down, +Z forward)
    // directly into Vulkan NDC ([-1, 1] x [-1, 1] with Y down, Z in [0, 1]).
    static Mat4 vulkanPerspectiveCV(float fx, float fy, float cx, float cy,
                                    float width, float height, float zNear, float zFar) {
        Mat4 p{};
        p.at(0, 0) = 2.0f * fx / width;
        p.at(0, 2) = 2.0f * cx / width - 1.0f;
        p.at(1, 1) = 2.0f * fy / height;
        p.at(1, 2) = 2.0f * cy / height - 1.0f;
        p.at(2, 2) = zFar / (zFar - zNear);
        p.at(2, 3) = -(zFar * zNear) / (zFar - zNear);
        p.at(3, 2) = 1.0f;
        return p;
    }

    // Standard Vulkan lookAt for viewport camera (world +Y up, maps to +Y down in Vulkan NDC).
    static Mat4 lookAtVulkan(const Vec3& eye, const Vec3& target, const Vec3& worldUp = {0.0f, 1.0f, 0.0f}) {
        Vec3 fwd = (target - eye).normalized();
        Vec3 right = fwd.cross(worldUp).normalized();
        if (right.normSq() < 1e-6f) {
            right = fwd.cross(Vec3{0.0f, 0.0f, 1.0f}).normalized();
        }
        Vec3 down = fwd.cross(right).normalized(); // right x down = fwd (right-handed CV frame)

        Mat4 view = identity();
        view.at(0, 0) = right.x; view.at(0, 1) = right.y; view.at(0, 2) = right.z;
        view.at(1, 0) = down.x;  view.at(1, 1) = down.y;  view.at(1, 2) = down.z;
        view.at(2, 0) = fwd.x;   view.at(2, 1) = fwd.y;   view.at(2, 2) = fwd.z;
        view.at(0, 3) = -right.dot(eye);
        view.at(1, 3) = -down.dot(eye);
        view.at(2, 3) = -fwd.dot(eye);
        return view;
    }
};

// Camera Coordinate Helpers:
// ISO 8855 Ego: +X fwd, +Y left, +Z up
// Vehicle Local (World-aligned when yaw=0): +X left, +Y up, +Z fwd
inline Vec3 isoEgoToVehicleLocal(const Vec3& iso) {
    return {iso.y, iso.z, iso.x};
}

inline Vec3 vehicleLocalToIsoEgo(const Vec3& loc) {
    return {loc.z, loc.x, loc.y};
}

// Forward Brown-Conrady radial/tangential distortion in normalized camera coordinates.
inline Vec2 distortNormalized(float xu, float yu, const std::array<float, 4>& d) {
    const float r2 = xu * xu + yu * yu;
    const float rad = 1.0f + d[0] * r2 + d[1] * r2 * r2;
    const float xd = xu * rad + 2.0f * d[2] * xu * yu + d[3] * (r2 + 2.0f * xu * xu);
    const float yd = yu * rad + d[2] * (r2 + 2.0f * yu * yu) + 2.0f * d[3] * xu * yu;
    return {xd, yd};
}

// Fixed-point iterative Brown-Conrady undistortion in normalized camera coordinates.
inline Vec2 undistortNormalized(float xd, float yd, const std::array<float, 4>& d) {
    float xu = xd;
    float yu = yd;
    for (int it = 0; it < 6; ++it) {
        const float r2 = xu * xu + yu * yu;
        const float rad = 1.0f + d[0] * r2 + d[1] * r2 * r2;
        const float dxT = 2.0f * d[2] * xu * yu + d[3] * (r2 + 2.0f * xu * xu);
        const float dyT = d[2] * (r2 + 2.0f * yu * yu) + 2.0f * d[3] * xu * yu;
        xu = (xd - dxT) / std::max(rad, 0.35f);
        yu = (yd - dyT) / std::max(rad, 0.35f);
    }
    return {xu, yu};
}

} // namespace cybercab
