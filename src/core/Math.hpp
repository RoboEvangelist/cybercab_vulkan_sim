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
