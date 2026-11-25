#pragma once
// Reusable zero-allocation 2D BEV grid spatial hash for fast O(1) neighborhood queries.
// Shared across Stage-1 VoxelToGaussianInit and DeformableAttention3D to avoid O(N^2) scans.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cybercab::sota {

class SpatialHashIndex3D {
public:
    static constexpr int kGridDim = 64;
    static constexpr int kGridMask = kGridDim - 1;
    static constexpr int kNumBuckets = kGridDim * kGridDim;

    explicit SpatialHashIndex3D(float cellSizeM = 2.0f)
        : cellSizeM_(std::max(0.25f, cellSizeM)),
          invCellSize_(1.0f / std::max(0.25f, cellSizeM)) {
        head_.fill(-1);
    }

    void reserve(size_t maxItems) {
        next_.reserve(maxItems);
    }

    void reset(size_t itemCount, float cellSizeM = 2.0f) {
        cellSizeM_ = std::max(0.25f, cellSizeM);
        invCellSize_ = 1.0f / cellSizeM_;
        head_.fill(-1);
        if (next_.capacity() < itemCount) next_.reserve(itemCount);
        next_.assign(itemCount, -1);
    }

    void insert(int idx, float wx, float wz) {
        if (idx < 0 || static_cast<size_t>(idx) >= next_.size()) return;
        const int b = bucketOf(cellCoord(wx), cellCoord(wz));
        next_[static_cast<size_t>(idx)] = head_[static_cast<size_t>(b)];
        head_[static_cast<size_t>(b)] = idx;
    }

    template <typename Fn>
    void forEachNeighbor(float wx, float wz, float radiusM, Fn&& fn) const {
        const int cx = cellCoord(wx);
        const int cz = cellCoord(wz);
        const int span = std::max(1, static_cast<int>(std::ceil(radiusM * invCellSize_)));
        for (int dz = -span; dz <= span; ++dz) {
            for (int dx = -span; dx <= span; ++dx) {
                int idx = head_[static_cast<size_t>(bucketOf(cx + dx, cz + dz))];
                while (idx >= 0) {
                    if (!fn(idx)) return;
                    idx = next_[static_cast<size_t>(idx)];
                }
            }
        }
    }

private:
    int cellCoord(float v) const {
        return static_cast<int>(std::floor(v * invCellSize_));
    }

    static int bucketOf(int gx, int gz) {
        return ((gz & kGridMask) << 6) | (gx & kGridMask);
    }

    float cellSizeM_ = 2.0f;
    float invCellSize_ = 0.5f;
    std::array<int32_t, kNumBuckets> head_{};
    std::vector<int32_t> next_;
};

} // namespace cybercab::sota
