#include "perception/ByteTrack.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cybercab {

ByteTracker::ByteTracker(const SimConfig& cfg) : cfg_(cfg) {
    const size_t maxTr = static_cast<size_t>(std::max(64, cfg_.btMaxTracks));
    const size_t maxDet = 128;
    tracks_.reserve(maxTr);
    highIdxScratch_.reserve(maxDet);
    lowIdxScratch_.reserve(maxDet);
    unmatchedScratch_.reserve(maxTr);
    trackMatchedScratch_.reserve(maxTr);
    highDetMatchedScratch_.reserve(maxDet);
    costFlatScratch_.reserve(maxTr * maxDet);
    const size_t maxDim = std::max(maxTr, maxDet) + 1;
    hungarian_.a.reserve(maxDim * maxDim);
    hungarian_.u.reserve(maxDim);
    hungarian_.v.reserve(maxDim);
    hungarian_.minv.reserve(maxDim);
    hungarian_.p.reserve(maxDim);
    hungarian_.way.reserve(maxDim);
    hungarian_.used.reserve(maxDim);
    hungarian_.assign.reserve(maxTr);
}

void ByteTracker::reset() {
    tracks_.clear();
    nextTrackId_ = 1;
}

void ByteTracker::HungarianWorkspace::solve(const float* flatCost, int nRows, int nCols, int strideCols, float gateThreshold) {
    assign.assign(static_cast<size_t>(nRows), -1);
    if (nRows == 0 || nCols == 0) return;

    const int n = std::max(nRows, nCols);
    const int dim = n + 1;
    const double kBig = static_cast<double>(gateThreshold) * 100.0 + 1e5;

    a.assign(static_cast<size_t>(dim) * dim, kBig);
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            const float c = flatCost[i * strideCols + j];
            a[static_cast<size_t>(i + 1) * dim + (j + 1)] = (c <= gateThreshold) ? static_cast<double>(c) : kBig;
        }
    }

    u.assign(static_cast<size_t>(dim), 0.0);
    v.assign(static_cast<size_t>(dim), 0.0);
    p.assign(static_cast<size_t>(dim), 0);
    way.assign(static_cast<size_t>(dim), 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        minv.assign(static_cast<size_t>(dim), std::numeric_limits<double>::infinity());
        used.assign(static_cast<size_t>(dim), 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    const double cur = a[static_cast<size_t>(i0) * dim + j] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    for (int j = 1; j <= nCols; ++j) {
        if (p[j] >= 1 && p[j] <= nRows) {
            const int row = p[j] - 1;
            const int col = j - 1;
            if (flatCost[row * strideCols + col] <= gateThreshold) {
                assign[row] = col;
            }
        }
    }
}

std::vector<int> ByteTracker::solveHungarian(const std::vector<std::vector<float>>& costMatrix, float gateThreshold) {
    const int nRows = static_cast<int>(costMatrix.size());
    if (nRows == 0) return {};
    const int nCols = static_cast<int>(costMatrix[0].size());
    if (nCols == 0) return std::vector<int>(static_cast<size_t>(nRows), -1);

    std::vector<float> flat(static_cast<size_t>(nRows) * nCols);
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            flat[static_cast<size_t>(i) * nCols + j] = costMatrix[i][j];
        }
    }
    HungarianWorkspace ws;
    ws.solve(flat.data(), nRows, nCols, nCols, gateThreshold);
    return ws.assign;
}

void ByteTracker::predictTrack(InternalTrack& tr, float dt) const {
    tr.x[0] += tr.x[2] * dt;
    tr.x[1] += tr.x[3] * dt;

    std::array<float, 16> F{};
    F[0] = F[5] = F[10] = F[15] = 1.0f;
    F[2] = dt;
    F[7] = dt;

    std::array<float, 16> FP{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += F[r * 4 + k] * tr.P[k * 4 + c];
            FP[r * 4 + c] = s;
        }
    }
