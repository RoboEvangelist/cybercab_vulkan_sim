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
    std::array<float, 16> Pnew{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += FP[r * 4 + k] * F[c * 4 + k];
            Pnew[r * 4 + c] = s;
        }
    }
    const float qp = cfg_.btProcessNoisePos * cfg_.btProcessNoisePos * dt;
    const float qv = cfg_.btProcessNoiseVel * cfg_.btProcessNoiseVel * dt;
    Pnew[0] += qp;
    Pnew[5] += qp;
    Pnew[10] += qv;
    Pnew[15] += qv;
    tr.P = Pnew;
    tr.pub.timeSinceUpdateSec += dt;
}

void ByteTracker::updateTrack(InternalTrack& tr, const Detection3D& det, bool lowConfMatch) {
    const float soilScale = 1.0f / std::max(det.camWeight, cfg_.soilMinTokenWeight);
    const float measStd = cfg_.btBaseMeasNoiseM * soilScale + det.epistemicSigma;
    const float Rdiag = measStd * measStd;

    const float S00 = tr.P[0] + Rdiag;
    const float S01 = tr.P[1];
    const float S10 = tr.P[4];
    const float S11 = tr.P[5] + Rdiag;
    const float detS = S00 * S11 - S01 * S10;
    if (std::abs(detS) < 1e-6f) return;

    const float invS00 = S11 / detS;
    const float invS01 = -S01 / detS;
    const float invS10 = -S10 / detS;
    const float invS11 = S00 / detS;

    float K[4][2]{};
    for (int r = 0; r < 4; ++r) {
        K[r][0] = tr.P[r * 4 + 0] * invS00 + tr.P[r * 4 + 1] * invS10;
        K[r][1] = tr.P[r * 4 + 0] * invS01 + tr.P[r * 4 + 1] * invS11;
    }

    const float y0 = det.worldPos.x - tr.x[0];
    const float y1 = det.worldPos.z - tr.x[1];
    for (int r = 0; r < 4; ++r) {
        tr.x[r] += K[r][0] * y0 + K[r][1] * y1;
    }

    std::array<float, 16> I_KH{};
    I_KH[0] = 1.0f - K[0][0]; I_KH[1] = -K[0][1];
    I_KH[4] = -K[1][0];       I_KH[5] = 1.0f - K[1][1];
    I_KH[8] = -K[2][0];       I_KH[9] = -K[2][1]; I_KH[10] = 1.0f;
    I_KH[12] = -K[3][0];      I_KH[13] = -K[3][1]; I_KH[15] = 1.0f;

    std::array<float, 16> Pnew{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += I_KH[r * 4 + k] * tr.P[k * 4 + c];
            Pnew[r * 4 + c] = s;
        }
    }
    tr.P = Pnew;

    tr.pub.hits += 1;
    tr.pub.timeSinceUpdateSec = 0.0f;
    tr.pub.score = det.score;
    tr.pub.matchedLowConf = lowConfMatch;
    if (det.camWeight > 0.45f) {
        tr.pub.semantic = det.semantic;
        tr.pub.halfExtents = det.halfExtents;
    }
    if (tr.pub.hits >= 2 || det.score >= cfg_.rtdetrHighConfThresh) {
        tr.pub.state = TrackState::Tracked;
    }
}

void ByteTracker::spawnTrack(const Detection3D& det) {
    if (static_cast<int>(tracks_.size()) >= cfg_.btMaxTracks) return;
    InternalTrack tr;
    tr.pub.trackId = nextTrackId_++;
    tr.pub.semantic = det.semantic;
    tr.pub.state = TrackState::Tentative;
    tr.pub.halfExtents = det.halfExtents;
    tr.pub.score = det.score;
    tr.pub.hits = 1;
    tr.pub.timeSinceUpdateSec = 0.0f;
    tr.pub.yaw = det.yaw;

    tr.x = {det.worldPos.x, det.worldPos.z, 0.0f, 0.0f};
    const float pPos = std::max(0.4f, det.epistemicSigma * det.epistemicSigma);
    tr.P = {
        pPos, 0, 0, 0,
        0, pPos, 0, 0,
        0, 0, 4.0f, 0,
        0, 0, 0, 4.0f
    };
    tracks_.push_back(tr);
}

void ByteTracker::refreshTrackPose(InternalTrack& tr, const EgoState& ego) {
    tr.pub.worldPos = {tr.x[0], tr.pub.halfExtents.y, tr.x[1]};
    tr.pub.worldVel = {tr.x[2], 0.0f, tr.x[3]};
    tr.pub.posSigmaM = std::sqrt(std::max(0.01f, tr.P[0] + tr.P[5]));
    if (std::hypot(tr.x[2], tr.x[3]) > 0.35f) {
        tr.pub.yaw = std::atan2(tr.x[2], tr.x[3]);
    }
    const float dx = tr.x[0] - ego.pos.x;
    const float dz = tr.x[1] - ego.pos.z;
    const float s = std::sin(ego.yaw), c = std::cos(ego.yaw);
    tr.pub.egoPos = {dx * s + dz * c, dx * c - dz * s, tr.pub.halfExtents.y};
}

void ByteTracker::update(float dt, const EgoState& ego, const std::vector<Detection3D>& detections) {
    for (auto& tr : tracks_) {
        predictTrack(tr, dt);
        tr.pub.matchedLowConf = false;
    }

    highIdxScratch_.clear();
    lowIdxScratch_.clear();
    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        if (detections[i].semantic == SemanticClass::SignalSign) continue;
        if (detections[i].score >= cfg_.rtdetrHighConfThresh) {
            highIdxScratch_.push_back(i);
        } else if (detections[i].score >= cfg_.rtdetrLowConfThresh) {
            lowIdxScratch_.push_back(i);
        }
    }

    trackMatchedScratch_.assign(tracks_.size(), 0);
    highDetMatchedScratch_.assign(highIdxScratch_.size(), 0);

    // Stage 1: High-Confidence matching
    if (!tracks_.empty() && !highIdxScratch_.empty()) {
        const int nRows = static_cast<int>(tracks_.size());
        const int nCols = static_cast<int>(highIdxScratch_.size());
        costFlatScratch_.resize(static_cast<size_t>(nRows) * nCols);
        for (int t = 0; t < nRows; ++t) {
            for (int d = 0; d < nCols; ++d) {
                const auto& det = detections[highIdxScratch_[d]];
