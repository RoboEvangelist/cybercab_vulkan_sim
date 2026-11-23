#include "perception/ByteTrack.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cybercab {
namespace {

struct RayDelta {
    bool valid = false;
    float rx = 0.0f;
    float rz = 0.0f;
    float dRad = 0.0f;
    float dLat = 0.0f;
    float minRange = 0.0f;
    float maxRange = 0.0f;
};

// Decompose detection-to-track offset into ego viewing ray (radial/lateral) components.
RayDelta computeRayDelta(const EgoState& ego, float tx, float tz, float dx, float dz) {
    RayDelta r;
    const float trRelX = tx - ego.pos.x;
    const float trRelZ = tz - ego.pos.z;
    const float trRange = std::hypot(trRelX, trRelZ);
    const float detRelX = dx - ego.pos.x;
    const float detRelZ = dz - ego.pos.z;
    const float detRange = std::hypot(detRelX, detRelZ);
    if (trRange <= 1.0f || detRange <= 1.0f) return r;
    r.valid = true;
    r.rx = trRelX / trRange;
    r.rz = trRelZ / trRange;
    const float dX = dx - tx;
    const float dZ = dz - tz;
    r.dRad = std::abs(dX * r.rx + dZ * r.rz);
    r.dLat = std::abs(-dX * r.rz + dZ * r.rx);
    r.minRange = std::min(trRange, detRange);
    r.maxRange = std::max(trRange, detRange);
    return r;
}

} // namespace

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
    if (!std::isfinite(dt) || dt <= 0.0f) return;
    tr.lastDt = dt;
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

void ByteTracker::updateTrack(InternalTrack& tr, const EgoState& ego, const Detection3D& det, bool lowConfMatch) {
    const float soilScale = 1.0f / std::max(det.camWeight, cfg_.soilMinTokenWeight);
    const float measStd = cfg_.btBaseMeasNoiseM * soilScale + det.epistemicSigma;
    const RayDelta ray = computeRayDelta(ego, tr.x[0], tr.x[1], det.worldPos.x, det.worldPos.z);
    const bool isVehTrack = (tr.pub.semantic == SemanticClass::Vehicle && det.semantic == SemanticClass::Vehicle);
    float sigmaRad = measStd;
    float radExcess = 0.0f;
    if (ray.valid && isVehTrack && tr.pub.hits >= 2) {
        // Measure sudden single-step radial jumps vs previous detection (predicted by current track velocity),
        // rather than innovation against tr.x which would starve smooth moving targets during warmup.
        const float elapsed = std::max(tr.lastDt, tr.pub.timeSinceUpdateSec);
        const float detDx = det.worldPos.x - tr.lastDetPos.x;
        const float detDz = det.worldPos.z - tr.lastDetPos.z;
        const float vRad = tr.x[2] * ray.rx + tr.x[3] * ray.rz;
        const float detJumpRad = std::abs((detDx * ray.rx + detDz * ray.rz) - vRad * elapsed);
        const float radThresh = ((tr.lastCamIndex >= 0 && det.camIndex != tr.lastCamIndex) ? 0.45f : 0.65f)
            + 4.0f * std::max(0.0f, elapsed - 0.05f);
        radExcess = std::max(0.0f, detJumpRad - radThresh);
        if (radExcess > 0.0f) {
            sigmaRad += std::min(1.0f, 1.5f * radExcess);
        }
    }
    const float rLat = measStd * measStd;
    const float rRad = sigmaRad * sigmaRad;
    const float R00 = ray.valid ? (rRad * ray.rx * ray.rx + rLat * ray.rz * ray.rz) : rLat;
    const float R01 = ray.valid ? ((rRad - rLat) * ray.rx * ray.rz) : 0.0f;
    const float R11 = ray.valid ? (rRad * ray.rz * ray.rz + rLat * ray.rx * ray.rx) : rLat;

    const float S00 = tr.P[0] + R00;
    const float S01 = tr.P[1] + R01;
    const float S10 = tr.P[4] + R01;
    const float S11 = tr.P[5] + R11;
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
    tr.x[0] += K[0][0] * y0 + K[0][1] * y1;
    tr.x[1] += K[1][0] * y0 + K[1][1] * y1;
    float dvx = K[2][0] * y0 + K[2][1] * y1;
    float dvz = K[3][0] * y0 + K[3][1] * y1;
    const float innovRadAbs = ray.valid ? std::abs(y0 * ray.rx + y1 * ray.rz) : 0.0f;
    if (ray.valid && isVehTrack && tr.pub.hits >= 3 && (radExcess > 0.0f || innovRadAbs > 0.65f)) {
        const float dvRad = dvx * ray.rx + dvz * ray.rz;
        const float dvLat = -dvx * ray.rz + dvz * ray.rx;
        const float maxDvRad = std::max(0.06f, 2.2f * tr.lastDt);
        const float clampedRad = std::clamp(dvRad, -maxDvRad, maxDvRad);
        dvx = clampedRad * ray.rx - dvLat * ray.rz;
        dvz = clampedRad * ray.rz + dvLat * ray.rx;
    }
    tr.x[2] += dvx;
    tr.x[3] += dvz;

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
    // Exponentially decayed multi-frame class evidence suppresses single-frame class flicker
    for (float& v : tr.classVotes) v *= 0.72f;
    const int detCls = std::clamp(static_cast<int>(det.semantic), 0, 3);
    if (det.camWeight > 0.45f) {
        tr.classVotes[detCls] += det.score;
        const int curCls = std::clamp(static_cast<int>(tr.pub.semantic), 0, 3);
        if (detCls == curCls || tr.pub.hits < 3 || tr.classVotes[detCls] > tr.classVotes[curCls] * 1.15f) {
            tr.pub.semantic = det.semantic;
            tr.pub.halfExtents = det.halfExtents;
        }
    }
    const float maxSpeed = (tr.pub.semantic == SemanticClass::Pedestrian) ? 2.6f : 22.0f;
    const float spd = std::hypot(tr.x[2], tr.x[3]);
    if (spd > maxSpeed) {
        const float s = maxSpeed / spd;
        tr.x[2] *= s;
        tr.x[3] *= s;
    }
    if (tr.pub.hits >= 2 || det.score >= cfg_.rtdetrHighConfThresh) {
        tr.pub.state = TrackState::Tracked;
    }
    const float alpha = (radExcess > 0.0f) ? 0.35f : 1.0f;
    tr.lastDetPos.x += alpha * (det.worldPos.x - tr.lastDetPos.x);
    tr.lastDetPos.y = det.worldPos.y;
    tr.lastDetPos.z += alpha * (det.worldPos.z - tr.lastDetPos.z);
    tr.lastCamIndex = det.camIndex;
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
    tr.lastDetPos = det.worldPos;
    tr.lastCamIndex = det.camIndex;
    tr.classVotes[std::clamp(static_cast<int>(det.semantic), 0, 3)] = det.score;

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
    const float safeDt = (std::isfinite(dt) && dt > 0.0f) ? dt : 0.0f;
    for (auto& tr : tracks_) {
        predictTrack(tr, safeDt);
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

    auto effectiveMatchDist = [&](int trackIdx, const Detection3D& det) -> float {
        const auto& tr = tracks_[trackIdx];
        const float dist = std::hypot(tr.x[0] - det.worldPos.x, tr.x[1] - det.worldPos.z);
        if (tr.pub.semantic == det.semantic && tr.pub.state != TrackState::Tentative) {
            const RayDelta ray = computeRayDelta(ego, tr.x[0], tr.x[1], det.worldPos.x, det.worldPos.z);
            if (ray.valid && ray.minRange >= 4.0f && ray.dLat <= 0.90f && ray.dRad <= 5.2f) {
                return std::min(dist, std::hypot(ray.dLat, 0.55f * ray.dRad));
            }
        }
        return dist;
    };

    // Stage 1: High-Confidence matching
    if (!tracks_.empty() && !highIdxScratch_.empty()) {
        const int nRows = static_cast<int>(tracks_.size());
        const int nCols = static_cast<int>(highIdxScratch_.size());
        costFlatScratch_.resize(static_cast<size_t>(nRows) * nCols);
        for (int t = 0; t < nRows; ++t) {
            for (int d = 0; d < nCols; ++d) {
                const auto& det = detections[highIdxScratch_[d]];
                const float dist = effectiveMatchDist(t, det);
                const float classPen = (tracks_[t].pub.semantic == det.semantic) ? 0.0f : 1.5f;
                costFlatScratch_[static_cast<size_t>(t) * nCols + d] = dist + classPen;
            }
        }
        hungarian_.solve(costFlatScratch_.data(), nRows, nCols, nCols, cfg_.btHighMatchDistM);
        for (int t = 0; t < nRows; ++t) {
            const int d = hungarian_.assign[t];
            if (d >= 0) {
                updateTrack(tracks_[t], ego, detections[highIdxScratch_[d]], false);
                trackMatchedScratch_[t] = 1;
                highDetMatchedScratch_[d] = 1;
            }
        }
    }

    // Stage 2 (ByteTrack): Low-Confidence matching for unmatched active/lost tracks
    unmatchedScratch_.clear();
    for (size_t t = 0; t < tracks_.size(); ++t) {
        if (!trackMatchedScratch_[t] && tracks_[t].pub.state != TrackState::Tentative) {
            unmatchedScratch_.push_back(static_cast<int>(t));
        }
    }
    if (!unmatchedScratch_.empty() && !lowIdxScratch_.empty()) {
        const int nRows = static_cast<int>(unmatchedScratch_.size());
        const int nCols = static_cast<int>(lowIdxScratch_.size());
        costFlatScratch_.resize(static_cast<size_t>(nRows) * nCols);
        for (int r = 0; r < nRows; ++r) {
            const int t = unmatchedScratch_[r];
            for (int d = 0; d < nCols; ++d) {
                const auto& det = detections[lowIdxScratch_[d]];
                const float dist = effectiveMatchDist(t, det);
                // Enforce semantic class consistency in Stage-2 association so pedestrians cannot hijack lost vehicles.
                const float classGate = (tracks_[t].pub.semantic == det.semantic) ? 0.0f : (cfg_.btLowMatchDistM + 100.0f);
                costFlatScratch_[static_cast<size_t>(r) * nCols + d] = dist + classGate;
            }
        }
        hungarian_.solve(costFlatScratch_.data(), nRows, nCols, nCols, cfg_.btLowMatchDistM);
        for (int r = 0; r < nRows; ++r) {
            const int d = hungarian_.assign[r];
            if (d >= 0) {
                const int t = unmatchedScratch_[r];
                updateTrack(tracks_[t], ego, detections[lowIdxScratch_[d]], true);
                trackMatchedScratch_[t] = 1;
            }
        }
    }

    for (size_t t = 0; t < tracks_.size(); ++t) {
        if (!trackMatchedScratch_[t]) {
            const float maxLostForTrack = (tracks_[t].pub.hits <= 3)
                ? std::min(cfg_.btMaxTimeLostSec, 0.38f)
                : cfg_.btMaxTimeLostSec;
            if (tracks_[t].pub.state == TrackState::Tentative) {
                tracks_[t].pub.state = TrackState::Removed;
            } else if (tracks_[t].pub.timeSinceUpdateSec > maxLostForTrack) {
                tracks_[t].pub.state = TrackState::Removed;
            } else {
                tracks_[t].pub.state = TrackState::Lost;
                const float damp = std::exp(-safeDt * 1.4f);
                tracks_[t].x[2] *= damp;
                tracks_[t].x[3] *= damp;
            }
        }
        refreshTrackPose(tracks_[t], ego);
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const InternalTrack& tr) { return tr.pub.state == TrackState::Removed; }),
        tracks_.end());

    for (size_t d = 0; d < highIdxScratch_.size(); ++d) {
        if (!highDetMatchedScratch_[d]) {
            const auto& det = detections[highIdxScratch_[d]];
            const float sameClassDupRadius = (det.semantic == SemanticClass::Pedestrian) ? 1.2f : 2.2f;
            bool duplicateOfExisting = false;
            for (const auto& tr : tracks_) {
                const float dist = std::hypot(tr.x[0] - det.worldPos.x, tr.x[1] - det.worldPos.z);
                if ((tr.pub.semantic == det.semantic && dist <= sameClassDupRadius) || dist <= 1.5f) {
                    duplicateOfExisting = true;
                    break;
                }
                const RayDelta ray = computeRayDelta(ego, tr.x[0], tr.x[1], det.worldPos.x, det.worldPos.z);
                if (ray.valid) {
                    if (tr.pub.semantic != det.semantic) {
                        const float maxJumpRad = std::min(8.5f, std::max(6.0f, 0.92f * ray.minRange));
                        if (ray.dLat <= 1.25f && ray.dRad <= maxJumpRad) {
                            duplicateOfExisting = true;
                            break;
                        }
                    } else if (det.semantic == SemanticClass::Pedestrian &&
                               tr.pub.state != TrackState::Tentative &&
                               ray.minRange > 12.0f && ray.minRange < 55.0f) {
                        const float pedSpreadRad = std::min(9.0f, 0.32f * ray.maxRange);
                        if (ray.dLat <= 1.25f && ray.dRad <= pedSpreadRad) {
                            duplicateOfExisting = true;
                            break;
                        }
                    } else if (det.semantic == SemanticClass::Vehicle &&
                               tr.pub.state != TrackState::Tentative &&
                               ray.minRange > 4.5f && ray.minRange < 55.0f) {
                        const float vehSpreadRad = std::min(11.5f, std::max(8.8f, 0.38f * ray.maxRange));
                        if (ray.dLat <= 1.60f && ray.dRad <= vehSpreadRad) {
                            duplicateOfExisting = true;
                            break;
                        }
                    }
                }
            }
            if (!duplicateOfExisting) {
                const size_t before = tracks_.size();
                spawnTrack(det);
                if (tracks_.size() > before) {
                    refreshTrackPose(tracks_.back(), ego);
                }
            }
        }
    }
}

void ByteTracker::collectActiveTracks(std::vector<Track3D>& out) const {
    out.clear();
    for (const auto& tr : tracks_) {
        if (tr.pub.state == TrackState::Tracked || tr.pub.state == TrackState::Lost) {
            out.push_back(tr.pub);
        }
    }
}

std::vector<Track3D> ByteTracker::activeTracks() const {
    std::vector<Track3D> out;
    out.reserve(tracks_.size());
    collectActiveTracks(out);
    return out;
}

void ByteTracker::exportTrackPriorsGpu(const EgoState& ego, std::vector<TrackPriorGpu>& out) const {
    out.assign(static_cast<size_t>(cfg_.btMaxTracks), TrackPriorGpu{});
    int idx = 0;
    const float s = std::sin(ego.yaw), c = std::cos(ego.yaw);
    for (const auto& tr : tracks_) {
        if (idx >= cfg_.btMaxTracks) break;
        if (tr.pub.state != TrackState::Tracked && tr.pub.state != TrackState::Lost) continue;
        const float dx = tr.x[0] - ego.pos.x;
        const float dz = tr.x[1] - ego.pos.z;
        const float egoX = dx * s + dz * c;
        const float egoY = dx * c - dz * s;

        TrackPriorGpu g{};
        g.posVel = {egoX, egoY, tr.x[2], tr.x[3]};
        g.sizeMeta = {tr.pub.halfExtents.x, tr.pub.halfExtents.z,
                      static_cast<float>(static_cast<int>(tr.pub.semantic)), tr.pub.posSigmaM};
        g.yawValid = {tr.pub.yaw, tr.pub.score, static_cast<float>(tr.pub.trackId), 1.0f};
        out[idx++] = g;
    }
}

std::vector<TrackPriorGpu> ByteTracker::exportTrackPriorsGpu(const EgoState& ego) const {
    std::vector<TrackPriorGpu> out;
    exportTrackPriorsGpu(ego, out);
    return out;
}

} // namespace cybercab
