#pragma once

#include "config/SimConfig.hpp"
#include "perception/PerceptionTypes.hpp"
#include "world/WorldTypes.hpp"
#include <array>
#include <vector>

namespace cybercab {

class ByteTracker {
public:
    explicit ByteTracker(const SimConfig& cfg);

    void reset();
    void update(float dt, const EgoState& ego, const std::vector<Detection3D>& detections);

    void collectActiveTracks(std::vector<Track3D>& out) const;
    std::vector<Track3D> activeTracks() const;
    void exportTrackPriorsGpu(const EgoState& ego, std::vector<TrackPriorGpu>& out) const;
    std::vector<TrackPriorGpu> exportTrackPriorsGpu(const EgoState& ego) const;

    // Optimal assignment via Kuhn-Munkres Hungarian algorithm (reusing scratch when provided)
    static std::vector<int> solveHungarian(const std::vector<std::vector<float>>& costMatrix, float gateThreshold);

private:
    struct InternalTrack {
        Track3D pub;
        std::array<float, 4> x{};
        std::array<float, 16> P{};
        std::array<float, 4> classVotes{}; // index: 1=Vehicle, 2=Pedestrian, 3=SignalSign
        Vec3 lastDetPos{};
        int lastCamIndex = -1;
        float lastDt = 1.0f / 36.0f;
    };

    struct HungarianWorkspace {
        std::vector<double> a; // flattened (n+1)*(n+1)
        std::vector<double> u, v, minv;
        std::vector<int> p, way, assign;
        std::vector<uint8_t> used;

        void solve(const float* flatCost, int nRows, int nCols, int strideCols, float gateThreshold);
    };

    const SimConfig& cfg_;
    std::vector<InternalTrack> tracks_;
    int nextTrackId_ = 1;

    // Pre-allocated hot-path scratch buffers (zero heap allocation after construction)
    std::vector<int> highIdxScratch_;
    std::vector<int> lowIdxScratch_;
    std::vector<int> unmatchedScratch_;
    std::vector<uint8_t> trackMatchedScratch_;
    std::vector<uint8_t> highDetMatchedScratch_;
    std::vector<float> costFlatScratch_;
    HungarianWorkspace hungarian_;

    void predictTrack(InternalTrack& tr, float dt) const;
    void updateTrack(InternalTrack& tr, const EgoState& ego, const Detection3D& det, bool lowConfMatch);
    void spawnTrack(const Detection3D& det);
    static void refreshTrackPose(InternalTrack& tr, const EgoState& ego);
};

} // namespace cybercab
