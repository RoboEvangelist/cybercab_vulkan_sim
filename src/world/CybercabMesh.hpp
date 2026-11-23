#pragma once
// Procedural compound-curved Tesla Cybercab (Robotaxi) coupe 3D mesh
// generator.
#include "world/WorldTypes.hpp"
#include <vector>

namespace cybercab {

/// Appends the sculpted Cybercab coupe bodywork, panoramic glass canopy, aero
/// wheels, and LEDs.
void appendCybercabVehicleMesh(std::vector<SceneVertex>& verts,
                               std::vector<uint32_t>& indices,
                               const EgoState& ego, const Mat4& egoMat,
                               const Mat4& sprungMat, bool fsdWireMode);

}  // namespace cybercab
