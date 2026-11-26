#pragma once
// Modular data types, FMEA fault flags & serialized weights for Stage-2 LeWM & H-JEPA.
#include <array>
#include <cstdint>
#include <string>

namespace cybercab::sota {

inline constexpr uint32_t kLeWmLatentDim = 16u;
inline constexpr uint32_t kLeWmMacroDim = 8u;
inline constexpr uint32_t kLeWmActionDim = 2u;
inline constexpr uint32_t kLeWmMaxRollout = 8u;
inline constexpr uint32_t kLeWmSigRegWindow = 32u;
inline constexpr float kLeWmMaxLatentNorm = 18.0f;
inline constexpr uint32_t kLeWmWeightMagic = 0x4C574D31u;  // "LWM1"
inline constexpr uint32_t kLeWmWeightVersion = 1u;

enum WmFmeaFault : uint32_t {
  kWmFaultNone = 0u,
  kWmFaultNanInput = 1u << 0,
  kWmFaultLatentCollapse = 1u << 1,
  kWmFaultRolloutDivergence = 1u << 2,
  kWmFaultWeightLoad = 1u << 3,
  kWmFaultInvalidConfig = 1u << 4,
};

enum class MacroSubgoal : int {
  KeepLane = 0,
  OvertakeLeft = 1,
  ReturnRight = 2,
  YieldSlow = 3,
  StopHold = 4,
};

inline const char* macroSubgoalName(int code) {
  switch (static_cast<MacroSubgoal>(code)) {
    case MacroSubgoal::KeepLane: return "KEEP_LANE";
    case MacroSubgoal::OvertakeLeft: return "OVERTAKE_LEFT";
    case MacroSubgoal::ReturnRight: return "RETURN_RIGHT";
    case MacroSubgoal::YieldSlow: return "YIELD_SLOW";
    case MacroSubgoal::StopHold: return "STOP_HOLD";
  }
  return "UNKNOWN";
}

using LatentVec16 = std::array<float, kLeWmLatentDim>;
using MacroVec8 = std::array<float, kLeWmMacroDim>;

struct LeWmAction {
  float accelMs2 = 0.0f;
  float latVelMs = 0.0f;
};

struct SigRegMetrics {
  float meanSqNorm = 0.0f;
  float varDeviation = 0.0f;
  float offDiagCovNorm = 0.0f;
  float sigregLoss = 0.0f;
  bool collapsed = false;
};

struct LeWmWeights {
  std::array<std::array<float, kLeWmLatentDim>, kLeWmLatentDim> dynA{};
  std::array<std::array<float, kLeWmActionDim>, kLeWmLatentDim> dynB{};
  LatentVec16 dynBias{};
  std::array<std::array<float, kLeWmLatentDim>, kLeWmLatentDim> projW{};
  std::array<std::array<float, kLeWmLatentDim>, kLeWmMacroDim> macroW{};
  LatentVec16 sigregMean{};
  LatentVec16 sigregInvStd{};
  uint32_t checksum = 0u;

  static LeWmWeights defaultAnalytical();
  uint32_t computeChecksum() const;
  bool isFiniteValid() const;
  bool saveBinary(const std::string& path) const;
  static bool loadBinary(const std::string& path, LeWmWeights& outWeights);
};

}  // namespace cybercab::sota
