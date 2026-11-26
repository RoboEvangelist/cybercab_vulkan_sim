#include "perception/sota/LeWmTypes.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>

namespace cybercab::sota {
namespace {

uint32_t fnv1aUpdate(uint32_t hash, const void* data, size_t bytes) {
  const auto* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= static_cast<uint32_t>(p[i]);
    hash *= 16777619u;
  }
  return hash;
}

template <size_t N>
bool allFiniteArr(const std::array<float, N>& a) {
  for (float v : a) {
    if (!std::isfinite(v)) return false;
  }
  return true;
}

template <size_t R, size_t C>
bool allFiniteMat(const std::array<std::array<float, C>, R>& m) {
  for (const auto& row : m) {
    if (!allFiniteArr(row)) return false;
  }
  return true;
}

struct WeightFileHeader {
  uint32_t magic = kLeWmWeightMagic;
  uint32_t version = kLeWmWeightVersion;
};

struct WeightFilePayload {
  std::array<std::array<float, kLeWmLatentDim>, kLeWmLatentDim> dynA{};
  std::array<std::array<float, kLeWmActionDim>, kLeWmLatentDim> dynB{};
  LatentVec16 dynBias{};
  std::array<std::array<float, kLeWmLatentDim>, kLeWmLatentDim> projW{};
  std::array<std::array<float, kLeWmLatentDim>, kLeWmMacroDim> macroW{};
  LatentVec16 sigregMean{};
  LatentVec16 sigregInvStd{};
};

WeightFilePayload toPayload(const LeWmWeights& w) {
  WeightFilePayload p{};
  p.dynA = w.dynA;
  p.dynB = w.dynB;
  p.dynBias = w.dynBias;
  p.projW = w.projW;
  p.macroW = w.macroW;
  p.sigregMean = w.sigregMean;
  p.sigregInvStd = w.sigregInvStd;
  return p;
}

void fromPayload(const WeightFilePayload& p, LeWmWeights& w) {
  w.dynA = p.dynA;
  w.dynB = p.dynB;
  w.dynBias = p.dynBias;
  w.projW = p.projW;
  w.macroW = p.macroW;
  w.sigregMean = p.sigregMean;
  w.sigregInvStd = p.sigregInvStd;
}

}  // namespace

LeWmWeights LeWmWeights::defaultAnalytical() {
  LeWmWeights w{};
  for (size_t i = 0; i < kLeWmLatentDim; ++i) {
    w.projW[i][i] = 1.0f;
    w.dynA[i][i] = 0.96f;
    w.sigregMean[i] = 0.0f;
    w.sigregInvStd[i] = 1.0f;
  }
  // Longitudinal dynamics over nominal 0.25s step (cruise ~ 10.5 m/s)
  w.dynA[0][0] = 0.99f;
  w.dynB[0][0] = 0.25f / 10.5f;  // accel -> speedNorm
  w.dynA[1][1] = 0.99f;
  w.dynA[1][2] = 0.12f;
  w.dynB[1][1] = 0.25f / 1.95f;  // latVel -> laneNorm
  w.dynA[4][4] = 0.98f;
  w.dynA[4][5] = 0.12f;          // closing speed grows sameInvGap
  w.dynB[4][0] = 0.018f;         // positive accel increases gap threat
  w.dynA[5][5] = 0.94f;
  w.dynB[5][0] = 0.25f / 12.0f;  // positive accel increases closing rate
  w.dynA[6][6] = 0.97f;
  w.dynA[6][0] = 0.06f;          // faster speed shrinks static margin
  w.dynB[6][0] = 0.012f;
  w.dynA[8][8] = 0.96f;
  w.dynA[9][9] = 0.98f;
  w.dynA[10][10] = 0.94f;
  w.dynA[12][12] = 0.96f;
  w.dynB[12][0] = 0.010f;
  w.dynA[13][13] = 0.97f;

  // Coarse 8D macro projection (slow tactical abstractions over 16D state)
  w.macroW[0][0] = 0.70f;
  w.macroW[0][7] = 0.30f;
  w.macroW[1][1] = 1.00f;
  w.macroW[2][4] = 0.55f;
  w.macroW[2][6] = 0.35f;
  w.macroW[2][5] = 0.25f;
  w.macroW[3][8] = 0.45f;
  w.macroW[3][9] = 0.65f;
  w.macroW[3][10] = 0.55f;
  w.macroW[4][4] = 0.50f;
  w.macroW[4][6] = 0.30f;
  w.macroW[4][8] = -0.45f;
  w.macroW[4][9] = -0.75f;
  w.macroW[4][10] = -0.60f;
  w.macroW[5][12] = 0.65f;
  w.macroW[5][13] = 0.60f;
  w.macroW[6][14] = 1.00f;
  w.macroW[7][15] = 1.00f;

  w.checksum = w.computeChecksum();
  return w;
}

uint32_t LeWmWeights::computeChecksum() const {
  const WeightFilePayload p = toPayload(*this);
  return fnv1aUpdate(2166136261u, &p, sizeof(p));
}

bool LeWmWeights::isFiniteValid() const {
  if (!allFiniteMat(dynA) || !allFiniteMat(dynB) || !allFiniteArr(dynBias) ||
      !allFiniteMat(projW) || !allFiniteMat(macroW) ||
      !allFiniteArr(sigregMean) || !allFiniteArr(sigregInvStd)) {
    return false;
  }
  for (float s : sigregInvStd) {
    if (s <= 0.0f) return false;
  }
  return true;
}

bool LeWmWeights::saveBinary(const std::string& path) const {
  if (!isFiniteValid()) return false;
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  if (!ofs.good()) return false;
  const WeightFileHeader hdr{};
  const WeightFilePayload p = toPayload(*this);
  const uint32_t cs = computeChecksum();
  ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  ofs.write(reinterpret_cast<const char*>(&p), sizeof(p));
  ofs.write(reinterpret_cast<const char*>(&cs), sizeof(cs));
  return ofs.good();
}

bool LeWmWeights::loadBinary(const std::string& path, LeWmWeights& outWeights) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs.good()) return false;
  const std::streamsize expectedSize = static_cast<std::streamsize>(
    sizeof(WeightFileHeader) + sizeof(WeightFilePayload) + sizeof(uint32_t));
  if (ifs.tellg() != expectedSize) return false;
  ifs.seekg(0, std::ios::beg);

  WeightFileHeader hdr{};
  WeightFilePayload p{};
  uint32_t cs = 0u;
  ifs.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
  ifs.read(reinterpret_cast<char*>(&p), sizeof(p));
  ifs.read(reinterpret_cast<char*>(&cs), sizeof(cs));
  if (!ifs.good() || hdr.magic != kLeWmWeightMagic ||
      hdr.version != kLeWmWeightVersion) {
    return false;
  }
  LeWmWeights cand{};
  fromPayload(p, cand);
  cand.checksum = cand.computeChecksum();
  if (cand.checksum != cs || !cand.isFiniteValid()) return false;
  outWeights = cand;
  return true;
}

}  // namespace cybercab::sota
