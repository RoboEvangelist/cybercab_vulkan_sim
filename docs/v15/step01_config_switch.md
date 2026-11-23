# Step 1: `simulation.pipeline_version` config switch

## Scope
Add a validated config selector `v14 | v15`. **Out of scope:** any consumer of the flag (later steps).
Files: `src/config/SimConfig.hpp`, `src/config/SimConfig.cpp`, `config/cybercab_sim.yaml`, `tests/test_v15_config.cpp`.

## Premises
- **P1** `SimConfig.hpp:100`: `std::string pipelineVersion = "v14";`, so the code default is v14.
- **P2** `SimConfig.hpp:101`: `isV15()` returns true iff the string equals `"v15"`.
- **P3** `SimConfig.cpp:78`: YAML `simulation.pipeline_version` is read via `readOr`, falling back to P1.
- **P4** `SimConfig.cpp:210-212`: any value other than `v14`/`v15` throws `std::runtime_error`.
- **P5** `cybercab_sim.yaml`: ships `pipeline_version: "v14"`.
- **P6** At this commit no source file outside `SimConfig.*` references `pipelineVersion` or `isV15`.

## Trace
- **T1 (v14 / shipped YAML):** P3 reads `"v14"`, P4 passes, field = `"v14"`. By P6 nothing reads it.
- **T2 (missing key, older YAMLs):** P3 falls back to P1, giving `"v14"`.
- **T3 (`"v99"`):** P4 throws before the config is returned.

## Claims
- **C-EQ:** by T1/T2 + P6, every runtime path is unchanged for v14 and for YAMLs without the key.
- **C-V15:** `v15` is accepted, other config sections are unaffected, and invalid versions are rejected.

## Evidence
- **E1** `V15ConfigTest.ShippedYamlDefaultsToV14`: PASS
- **E2** `V15ConfigTest.YamlCanSelectV15`: PASS (8 cameras still loaded)
- **E3** `V15ConfigTest.UnknownVersionIsRejected`: PASS
- **E4** ctest: **99/99** pass.

## Known limits
None. This is pure configuration.
