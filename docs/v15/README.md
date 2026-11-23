# FSD v15 Change Certificates

Every v15 step is committed together with a **semi-formal change certificate**, following
*Agentic Code Reasoning* (Ugare & Chandra, Meta, arXiv:2603.01896). That paper replaces free-form
reasoning with a certificate the author cannot fill in by guessing. Here the certificate has:

1. **Scope**: one feature, the files touched, and what is explicitly *out* of scope.
2. **Premises (P#)**: concrete facts with `file:line` citations, no claims from function names alone.
3. **Trace (T#)**: symbolic execution of the changed paths, done **separately for `v14` and `v15`**.
4. **Claims (C#)**:
   - **C-EQ (patch equivalence)**: with `pipeline_version: v14`, observable outputs are *identical* to the parent commit, derived only from P/T.
   - **C-V15**: the new behavior, stated as measurable properties.
5. **Evidence (E#)**: test names and the observed numbers (gtest `RecordProperty` values / ctest totals).
   A claim without a test stays marked **UNVERIFIED**.
6. **Known limits / risks**: what is approximate, analog-only, or deferred.

## Step index

| Step | Module | Certificate | Commit |
|---|---|---|---|
| 1 | Config switch `pipeline_version` | [step01_config_switch.md](step01_config_switch.md) | `84b73e4` |
| 2 | M1a ISP-bypass linear 14-bit photon tensor | [step02_m1a_linear_photon_tensor.md](step02_m1a_linear_photon_tensor.md) | `47f26e2` |
| 3 | M1b Poisson-Gaussian sensor noise | [step03_m1b_sensor_noise.md](step03_m1b_sensor_noise.md) | this commit |

## Open UNVERIFIED items
- step03 R1: read-noise floor `b` only loosely identified (needs dark-scene test).
- step02 R2: v14 perception quality under `pipeline_version: v15` not yet re-validated.

## Regression gate (every step)
- `ctest --test-dir build` must show **100%** pass.
- The 96 pre-v15 tests are the v14 equivalence oracle; they run with `pipeline_version: v14`.
- New files are kept to **350 lines or fewer**.
