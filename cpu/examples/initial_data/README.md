# Packaged CPU initial data

Initial data specify the distribution at the start of a calculation. A binary
file stores numbers in a program-defined byte format; CSV stores a table
as comma-separated text. A header describes the file contents. Phase space
combines position and velocity. An INI file is a plain-text list of settings;
the [CPU guide](../../README.md) explains how to select one.

Additional terms are defined in the [plain-language glossary](../../../docs/TERMS.md).

Developed by Uddipan Banik and Frank van den Bosch

These binary and CSV inputs supply the documented examples. A binary header
identifies its format, dimensions and coordinates. Use the matching VENUS
initial-condition or restart reader so these quantities are checked before
the numerical arrays are interpreted.

- `bp_zero_flux_gamma2p5_eds_..._discrete.bin` is the `gamma_i=2.5`
  phase-space distribution used by the expanding VP and prompt-cusp BP
  examples.
- `bp_zero_flux_b1_gamma2_...bin` and
  `bp_zero_flux_b4o3_gamma1p5_...bin` are expanding power-law grid inputs.
- `frank_core_exact_...restart.bin` is a cell-integrated stationary Frank
  state with delayed-inner-carrier data.
- `frank_core_eddington_8192.csv` tabulates the Frank equilibrium distribution
  versus binding energy.
- `frank_core_eq2_...bin`, `sidm_nfw_yukawa_corecollapse_R100.bin`, and the
  shell files support the corresponding SphDSMC, SIDM, and infall examples.

The grid dimensions and mass normalization in an INI file must match its
binary input.  VENUS checks the binary header before allocating solver state.
