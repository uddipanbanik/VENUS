# Relativistic material collisions

This is the C++ library reference. An adapter connects the collision routines
to a coordinate-specific trajectory solver. Elastic encounters preserve local
energy and momentum; isotropic scattering has equal probability per solid
angle. A cross section is an effective interaction area. Schwarzschild (SC)
and ingoing Eddington--Finkelstein (EF) are relativistic coordinates. A
constraint is an equation the spacetime geometry and matter must satisfy.
See the [physical explanation](../../../../docs/relativistic_boltzmann.md)
for the rate equations and the meaning of the local frame.

Additional terms are defined in the [plain-language glossary](../../../../docs/TERMS.md).

This module applies local elastic material collisions to spherical relativistic
particle distributions. It provides adapters for Schwarzschild and advanced-EF
coordinates. Each physical material particle has rest mass m_chi. A statistical
marker represents rest mass W and particle number W/m_chi, which determines
its collision-rate weight. The spherical sampler uses a constant isotropic
material cross section. The local amplitude interface accepts reciprocal scalar
amplitudes with certified bounds on their differential rates.

Include the module through `ev/relativistic_boltzmann/coupled_evolution.hpp`.
The coupled update first advances the particles and solves the Einstein
constraints. It then holds that endpoint geometry fixed during local collisions
and reconstructs the matter sources from the outgoing particles. A zero cross
section calls Einstein–Vlasov streaming directly. `eb::advance_sc` uses a
spacelike proper cell volume and lapse. `eb::advance_ef` uses the null crossing
measure, coordinate shell volume and EF metric b. Its rate bound is refreshed
when the particle state changes.

`eb::advance_cell` provides the local collision process. Its strict spatial
policy requires equal radial positions, representing a local spherical
distribution. Pairing particles across a finite radial cell requires explicit
opt-in and records their radial separation. Each accepted event scatters equal
portions of represented rest mass from the two packets and retains the
unscattered remainder of the larger packet. The rate bound on represented
particle number, random-number state, remaining optical depth to the next
proposal and event counters persist between calls. A new exponential waiting
depth replaces the used depth after each proposal.

`eb::serialize_cell` and `eb::deserialize_cell` save and restore the local
collision state together with its material model, time slice and frozen cell
geometry. A full coupled restart also needs the global marker records, mesh,
evolution controls and persistent cell clocks. A restart using the local
amplitude interface additionally identifies the amplitude model and its
parameters.

The coupled algorithm applies streaming and collisions in separate first-order
steps. Component tests check local conservation, future-directed mass-shell
momenta and covariant rates. Pairing within a finite cell approximates a local
collision by using particles at different radii. The internal event times
describe the collision process with its geometry held fixed. Extending that
description to chronological transport in an evolving spacetime requires a
coupled causality check. Full coupled restart, resolution convergence and
compact-halo evolution also require their corresponding validation. The SC
adapter retains the polar-areal stopping threshold. The EF adapter retains
every particle and checks that the outer boundary lies in isolated vacuum.

The component test checks null-rate normalization, the prescribed
horizon-crossing observer, model-bound local restart, exact collision-off
controls and collision/source feedback in both coordinate systems. These
checks establish the local collision and adapter behavior. Claims about
self-gravitating collapse require a resolved coupled evolution and its
conservation, boundary and refinement evidence.
