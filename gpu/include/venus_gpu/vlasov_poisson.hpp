#pragma once

// Advance spherical collisionless motion in a self-consistent Newtonian field.
// The input particle state carries radius r, radial peculiar velocity w, and
// eta=L^2/2, where L is specific angular momentum. Deposition converts particle
// weights into shell moments; a spherical Poisson solve returns potential and
// its radial derivative. Kick-drift-kick then updates the orbits while keeping
// L fixed between encounters. Options specify the mesh assignment, physical
// background, central mass, and boundary treatment. Device fields persist
// between calls; explicit download produces host diagnostics and output.

#include "venus_gpu/radial_mesh.hpp"
#include "venus_gpu/state.hpp"

#include <memory>
#include <vector>

namespace venus_gpu {

enum class ParticleShape { cic, tsc };
enum class InnerBoundary { reflecting_wall, origin_parity };
enum class OuterBoundary { reflecting_wall, outflow };

// A power-law scale factor has H=alpha/(t+time_offset). This supplies
// Hubble drift and drag in peculiar-velocity variables when enabled.
struct Expansion {
    bool enabled = false;
    double alpha = 0.0;
    double time_offset = 1.0;
};

struct VlasovPoissonOptions {
    double gravity = 1.0;
    double central_mass = 0.0;
    ParticleShape particle_shape = ParticleShape::cic;
    InnerBoundary inner_boundary = InnerBoundary::reflecting_wall;
    OuterBoundary outer_boundary = OuterBoundary::reflecting_wall;
    Expansion expansion{};
    // Subtract a homogeneous density only when the marker weights still
    // include it. Density-contrast weights already represent the Poisson source.
    bool subtract_homogeneous_background = false;
    // Retain the unresolved central fraction of active particle clouds in
    // the source ledger and the interior continuation of the radial field.
    bool carry_active_mass_inside_inner_face = false;
};

// These extensive masses distinguish resolved deposition, unresolved
// interior mass, and escaped particles, so their sum can be checked.
struct DepositionDiagnostics {
    double resolved_active_mass = 0.0;
    double deposited_mesh_mass = 0.0;
    double hidden_inner_mass = 0.0;
    double escaped_mass = 0.0;
    double mass_roundoff = 0.0;
};

// Shell arrays are extensive moments before division by volume or mass.
// source_mass_edge and source_line_density define the finite-volume field;
// phi_r is dPhi/dr, so gravitational radial acceleration is -phi_r.
struct FieldHost {
    std::vector<double> shell_mass;
    std::vector<double> shell_radial_momentum;
    std::vector<double> shell_radial_second_moment;
    std::vector<double> shell_tangential_second_moment;
    std::vector<double> shell_radial_energy_flux;
    // The sum of squared deposited particle weights determines the effective
    // count, Neff=(sum weight)^2/sum(weight^2), for each shell.
    std::vector<double> shell_mass_square;
    std::vector<double> effective_marker_number;
    std::vector<double> source_mass_edge;
    std::vector<double> source_line_density;
    std::vector<double> phi;
    std::vector<double> phi_r;
    DepositionDiagnostics diagnostics{};
};

class VlasovPoissonModule {
public:
    explicit VlasovPoissonModule(const RadialMesh& mesh);
    ~VlasovPoissonModule();
    VlasovPoissonModule(VlasovPoissonModule&&) noexcept;
    VlasovPoissonModule& operator=(VlasovPoissonModule&&) noexcept;
    VlasovPoissonModule(const VlasovPoissonModule&) = delete;
    VlasovPoissonModule& operator=(const VlasovPoissonModule&) = delete;

    // Recompute the density moments and their field from one completed
    // particle state. No characteristic motion occurs in this operation.
    DepositionDiagnostics deposit_and_solve(
        const WpicState& state,
        double time,
        const VlasovPoissonOptions& options);
    // MPI supplies moments summed over ranks; solving those moments gives
    // each device the same potential before its local particle push.
    void solve_from_global_moments(
        const FieldHost& global_moments,
        double time,
        const VlasovPoissonOptions& options);
    // Split KDK around the endpoint field solve: begin applies the incoming
    // half kick and drift; finish uses the newly deposited endpoint field.
    void begin_kdk_with_current_field(
        WpicState& state,
        double time,
        double dt,
        const VlasovPoissonOptions& options);
    void finish_kdk_with_current_field(
        WpicState& state,
        double endpoint_time,
        double dt,
        const VlasovPoissonOptions& options);
    // The complete single-device path performs both field observations and
    // both half kicks, returning the final-state deposition mass ledger.
    DepositionDiagnostics advance_kdk(
        WpicState& state,
        double time,
        double dt,
        const VlasovPoissonOptions& options);
    FieldHost download_field() const;
    // Observational access to the already computed inner face potential.
    double download_inner_face_potential() const;
    const RadialMesh& mesh() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace venus_gpu
