#pragma once

// Describe the spherical finite-volume mesh used by gravity and collisions.
// from_edges() takes increasing physical radii and returns cell centers in
// radius and in q=r^3/3, together with shell volumes. Equal intervals in q
// have equal volume because dV=4*pi*dq. The field and collision modules use
// these same volumes when converting deposited mass to a density.

#include <cstddef>
#include <vector>

namespace venus_gpu {

/// Radial finite-volume mesh in q=r^3/3 and ordinary radius.
struct RadialMesh {
    // N cells have N+1 faces. q_center is the midpoint between volume
    // coordinates of adjacent faces. radius_center is the arithmetic midpoint
    // of the two physical radii, used for radial field samples.
    std::vector<double> edge;
    std::vector<double> q_center;
    std::vector<double> radius_center;
    std::vector<double> shell_volume;

    // Construction validates the face ordering before any CUDA allocation,
    // so deposition and field reconstruction share one consistent geometry.
    static RadialMesh from_edges(const std::vector<double>& radial_edges);
    std::size_t cells() const noexcept { return shell_volume.size(); }
};

}  // namespace venus_gpu
