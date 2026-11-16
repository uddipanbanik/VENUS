#pragma once

// Construct the local observer frame for the ingoing EF metric
//  ds^2=-a b^2 dv^2+2b dv dr+r^2 dOmega^2, with c=1.
// Inputs are a,b,r at one event and a future canonical radial momentum.
// Outputs are the orthonormal frame, the positive shell measure and the
// observer number density. This prescribed timelike frame is regular at a=0;
// it is distinct from a static observer, whose worldline need not remain timelike.

#include "spherical_collisions.hpp"

namespace eb {

// Metric coefficients in ds^2=-a*b^2*dv^2+2*b*dv*dr+r^2*dOmega^2.
struct EFPoint {
    double a = 1;
    double b = 1;
    double r = 1;
};

// Columns of the returned tetrad are the prescribed timelike observer and
// its outward radial and angular unit vectors at the same spacetime event.
inline rba::Geometry ef_frame(const EFPoint& x, double v = 0, double theta = rb::pi/2) {
    rb::require(std::isfinite(x.a) && x.b > 0 && std::isfinite(x.b) && x.r > 0
                && std::isfinite(x.r) && theta > 0 && theta < rb::pi,
                "Invalid horizon-regular EF frame point.");
    rba::Geometry g{{v,x.r,theta,0},{},{}};
    g.metric[0][0] = -x.a*x.b*x.b;
    g.metric[0][1] = g.metric[1][0] = x.b;
    g.metric[2][2] = x.r*x.r;
    g.metric[3][3] = x.r*x.r*std::sin(theta)*std::sin(theta);
    // The first two columns are U=(1/b,(a-1)/2) and e_r=(1/b,(a+1)/2).
    // Their metric norms are -1 and +1 and their scalar product is zero.
    // Neither formula divides by a at an apparent horizon.
    g.tetrad[0][0] = g.tetrad[0][1] = 1/x.b;
    g.tetrad[1][0] = (x.a-1)/2;
    g.tetrad[1][1] = (x.a+1)/2;
    g.tetrad[2][2] = 1/x.r;
    g.tetrad[3][3] = 1/(x.r*std::sin(theta));
    rba::validate_geometry(g);
    return g;
}

// On an ingoing null time slice, the future massive mass shell has k_r>0.
// Eliminating k_v leaves a 1/k_r Jacobian. The angular factors belong to
// the separate spherical measure.
inline double ef_shell_density_physical_canonical(double physical_k_r) {
    rb::require(physical_k_r > 0 && std::isfinite(physical_k_r),"Future EF k_r must be positive.");
    // dPi=dkr d²k_perp/kr. The angular-coordinate Jacobian is separate.
    return 1/physical_k_r;
}

// Contract the future momentum with the EF coordinate time direction.
// This returns k_v, which is a covariant component; the local observer
// energy instead follows from the tetrad contraction.
inline double ef_covariant_energy(const Packet& p, const Model& model, const EFPoint& x) {
    rb::require(p.r == x.r,"Momentum and metric must share an event radius.");
    (void)lift(p,model,Slice::advanced_null,0);
    const double chi = 1+2*p.eta/(p.r*p.r);
    return -0.5*model.material_mass*x.b*(x.a*p.radial+chi/p.radial);
}

// One null-slice crossing count q corresponds to q E_U/(P V0) local observer number.
inline double ef_observer_number_density(const Packet& p, const Model& model, double V0) {
    rb::require(V0 > 0 && std::isfinite(V0),"Invalid coordinate shell volume.");
    // The packet represents W/m_chi material particles crossing this null
    // slice. Multiplication by E_U/(m_chi P) supplies the observer density.
    return p.rest_weight/model.material_mass*null_D(p,model)/V0;
}

}  // namespace eb
