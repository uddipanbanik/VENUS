#pragma once

// Compute an elastic encounter in one local orthonormal frame, with c=1 and
// metric signature (-,+,+,+). Inputs are physical material masses and future
// four-momenta. Outputs are outgoing momenta on the same mass shells.
// Statistical marker weights enter the sampling modules; they never replace
// the material mass in these two-body kinematics.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rb {

constexpr double pi = 3.141592653589793238462643383279502884;

inline void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 operator*(double a, Vec3 b) { return {a*b.x,a*b.y,a*b.z}; }
inline Vec3 operator*(Vec3 b, double a) { return a*b; }
inline Vec3 operator/(Vec3 b, double a) { return (1.0/a)*b; }
inline double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
inline double norm(Vec3 a) { return std::hypot(a.x,std::hypot(a.y,a.z)); }
inline bool finite(Vec3 a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

// Physical material four-momentum (energy, spatial momentum) in one local
// orthonormal frame. Energies and momenta share the c=1 unit convention.
struct Four {
    double t = 0.0;
    Vec3 p;
};

inline Four operator+(Four a, Four b) { return {a.t+b.t,a.p+b.p}; }
inline Four operator-(Four a, Four b) { return {a.t-b.t,a.p-b.p}; }
inline Four operator*(double a, Four b) { return {a*b.t,a*b.p}; }
inline double minkowski(Four a, Four b) { return -a.t*b.t+dot(a.p,b.p); }

// The future branch has E=sqrt(m^2+|p|^2). Nested hypot evaluates this
// relation without forming unnecessarily large squared intermediate values.
inline Four on_shell(double mass, Vec3 momentum) {
    require(mass > 0.0 && std::isfinite(mass) && finite(momentum),
            "A massive future momentum requires finite mass > 0 and momentum.");
    const double energy = std::hypot(mass,norm(momentum));
    require(std::isfinite(energy),"Momentum energy overflow.");
    return {energy,momentum};
}

// Compare the supplied energy with the future-shell energy computed from
// its spatial momentum. The tolerance resolves floating-point projection.
inline void validate_momentum(double mass, Four p) {
    const Four expected = on_shell(mass,p.p);
    require(p.t > 0.0 && std::isfinite(p.t)
            && std::abs(p.t-expected.t)
               <= 128.0*std::numeric_limits<double>::epsilon()*expected.t,
            "A collision input must be future directed and on shell.");
}

// Passive boost to the frame moving with velocity beta in the input frame.
inline Four boost(Four p, Vec3 beta) {
    const double b2 = dot(beta,beta);
    require(finite(beta) && b2 < 1.0 && std::isfinite(p.t) && finite(p.p),
            "Invalid Lorentz boost.");
    if (b2 == 0.0) {
        return p;
    }
    const double gamma = 1.0/std::sqrt(1.0-b2);
    const double bp = dot(beta,p.p);
    // gamma^2/(gamma+1) equals (gamma-1)/beta^2 at small beta.
    const double coefficient = gamma*gamma/(gamma+1.0)*bp-gamma*p.t;
    return {gamma*(p.t-bp),p.p+coefficient*beta};
}

// The pair center-of-momentum data are shared by all scattering angles.
// root_s is the invariant total energy; momentum_cm is either outgoing
// particle momentum magnitude in that frame.
struct PairGeometry {
    double root_s = 0.0;
    double momentum_cm = 0.0;
    double energy1_cm = 0.0;
    double energy2_cm = 0.0;
    Four unit_total;
    Vec3 cm_velocity;
};

inline double moller_flux_speed(Four p1, Four p2);

inline PairGeometry pair_geometry(double m1, Four p1, double m2, Four p2) {
    validate_momentum(m1,p1);
    validate_momentum(m2,p2);
    const Four total = p1+p2;
    const long double e = total.t;
    const long double px = total.p.x;
    const long double py = total.p.y;
    const long double pz = total.p.z;
    const long double measured_s = e*e-px*px-py*py-pz*pz;
    // F from velocity differences resolves co-moving small relative speeds.
    // Reconstruct p1.p2 from F and the known material masses without subtracting
    // two nearly equal large laboratory energies.
    const long double flux = static_cast<long double>(p1.t)*p2.t*moller_flux_speed(p1,p2);
    const long double product = static_cast<long double>(m1)*m2;
    const long double invariant_dot = std::hypot(product,flux);
    const long double s = static_cast<long double>(m1)*m1
                         +static_cast<long double>(m2)*m2+2.0L*invariant_dot;
    const long double threshold = static_cast<long double>(m1+m2)*(m1+m2);
    const long double tolerance = 256.0L*std::numeric_limits<double>::epsilon()
                                *std::max(s,threshold);
    require(s > 0.0L && s+ tolerance >= threshold,
            "Pair invariant lies below its two-body mass threshold.");
    require(std::isfinite(static_cast<double>(s)) && measured_s > 0.0L
            && std::abs(measured_s-s) <= 1.0e-8L*s,
            "Laboratory four-momenta cannot resolve the pair invariant accurately.");
    // The two-body threshold is (m1+m2)^2. Elastic scattering keeps the pair
    // invariant energy, so the CM momentum follows from invariant flux/sqrt(s).
    const long double physical_s = std::max(s,threshold);
    const double root_s = static_cast<double>(std::sqrt(physical_s));
    const double q = static_cast<double>(flux/root_s);
    return {root_s,q,
            std::hypot(m1,q),std::hypot(m2,q),
            (1.0/root_s)*total,total.p/total.t};
}

// The invariant flux divided by E1 E2 is the Moller flux factor. It can exceed 1.
inline double moller_flux_speed(Four p1, Four p2) {
    require(p1.t > norm(p1.p) && p2.t > norm(p2.p)
            && std::isfinite(p1.t) && std::isfinite(p2.t)
            && finite(p1.p) && finite(p2.p),
            "Flux requires finite future timelike momenta resolvable in this frame.");
    // The Moller factor is sqrt(|v1-v2|^2-|v1 cross v2|^2). It multiplies
    // number densities in the encounter rate and is a flux factor, not the
    // relative speed measured by either particle.
    const Vec3 v1 = p1.p/p1.t;
    const Vec3 v2 = p2.p/p2.t;
    const Vec3 dv = v1-v2;
    const Vec3 cv = cross(v1,v2);
    const double radicand = dot(dv,dv)-dot(cv,cv);
    require(radicand >= -64.0*std::numeric_limits<double>::epsilon(),
            "Invalid Moller flux radicand.");
    return std::sqrt(std::max(0.0,radicand));
}

inline double invariant_flux(Four p1, Four p2) {
    return p1.t*p2.t*moller_flux_speed(p1,p2);
}

struct PairOutput {
    Four first;
    Four second;
};

// direction is a unit spacelike four-vector orthogonal to the total momentum.
// Transforming this vector with both incoming momenta gives a fixed-event
// boost-equivariance test without introducing a frame-dependent angle label.
inline PairOutput scatter_with_direction(
    double m1, Four p1, double m2, Four p2, Four direction) {
    const PairGeometry g = pair_geometry(m1,p1,m2,p2);
    const double scale = std::max(1.0,std::abs(direction.t));
    require(std::abs(minkowski(direction,g.unit_total)) < 2.0e-10*scale*scale
            && std::abs(minkowski(direction,direction)-1.0) < 2.0e-10*scale*scale,
            "The outgoing direction must be unit spacelike in the pair CM plane.");
    // The total four-velocity supplies the CM energy part. Opposite spacelike
    // parts conserve the pair four-momentum for every admitted outgoing angle.
    const Four first = g.energy1_cm*g.unit_total+g.momentum_cm*direction;
    const Four second = g.energy2_cm*g.unit_total-g.momentum_cm*direction;
    const Four first_shell = on_shell(m1,first.p);
    const Four second_shell = on_shell(m2,second.p);
    // Store the mass-shell energy from the final spatial momentum. The
    // covariant construction supplies the analytic energy; their difference
    // is checked before this floating-point projection is committed.
    require(first.t > 0.0 && second.t > 0.0
            && std::abs(first.t-first_shell.t)+std::abs(second.t-second_shell.t)
               <= 1.0e-10*(p1.t+p2.t),
            "Outgoing mass-shell projection exceeds the conditioning guard.");
    return {first_shell,second_shell};
}

// The complete random stream is identified by its 64-bit state and draw
// count. Saving both lets a restart preserve the actual stochastic path.
struct Random {
    std::uint64_t state = 0;
    std::uint64_t draws = 0;

    std::uint64_t next() {
        ++draws;
        state += UINT64_C(0x9e3779b97f4a7c15);
        std::uint64_t z = state;
        z = (z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        z = (z^(z>>27))*UINT64_C(0x94d049bb133111eb);
        return z^(z>>31);
    }

    double uniform() {
        // Midpoints of 2^52 equal intervals lie strictly inside (0,1).
        return (static_cast<double>(next()>>12)+0.5)*0x1.0p-52;
    }

    // Uniform cos(theta) and uniform azimuth give uniform solid angle.
    // A uniform polar angle itself would overweight the poles.
    Vec3 unit_direction() {
        const double z = 2.0*uniform()-1.0;
        const double phi = 2.0*pi*uniform();
        const double transverse = std::sqrt(std::max(0.0,1.0-z*z));
        return {transverse*std::cos(phi),transverse*std::sin(phi),z};
    }
};

inline PairOutput scatter_isotropic(
    double m1, Four p1, double m2, Four p2, Random& random) {
    const PairGeometry g = pair_geometry(m1,p1,m2,p2);
    // The sampled direction begins in the CM frame. The inverse passive boost
    // expresses this same spacelike direction in the incoming observer frame.
    const Four direction = boost({0.0,random.unit_direction()},-1.0*g.cm_velocity);
    return scatter_with_direction(m1,p1,m2,p2,direction);
}

// Coordinate components in ds^2=-alpha^2 dt^2+A^2 dr^2+r^2 dOmega^2.
// Angular inputs/outputs are physical orthonormal components at this point.
// For ds^2=-alpha^2 dt^2+A^2 dr^2+r^2 dOmega^2, divide each physical
// component by its coordinate scale factor. The angular axes are local.
inline Four sc_tetrad_to_coordinate(
    Four p, double alpha, double A, double radius, double theta) {
    require(alpha > 0.0 && A > 0.0 && std::isfinite(alpha) && std::isfinite(A)
            && radius > 0.0 && std::isfinite(radius)
            && theta > 0.0 && theta < pi,
            "A spacelike Schwarzschild slice requires positive finite metric factors.");
    return {p.t/alpha,{p.p.x/A,p.p.y/radius,p.p.z/(radius*std::sin(theta))}};
}

// The inverse map multiplies by those scale factors, recovering the
// physical energy and momenta used by the local collision rule.
inline Four sc_coordinate_to_tetrad(
    Four p, double alpha, double A, double radius, double theta) {
    require(alpha > 0.0 && A > 0.0 && std::isfinite(alpha) && std::isfinite(A)
            && radius > 0.0 && std::isfinite(radius)
            && theta > 0.0 && theta < pi,
            "Invalid Schwarzschild metric factors.");
    return {alpha*p.t,{A*p.p.x,radius*p.p.y,radius*std::sin(theta)*p.p.z}};
}

}  // namespace rb
