#pragma once

// Evaluate field-change contributions for particle mass inside the mesh face.
// Inputs are a particle weight and radius, the inner mesh geometry, and the
// previous/current interior fields. terms() returns mass-weighted squares of
// potential, radial field derivative, and their changes, plus mass counts.
// The uniform interior continuation assumes zero central point mass. The
// hidden cloud fraction uses exactly the deposition rounding so receiver
// weights and gravitational source mass close the same ledger.

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define VENUS_RECEIVER_HD __host__ __device__
#else
#define VENUS_RECEIVER_HD
#endif

namespace venus_gpu_receiver {
// The inner-face potential fixes the additive potential constant. The
// enclosed source mass fixes the uniform-interior force coefficient.
struct InnerField {
    double inner_radius;
    double inner_face_potential;
    double inner_source_mass;
    double gravity;
};

// Each channel is an extensive sum, ready for a fixed-order reduction.
// physical_mass2 also records the unequal-weight effective population.
struct ReceiverTerms {
    double physical_mass;
    double physical_mass2;
    double hidden_mass;
    double phi0_square;
    double phi1_square;
    double phi_difference_square;
    double force0_square;
    double force1_square;
    double force_difference_square;
    std::uint64_t active_count;
    std::uint64_t hidden_count;
};

// Combine independent receiver contributions without changing their
// normalization; normalization follows the complete mass-closure check.
VENUS_RECEIVER_HD inline ReceiverTerms add(ReceiverTerms a, const ReceiverTerms& b) {
    a.physical_mass += b.physical_mass;
    a.physical_mass2 += b.physical_mass2;
    a.hidden_mass += b.hidden_mass;
    a.phi0_square += b.phi0_square;
    a.phi1_square += b.phi1_square;
    a.phi_difference_square += b.phi_difference_square;
    a.force0_square += b.force0_square;
    a.force1_square += b.force1_square;
    a.force_difference_square += b.force_difference_square;
    a.active_count += b.active_count;
    a.hidden_count += b.hidden_count;
    return a;
}

// Match the deposition kernel's separately rounded resolved fraction.
// Near the inner face, fusing its complement changes a tiny hidden weight.
VENUS_RECEIVER_HD inline double deposition_rounded_hidden_fraction(double s) {
#if defined(__CUDA_ARCH__)
    const double square = __dmul_rn(s,s);
    const double shape = __dsub_rn(3.0,__dadd_rn(s,s));
    const double resolved = __dmul_rn(square,shape);
    return __dsub_rn(1.0,resolved);
#else
    volatile double square = s*s;
    volatile double shape = 3.0-(s+s);
    volatile double resolved = square*shape;
    return 1.0-resolved;
#endif
}

// TSC deposition smoothly divides a cloud between resolved cells and the
// unresolved interior. q=r^3/3 makes this division a volume-coordinate weight.
VENUS_RECEIVER_HD inline double hidden_fraction(double radius, double inner, double qcenter0) {
    if (!(radius < inner)) return 0;
    const double face_q = inner * inner * inner / 3.0;
    const double first_width = 2.0 * (qcenter0 - face_q);
    const double transition = fmin(first_width, face_q);
    const double q = radius * radius * radius / 3.0;
    const double s = fmin(1.0, fmax(0.0, (q - (face_q - transition)) / transition));
    return deposition_rounded_hidden_fraction(s);
}

VENUS_RECEIVER_HD inline double potential(double r, const InnerField& f) {
    // The supported uniform interior continuation has zero central point mass.
    const double coefficient = f.gravity * f.inner_source_mass /
        (f.inner_radius * f.inner_radius * f.inner_radius);
    return f.inner_face_potential + .5 * coefficient *
        (r * r - f.inner_radius * f.inner_radius);
}

// Return dPhi/dr; the orbit equation uses its negative as acceleration.
VENUS_RECEIVER_HD inline double force(double r, const InnerField& f) {
    return f.gravity * f.inner_source_mass * r /
        (f.inner_radius * f.inner_radius * f.inner_radius);
}

// Every active particle contributes to the physical mass ledger. Only its
// positive hidden cloud weight contributes these interior field moments.
VENUS_RECEIVER_HD inline ReceiverTerms terms(double mass, double radius,
        double qcenter0, const InnerField& previous, const InnerField& current) {
    ReceiverTerms x{};
    x.physical_mass = mass;
    x.physical_mass2 = mass * mass;
    x.active_count = 1;
    const double beta = mass * hidden_fraction(radius, current.inner_radius, qcenter0);
    if (!(beta > 0)) return x;
    const double p0 = potential(radius, previous);
    const double p1 = potential(radius, current);
    const double g0 = force(radius, previous);
    const double g1 = force(radius, current);
    x.hidden_mass = beta;
    x.phi0_square = beta * p0 * p0;
    x.phi1_square = beta * p1 * p1;
    x.phi_difference_square = beta * (p1-p0) * (p1-p0);
    x.force0_square = beta * g0 * g0;
    x.force1_square = beta * g1 * g1;
    x.force_difference_square = beta * (g1-g0) * (g1-g0);
    x.hidden_count = 1;
    return x;
}
} // namespace venus_gpu_receiver
#undef VENUS_RECEIVER_HD
