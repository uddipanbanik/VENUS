#pragma once

// Evaluate finite-angle elastic Boltzmann scattering on spherical velocity cells.
// Inputs are f(r,w,L), velocity quadrature, local shell radius, collision law,
// angular resolution, and elapsed time. Here w is radial velocity and L is
// specific angular momentum, so v_t=L/r and d^3v=2*pi*L*dL*dw/r^2.
// Shared helpers construct 3-D pair velocities, total/differential rates,
// outgoing directions, remap weights, and collision loss times. The direct
// backend integrates gain/loss over angular quadrature, updates f, and returns
// substep and unrepresented-velocity budgets. Its optional moment repair
// addresses direct-remap defects; reversible backends use their own discrete
// reaction construction and conservation/entropy conditions.
// Local radial shells are independent. This direct backend uses serial or
// threaded host execution and an untraced physical distribution.

#include <array>
#include <sstream>

#include "ev/grids.hpp"
#include "ev/mpi.hpp"
#include "ev/parallel.hpp"

namespace ev {

// The x axis is local radial direction; y and z span the tangent plane
// reconstructed during an encounter in a spherically symmetric shell.
struct BPFAVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

// One active reduced velocity cell carries n=f*velocity_volume, its local
// radial/tangential speeds, and its original shell-grid destination index.
struct BPFAVelocityCell {
    int j = 0;
    int k = 0;
    int shell_index = 0;
    double wr = 0.0;
    double vt = 0.0;
    double volume = 0.0;
    double n = 0.0;
};

// A bilinear (w,eta) stencil has at most four positive destination weights,
// with eta=L^2/2 evaluated from the outgoing tangential velocity.
struct BPFARemapStencil {
    int index[4] = {0, 0, 0, 0};
    double weight[4] = {0.0, 0.0, 0.0, 0.0};
    int count = 0;
};

struct BPFAShellMoments {
    double mass = 0.0;
    double radial_momentum = 0.0;
    double kinetic_energy = 0.0;
};

/// Tracks finite-angle outcomes that cannot be represented on the velocity grid.
struct BPFARemapBudget {
    double attempted_event_density = 0.0;
    double skipped_event_density = 0.0;
    double skipped_w_event_density = 0.0;
    double skipped_L_event_density = 0.0;
    double max_shell_skipped_fraction = 0.0;
};

// Event-density budgets add over shells/substeps; the worst skipped
// fraction remains a maximum so one unresolved shell stays visible.
inline void bp_fa_merge_remap_budget(BPFARemapBudget& total, const BPFARemapBudget& part) {
    total.attempted_event_density += part.attempted_event_density;
    total.skipped_event_density += part.skipped_event_density;
    total.skipped_w_event_density += part.skipped_w_event_density;
    total.skipped_L_event_density += part.skipped_L_event_density;
    total.max_shell_skipped_fraction =
        std::max(total.max_shell_skipped_fraction, part.max_shell_skipped_fraction);
}

// Small vector operations keep incoming relative motion, center-of-mass
// motion, and outgoing rotation in the same local Cartesian frame.
inline BPFAVec3 bp_fa_add(BPFAVec3 a, BPFAVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline BPFAVec3 bp_fa_sub(BPFAVec3 a, BPFAVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline BPFAVec3 bp_fa_scale(BPFAVec3 a, double s) {
    return {s * a.x, s * a.y, s * a.z};
}

inline double bp_fa_dot(BPFAVec3 a, BPFAVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline BPFAVec3 bp_fa_cross(BPFAVec3 a, BPFAVec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double bp_fa_norm(BPFAVec3 a) {
    return std::sqrt(std::max(0.0, bp_fa_dot(a, a)));
}

inline BPFAVec3 bp_fa_unit(BPFAVec3 a) {
    const double n = bp_fa_norm(a);
    if (!(n > 0.0)) {
        return {1.0, 0.0, 0.0};
    }
    return bp_fa_scale(a, 1.0 / n);
}

// The auto angular law follows the selected physical cross section:
// Born uses its directional kernel; the other total laws use isotropic angles.
inline std::string bp_fa_kernel_name(const Params& p) {
    if (p.bp_boltzmann_kernel == "auto") {
        return (p.bp_cross_section == "yukawa_born") ? "yukawa_born" : "isotropic";
    }
    return p.bp_boltzmann_kernel;
}

// Evaluate the selected total rate area at relative speed, including its
// explicit strength and low-speed regularization or saturation.
inline double bp_fa_total_cross_section(double u, const Params& p) {
    const double us = std::max(std::abs(u), 1.0e-14);
    if (p.bp_cross_section == "constant") {
        return p.bp_collision_strength * p.bp_sigma0;
    }
    if (p.bp_cross_section == "powerlaw") {
        const double u2 = us * us + p.bp_sigma_u0 * p.bp_sigma_u0;
        return p.bp_collision_strength * p.bp_sigma0 * std::pow(u2, -0.5 * p.bp_sigma_n);
    }
    if (p.bp_cross_section == "softened_gravity") {
        // Match the FP softened-gravity transfer cross section.  With the
        // isotropic finite-angle kernel, sigma_total is also sigma_T.
        const double eps = p.bp_subhalo_radius;
        const double m = p.bp_subhalo_mass;
        const double sigma_sat = (p.bp_sigma0 > 0.0) ? p.bp_sigma0 : (2.0 * PI * eps * eps);
        const double high = 4.0 * PI * p.vp_gravity * p.vp_gravity * m * m
            * std::max(p.bp_coulomb_log, 0.0) / std::pow(us, 4);
        if (!(high > 0.0)) {
            return 0.0;
        }
        return p.bp_collision_strength / (1.0 / sigma_sat + 1.0 / high);
    }
    if (p.bp_cross_section == "yukawa_born") {
        const double alpha = p.bp_yukawa_alpha;
        const double mchi = p.bp_yukawa_mchi;
        const double mphi = p.bp_yukawa_mphi;
        const double mphi2 = mphi * mphi;
        const double x = (mchi * mchi * us * us) / mphi2;
        const double denom = mphi2 * mphi2 * (1.0 + x);
        if (!(denom > 0.0)) {
            return 0.0;
        }
        return p.bp_collision_strength * 4.0 * PI * alpha * alpha * mchi * mchi / denom;
    }
    return 0.0;
}

// The Born differential rate depends on mu=cos(theta) through momentum
// transfer. Integrating this solid-angle law gives the total area above.
inline double bp_fa_yukawa_born_dsigma_domega(double mu, double u, const Params& p) {
    const double us = std::max(std::abs(u), 1.0e-14);
    const double alpha = p.bp_yukawa_alpha;
    const double mchi = p.bp_yukawa_mchi;
    const double mphi = p.bp_yukawa_mphi;
    const double denom = mphi * mphi + 0.5 * mchi * mchi * us * us * (1.0 - mu);
    if (!(denom > 0.0)) {
        return 0.0;
    }
    return p.bp_collision_strength * alpha * alpha * mchi * mchi / (denom * denom);
}

// Inverse of the Yukawa angular survival probability Pr(cos(theta) >= mu).
// Inputs are finite nonnegative speed, positive finite shape parameters, and
// a unit deviate. Factoring the integrated law avoids subtracting inverse
// denominators when scattering is almost isotropic. The reciprocal branch
// also keeps the expression bounded in the strongly forward-peaked limit.
inline double bp_fa_yukawa_born_mu_from_unit(
    double relative_speed, const Params& p, double unit) {
    const double z = std::clamp(unit, 0.0, 1.0);
    if (z <= 0.0) return 1.0;
    if (z >= 1.0) return -1.0;
    const double speed_ratio =
        std::abs(relative_speed) * p.bp_yukawa_mchi / p.bp_yukawa_mphi;
    const double y = speed_ratio * speed_ratio;
    double one_minus_mu;
    if (y <= 1.0) {
        one_minus_mu = 2.0 * z / (1.0 + y * (1.0 - z));
    } else {
        const double inverse_y = 1.0 / y;
        one_minus_mu = 2.0 * z * inverse_y / (inverse_y + (1.0 - z));
    }
    return std::clamp(1.0 - one_minus_mu, -1.0, 1.0);
}

// Direct angular quadrature later normalizes these weights to unit
// angular probability, while the total cross section sets event amplitude.
inline double bp_fa_scatter_weight_raw(double mu, double u, const Params& p, const std::string& kernel) {
    if (kernel == "yukawa_born") {
        return bp_fa_yukawa_born_dsigma_domega(mu, u, p);
    }
    return 1.0;
}

// Choose a helper separated from the incoming direction and construct
// two orthonormal transverse vectors for its polar/azimuthal scattering frame.
inline std::array<BPFAVec3, 2> bp_fa_perpendicular_basis_unit(BPFAVec3 uhat) {
    const BPFAVec3 helper = (std::abs(uhat.x) < 0.75) ? BPFAVec3{1.0, 0.0, 0.0} : BPFAVec3{0.0, 1.0, 0.0};
    const BPFAVec3 e1 = bp_fa_unit(bp_fa_cross(uhat, helper));
    const BPFAVec3 e2 = bp_fa_cross(uhat, e1);
    return {e1, e2};
}

inline std::array<BPFAVec3, 2> bp_fa_perpendicular_basis(BPFAVec3 axis) {
    return bp_fa_perpendicular_basis_unit(bp_fa_unit(axis));
}

// Rotate relative velocity about its own incoming direction; this retains
// the forward-scattering meaning of mu for an anisotropic angular law.
inline BPFAVec3 bp_fa_scattered_relative_direction(BPFAVec3 incoming, double mu, double psi) {
    const BPFAVec3 uhat = bp_fa_unit(incoming);
    const std::array<BPFAVec3, 2> basis = bp_fa_perpendicular_basis(uhat);
    const double sint = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    return bp_fa_add(
        bp_fa_scale(uhat, mu),
        bp_fa_scale(
            bp_fa_add(bp_fa_scale(basis[0], std::cos(psi)), bp_fa_scale(basis[1], std::sin(psi))),
            sint));
}

inline BPFAVec3 bp_fa_scattered_relative_direction_from_basis(
    BPFAVec3 uhat,
    BPFAVec3 e1,
    BPFAVec3 e2,
    double mu,
    double sint,
    double cospsi,
    double sinpsi) {
    return bp_fa_add(
        bp_fa_scale(uhat, mu),
        bp_fa_scale(bp_fa_add(bp_fa_scale(e1, cospsi), bp_fa_scale(e2, sinpsi)), sint));
}

// Uniform azimuth midpoint nodes give an average over one complete circle
// while keeping quadrature samples away from duplicated periodic endpoints.
inline void bp_fa_midpoint_angle_table(int n, std::vector<double>& cosv, std::vector<double>& sinv) {
    const int nq = std::max(1, n);
    cosv.resize(static_cast<std::size_t>(nq));
    sinv.resize(static_cast<std::size_t>(nq));
    for (int q = 0; q < nq; ++q) {
        const double angle = (static_cast<double>(q) + 0.5) * 2.0 * PI / static_cast<double>(nq);
        cosv[static_cast<std::size_t>(q)] = std::cos(angle);
        sinv[static_cast<std::size_t>(q)] = std::sin(angle);
    }
}

/// Gauss-Legendre nodes and weights normalized for an average over mu in [-1,1].
inline void bp_fa_gauss_legendre_mu_table(
    int n,
    std::vector<double>& nodes,
    std::vector<double>& weights) {
    const int nq = std::max(1, n);
    nodes.assign(static_cast<std::size_t>(nq), 0.0);
    weights.assign(static_cast<std::size_t>(nq), 0.0);
    // Legendre roots occur in symmetric pairs. Newton iteration gives each
    // root and derivative; the weights below sum to one for a mu average.
    const int half = (nq + 1) / 2;
    for (int i = 0; i < half; ++i) {
        double z = std::cos(
            PI * (static_cast<double>(i) + 0.75)
            / (static_cast<double>(nq) + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 64; ++iteration) {
            double pnm1 = 1.0;
            double pn = z;
            if (nq == 1) {
                pn = z;
                pnm1 = 1.0;
            } else {
                for (int order = 2; order <= nq; ++order) {
                    const double pnext =
                        ((2.0 * order - 1.0) * z * pn
                            - (order - 1.0) * pnm1)
                        / static_cast<double>(order);
                    pnm1 = pn;
                    pn = pnext;
                }
            }
            derivative = static_cast<double>(nq) * (z * pn - pnm1)
                / (z * z - 1.0);
            const double step = pn / derivative;
            z -= step;
            if (std::abs(step) <= 8.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(z))) {
                break;
            }
        }
        const double average_weight =
            1.0 / ((1.0 - z * z) * derivative * derivative);
        nodes[static_cast<std::size_t>(i)] = -z;
        nodes[static_cast<std::size_t>(nq - 1 - i)] = z;
        weights[static_cast<std::size_t>(i)] = average_weight;
        weights[static_cast<std::size_t>(nq - 1 - i)] = average_weight;
    }
}

// Integrate the spherical tangent-plane angle at fixed r. Multiplying
// each f value by this velocity-cell measure gives its local population.
inline std::vector<double> bp_fa_shell_velocity_volumes(
    double ri,
    const std::vector<double>& wq,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    const Params& p) {
    std::vector<double> volume(static_cast<std::size_t>(p.Nw * p.NL), 0.0);
    const double factor = 2.0 * PI / std::max(ri * ri, 1.0e-300);
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            const double vcell = factor
                * wq[static_cast<std::size_t>(j)]
                * Lq[static_cast<std::size_t>(k)]
                * std::max(L[static_cast<std::size_t>(k)], 0.0);
            volume[static_cast<std::size_t>(j * p.NL + k)] = std::max(0.0, vcell);
        }
    }
    return volume;
}

// Integrate 1, w, and (w^2+v_t^2)/2 against the same local velocity
// measure, supplying the invariants targeted by the optional shell repair.
inline BPFAShellMoments bp_fa_shell_moments(
    const std::vector<double>& f,
    int i,
    double ri,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& volume,
    const Params& p) {
    BPFAShellMoments m;
    for (int j = 0; j < p.Nw; ++j) {
        const double wr = w[static_cast<std::size_t>(j)];
        for (int k = 0; k < p.NL; ++k) {
            const int s = j * p.NL + k;
            const double vol = volume[static_cast<std::size_t>(s)];
            if (!(vol > 0.0)) {
                continue;
            }
            const double vt = L[static_cast<std::size_t>(k)] / ri;
            const double n = std::max(0.0, f[idx(i, j, k, p)]) * vol;
            const double e = 0.5 * (wr * wr + vt * vt);
            m.mass += n;
            m.radial_momentum += n * wr;
            m.kinetic_energy += n * e;
        }
    }
    return m;
}

// Build the represented collision population above the configured f floor.
// An optional cumulative low-population omission uses its explicit mass budget.
inline std::vector<BPFAVelocityCell> bp_fa_active_cells(
    const std::vector<double>& f,
    int i,
    double ri,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& volume,
    const Params& p) {
    std::vector<BPFAVelocityCell> cells;
    cells.reserve(static_cast<std::size_t>(p.Nw * p.NL));
    double total_n = 0.0;
    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            const int s = j * p.NL + k;
            const double vol = volume[static_cast<std::size_t>(s)];
            if (!(vol > 0.0)) {
                continue;
            }
            const double fq = f[idx(i, j, k, p)];
            if (!(fq > std::max(0.0, p.f_floor))) {
                continue;
            }
            const double n = fq * vol;
            if (!(n > 0.0) || !std::isfinite(n)) {
                continue;
            }
            BPFAVelocityCell c;
            c.j = j;
            c.k = k;
            c.shell_index = s;
            c.wr = w[static_cast<std::size_t>(j)];
            c.vt = L[static_cast<std::size_t>(k)] / ri;
            c.volume = vol;
            c.n = n;
            cells.push_back(c);
            total_n += n;
        }
    }
    // Order by represented population and omit only a cumulative amount
    // within the configured cumulative fraction of shell population.
    if (p.bp_boltzmann_mass_tol > 0.0 && total_n > 0.0) {
        const double omission_budget =
            std::clamp(p.bp_boltzmann_mass_tol, 0.0, 1.0) * total_n;
        std::sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) {
            return a.n < b.n;
        });
        double omitted_n = 0.0;
        std::size_t first_kept = 0;
        while (first_kept < cells.size()
               && omitted_n + cells[first_kept].n <= omission_budget) {
            omitted_n += cells[first_kept].n;
            ++first_kept;
        }
        cells.erase(cells.begin(), cells.begin() + static_cast<std::ptrdiff_t>(first_kept));
    }
    return cells;
}

// Locate positive interpolation weights on the represented node range.
// Values outside the stated endpoint roundoff tolerance are unrepresentable.
inline bool bp_fa_bracket(
    const std::vector<double>& x,
    int first,
    double value,
    int& a,
    int& b,
    double& t) {
    const int n = static_cast<int>(x.size());
    if (n <= 0 || first < 0 || first >= n) {
        return false;
    }
    const double lo = x[static_cast<std::size_t>(first)];
    const double hi = x.back();
    const double tol = 1.0e-12 * std::max(1.0, std::max(std::abs(lo), std::abs(hi)));
    if (value < lo - tol || value > hi + tol) {
        return false;
    }
    if (value <= lo) {
        a = first;
        b = first;
        t = 0.0;
        return true;
    }
    if (value >= hi) {
        a = n - 1;
        b = n - 1;
        t = 0.0;
        return true;
    }
    auto begin = x.begin() + first;
    auto it = std::lower_bound(begin, x.end(), value);
    const int upper = static_cast<int>(it - x.begin());
    if (upper <= first) {
        a = first;
        b = first;
        t = 0.0;
        return true;
    }
    a = upper - 1;
    b = upper;
    const double xa = x[static_cast<std::size_t>(a)];
    const double xb = x[static_cast<std::size_t>(b)];
    t = (xb > xa) ? std::clamp((value - xa) / (xb - xa), 0.0, 1.0) : 0.0;
    return true;
}

// The nodal L=0 point has zero L*dL measure in this quadrature; locate
// the first positive-measure tangential node for outgoing remaps.
inline int bp_fa_first_positive_L_cell(const std::vector<double>& L) {
    for (int k = 0; k < static_cast<int>(L.size()); ++k) {
        if (L[static_cast<std::size_t>(k)] > 0.0) {
            return k;
        }
    }
    return static_cast<int>(L.size());
}

struct BPFABracketLookup {
    int first = 0;
    double lo = 0.0;
    double hi = 0.0;
    double bin_scale = 0.0;
    std::vector<int> upper;
};

/// Builds a coarse direct locator whose final local adjustment is exact.
inline BPFABracketLookup bp_fa_make_bracket_lookup(
    const std::vector<double>& x,
    int first) {
    BPFABracketLookup lookup;
    lookup.first = first;
    if (first < 0 || first >= static_cast<int>(x.size())) {
        return lookup;
    }
    lookup.lo = x[static_cast<std::size_t>(first)];
    lookup.hi = x.back();
    const int bins = std::max(256, 64 * static_cast<int>(x.size()));
    lookup.upper.resize(static_cast<std::size_t>(bins), first);
    if (!(lookup.hi > lookup.lo)) {
        return lookup;
    }
    lookup.bin_scale = static_cast<double>(bins) / (lookup.hi - lookup.lo);
    const auto begin = x.begin() + first;
    for (int q = 0; q < bins; ++q) {
        const double value = lookup.lo
            + (lookup.hi - lookup.lo) * static_cast<double>(q) / static_cast<double>(bins);
        lookup.upper[static_cast<std::size_t>(q)] =
            static_cast<int>(std::lower_bound(begin, x.end(), value) - x.begin());
    }
    return lookup;
}

// Use the coarse lookup only as a starting locator, then adjust against
// actual nodes so its final bracket agrees with an ordered search.
inline bool bp_fa_bracket_lookup(
    const std::vector<double>& x,
    const BPFABracketLookup& lookup,
    double value,
    int& a,
    int& b,
    double& t) {
    const int n = static_cast<int>(x.size());
    if (n <= 0 || lookup.first < 0 || lookup.first >= n || lookup.upper.empty()) {
        return false;
    }
    const double tol = 1.0e-12
        * std::max(1.0, std::max(std::abs(lookup.lo), std::abs(lookup.hi)));
    if (value < lookup.lo - tol || value > lookup.hi + tol) {
        return false;
    }
    if (value <= lookup.lo) {
        a = lookup.first;
        b = lookup.first;
        t = 0.0;
        return true;
    }
    if (value >= lookup.hi) {
        a = n - 1;
        b = n - 1;
        t = 0.0;
        return true;
    }
    const int bin = std::clamp(
        static_cast<int>((value - lookup.lo) * lookup.bin_scale),
        0,
        static_cast<int>(lookup.upper.size()) - 1);
    int upper = lookup.upper[static_cast<std::size_t>(bin)];
    upper = std::clamp(upper, lookup.first, n - 1);
    while (upper < n && x[static_cast<std::size_t>(upper)] < value) {
        ++upper;
    }
    while (upper > lookup.first
           && x[static_cast<std::size_t>(upper - 1)] >= value) {
        --upper;
    }
    if (upper <= lookup.first) {
        a = lookup.first;
        b = lookup.first;
        t = 0.0;
        return true;
    }
    if (upper >= n) {
        a = n - 1;
        b = n - 1;
        t = 0.0;
        return true;
    }
    a = upper - 1;
    b = upper;
    const double xa = x[static_cast<std::size_t>(a)];
    const double xb = x[static_cast<std::size_t>(b)];
    t = (xb > xa) ? std::clamp((value - xa) / (xb - xa), 0.0, 1.0) : 0.0;
    return true;
}

// Convert an outgoing 3-D velocity to (w,eta), find its neighboring
// grid nodes, and keep positive-volume destinations. Report radial and
// angular-domain failures separately for the skipped-event budget.
inline bool bp_fa_remap_stencil(
    BPFAVec3 v,
    double ri,
    const std::vector<double>& w,
    const std::vector<double>& eta,
    const BPFABracketLookup& w_lookup,
    const BPFABracketLookup& eta_lookup,
    const std::vector<double>& volume,
    const Params& p,
    BPFARemapStencil& stencil,
    bool* w_outside = nullptr,
    bool* L_outside = nullptr) {
    stencil = {};
    if (w_outside != nullptr) {
        *w_outside = false;
    }
    if (L_outside != nullptr) {
        *L_outside = false;
    }
    int j0 = 0;
    int j1 = 0;
    double tw = 0.0;
    if (!bp_fa_bracket_lookup(w, w_lookup, v.x, j0, j1, tw)) {
        if (w_outside != nullptr) {
            *w_outside = true;
        }
        return false;
    }

    const double et = 0.5 * ri * ri * std::max(0.0, v.y * v.y + v.z * v.z);
    int k0 = 0;
    int k1 = 0;
    double te = 0.0;
    if (!bp_fa_bracket_lookup(eta, eta_lookup, et, k0, k1, te)) {
        if (L_outside != nullptr) {
            *L_outside = true;
        }
        return false;
    }

    const int js[2] = {j0, j1};
    const double jw[2] = {(j0 == j1) ? 1.0 : (1.0 - tw), (j0 == j1) ? 0.0 : tw};
    const int ks[2] = {k0, k1};
    const double kw[2] = {(k0 == k1) ? 1.0 : (1.0 - te), (k0 == k1) ? 0.0 : te};
    for (int a = 0; a < 2; ++a) {
        if (!(jw[a] > 0.0)) {
            continue;
        }
        for (int b = 0; b < 2; ++b) {
            if (!(kw[b] > 0.0)) {
                continue;
            }
            const int s = js[a] * p.NL + ks[b];
            if (!(volume[static_cast<std::size_t>(s)] > 0.0)) {
                continue;
            }
            stencil.index[stencil.count] = s;
            stencil.weight[stencil.count] = jw[a] * kw[b];
            ++stencil.count;
        }
    }
    return stencil.count > 0;
}

// Add an extensive event population to destination cells. Division by
// their velocity volume occurs only when installing the final f increment.
inline void bp_fa_deposit(std::vector<double>& delta_n, const BPFARemapStencil& stencil, double n) {
    for (int q = 0; q < stencil.count; ++q) {
        delta_n[static_cast<std::size_t>(stencil.index[q])] += n * stencil.weight[q];
    }
}

// Pivoted elimination solves the three invariant correction coefficients.
// Degenerate moment geometry returns failure to the optional repair caller.
inline bool bp_fa_solve_3x3(double A[3][3], double b[3], double x[3]) {
    double m[3][4] = {
        {A[0][0], A[0][1], A[0][2], b[0]},
        {A[1][0], A[1][1], A[1][2], b[1]},
        {A[2][0], A[2][1], A[2][2], b[2]},
    };
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row) {
            if (std::abs(m[row][col]) > std::abs(m[pivot][col])) {
                pivot = row;
            }
        }
        if (!(std::abs(m[pivot][col]) > 1.0e-300)) {
            return false;
        }
        if (pivot != col) {
            for (int q = col; q < 4; ++q) {
                std::swap(m[col][q], m[pivot][q]);
            }
        }
        const double inv = 1.0 / m[col][col];
        for (int q = col; q < 4; ++q) {
            m[col][q] *= inv;
        }
        for (int row = 0; row < 3; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = m[row][col];
            for (int q = col; q < 4; ++q) {
                m[row][q] -= factor * m[col][q];
            }
        }
    }
    x[0] = m[0][3];
    x[1] = m[1][3];
    x[2] = m[2][3];
    return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
}

// Iteratively correct direct-remap mass, radial momentum, and kinetic
// energy through a distribution-weighted basis {1,w,e}. The step fraction
// limits the multiplicative correction to keep positive f values nonnegative.
inline void bp_fa_repair_shell_moments(
    std::vector<double>& f,
    int i,
    double ri,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& volume,
    const BPFAShellMoments& target,
    const Params& p) {
    const double relative_tolerance = 1.0e-13;
    const double mass_scale = std::max(std::abs(target.mass), 1.0e-30);
    const double momentum_scale = std::max(
        std::sqrt(std::max(
            0.0,
            2.0 * std::abs(target.mass) * std::abs(target.kinetic_energy))),
        1.0e-30);
    const double energy_scale = std::max(std::abs(target.kinetic_energy), 1.0e-30);
    for (int iter = 0; iter < 8; ++iter) {
        const BPFAShellMoments current = bp_fa_shell_moments(f, i, ri, w, L, volume, p);
        double rhs[3] = {
            target.mass - current.mass,
            target.radial_momentum - current.radial_momentum,
            target.kinetic_energy - current.kinetic_energy,
        };
        if (std::abs(rhs[0]) <= relative_tolerance * mass_scale
            && std::abs(rhs[1]) <= relative_tolerance * momentum_scale
            && std::abs(rhs[2]) <= relative_tolerance * energy_scale) {
            return;
        }
        // The Gram matrix of the invariant basis maps coefficient changes
        // to the three measured moment deficits on currently positive cells.
        double A[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        for (int j = 0; j < p.Nw; ++j) {
            const double wr = w[static_cast<std::size_t>(j)];
            for (int k = 0; k < p.NL; ++k) {
                const int s = j * p.NL + k;
                const double vol = volume[static_cast<std::size_t>(s)];
                const double fq = f[idx(i, j, k, p)];
                if (!(vol > 0.0) || !(fq > 0.0)) {
                    continue;
                }
                const double vt = L[static_cast<std::size_t>(k)] / ri;
                const double e = 0.5 * (wr * wr + vt * vt);
                const double basis[3] = {1.0, wr, e};
                const double weight = fq * vol;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        A[a][b] += weight * basis[a] * basis[b];
                    }
                }
            }
        }
        double lambda[3] = {0.0, 0.0, 0.0};
        if (!bp_fa_solve_3x3(A, rhs, lambda)) {
            return;
        }
        // Restrict the correction before applying it, preserving positive
        // cells while allowing subsequent iterations to close the residual.
        double alpha = 1.0;
        for (int j = 0; j < p.Nw; ++j) {
            const double wr = w[static_cast<std::size_t>(j)];
            for (int k = 0; k < p.NL; ++k) {
                const double fq = f[idx(i, j, k, p)];
                if (!(fq > 0.0)) {
                    continue;
                }
                const double vt = L[static_cast<std::size_t>(k)] / ri;
                const double e = 0.5 * (wr * wr + vt * vt);
                const double s = lambda[0] + lambda[1] * wr + lambda[2] * e;
                if (s < -0.95) {
                    alpha = std::min(alpha, 0.95 / (-s));
                }
            }
        }
        for (int j = 0; j < p.Nw; ++j) {
            const double wr = w[static_cast<std::size_t>(j)];
            for (int k = 0; k < p.NL; ++k) {
                const double vt = L[static_cast<std::size_t>(k)] / ri;
                const double e = 0.5 * (wr * wr + vt * vt);
                const double s = lambda[0] + lambda[1] * wr + lambda[2] * e;
                double& fq = f[idx(i, j, k, p)];
                fq = std::max(0.0, fq * (1.0 + alpha * s));
            }
        }
    }
}

// Average u*sigma(u) over the unresolved relative tangent-plane azimuth
// for this pair of reduced velocity cells.
inline double bp_fa_pair_u_sigma_phi_average(
    const BPFAVelocityCell& a,
    const BPFAVelocityCell& b,
    const std::vector<double>& phi_cos,
    const std::vector<double>& phi_sin,
    const Params& p) {
    const int nphi = static_cast<int>(phi_cos.size());
    double avg = 0.0;
    for (int q = 0; q < nphi; ++q) {
        const BPFAVec3 v1{a.wr, a.vt, 0.0};
        const BPFAVec3 v2{
            b.wr,
            b.vt * phi_cos[static_cast<std::size_t>(q)],
            b.vt * phi_sin[static_cast<std::size_t>(q)]};
        const double u = bp_fa_norm(bp_fa_sub(v1, v2));
        avg += u * bp_fa_total_cross_section(u, p);
    }
    return avg / static_cast<double>(nphi);
}

/// Rigorous O(Nv) upper bound on the loss rate for a constant cross section.
inline double bp_fa_constant_shell_loss_rate_upper_bound(
    const std::vector<BPFAVelocityCell>& cells,
    const Params& p) {
    const double sigma = bp_fa_total_cross_section(1.0, p);
    if (!(sigma > 0.0) || cells.empty()) {
        return 0.0;
    }
    double total_n = 0.0;
    double speed_moment = 0.0;
    double max_speed = 0.0;
    for (const BPFAVelocityCell& cell : cells) {
        const double speed = std::hypot(cell.wr, cell.vt);
        total_n += cell.n;
        speed_moment += cell.n * speed;
        max_speed = std::max(max_speed, speed);
    }
    // |v_a-v_b| <= |v_a|+|v_b| for every pair azimuth, hence
    // nu_a <= sigma (|v_a| sum_b n_b + sum_b n_b |v_b|).
    return sigma * (total_n * max_speed + speed_moment);
}

// Sum partner populations times the discrete pair-azimuth averaged rate
// for each active velocity cell, then take the largest represented loss rate.
inline double bp_fa_shell_max_loss_rate(
    const std::vector<BPFAVelocityCell>& cells,
    const std::vector<double>& phi_cos,
    const std::vector<double>& phi_sin,
    const Params& p,
    int requested_threads) {
    if (cells.empty()) {
        return 0.0;
    }
    const int cell_count = static_cast<int>(cells.size());
    const int nthreads = std::max(1, std::min(requested_threads, cell_count));
    if (nthreads == 1) {
        std::vector<double> loss_rate(cells.size(), 0.0);
        for (std::size_t a = 0; a < cells.size(); ++a) {
            for (std::size_t b = a; b < cells.size(); ++b) {
                const double avg = bp_fa_pair_u_sigma_phi_average(cells[a], cells[b], phi_cos, phi_sin, p);
                if (!(avg > 0.0)) {
                    continue;
                }
                if (a == b) {
                    loss_rate[a] += cells[a].n * avg;
                } else {
                    loss_rate[a] += cells[b].n * avg;
                    loss_rate[b] += cells[a].n * avg;
                }
            }
        }
        return loss_rate.empty() ? 0.0 : *std::max_element(loss_rate.begin(), loss_rate.end());
    }

    // Each worker owns a complete partial rate array. Reducing worker
    // arrays afterward avoids concurrent updates from shared pair endpoints.
    std::vector<std::vector<double>> partial(
        static_cast<std::size_t>(nthreads),
        std::vector<double>(cells.size(), 0.0));
    parallel_for_dynamic_chunks(0, cell_count, nthreads, 1,
        [&](int alo, int ahi, int tid) {
            std::vector<double>& loss_rate = partial[static_cast<std::size_t>(tid)];
            for (int ia = alo; ia < ahi; ++ia) {
                const std::size_t a = static_cast<std::size_t>(ia);
                for (std::size_t b = a; b < cells.size(); ++b) {
                    const double avg = bp_fa_pair_u_sigma_phi_average(cells[a], cells[b], phi_cos, phi_sin, p);
                    if (!(avg > 0.0)) {
                        continue;
                    }
                    if (a == b) {
                        loss_rate[a] += cells[a].n * avg;
                    } else {
                        loss_rate[a] += cells[b].n * avg;
                        loss_rate[b] += cells[a].n * avg;
                    }
                }
            }
        });

    double local_max = 0.0;
    for (std::size_t s = 0; s < cells.size(); ++s) {
        double rate = 0.0;
        for (const std::vector<double>& part : partial) {
            rate += part[s];
        }
        local_max = std::max(local_max, rate);
    }
    return local_max;
}

/// Selects radially local pair sums that would otherwise dominate one shell worker.
inline std::vector<int> bp_fa_heavy_pair_shells(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p,
    int requested_threads) {
    if (requested_threads <= 1) {
        return {};
    }
    std::vector<std::size_t> work(static_cast<std::size_t>(p.Nr), 0);
    std::size_t total_work = 0;
    for (int i = 0; i < p.Nr; ++i) {
        const double ri = std::max(r[static_cast<std::size_t>(i)], 1.0e-300);
        const std::vector<double> volume = bp_fa_shell_velocity_volumes(ri, wq, L, Lq, p);
        const std::size_t n = bp_fa_active_cells(f, i, ri, w, L, volume, p).size();
        work[static_cast<std::size_t>(i)] = n * (n + 1) / 2;
        total_work += work[static_cast<std::size_t>(i)];
    }
    if (total_work == 0) {
        return {};
    }

    // A shell is pair-parallel only when its serial work exceeds four ideal
    // thread shares. Uniform shells stay on the cheaper radial scheduler.
    const std::size_t threshold = std::max<std::size_t>(
        1,
        (4 * total_work + static_cast<std::size_t>(requested_threads) - 1)
            / static_cast<std::size_t>(requested_threads));
    std::vector<int> heavy;
    for (int i = 0; i < p.Nr; ++i) {
        if (work[static_cast<std::size_t>(i)] >= threshold) {
            heavy.push_back(i);
        }
    }
    std::sort(heavy.begin(), heavy.end(), [&](int a, int b) {
        return work[static_cast<std::size_t>(a)] > work[static_cast<std::size_t>(b)];
    });
    return heavy;
}

// Constant area admits the linear triangle-inequality upper bound over
// active cells. Other laws evaluate the represented pair-angle quadrature,
// splitting unusually dense velocity shells across workers when useful.
inline double bp_fa_estimate_max_loss_rate(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p) {
    const int nthreads = effective_threads(p, p.Nr);
    if (p.bp_cross_section == "constant") {
        std::vector<double> partial(static_cast<std::size_t>(nthreads), 0.0);
        parallel_for_dynamic_chunks(0, p.Nr, nthreads, 1,
            [&](int ilo, int ihi, int tid) {
                double local_max = 0.0;
                for (int i = ilo; i < ihi; ++i) {
                    const double ri =
                        std::max(r[static_cast<std::size_t>(i)], 1.0e-300);
                    const std::vector<double> volume =
                        bp_fa_shell_velocity_volumes(ri, wq, L, Lq, p);
                    const std::vector<BPFAVelocityCell> cells =
                        bp_fa_active_cells(f, i, ri, w, L, volume, p);
                    local_max = std::max(
                        local_max,
                        bp_fa_constant_shell_loss_rate_upper_bound(cells, p));
                }
                partial[static_cast<std::size_t>(tid)] = local_max;
            });
        return *std::max_element(partial.begin(), partial.end());
    }

    const int nphi = std::max(1, p.bp_boltzmann_pair_phi_points);
    std::vector<double> phi_cos;
    std::vector<double> phi_sin;
    bp_fa_midpoint_angle_table(nphi, phi_cos, phi_sin);
    const int requested_threads = effective_threads(
        p, std::max(1, p.Nr * std::max(1, p.Nw * p.NL)));
    const std::vector<int> heavy = bp_fa_heavy_pair_shells(
        f, r, w, L, wq, Lq, p, requested_threads);
    std::vector<unsigned char> is_heavy(static_cast<std::size_t>(p.Nr), 0);
    double heavy_max = 0.0;
    for (int i : heavy) {
        is_heavy[static_cast<std::size_t>(i)] = 1;
        const double ri = std::max(r[static_cast<std::size_t>(i)], 1.0e-300);
        const std::vector<double> volume = bp_fa_shell_velocity_volumes(ri, wq, L, Lq, p);
        const std::vector<BPFAVelocityCell> cells = bp_fa_active_cells(f, i, ri, w, L, volume, p);
        heavy_max = std::max(
            heavy_max,
            bp_fa_shell_max_loss_rate(cells, phi_cos, phi_sin, p, requested_threads));
    }

    std::vector<double> partial(static_cast<std::size_t>(nthreads), 0.0);
    parallel_for_dynamic_chunks(0, p.Nr, nthreads, 1,
        [&](int ilo, int ihi, int tid) {
            double local_max = 0.0;
            for (int i = ilo; i < ihi; ++i) {
                if (is_heavy[static_cast<std::size_t>(i)]) {
                    continue;
                }
                const double ri = std::max(r[static_cast<std::size_t>(i)], 1.0e-300);
                const std::vector<double> volume = bp_fa_shell_velocity_volumes(ri, wq, L, Lq, p);
                const std::vector<BPFAVelocityCell> cells = bp_fa_active_cells(f, i, ri, w, L, volume, p);
                local_max = std::max(local_max, bp_fa_shell_max_loss_rate(cells, phi_cos, phi_sin, p, 1));
            }
            partial[static_cast<std::size_t>(tid)] =
                std::max(partial[static_cast<std::size_t>(tid)], local_max);
        });
    return std::max(heavy_max, *std::max_element(partial.begin(), partial.end()));
}

// Integrate one local shell's direct gain/loss update from its incoming
// population, preserving a separate budget for outgoing velocity-domain loss.
inline BPFARemapBudget bp_fa_apply_shell_substep(
    std::vector<double>& f,
    int i,
    double dt,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p,
    int inner_threads) {
    BPFARemapBudget shell_budget;
    const double ri = std::max(r[static_cast<std::size_t>(i)], 1.0e-300);
    const std::vector<double> volume = bp_fa_shell_velocity_volumes(ri, wq, L, Lq, p);
    const BPFAShellMoments target = bp_fa_shell_moments(f, i, ri, w, L, volume, p);
    const std::vector<BPFAVelocityCell> cells = bp_fa_active_cells(f, i, ri, w, L, volume, p);
    if (cells.size() < 1) {
        return shell_budget;
    }
    std::vector<double> eta(L.size(), 0.0);
    for (std::size_t k = 0; k < L.size(); ++k) {
        eta[k] = 0.5 * L[k] * L[k];
    }
    const int first_L = bp_fa_first_positive_L_cell(L);
    if (first_L >= p.NL) {
        return shell_budget;
    }
    const BPFABracketLookup w_lookup = bp_fa_make_bracket_lookup(w, 0);
    const BPFABracketLookup eta_lookup = bp_fa_make_bracket_lookup(eta, first_L);

    // Quadrature averages the incoming relative azimuth and outgoing
    // azimuth, while Gauss nodes integrate mu. Normalize each angular law
    // separately from the physical total rate.
    const std::string kernel = bp_fa_kernel_name(p);
    const int npair_phi = std::max(1, p.bp_boltzmann_pair_phi_points);
    const int nmu = std::max(1, p.bp_boltzmann_scatter_mu_points);
    const int npsi = std::max(1, p.bp_boltzmann_scatter_phi_points);
    const double inv_pair_phi = 1.0 / static_cast<double>(npair_phi);
    const double inv_npsi = 1.0 / static_cast<double>(npsi);
    std::vector<double> pair_phi_cos;
    std::vector<double> pair_phi_sin;
    std::vector<double> psi_cos;
    std::vector<double> psi_sin;
    bp_fa_midpoint_angle_table(npair_phi, pair_phi_cos, pair_phi_sin);
    bp_fa_midpoint_angle_table(npsi, psi_cos, psi_sin);
    std::vector<double> mu_node(static_cast<std::size_t>(nmu), 0.0);
    std::vector<double> mu_quadrature_weight(static_cast<std::size_t>(nmu), 0.0);
    std::vector<double> mu_sint(static_cast<std::size_t>(nmu), 0.0);
    std::vector<double> raw_mu(static_cast<std::size_t>(nmu), 0.0);
    bp_fa_gauss_legendre_mu_table(nmu, mu_node, mu_quadrature_weight);
    for (int imu = 0; imu < nmu; ++imu) {
        const double mu = mu_node[static_cast<std::size_t>(imu)];
        mu_sint[static_cast<std::size_t>(imu)] = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    }
    std::vector<double> delta_n(static_cast<std::size_t>(p.Nw * p.NL), 0.0);

    auto accumulate_pair = [&](std::size_t ia, std::size_t ib, std::vector<double>& delta,
                               std::vector<double>& raw_mu_work, BPFARemapBudget& budget) {
            const BPFAVelocityCell& a = cells[ia];
            const BPFAVelocityCell& b = cells[ib];
            // Sum unordered incoming cell pairs. A same-cell pair receives
            // the factor one half; each accepted angular outcome removes
            // two incoming populations and adds two outgoing populations.
            const double pair_density = (ia == ib) ? (0.5 * a.n * a.n) : (a.n * b.n);
            if (!(pair_density > 0.0)) {
                return;
            }
            for (int iphi = 0; iphi < npair_phi; ++iphi) {
                const BPFAVec3 v1{a.wr, a.vt, 0.0};
                const BPFAVec3 v2{
                    b.wr,
                    b.vt * pair_phi_cos[static_cast<std::size_t>(iphi)],
                    b.vt * pair_phi_sin[static_cast<std::size_t>(iphi)]};
                const BPFAVec3 rel = bp_fa_sub(v1, v2);
                const double u = bp_fa_norm(rel);
                const double sigma = bp_fa_total_cross_section(u, p);
                const double event_prefactor = pair_density * u * sigma * dt * inv_pair_phi;
                if (!(event_prefactor > 0.0)) {
                    continue;
                }
                // Finite angular quadrature uses a normalized directional
                // distribution; u*sigma_total already fixes the event rate.
                double raw_sum = 0.0;
                for (int imu = 0; imu < nmu; ++imu) {
                    const double mu = mu_node[static_cast<std::size_t>(imu)];
                    raw_mu_work[static_cast<std::size_t>(imu)] =
                        mu_quadrature_weight[static_cast<std::size_t>(imu)]
                        * bp_fa_scatter_weight_raw(mu, u, p, kernel);
                    raw_sum += raw_mu_work[static_cast<std::size_t>(imu)];
                }
                if (!(raw_sum > 0.0)) {
                    continue;
                }
                const BPFAVec3 com = bp_fa_scale(bp_fa_add(v1, v2), 0.5);
                const BPFAVec3 uhat = bp_fa_scale(rel, 1.0 / u);
                const std::array<BPFAVec3, 2> basis = bp_fa_perpendicular_basis_unit(uhat);
                for (int imu = 0; imu < nmu; ++imu) {
                    const double mu = mu_node[static_cast<std::size_t>(imu)];
                    const double sint = mu_sint[static_cast<std::size_t>(imu)];
                    const double mu_weight = raw_mu_work[static_cast<std::size_t>(imu)] / raw_sum;
                    if (!(mu_weight > 0.0)) {
                        continue;
                    }
                    for (int ipsi = 0; ipsi < npsi; ++ipsi) {
                        const BPFAVec3 ndir = bp_fa_scattered_relative_direction_from_basis(
                            uhat,
                            basis[0],
                            basis[1],
                            mu,
                            sint,
                            psi_cos[static_cast<std::size_t>(ipsi)],
                            psi_sin[static_cast<std::size_t>(ipsi)]);
                        const BPFAVec3 rel_after = bp_fa_scale(ndir, u);
                        const BPFAVec3 out1 = bp_fa_add(com, bp_fa_scale(rel_after, 0.5));
                        const BPFAVec3 out2 = bp_fa_sub(com, bp_fa_scale(rel_after, 0.5));
                        const double event_n = event_prefactor * mu_weight * inv_npsi;
                        budget.attempted_event_density += event_n;
                        BPFARemapStencil s1;
                        BPFARemapStencil s2;
                        bool out1_w = false;
                        bool out1_L = false;
                        bool out2_w = false;
                        bool out2_L = false;
                        const bool mapped1 = bp_fa_remap_stencil(
                            out1, ri, w, eta, w_lookup, eta_lookup, volume, p, s1, &out1_w, &out1_L);
                        const bool mapped2 = bp_fa_remap_stencil(
                            out2, ri, w, eta, w_lookup, eta_lookup, volume, p, s2, &out2_w, &out2_L);
                        // Keep the pair event whole: an unrepresented
                        // partner skips both gain and loss, and its attempted
                        // population enters the explicit remap-loss ledger.
                        if (!mapped1 || !mapped2) {
                            budget.skipped_event_density += event_n;
                            if (out1_w || out2_w) {
                                budget.skipped_w_event_density += event_n;
                            }
                            if (out1_L || out2_L) {
                                budget.skipped_L_event_density += event_n;
                            }
                            continue;
                        }
                        // Equal-mass elastic kinematics conserve momentum
                        // and energy before remap. Accumulate extensive gain
                        // and loss before changing any incoming f values.
                        delta[static_cast<std::size_t>(a.shell_index)] -= event_n;
                        delta[static_cast<std::size_t>(b.shell_index)] -= event_n;
                        bp_fa_deposit(delta, s1, event_n);
                        bp_fa_deposit(delta, s2, event_n);
                    }
                }
            }
        };

    const int cell_count = static_cast<int>(cells.size());
    const int nthreads = std::max(1, std::min(inner_threads, cell_count));
    if (nthreads == 1) {
        for (std::size_t ia = 0; ia < cells.size(); ++ia) {
            for (std::size_t ib = ia; ib < cells.size(); ++ib) {
                accumulate_pair(ia, ib, delta_n, raw_mu, shell_budget);
            }
        }
    } else {
        // Use disjoint worker gain/loss arrays and angular scratch; combine
        // them in worker order after all pairs have used the incoming state.
        std::vector<std::vector<double>> partial_delta(
            static_cast<std::size_t>(nthreads),
            std::vector<double>(delta_n.size(), 0.0));
        std::vector<BPFARemapBudget> partial_budget(static_cast<std::size_t>(nthreads));
        parallel_for_dynamic_chunks(0, cell_count, nthreads, 1,
            [&](int alo, int ahi, int tid) {
                std::vector<double>& local_delta = partial_delta[static_cast<std::size_t>(tid)];
                std::vector<double> raw_mu_local(static_cast<std::size_t>(nmu), 0.0);
                BPFARemapBudget& local_budget = partial_budget[static_cast<std::size_t>(tid)];
                for (int ia_int = alo; ia_int < ahi; ++ia_int) {
                    const std::size_t ia = static_cast<std::size_t>(ia_int);
                    for (std::size_t ib = ia; ib < cells.size(); ++ib) {
                        accumulate_pair(ia, ib, local_delta, raw_mu_local, local_budget);
                    }
                }
            });
        for (int tid = 0; tid < nthreads; ++tid) {
            const std::vector<double>& part = partial_delta[static_cast<std::size_t>(tid)];
            for (std::size_t s = 0; s < delta_n.size(); ++s) {
                delta_n[s] += part[s];
            }
            bp_fa_merge_remap_budget(shell_budget, partial_budget[static_cast<std::size_t>(tid)]);
        }
    }

    for (int j = 0; j < p.Nw; ++j) {
        for (int k = 0; k < p.NL; ++k) {
            const int s = j * p.NL + k;
            const double vol = volume[static_cast<std::size_t>(s)];
            if (!(vol > 0.0)) {
                continue;
            }
            double& fq = f[idx(i, j, k, p)];
            fq = std::max(0.0, fq + delta_n[static_cast<std::size_t>(s)] / vol);
        }
    }
    // Direct interpolation can alter discrete moments. Apply the optional
    // positive correction against the incoming shell target when selected.
    if (p.bp_boltzmann_conserve_moments) {
        bp_fa_repair_shell_moments(f, i, ri, w, L, volume, target, p);
    }
    if (shell_budget.attempted_event_density > 0.0) {
        shell_budget.max_shell_skipped_fraction =
            shell_budget.skipped_event_density / shell_budget.attempted_event_density;
    }
    return shell_budget;
}

// Parallelize independent shells normally, and pair-parallelize only the
// unusually expensive shells selected by the measured active-cell work model.
inline BPFARemapBudget bp_fa_apply_substep(
    std::vector<double>& f,
    double dt,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p) {
    BPFARemapBudget budget;
    const int requested_threads = effective_threads(
        p, std::max(1, p.Nr * std::max(1, p.Nw * p.NL)));
    const std::vector<int> heavy = bp_fa_heavy_pair_shells(
        f, r, w, L, wq, Lq, p, requested_threads);
    std::vector<unsigned char> is_heavy(static_cast<std::size_t>(p.Nr), 0);
    std::vector<BPFARemapBudget> shell_budget(static_cast<std::size_t>(p.Nr));
    for (int i : heavy) {
        is_heavy[static_cast<std::size_t>(i)] = 1;
        shell_budget[static_cast<std::size_t>(i)] =
            bp_fa_apply_shell_substep(f, i, dt, r, w, L, wq, Lq, p, requested_threads);
    }

    parallel_for_dynamic_chunks(0, p.Nr, effective_threads(p, p.Nr), 1,
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                if (!is_heavy[static_cast<std::size_t>(i)]) {
                    shell_budget[static_cast<std::size_t>(i)] =
                        bp_fa_apply_shell_substep(f, i, dt, r, w, L, wq, Lq, p, 1);
                }
            }
        });
    for (const BPFARemapBudget& part : shell_budget) {
        bp_fa_merge_remap_budget(budget, part);
    }
    return budget;
}

// Accumulate the configured call cadence, then cover its full physical
// interval with rate-limited explicit collision substeps and remap checks.
inline BPCollisionStepInfo apply_bp_finite_angle_boltzmann_collision_step(
    std::vector<double>& f,
    std::vector<double>* tracer_f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    double dt,
    const Params& p,
    const MpiContext& mpi) {
    BPCollisionStepInfo info;
    if (!using_bp_finite_angle_boltzmann(p) || dt <= 0.0 || p.bp_collision_strength == 0.0) {
        return info;
    }
    if (mpi.size > 1) {
        throw std::runtime_error(
            "Finite-angle Boltzmann collisions are local in r but dense in velocity space; "
            "the current implementation requires a serial/OpenMP run.");
    }
    if (tracer_f != nullptr && !tracer_f->empty()) {
        throw std::runtime_error(
            "Finite-angle Boltzmann collisions for passive tracer distributions are not implemented yet.");
    }

    // This process-local cadence state groups successive driver calls. A
    // changed grid or collision law resets the accumulated-call configuration.
    struct CollisionCadence {
        int calls_left = 0;
        int Nr = 0;
        int Nw = 0;
        int NL = 0;
        int apply_every = 1;
        double dt_accum = 0.0;
        double strength = 0.0;
        double sigma0 = 0.0;
        double sigma_u0 = 0.0;
        double sigma_n = 0.0;
        double subhalo_mass = 0.0;
        double subhalo_radius = 0.0;
        double coulomb_log = 0.0;
        double yukawa_alpha = 0.0;
        double yukawa_mchi = 0.0;
        double yukawa_mphi = 0.0;
        double boltzmann_mass_tol = 0.0;
        double boltzmann_remap_max_skipped_fraction = 0.0;
        std::string collision_operator;
        std::string cross_section;
        std::string boltzmann_kernel;
    };
    static CollisionCadence cadence;
    const bool cadence_matches =
        cadence.Nr == p.Nr
        && cadence.Nw == p.Nw
        && cadence.NL == p.NL
        && cadence.apply_every == p.bp_collision_apply_every
        && cadence.strength == p.bp_collision_strength
        && cadence.sigma0 == p.bp_sigma0
        && cadence.sigma_u0 == p.bp_sigma_u0
        && cadence.sigma_n == p.bp_sigma_n
        && cadence.subhalo_mass == p.bp_subhalo_mass
        && cadence.subhalo_radius == p.bp_subhalo_radius
        && cadence.coulomb_log == p.bp_coulomb_log
        && cadence.yukawa_alpha == p.bp_yukawa_alpha
        && cadence.yukawa_mchi == p.bp_yukawa_mchi
        && cadence.yukawa_mphi == p.bp_yukawa_mphi
        && cadence.boltzmann_mass_tol == p.bp_boltzmann_mass_tol
        && cadence.boltzmann_remap_max_skipped_fraction
            == p.bp_boltzmann_remap_max_skipped_fraction
        && cadence.collision_operator == p.bp_collision_operator
        && cadence.cross_section == p.bp_cross_section
        && cadence.boltzmann_kernel == p.bp_boltzmann_kernel;
    if (!cadence_matches) {
        cadence = CollisionCadence{};
        cadence.Nr = p.Nr;
        cadence.Nw = p.Nw;
        cadence.NL = p.NL;
        cadence.apply_every = p.bp_collision_apply_every;
        cadence.strength = p.bp_collision_strength;
        cadence.sigma0 = p.bp_sigma0;
        cadence.sigma_u0 = p.bp_sigma_u0;
        cadence.sigma_n = p.bp_sigma_n;
        cadence.subhalo_mass = p.bp_subhalo_mass;
        cadence.subhalo_radius = p.bp_subhalo_radius;
        cadence.coulomb_log = p.bp_coulomb_log;
        cadence.yukawa_alpha = p.bp_yukawa_alpha;
        cadence.yukawa_mchi = p.bp_yukawa_mchi;
        cadence.yukawa_mphi = p.bp_yukawa_mphi;
        cadence.boltzmann_mass_tol = p.bp_boltzmann_mass_tol;
        cadence.boltzmann_remap_max_skipped_fraction =
            p.bp_boltzmann_remap_max_skipped_fraction;
        cadence.collision_operator = p.bp_collision_operator;
        cadence.cross_section = p.bp_cross_section;
        cadence.boltzmann_kernel = p.bp_boltzmann_kernel;
    }
    // Skipped callbacks still contribute their physical time to the next
    // applied collision interval. The apply counter controls only call grouping.
    cadence.dt_accum += dt;
    if (p.bp_collision_apply_every > 1 && cadence.calls_left > 0) {
        --cadence.calls_left;
        return info;
    }
    const double source_dt = cadence.dt_accum;
    cadence.dt_accum = 0.0;
    cadence.calls_left = std::max(0, p.bp_collision_apply_every - 1);

    double remaining = source_dt;
    double min_limit = std::numeric_limits<double>::infinity();
    double last_subdt = 0.0;
    // Recompute the represented loss rate after each installed collision
    // substep because its velocity populations have changed.
    while (remaining > 1.0e-14 * std::max(1.0, source_dt)) {
        const double max_rate = bp_fa_estimate_max_loss_rate(f, r, w, L, wq, Lq, p);
        if (!(max_rate > 0.0) || !std::isfinite(max_rate)) {
            break;
        }
        const double limit = p.bp_collision_cfl / max_rate;
        if (!(limit > 0.0) || !std::isfinite(limit)) {
            throw std::runtime_error("Finite-angle Boltzmann collision step has non-finite collision time.");
        }
        min_limit = std::min(min_limit, limit);
        if (info.substeps >= p.bp_collision_max_substeps) {
            throw std::runtime_error(
                "Finite-angle Boltzmann collision step needs more substeps than bp_collision_max_substeps. "
                "Reduce dt_max, lower bp_collision_strength, or increase bp_collision_max_substeps.");
        }
        const double subdt = std::min(remaining, limit);
        const BPFARemapBudget remap = bp_fa_apply_substep(f, subdt, r, w, L, wq, Lq, p);
        info.remap_attempted_event_density += remap.attempted_event_density;
        info.remap_skipped_event_density += remap.skipped_event_density;
        info.remap_skipped_w_event_density += remap.skipped_w_event_density;
        info.remap_skipped_L_event_density += remap.skipped_L_event_density;
        info.max_shell_remap_skipped_fraction = std::max(
            info.max_shell_remap_skipped_fraction, remap.max_shell_skipped_fraction);
        const double remap_skipped_fraction =
            (remap.attempted_event_density > 0.0)
            ? remap.skipped_event_density / remap.attempted_event_density
            : 0.0;
        // A finite velocity grid may omit outgoing collision products.
        // Enforce the configured attempted-event fraction before continuing
        // to another substep, retaining separate w/L failure diagnostics.
        if (remap_skipped_fraction > p.bp_boltzmann_remap_max_skipped_fraction) {
            std::ostringstream message;
            message
                << "Finite-angle Boltzmann post-collision remap skipped "
                << remap_skipped_fraction
                << " of attempted event density, above "
                << "bp_boltzmann_remap_max_skipped_fraction="
                << p.bp_boltzmann_remap_max_skipped_fraction
                << " (w=" << remap.skipped_w_event_density
                << ", L=" << remap.skipped_L_event_density
                << ", max_shell_fraction=" << remap.max_shell_skipped_fraction
                << "). Enlarge the velocity-domain guard cells; do not "
                << "continue with a silently truncated collision kernel.";
            throw std::runtime_error(message.str());
        }
        remaining = std::max(0.0, remaining - subdt);
        last_subdt = subdt;
        ++info.substeps;
        ++info.explicit_substeps;
    }
    // Report the final executed substep, the smallest encountered stability
    // limit, and its rate timescale in the common collision-step record.
    info.collision_dt = last_subdt;
    info.explicit_dt_limit = std::isfinite(min_limit) ? min_limit : source_dt;
    info.collision_time_estimate = (info.explicit_dt_limit > 0.0)
        ? info.explicit_dt_limit / std::max(p.bp_collision_cfl, 1.0e-300)
        : 0.0;
    return info;
}

}  // namespace ev
