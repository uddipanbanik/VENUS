#pragma once

// Velocity-space collision operators for spherical Boltzmann--Poisson.
// The Fokker--Planck path integrates the configured transfer-cross-section law
// against a local distribution or Maxwellian bath, projects drift/diffusion into
// radial velocity and angular momentum, and advances conservative face fluxes.
// Finite-angle operators enter through separate dispatch branches. Inputs are f,
// velocity quadrature, physical collision parameters and an elapsed source time.
// Outputs are the changed distribution/tracer and timestep/operator diagnostics.
// The selected bath, angular law, coefficient refresh and splitting controls state
// the collision approximation used alongside the separate gravitational transport.

#include "ev/finite_angle_boltzmann.hpp"
#include "ev/reversible_finite_angle_boltzmann.hpp"
#include "ev/sampled_reversible_boltzmann.hpp"
#include "ev/streamed_reversible_boltzmann.hpp"
#include "ev/vlasov_poisson.hpp"

namespace ev {

/// Local velocity moments used to build a Maxwellian bath for the collision coefficients.
struct BPLocalBath {
    double rho = 0.0;
    double mean_vr = 0.0;
    double theta = 0.0;
};

/// Projected spherical Fokker-Planck coefficients at one (r,v_r,L) grid point.
struct BPCoeff {
    double D_r = 0.0;
    double D_L = 0.0;
    double D_rr = 0.0;
    double D_rL = 0.0;
    double D_LL = 0.0;
    double D_aa = 0.0;
};

/// Same local FP tensor after changing the angular coordinate to eta=L^2/2.
struct BPEtaCoeff {
    double D_w = 0.0;
    double D_eta = 0.0;
    double D_ww = 0.0;
    double D_weta = 0.0;
    double D_etaeta = 0.0;
};

/// Applies diagnostic term masks to isolate pieces of the velocity-space FP flux.
BPCoeff mask_bp_fp_terms(BPCoeff c, const Params& p) {
    if (p.bp_fp_terms == "full") {
        return c;
    }
    if (p.bp_fp_terms == "vr_only") {
        c.D_L = 0.0;
        c.D_LL = 0.0;
        c.D_rL = 0.0;
        c.D_aa = 0.0;
        return c;
    }
    if (p.bp_fp_terms == "L_only") {
        c.D_r = 0.0;
        c.D_rr = 0.0;
        c.D_rL = 0.0;
        return c;
    }
    if (p.bp_fp_terms == "no_mixed") {
        c.D_rL = 0.0;
        return c;
    }
    if (p.bp_fp_terms == "no_azimuth") {
        c.D_aa = 0.0;
        return c;
    }
    if (p.bp_fp_terms == "no_drift") {
        c.D_r = 0.0;
        c.D_L = 0.0;
        return c;
    }
    if (p.bp_fp_terms == "drift_only") {
        c.D_rr = 0.0;
        c.D_rL = 0.0;
        c.D_LL = 0.0;
        c.D_aa = 0.0;
        return c;
    }
    if (p.bp_fp_terms == "Drr_only") {
        c.D_r = 0.0;
        c.D_L = 0.0;
        c.D_rL = 0.0;
        c.D_LL = 0.0;
        c.D_aa = 0.0;
        return c;
    }
    if (p.bp_fp_terms == "DLL_only") {
        c.D_r = 0.0;
        c.D_L = 0.0;
        c.D_rr = 0.0;
        c.D_rL = 0.0;
        c.D_aa = 0.0;
        return c;
    }
    if (p.bp_fp_terms == "Daa_only") {
        c.D_r = 0.0;
        c.D_L = 0.0;
        c.D_rr = 0.0;
        c.D_rL = 0.0;
        c.D_LL = 0.0;
        return c;
    }
    return c;
}

// For eta=L^2/2, the transformed drift includes the second derivative of
// eta with respect to the two transverse velocity components. This Itô term
// contributes (D_LL+D_aa)/2 to D_eta alongside L*D_L.
BPEtaCoeff to_eta_coeff(BPCoeff c, double L_value) {
    BPEtaCoeff out;
    out.D_w = c.D_r;
    out.D_eta = L_value * c.D_L + 0.5 * (c.D_LL + c.D_aa);
    out.D_ww = c.D_rr;
    out.D_weta = L_value * c.D_rL;
    out.D_etaeta = L_value * L_value * c.D_LL;
    return out;
}

int bp_first_active_collision_L_cell(const std::vector<double>& L, const std::vector<double>& Lq, const Params& p) {
    const double Lmax = std::max({std::abs(p.L_max), L.empty() ? 0.0 : std::abs(L.back()), 0.0});
    const double floor = std::max(0.0, p.bp_collision_l_min_rel) * Lmax;
    for (int k = 0; k < static_cast<int>(L.size()); ++k) {
        const double weight = L[static_cast<std::size_t>(k)] * Lq[static_cast<std::size_t>(k)];
        if (weight > 0.0 && L[static_cast<std::size_t>(k)] >= floor) {
            return k;
        }
    }
    return static_cast<int>(L.size());
}

// The Bernoulli function x/(exp(x)-1) supplies exponentially fitted face
// weights. Use its series near zero and limiting forms at large signed arguments.
double bp_bernoulli(double x) {
    if (std::abs(x) < 1.0e-6) {
        return 1.0 - 0.5 * x + x * x / 12.0;
    }
    if (x > 60.0) {
        return 0.0;
    }
    if (x < -60.0) {
        return -x;
    }
    return x / std::expm1(x);
}

/// Conservative one-dimensional FP face flux for F=A f - d(B f)/dx.
// Build the drift and diagonal diffusion contribution to a common face.
// The centered option uses upwind drift with a centered B*f difference;
// Chang--Cooper weights resolve the local drift/diffusion balance.
double bp_diagonal_face_flux(
    double drift,
    double B_left,
    double B_right,
    double f_left,
    double f_right,
    double dx,
    const Params& p) {
    const double safe_dx = std::max(dx, 1.0e-300);
    if (p.bp_collision_flux == "centered") {
        const double f_upwind = (drift >= 0.0) ? f_left : f_right;
        return drift * f_upwind - (B_right * f_right - B_left * f_left) / safe_dx;
    }
    const double B_face = 0.5 * (std::max(0.0, B_left) + std::max(0.0, B_right));
    if (!(B_face > 0.0)) {
        const double f_upwind = (drift >= 0.0) ? f_left : f_right;
        return drift * f_upwind;
    }
    const double peclet = drift * safe_dx / B_face;
    return (bp_bernoulli(-peclet) * B_left * f_left
            - bp_bernoulli(peclet) * B_right * f_right)
        / safe_dx;
}

/// Drift/diffusion kernel in a frame where the test velocity is along one axis.
struct BPKernel {
    double drift_parallel = 0.0;
    double diff_parallel = 0.0;
    double diff_perp = 0.0;
};

/// One local interpolation table for the expensive velocity integrals.
struct BPKernelTable {
    std::vector<double> speed;
    std::vector<double> drift_parallel;
    std::vector<double> diff_parallel;
    std::vector<double> diff_perp;
};

/// One local 2D interpolation table for coefficients from the instantaneous DF.
struct BPCoeffTable2D {
    std::vector<double> w_nodes;
    std::vector<double> L_nodes;
    std::vector<BPCoeff> coeff;
    int Nw_table = 0;
    int NL_table = 0;
};

struct BPPartnerCell {
    double wr = 0.0;
    double vt = 0.0;
    double vt2 = 0.0;
    double weight_f = 0.0;
};

struct BPAzimuthNode {
    double cos_phi = 1.0;
    double sin2_phi = 0.0;
    double weight = 0.0;
};

struct BPInterpStencil {
    int index[4] = {0, 0, 0, 0};
    double weight[4] = {0.0, 0.0, 0.0, 0.0};
    int count = 0;
};

struct BPSigmaEvaluator {
    int model = -1;
    double strength = 0.0;
    double sigma0 = 0.0;
    double u0_sq = 0.0;
    double half_n = 0.0;
    double softened_inv_sat = 0.0;
    double softened_inv_A = 0.0;
    double yukawa_x_coeff = 0.0;
    double yukawa_prefactor = 0.0;

    double operator()(double u2) const {
        const double us2 = std::max(u2, 1.0e-28);
        if (model == 0) {
            return strength * sigma0;
        }
        if (model == 1) {
            return strength * sigma0 * std::pow(us2 + u0_sq, -half_n);
        }
        if (model == 2) {
            if (!(softened_inv_A > 0.0)) {
                return 0.0;
            }
            return strength / (softened_inv_sat + us2 * us2 * softened_inv_A);
        }
        if (model == 3) {
            const double x = yukawa_x_coeff * us2;
            const double bracket = std::log1p(x) - x / (1.0 + x);
            return strength * yukawa_prefactor * bracket / (us2 * us2);
        }
        return 0.0;
    }
};

/// Smooth generic transfer cross section sigma_T(u).
// This transfer area weights the small-angle drift/diffusion approximation.
// The finite-angle backends use their corresponding differential or total law
// through their separate collision modules.
double bp_transfer_cross_section(double u, const Params& p) {
    const double us = std::max(std::abs(u), 1.0e-14);
    if (p.bp_cross_section == "constant") {
        return p.bp_collision_strength * p.bp_sigma0;
    }
    if (p.bp_cross_section == "powerlaw") {
        const double u2 = us * us + p.bp_sigma_u0 * p.bp_sigma_u0;
        return p.bp_collision_strength * p.bp_sigma0 * std::pow(u2, -0.5 * p.bp_sigma_n);
    }
    if (p.bp_cross_section == "softened_gravity") {
        // Subhalos are represented as extended Plummer-like scatterers with
        // radius epsilon=bp_subhalo_radius and effective mass bp_subhalo_mass.
        // The harmonic bridge gives a geometric low-speed plateau and the
        // Rutherford v^{-4} high-speed tail.
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
        const double x = (mchi * mchi * us * us) / (mphi * mphi);
        const double bracket = std::log1p(x) - x / (1.0 + x);
        const double sigma = 8.0 * PI * alpha * alpha * bracket / (mchi * mchi * std::pow(us, 4));
        return p.bp_collision_strength * sigma;
    }
    return 0.0;
}

BPSigmaEvaluator make_bp_sigma_evaluator(const Params& p) {
    BPSigmaEvaluator eval;
    eval.strength = p.bp_collision_strength;
    eval.sigma0 = p.bp_sigma0;
    if (p.bp_cross_section == "constant") {
        eval.model = 0;
    } else if (p.bp_cross_section == "powerlaw") {
        eval.model = 1;
        eval.u0_sq = p.bp_sigma_u0 * p.bp_sigma_u0;
        eval.half_n = 0.5 * p.bp_sigma_n;
    } else if (p.bp_cross_section == "softened_gravity") {
        eval.model = 2;
        const double sigma_sat = (p.bp_sigma0 > 0.0)
            ? p.bp_sigma0
            : (2.0 * PI * p.bp_subhalo_radius * p.bp_subhalo_radius);
        const double A = 4.0 * PI * p.vp_gravity * p.vp_gravity
            * p.bp_subhalo_mass * p.bp_subhalo_mass * std::max(p.bp_coulomb_log, 0.0);
        eval.softened_inv_sat = (sigma_sat > 0.0) ? (1.0 / sigma_sat) : std::numeric_limits<double>::infinity();
        eval.softened_inv_A = (A > 0.0) ? (1.0 / A) : 0.0;
    } else if (p.bp_cross_section == "yukawa_born") {
        eval.model = 3;
        eval.yukawa_x_coeff = (p.bp_yukawa_mchi * p.bp_yukawa_mchi)
            / (p.bp_yukawa_mphi * p.bp_yukawa_mphi);
        eval.yukawa_prefactor = 8.0 * PI * p.bp_yukawa_alpha * p.bp_yukawa_alpha
            / (p.bp_yukawa_mchi * p.bp_yukawa_mchi);
    }
    return eval;
}

std::vector<BPAzimuthNode> make_bp_azimuth_nodes(const Params& p) {
    const int nphi = std::max(2, p.bp_coeff_mu_points);
    const double dphi = 2.0 * PI / static_cast<double>(nphi);
    std::vector<BPAzimuthNode> nodes;
    nodes.reserve(static_cast<std::size_t>((nphi + 1) / 2));
    for (int iphi = 0; iphi <= (nphi - 1) / 2; ++iphi) {
        const double phi = (static_cast<double>(iphi) + 0.5) * dphi;
        BPAzimuthNode node;
        node.cos_phi = std::cos(phi);
        const double s = std::sin(phi);
        node.sin2_phi = s * s;
        const bool has_mirror = (iphi != nphi - 1 - iphi);
        node.weight = has_mirror ? (2.0 * dphi) : dphi;
        nodes.push_back(node);
    }
    return nodes;
}

/// Computes local density, radial bulk velocity, and scalar temperature from f(r,v_r,L).
// Integrate the spherical local velocity measure 2*pi*L*dL*dw/r^2.
// Subtract radial streaming from the second moment before defining the scalar
// thermal variance theta=(Var(w)+<v_t^2>)/3.
std::vector<BPLocalBath> compute_bp_local_bath(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p) {
    std::vector<BPLocalBath> bath(p.Nr);
    struct BathSums {
        double mass0 = 0.0;
        double mass1 = 0.0;
        double mass2r = 0.0;
        double mass2t = 0.0;
    };
    const int work = (p.NL > 0) ? p.Nr * p.Nw : 0;
    const int nthreads = effective_threads(p, work);
    const std::size_t stride = static_cast<std::size_t>(p.Nr);
    std::vector<BathSums> partial(static_cast<std::size_t>(nthreads) * stride);
    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int tid) {
            const std::size_t base = static_cast<std::size_t>(tid) * stride;
            for (int q = lo; q < hi; ++q) {
                const int i = q / p.Nw;
                const int j = q - i * p.Nw;
                BathSums cell;
                const double ri = r[static_cast<std::size_t>(i)];
                for (int k = 0; k < p.NL; ++k) {
                    const double weight = wq[static_cast<std::size_t>(j)]
                        * Lq[static_cast<std::size_t>(k)] * L[static_cast<std::size_t>(k)]
                        * f[idx(i, j, k, p)];
                    cell.mass0 += weight;
                    cell.mass1 += w[static_cast<std::size_t>(j)] * weight;
                    cell.mass2r += w[static_cast<std::size_t>(j)] * w[static_cast<std::size_t>(j)] * weight;
                    const double vt = L[static_cast<std::size_t>(k)] / ri;
                    cell.mass2t += vt * vt * weight;
                }
                BathSums& sum = partial[base + static_cast<std::size_t>(i)];
                sum.mass0 += cell.mass0;
                sum.mass1 += cell.mass1;
                sum.mass2r += cell.mass2r;
                sum.mass2t += cell.mass2t;
            }
        });

    for (int i = 0; i < p.Nr; ++i) {
        BathSums total;
        for (int tid = 0; tid < nthreads; ++tid) {
            const BathSums& sum =
                partial[static_cast<std::size_t>(tid) * stride + static_cast<std::size_t>(i)];
            total.mass0 += sum.mass0;
            total.mass1 += sum.mass1;
            total.mass2r += sum.mass2r;
            total.mass2t += sum.mass2t;
        }
        const double ri = r[static_cast<std::size_t>(i)];
        const double factor = 2.0 * PI / (ri * ri);
        const double rho = factor * total.mass0;
        BPLocalBath bi;
        bi.rho = std::max(0.0, rho);
        if (rho > 0.0) {
            bi.mean_vr = factor * total.mass1 / rho;
            const double vr2 = factor * total.mass2r / rho;
            const double vt2 = factor * total.mass2t / rho;
            const double var_r = std::max(0.0, vr2 - bi.mean_vr * bi.mean_vr);
            bi.theta = std::max(p.bp_temperature_floor, (var_r + vt2) / 3.0);
        } else {
            bi.mean_vr = 0.0;
            bi.theta = p.bp_temperature_floor;
        }
        bath[static_cast<std::size_t>(i)] = bi;
    }
    return bath;
}

/// Replaces the local bath by one global Maxwellian bath when requested.
void apply_global_bp_bath(std::vector<BPLocalBath>& bath, const std::vector<double>& r, const Params& p) {
    std::vector<double> r_edges = control_volume_edges(r);
    if (using_origin_parity_inner(p)) {
        r_edges.front() = 0.0;
    }
    double volume_weight = 0.0;
    double mass_weight = 0.0;
    double momentum_weight = 0.0;
    double speed2_weight = 0.0;
    for (int i = 0; i < p.Nr; ++i) {
        const double shell_volume =
            (std::pow(r_edges[static_cast<std::size_t>(i + 1)], 3)
             - std::pow(r_edges[static_cast<std::size_t>(i)], 3)) / 3.0;
        const BPLocalBath& bi = bath[static_cast<std::size_t>(i)];
        const double rho_i = std::max(bi.rho, 0.0);
        const double mass_i = std::max(shell_volume, 0.0) * rho_i;
        volume_weight += std::max(shell_volume, 0.0);
        mass_weight += mass_i;
        momentum_weight += mass_i * bi.mean_vr;
        // Store <v^2> rather than just theta so radial streaming between shells
        // contributes to the temperature of the single global Maxwellian bath.
        speed2_weight += mass_i * (3.0 * bi.theta + bi.mean_vr * bi.mean_vr);
    }
    if (!(volume_weight > 0.0) || !(mass_weight > 0.0)) {
        return;
    }
    const double rho = std::max(mass_weight / volume_weight, 0.0);
    const double mean = momentum_weight / mass_weight;
    const double theta = std::max(
        p.bp_temperature_floor,
        (speed2_weight / mass_weight - mean * mean) / 3.0);
    for (BPLocalBath& b : bath) {
        b.rho = rho;
        b.mean_vr = mean;
        b.theta = theta;
    }
}

/// Evaluates the expensive bath integral for one relative test speed.
// Use a Maxwellian normalized to the bath density. Relative-speed and
// angular quadrature supply friction and parallel/perpendicular diffusion,
// with the zero-test-speed limit made isotropic.
BPKernel bp_kernel_from_maxwellian_speed(double V, const BPLocalBath& bath, const Params& p) {
    BPKernel kernel;
    const double rho_floor = std::max(p.bp_density_floor_abs, p.bp_density_floor_rel * bath.rho);
    if (!(bath.rho > rho_floor) || !(bath.theta > 0.0)) {
        return kernel;
    }
    const double sigma = std::sqrt(std::max(bath.theta, p.bp_temperature_floor));
    const double smax = std::max(p.bp_bath_speed_cutoff * sigma, 1.0e-12);
    const int Ns = std::max(2, p.bp_coeff_speed_points);
    const int Nmu = std::max(2, p.bp_coeff_mu_points);
    const double ds = smax / static_cast<double>(Ns);
    const double dmu = 2.0 / static_cast<double>(Nmu);
    const double norm = bath.rho / std::pow(2.0 * PI * bath.theta, 1.5);

    double drift_parallel_integral = 0.0;
    double diff_parallel_integral = 0.0;
    double diff_perp_integral = 0.0;
    for (int is = 0; is < Ns; ++is) {
        const double s = (static_cast<double>(is) + 0.5) * ds;
        const double fstar = norm * std::exp(-0.5 * s * s / bath.theta);
        const double speed_weight = 2.0 * PI * s * s * ds * fstar;
        for (int im = 0; im < Nmu; ++im) {
            const double mu = -1.0 + (static_cast<double>(im) + 0.5) * dmu;
            const double u2 = std::max(V * V + s * s - 2.0 * V * s * mu, 1.0e-28);
            const double u = std::sqrt(u2);
            const double sigma_T = bp_transfer_cross_section(u, p);
            if (!(sigma_T > 0.0)) {
                continue;
            }
            const double u_parallel = V - s * mu;
            const double u_perp2 = std::max(0.0, s * s * (1.0 - mu * mu));
            const double weight = speed_weight * dmu;
            drift_parallel_integral += weight * u * sigma_T * u_parallel;
            diff_parallel_integral += weight * u * u * u * sigma_T
                * std::max(0.0, 1.0 - u_parallel * u_parallel / u2);
            diff_perp_integral += weight * u * u * u * sigma_T
                * std::max(0.0, 1.0 - 0.5 * u_perp2 / u2);
        }
    }

    kernel.drift_parallel = -0.5 * drift_parallel_integral;
    kernel.diff_parallel = 0.25 * diff_parallel_integral;
    kernel.diff_perp = 0.25 * diff_perp_integral;
    if (V < 1.0e-10) {
        kernel.drift_parallel = 0.0;
        const double iso = std::max(0.0, (kernel.diff_parallel + 2.0 * kernel.diff_perp) / 3.0);
        kernel.diff_parallel = iso;
        kernel.diff_perp = iso;
    }
    kernel.diff_parallel = std::max(0.0, kernel.diff_parallel);
    kernel.diff_perp = std::max(0.0, kernel.diff_perp);
    return kernel;
}

/// Projects a parallel/perpendicular velocity-space kernel onto spherical (v_r,L) coordinates.
// Project the diffusion tensor relative to the test velocity onto (w,L).
// Factors of r convert transverse velocities to angular momentum, while D_aa
// retains the second transverse direction required by the eta drift.
BPCoeff bp_project_kernel(double r, double cr, double ct, const BPKernel& kernel) {
    BPCoeff coeff;
    const double V = std::sqrt(cr * cr + ct * ct);
    const double invV = (V > 1.0e-10) ? (1.0 / V) : 0.0;
    const double er = cr * invV;
    const double et = ct * invV;
    const double anis = kernel.diff_parallel - kernel.diff_perp;
    coeff.D_r = kernel.drift_parallel * er;
    coeff.D_L = r * kernel.drift_parallel * et;
    coeff.D_rr = std::max(0.0, kernel.diff_perp + anis * er * er);
    coeff.D_rL = r * anis * er * et;
    coeff.D_LL = std::max(0.0, r * r * (kernel.diff_perp + anis * et * et));
    coeff.D_aa = std::max(0.0, r * r * kernel.diff_perp);
    return coeff;
}

/// Cubic Hermite interpolation through tabulated values, with linear endpoint fallback.
double bp_cubic_interpolate(const std::vector<double>& x, const std::vector<double>& y, double xp) {
    const int n = static_cast<int>(x.size());
    if (n == 0) {
        return 0.0;
    }
    if (n == 1 || xp <= x.front()) {
        return y.front();
    }
    if (xp >= x.back()) {
        return y.back();
    }
    const int i = lower_cell(x, xp);
    const double x0 = x[static_cast<std::size_t>(i)];
    const double x1 = x[static_cast<std::size_t>(i + 1)];
    const double h = x1 - x0;
    if (!(h > 0.0)) {
        return y[static_cast<std::size_t>(i)];
    }
    const double t = (xp - x0) / h;
    const double y0 = y[static_cast<std::size_t>(i)];
    const double y1 = y[static_cast<std::size_t>(i + 1)];
    if (i == 0 || i + 2 >= n) {
        return (1.0 - t) * y0 + t * y1;
    }
    const double m0 = (y[static_cast<std::size_t>(i + 1)] - y[static_cast<std::size_t>(i - 1)])
        / (x[static_cast<std::size_t>(i + 1)] - x[static_cast<std::size_t>(i - 1)]);
    const double m1 = (y[static_cast<std::size_t>(i + 2)] - y[static_cast<std::size_t>(i)])
        / (x[static_cast<std::size_t>(i + 2)] - x[static_cast<std::size_t>(i)]);
    const double h00 = (1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t);
    const double h10 = t * (1.0 - t) * (1.0 - t);
    const double h01 = t * t * (3.0 - 2.0 * t);
    const double h11 = t * t * (t - 1.0);
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
}

BPInterpStencil bp_cubic_stencil(const std::vector<double>& x, double xp) {
    BPInterpStencil stencil;
    const int n = static_cast<int>(x.size());
    if (n == 0) {
        return stencil;
    }
    if (n == 1 || xp <= x.front()) {
        stencil.index[0] = 0;
        stencil.weight[0] = 1.0;
        stencil.count = 1;
        return stencil;
    }
    if (xp >= x.back()) {
        stencil.index[0] = n - 1;
        stencil.weight[0] = 1.0;
        stencil.count = 1;
        return stencil;
    }
    const int i = lower_cell(x, xp);
    const double x0 = x[static_cast<std::size_t>(i)];
    const double x1 = x[static_cast<std::size_t>(i + 1)];
    const double h = x1 - x0;
    if (!(h > 0.0)) {
        stencil.index[0] = i;
        stencil.weight[0] = 1.0;
        stencil.count = 1;
        return stencil;
    }
    const double t = (xp - x0) / h;
    if (i == 0 || i + 2 >= n) {
        stencil.index[0] = i;
        stencil.index[1] = i + 1;
        stencil.weight[0] = 1.0 - t;
        stencil.weight[1] = t;
        stencil.count = 2;
        return stencil;
    }

    const double h00 = (1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t);
    const double h10 = t * (1.0 - t) * (1.0 - t);
    const double h01 = t * t * (3.0 - 2.0 * t);
    const double h11 = t * t * (t - 1.0);
    const double d0 = x[static_cast<std::size_t>(i + 1)] - x[static_cast<std::size_t>(i - 1)];
    const double d1 = x[static_cast<std::size_t>(i + 2)] - x[static_cast<std::size_t>(i)];
    stencil.index[0] = i - 1;
    stencil.index[1] = i;
    stencil.index[2] = i + 1;
    stencil.index[3] = i + 2;
    stencil.weight[0] = -h10 * h / d0;
    stencil.weight[1] = h00 - h11 * h / d1;
    stencil.weight[2] = h01 + h10 * h / d0;
    stencil.weight[3] = h11 * h / d1;
    stencil.count = 4;
    return stencil;
}

std::vector<BPInterpStencil> bp_interpolation_stencils(
    const std::vector<double>& nodes,
    const std::vector<double>& grid) {
    std::vector<BPInterpStencil> stencils(grid.size());
    for (std::size_t q = 0; q < grid.size(); ++q) {
        stencils[q] = bp_cubic_stencil(nodes, grid[q]);
    }
    return stencils;
}

/// Builds a local interpolation table in relative speed for one radial shell.
BPKernelTable build_bp_kernel_table(
    int i,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const BPLocalBath& bath,
    const Params& p) {
    BPKernelTable table;
    const int n = std::max(4, p.bp_coeff_table_points);
    table.speed.assign(n, 0.0);
    table.drift_parallel.assign(n, 0.0);
    table.diff_parallel.assign(n, 0.0);
    table.diff_perp.assign(n, 0.0);

    double vmax = 0.0;
    const double ri = r[static_cast<std::size_t>(i)];
    for (double vri : w) {
        for (double Li : L) {
            const double cr = vri - bath.mean_vr;
            const double ct = Li / ri;
            vmax = std::max(vmax, std::sqrt(cr * cr + ct * ct));
        }
    }
    vmax = std::max(vmax, std::sqrt(std::max(bath.theta, p.bp_temperature_floor)));
    for (int q = 0; q < n; ++q) {
        const double x = static_cast<double>(q) / static_cast<double>(n - 1);
        table.speed[static_cast<std::size_t>(q)] = vmax * x;
        const BPKernel kernel = bp_kernel_from_maxwellian_speed(table.speed[static_cast<std::size_t>(q)], bath, p);
        table.drift_parallel[static_cast<std::size_t>(q)] = kernel.drift_parallel;
        table.diff_parallel[static_cast<std::size_t>(q)] = kernel.diff_parallel;
        table.diff_perp[static_cast<std::size_t>(q)] = kernel.diff_perp;
    }
    return table;
}

/// Interpolates the precomputed speed table without adding smoothing.
BPKernel interpolate_bp_kernel_table(const BPKernelTable& table, double V) {
    BPKernel kernel;
    kernel.drift_parallel = bp_cubic_interpolate(table.speed, table.drift_parallel, V);
    kernel.diff_parallel = std::max(0.0, bp_cubic_interpolate(table.speed, table.diff_parallel, V));
    kernel.diff_perp = std::max(0.0, bp_cubic_interpolate(table.speed, table.diff_perp, V));
    return kernel;
}

std::vector<int> bp_table_indices(int n_grid, int n_table) {
    const int n = std::clamp(n_table, 2, std::max(2, n_grid));
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(n));
    for (int q = 0; q < n; ++q) {
        const int index = static_cast<int>(std::lround(
            static_cast<double>(q) * static_cast<double>(n_grid - 1) / static_cast<double>(n - 1)));
        if (out.empty() || index != out.back()) {
            out.push_back(index);
        }
    }
    return out;
}

// Integrate actual partner velocities instead of replacing their distribution
// by a Maxwellian. The relative tangential azimuth restores the third velocity
// direction in the spherical representation.
BPCoeff bp_instantaneous_coeff_at(
    double r,
    double vr,
    double Ltest,
    const std::vector<BPPartnerCell>& partners,
    const std::vector<BPAzimuthNode>& azimuth,
    const BPSigmaEvaluator& sigma_eval) {
    BPCoeff out;
    if (partners.empty()) {
        return out;
    }
    const double vt = Ltest / r;
    const double vt2 = vt * vt;
    for (const BPPartnerCell& partner : partners) {
        if (!(partner.weight_f > 0.0)) {
            continue;
        }
        const double dv_r = vr - partner.wr;
        const double dv_r2 = dv_r * dv_r;
        const double base = dv_r2 + vt2 + partner.vt2;
        const double cross = 2.0 * vt * partner.vt;
        for (const BPAzimuthNode& node : azimuth) {
            const double c = node.cos_phi;
            const double dv_y = vt - partner.vt * c;
            const double dv_z2 = partner.vt2 * node.sin2_phi;
            const double u2 = std::max(base - cross * c, 1.0e-28);
            const double u = std::sqrt(u2);
            const double sigma_T = sigma_eval(u2);
            if (!(sigma_T > 0.0)) {
                continue;
            }
            const double weight = partner.weight_f * node.weight;
            const double drift_weight = -0.5 * weight * u * sigma_T;
            const double diff_weight = 0.25 * weight * u * u * u * sigma_T;
            out.D_r += drift_weight * dv_r;
            out.D_L += r * drift_weight * dv_y;
            out.D_rr += diff_weight * std::max(0.0, 1.0 - dv_r * dv_r / u2);
            out.D_rL += r * diff_weight * (-dv_r * dv_y / u2);
            out.D_LL += r * r * diff_weight * std::max(0.0, 1.0 - dv_y * dv_y / u2);
            out.D_aa += r * r * diff_weight * std::max(0.0, 1.0 - dv_z2 / u2);
        }
    }
    out.D_rr = std::max(0.0, out.D_rr);
    out.D_LL = std::max(0.0, out.D_LL);
    out.D_aa = std::max(0.0, out.D_aa);
    return out;
}

BPCoeffTable2D build_bp_instantaneous_table(
    int i,
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& w_nodes,
    const std::vector<double>& L_nodes,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const std::vector<BPAzimuthNode>& azimuth,
    const BPSigmaEvaluator& sigma_eval,
    const Params& p) {
    BPCoeffTable2D table;
    table.w_nodes = w_nodes;
    table.L_nodes = L_nodes;
    table.Nw_table = static_cast<int>(w_nodes.size());
    table.NL_table = static_cast<int>(L_nodes.size());
    table.coeff.assign(w_nodes.size() * L_nodes.size(), BPCoeff{});

    const double ri = r[static_cast<std::size_t>(i)];
    double fmax = 0.0;
    for (int jp = 0; jp < p.Nw; ++jp) {
        for (int kp = 0; kp < p.NL; ++kp) {
            fmax = std::max(fmax, f[idx(i, jp, kp, p)]);
        }
    }
    const double f_threshold = std::max(p.f_floor, p.bp_density_floor_rel * fmax);
    std::vector<BPPartnerCell> partners;
    partners.reserve(static_cast<std::size_t>(p.Nw) * static_cast<std::size_t>(p.NL));
    for (int jp = 0; jp < p.Nw; ++jp) {
        for (int kp = 0; kp < p.NL; ++kp) {
            const double fp = f[idx(i, jp, kp, p)];
            if (!(fp > f_threshold)) {
                continue;
            }
            BPPartnerCell partner;
            partner.wr = w[static_cast<std::size_t>(jp)];
            partner.vt = L[static_cast<std::size_t>(kp)] / ri;
            partner.vt2 = partner.vt * partner.vt;
            partner.weight_f = wq[static_cast<std::size_t>(jp)]
                * Lq[static_cast<std::size_t>(kp)]
                * L[static_cast<std::size_t>(kp)]
                * fp / (ri * ri);
            if (partner.weight_f > 0.0) {
                partners.push_back(partner);
            }
        }
    }
    // An optional coefficient approximation retains partner cells by descending
    // quadrature mass up to the configured fraction. Its omitted bath support is
    // part of this approximation and remains distinct from transport of f.
    if (p.bp_partner_mass_tol > 0.0 && partners.size() > 1) {
        double total_weight = 0.0;
        for (const BPPartnerCell& partner : partners) {
            total_weight += partner.weight_f;
        }
        if (total_weight > 0.0) {
            const double target_weight = std::clamp(1.0 - p.bp_partner_mass_tol, 0.0, 1.0) * total_weight;
            std::sort(partners.begin(), partners.end(),
                [](const BPPartnerCell& a, const BPPartnerCell& b) {
                    return a.weight_f > b.weight_f;
                });
            double kept_weight = 0.0;
            std::size_t keep_count = 0;
            while (keep_count < partners.size() && kept_weight < target_weight) {
                kept_weight += partners[keep_count].weight_f;
                ++keep_count;
            }
            partners.resize(std::max<std::size_t>(1, keep_count));
        }
    }

    for (int jw = 0; jw < table.Nw_table; ++jw) {
        for (int jL = 0; jL < table.NL_table; ++jL) {
            table.coeff[static_cast<std::size_t>(jw) * static_cast<std::size_t>(table.NL_table)
                        + static_cast<std::size_t>(jL)] =
                bp_instantaneous_coeff_at(
                    ri,
                    table.w_nodes[static_cast<std::size_t>(jw)],
                    table.L_nodes[static_cast<std::size_t>(jL)],
                    partners,
                    azimuth,
                    sigma_eval);
        }
    }
    return table;
}

BPCoeff interpolate_bp_coeff_table2d(
    const BPCoeffTable2D& table,
    const BPInterpStencil& w_stencil,
    const BPInterpStencil& L_stencil) {
    BPCoeff out;
    if (table.Nw_table == 0 || table.NL_table == 0) {
        return out;
    }
    for (int aw = 0; aw < w_stencil.count; ++aw) {
        const int jw = std::clamp(w_stencil.index[aw], 0, table.Nw_table - 1);
        const double ww = w_stencil.weight[aw];
        for (int aL = 0; aL < L_stencil.count; ++aL) {
            const int jL = std::clamp(L_stencil.index[aL], 0, table.NL_table - 1);
            const double weight = ww * L_stencil.weight[aL];
            const BPCoeff& c = table.coeff[
                static_cast<std::size_t>(jw) * static_cast<std::size_t>(table.NL_table)
                + static_cast<std::size_t>(jL)];
            out.D_r += weight * c.D_r;
            out.D_L += weight * c.D_L;
            out.D_rr += weight * c.D_rr;
            out.D_rL += weight * c.D_rL;
            out.D_LL += weight * c.D_LL;
            out.D_aa += weight * c.D_aa;
        }
    }
    out.D_rr = std::max(0.0, out.D_rr);
    out.D_LL = std::max(0.0, out.D_LL);
    out.D_aa = std::max(0.0, out.D_aa);
    return out;
}

std::vector<BPCoeff> compute_bp_coefficients_instantaneous_df(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p) {
    std::vector<BPCoeff> coeff(f.size());
    const std::vector<int> w_indices = bp_table_indices(p.Nw, p.bp_coeff_table_points);
    const std::vector<int> L_indices = bp_table_indices(p.NL, p.bp_coeff_table_points);
    std::vector<double> w_nodes(w_indices.size());
    std::vector<double> L_nodes(L_indices.size());
    for (std::size_t q = 0; q < w_indices.size(); ++q) {
        w_nodes[q] = w[static_cast<std::size_t>(w_indices[q])];
    }
    for (std::size_t q = 0; q < L_indices.size(); ++q) {
        L_nodes[q] = L[static_cast<std::size_t>(L_indices[q])];
    }
    const std::vector<BPInterpStencil> w_stencils = bp_interpolation_stencils(w_nodes, w);
    const std::vector<BPInterpStencil> L_stencils = bp_interpolation_stencils(L_nodes, L);
    const std::vector<BPAzimuthNode> azimuth = make_bp_azimuth_nodes(p);
    const BPSigmaEvaluator sigma_eval = make_bp_sigma_evaluator(p);

    std::vector<std::vector<BPPartnerCell>> partners_by_shell(static_cast<std::size_t>(p.Nr));
    parallel_for_dynamic_chunks(0, p.Nr, effective_threads(p, p.Nr), 1,
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                const double ri = r[static_cast<std::size_t>(i)];
                double fmax = 0.0;
                for (int jp = 0; jp < p.Nw; ++jp) {
                    for (int kp = 0; kp < p.NL; ++kp) {
                        fmax = std::max(fmax, f[idx(i, jp, kp, p)]);
                    }
                }
                const double f_threshold = std::max(p.f_floor, p.bp_density_floor_rel * fmax);
                std::vector<BPPartnerCell>& partners = partners_by_shell[static_cast<std::size_t>(i)];
                partners.reserve(static_cast<std::size_t>(p.Nw) * static_cast<std::size_t>(p.NL));
                for (int jp = 0; jp < p.Nw; ++jp) {
                    for (int kp = 0; kp < p.NL; ++kp) {
                        const double fp = f[idx(i, jp, kp, p)];
                        if (!(fp > f_threshold)) {
                            continue;
                        }
                        BPPartnerCell partner;
                        partner.wr = w[static_cast<std::size_t>(jp)];
                        partner.vt = L[static_cast<std::size_t>(kp)] / ri;
                        partner.vt2 = partner.vt * partner.vt;
                        partner.weight_f = wq[static_cast<std::size_t>(jp)]
                            * Lq[static_cast<std::size_t>(kp)]
                            * L[static_cast<std::size_t>(kp)]
                            * fp / (ri * ri);
                        if (partner.weight_f > 0.0) {
                            partners.push_back(partner);
                        }
                    }
                }
                if (p.bp_partner_mass_tol > 0.0 && partners.size() > 1) {
                    double total_weight = 0.0;
                    for (const BPPartnerCell& partner : partners) {
                        total_weight += partner.weight_f;
                    }
                    if (total_weight > 0.0) {
                        const double target_weight =
                            std::clamp(1.0 - p.bp_partner_mass_tol, 0.0, 1.0) * total_weight;
                        std::sort(partners.begin(), partners.end(),
                            [](const BPPartnerCell& a, const BPPartnerCell& b) {
                                return a.weight_f > b.weight_f;
                            });
                        double kept_weight = 0.0;
                        std::size_t keep_count = 0;
                        while (keep_count < partners.size() && kept_weight < target_weight) {
                            kept_weight += partners[keep_count].weight_f;
                            ++keep_count;
                        }
                        partners.resize(std::max<std::size_t>(1, keep_count));
                    }
                }
            }
        });

    std::vector<BPCoeffTable2D> tables(static_cast<std::size_t>(p.Nr));
    for (int i = 0; i < p.Nr; ++i) {
        BPCoeffTable2D& table = tables[static_cast<std::size_t>(i)];
        table.w_nodes = w_nodes;
        table.L_nodes = L_nodes;
        table.Nw_table = static_cast<int>(w_nodes.size());
        table.NL_table = static_cast<int>(L_nodes.size());
        table.coeff.assign(w_nodes.size() * L_nodes.size(), BPCoeff{});
    }

    const int table_Nw = static_cast<int>(w_nodes.size());
    const int table_NL = static_cast<int>(L_nodes.size());
    const int table_cells = table_Nw * table_NL;
    const int table_work = p.Nr * table_cells;
    parallel_for_dynamic_chunks(0, table_work, effective_threads(p, table_work), 1,
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / table_cells;
                const int rem = q - i * table_cells;
                const int jw = rem / table_NL;
                const int jL = rem - jw * table_NL;
                BPCoeffTable2D& table = tables[static_cast<std::size_t>(i)];
                table.coeff[static_cast<std::size_t>(jw) * static_cast<std::size_t>(table_NL)
                            + static_cast<std::size_t>(jL)] =
                    bp_instantaneous_coeff_at(
                        r[static_cast<std::size_t>(i)],
                        w_nodes[static_cast<std::size_t>(jw)],
                        L_nodes[static_cast<std::size_t>(jL)],
                        partners_by_shell[static_cast<std::size_t>(i)],
                        azimuth,
                        sigma_eval);
            }
        });

    const int coeff_work = static_cast<int>(coeff.size());
    parallel_for_chunks(0, coeff_work, effective_threads(p, coeff_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                coeff[static_cast<std::size_t>(q)] = interpolate_bp_coeff_table2d(
                    tables[static_cast<std::size_t>(i)],
                    w_stencils[static_cast<std::size_t>(j)],
                    L_stencils[static_cast<std::size_t>(k)]);
            }
        });
    return coeff;
}

/// Computes all local spherical FP coefficients for the current distribution.
std::vector<BPCoeff> compute_bp_coefficients(
    const std::vector<double>& f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p) {
    if (p.bp_bath == "instantaneous_df") {
        return compute_bp_coefficients_instantaneous_df(f, r, w, L, wq, Lq, p);
    }
    std::vector<BPLocalBath> bath = compute_bp_local_bath(f, r, w, L, wq, Lq, p);
    if (p.bp_bath == "global_maxwellian") {
        apply_global_bp_bath(bath, r, p);
    }

    const int table_points = std::max(4, p.bp_coeff_table_points);
    std::vector<BPKernelTable> tables(static_cast<std::size_t>(p.Nr));
    parallel_for_chunks(0, p.Nr, effective_threads(p, p.Nr),
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                const double ri = r[static_cast<std::size_t>(i)];
                const BPLocalBath& bi = bath[static_cast<std::size_t>(i)];
                BPKernelTable& table = tables[static_cast<std::size_t>(i)];
                table.speed.assign(table_points, 0.0);
                table.drift_parallel.assign(table_points, 0.0);
                table.diff_parallel.assign(table_points, 0.0);
                table.diff_perp.assign(table_points, 0.0);
                double local_vmax = 0.0;
                for (double vri : w) {
                    for (double Li : L) {
                        const double cr = vri - bi.mean_vr;
                        const double ct = Li / ri;
                        local_vmax = std::max(local_vmax, std::sqrt(cr * cr + ct * ct));
                    }
                }
                local_vmax = std::max(local_vmax, std::sqrt(std::max(bi.theta, p.bp_temperature_floor)));
                for (int q = 0; q < table_points; ++q) {
                    const double x = static_cast<double>(q) / static_cast<double>(table_points - 1);
                    table.speed[static_cast<std::size_t>(q)] = local_vmax * x;
                }
            }
        });

    const int table_work = p.Nr * table_points;
    parallel_for_chunks(0, table_work, effective_threads(p, table_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / table_points;
                const int s = q - i * table_points;
                const BPKernel kernel = bp_kernel_from_maxwellian_speed(
                    tables[static_cast<std::size_t>(i)].speed[static_cast<std::size_t>(s)],
                    bath[static_cast<std::size_t>(i)],
                    p);
                BPKernelTable& table = tables[static_cast<std::size_t>(i)];
                table.drift_parallel[static_cast<std::size_t>(s)] = kernel.drift_parallel;
                table.diff_parallel[static_cast<std::size_t>(s)] = kernel.diff_parallel;
                table.diff_perp[static_cast<std::size_t>(s)] = kernel.diff_perp;
            }
        });

    std::vector<BPCoeff> coeff(f.size());
    const int coeff_work = static_cast<int>(coeff.size());
    parallel_for_chunks(0, coeff_work, effective_threads(p, coeff_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double ri = r[static_cast<std::size_t>(i)];
                const BPLocalBath& bi = bath[static_cast<std::size_t>(i)];
                const double cr = w[static_cast<std::size_t>(j)] - bi.mean_vr;
                const double ct = L[static_cast<std::size_t>(k)] / ri;
                const double V = std::sqrt(cr * cr + ct * ct);
                coeff[static_cast<std::size_t>(q)] = bp_project_kernel(
                    ri, cr, ct, interpolate_bp_kernel_table(tables[static_cast<std::size_t>(i)], V));
            }
        });
    return coeff;
}

/// Returns a stable explicit source-step estimate for the fixed coefficient table.
// Estimate explicit stability from drift divided by cell width and diffusion
// divided by squared width, including mixed w--eta diffusion. The selected
// collision integrator uses these rates to choose substeps or RKL2 stages.
BPCollisionStepInfo bp_collision_step_info(
    const std::vector<BPCoeff>& coeff,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    double dt,
    const Params& p) {
    BPCollisionStepInfo info;
    auto local_spacing = [](const std::vector<double>& x, int i) {
        if (x.size() < 2) {
            return 1.0;
        }
        if (i <= 0) {
            return std::max(x[1] - x[0], 1.0e-14);
        }
        if (i + 1 >= static_cast<int>(x.size())) {
            return std::max(x[x.size() - 1] - x[x.size() - 2], 1.0e-14);
        }
        return std::max(std::min(x[static_cast<std::size_t>(i + 1)] - x[static_cast<std::size_t>(i)],
                                 x[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i - 1)]),
                        1.0e-14);
    };
    std::vector<double> eta(L.size(), 0.0);
    for (std::size_t k = 0; k < L.size(); ++k) {
        eta[k] = 0.5 * L[k] * L[k];
    }
    const int first_active_L = bp_first_active_collision_L_cell(L, Lq, p);
    double max_D_rr_rate = 0.0;
    double max_D_etaeta_rate = 0.0;
    double max_D_weta_rate = 0.0;
    double max_D_w_rate = 0.0;
    double max_D_eta_rate = 0.0;
    struct RateStats {
        double max_D_rr = 0.0;
        double max_D_LL = 0.0;
        double max_abs_D_rL = 0.0;
        double max_abs_D_r = 0.0;
        double max_abs_D_L = 0.0;
        double max_D_rr_rate = 0.0;
        double max_D_etaeta_rate = 0.0;
        double max_D_weta_rate = 0.0;
        double max_D_w_rate = 0.0;
        double max_D_eta_rate = 0.0;
    };
    const int work = static_cast<int>(coeff.size());
    const int nthreads = effective_threads(p, work);
    std::vector<RateStats> local(static_cast<std::size_t>(nthreads));
    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int tid) {
            RateStats stats;
            for (int q_int = lo; q_int < hi; ++q_int) {
                const std::size_t q = static_cast<std::size_t>(q_int);
                const BPCoeff c = mask_bp_fp_terms(coeff[q], p);
                stats.max_D_rr = std::max(stats.max_D_rr, c.D_rr);
                stats.max_D_LL = std::max(stats.max_D_LL, c.D_LL);
                stats.max_abs_D_rL = std::max(stats.max_abs_D_rL, std::abs(c.D_rL));
                stats.max_abs_D_r = std::max(stats.max_abs_D_r, std::abs(c.D_r));
                stats.max_abs_D_L = std::max(stats.max_abs_D_L, std::abs(c.D_L));
                const int rem = q_int - (q_int / (p.Nw * p.NL)) * (p.Nw * p.NL);
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double dw = local_spacing(w, j);
                const double deta = local_spacing(eta, k);
                const BPEtaCoeff ce = to_eta_coeff(c, L[static_cast<std::size_t>(k)]);
                stats.max_D_rr_rate = std::max(stats.max_D_rr_rate, ce.D_ww / (dw * dw));
                if (p.bp_collision_flux == "centered") {
                    stats.max_D_w_rate = std::max(stats.max_D_w_rate, std::abs(ce.D_w) / dw);
                }
                if (k >= first_active_L) {
                    stats.max_D_etaeta_rate = std::max(stats.max_D_etaeta_rate, ce.D_etaeta / (deta * deta));
                    stats.max_D_weta_rate = std::max(stats.max_D_weta_rate, std::abs(ce.D_weta) / (dw * deta));
                    if (p.bp_collision_flux == "centered") {
                        stats.max_D_eta_rate = std::max(stats.max_D_eta_rate, std::abs(ce.D_eta) / deta);
                    }
                }
            }
            local[static_cast<std::size_t>(tid)] = stats;
        });
    for (const RateStats& stats : local) {
        info.max_D_rr = std::max(info.max_D_rr, stats.max_D_rr);
        info.max_D_LL = std::max(info.max_D_LL, stats.max_D_LL);
        info.max_abs_D_rL = std::max(info.max_abs_D_rL, stats.max_abs_D_rL);
        info.max_abs_D_r = std::max(info.max_abs_D_r, stats.max_abs_D_r);
        info.max_abs_D_L = std::max(info.max_abs_D_L, stats.max_abs_D_L);
        max_D_rr_rate = std::max(max_D_rr_rate, stats.max_D_rr_rate);
        max_D_etaeta_rate = std::max(max_D_etaeta_rate, stats.max_D_etaeta_rate);
        max_D_weta_rate = std::max(max_D_weta_rate, stats.max_D_weta_rate);
        max_D_w_rate = std::max(max_D_w_rate, stats.max_D_w_rate);
        max_D_eta_rate = std::max(max_D_eta_rate, stats.max_D_eta_rate);
    }
    double limit = dt;
    if (max_D_rr_rate > 0.0) {
        limit = std::min(limit, p.bp_collision_cfl / max_D_rr_rate);
    }
    if (max_D_etaeta_rate > 0.0) {
        limit = std::min(limit, p.bp_collision_cfl / max_D_etaeta_rate);
    }
    if (max_D_weta_rate > 0.0) {
        limit = std::min(limit, p.bp_collision_cfl / max_D_weta_rate);
    }
    if (max_D_w_rate > 0.0) {
        limit = std::min(limit, p.bp_collision_cfl / max_D_w_rate);
    }
    if (max_D_eta_rate > 0.0) {
        limit = std::min(limit, p.bp_collision_cfl / max_D_eta_rate);
    }
    if (!(limit > 0.0) || !std::isfinite(limit)) {
        limit = dt;
    }
    info.explicit_dt_limit = limit;
    info.explicit_substeps = std::max(1, static_cast<int>(std::ceil(dt / std::max(limit, 1.0e-300))));
    if (p.bp_collision_integrator == "explicit"
        && info.explicit_substeps > p.bp_collision_max_substeps) {
        throw std::runtime_error(
            "Fokker-Planck collision step needs more explicit substeps than bp_collision_max_substeps. "
            "Reduce dt_max or bp_collision_strength, increase bp_collision_cfl cautiously, "
            "increase bp_collision_max_substeps, or use bp_collision_integrator=sts_rkl2.");
    }
    info.substeps = (p.bp_collision_integrator == "explicit")
        ? info.explicit_substeps
        : std::min(info.explicit_substeps, std::max(1, p.bp_collision_max_substeps));
    info.collision_dt = dt / static_cast<double>(info.substeps);
    return info;
}

// Coefficient refresh uses the estimated times for order-unity speed drift
// or velocity diffusion. A configured distribution-weighted quantile defines
// this cache clock separately from the explicit flux stability limit.
double estimate_bp_collision_time(
    const std::vector<double>& f,
    const std::vector<BPCoeff>& coeff,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    const Params& p) {
    struct TimeWeight {
        double time = 0.0;
        double weight = 0.0;
    };
    std::vector<TimeWeight> samples;
    const double fmax = max_value_parallel(f, p);
    const double threshold = std::max(p.f_floor, p.active_floor_rel * fmax);
    const double v_floor2 = 1.0e-20;
    const int nthreads = effective_threads(p, p.Nr);
    std::vector<std::vector<TimeWeight>> partial(static_cast<std::size_t>(nthreads));
    parallel_for_chunks(0, p.Nr, nthreads,
        [&](int ilo, int ihi, int tid) {
            std::vector<TimeWeight>& local_samples = partial[static_cast<std::size_t>(tid)];
            local_samples.reserve(
                static_cast<std::size_t>(std::max(1, ihi - ilo))
                * static_cast<std::size_t>(p.Nw)
                * static_cast<std::size_t>(std::max(1, p.NL / 8)));
            for (int i = ilo; i < ihi; ++i) {
                const double ri = std::max(r[static_cast<std::size_t>(i)], 1.0e-300);
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        const std::size_t q = idx(i, j, k, p);
                        if (!(f[q] > threshold)) {
                            continue;
                        }
                        const BPCoeff c = mask_bp_fp_terms(coeff[q], p);
                        const double vt = L[static_cast<std::size_t>(k)] / ri;
                        const double v2 = std::max(
                            w[static_cast<std::size_t>(j)] * w[static_cast<std::size_t>(j)] + vt * vt,
                            v_floor2);
                        const double speed = std::sqrt(v2);
                        const double trace = c.D_rr + (c.D_LL + c.D_aa) / (ri * ri);
                        const double drift = std::sqrt(c.D_r * c.D_r + (c.D_L / ri) * (c.D_L / ri));
                        double time = std::numeric_limits<double>::infinity();
                        if (trace > 0.0) {
                            time = std::min(time, v2 / trace);
                        }
                        if (drift > 0.0) {
                            time = std::min(time, speed / drift);
                        }
                        const double weight = wq[static_cast<std::size_t>(j)] * Lq[static_cast<std::size_t>(k)]
                            * L[static_cast<std::size_t>(k)] * f[q];
                        if (time > 0.0 && std::isfinite(time) && weight > 0.0) {
                            local_samples.push_back({time, weight});
                        }
                    }
                }
            }
        });
    std::size_t sample_count = 0;
    for (const std::vector<TimeWeight>& part : partial) {
        sample_count += part.size();
    }
    samples.reserve(sample_count);
    for (std::vector<TimeWeight>& part : partial) {
        samples.insert(samples.end(), part.begin(), part.end());
    }
    if (samples.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    std::sort(samples.begin(), samples.end(), [](const TimeWeight& a, const TimeWeight& b) {
        return a.time < b.time;
    });
    const double q = std::clamp(p.bp_coeff_refresh_time_quantile, 0.0, 1.0);
    if (q <= 0.0) {
        return samples.front().time;
    }
    double total_weight = 0.0;
    for (const TimeWeight& sample : samples) {
        total_weight += sample.weight;
    }
    const double target = q * total_weight;
    double accum = 0.0;
    for (const TimeWeight& sample : samples) {
        accum += sample.weight;
        if (accum >= target) {
            return sample.time;
        }
    }
    return samples.back().time;
}

/// Applies one fixed-coefficient conservative spherical FP update to a distribution.
// At fixed radius, f evolves as the divergence of w and eta fluxes.
// Interior faces share one flux, zero-initialized boundary faces close the
// collision domain, and Delta_eta=L*Lq gives the angular cell measure.
void apply_bp_fixed_coefficients_once(
    std::vector<double>& f,
    const std::vector<BPCoeff>& coeff,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    double dt,
    const Params& p) {
    const std::vector<double> w_edges = control_volume_edges(w);
    std::vector<double> eta(L.size(), 0.0);
    for (std::size_t k = 0; k < L.size(); ++k) {
        eta[k] = 0.5 * L[k] * L[k];
    }
    std::vector<double> flux_w(
        static_cast<std::size_t>(p.Nr) * static_cast<std::size_t>(p.Nw + 1) * static_cast<std::size_t>(p.NL),
        0.0);
    std::vector<double> flux_eta(
        static_cast<std::size_t>(p.Nr) * static_cast<std::size_t>(p.Nw) * static_cast<std::size_t>(p.NL + 1),
        0.0);

    // Mixed diffusion appears in both flux directions. Interpolate D_weta*f
    // at each face before taking the transverse derivative on the same grid.
    auto cross_at_w_face = [&](int i, int j_face, int k) {
        if (j_face <= 0 || j_face >= p.Nw || k < 0 || k >= p.NL) {
            return 0.0;
        }
        const std::size_t a = idx(i, j_face - 1, k, p);
        const std::size_t b = idx(i, j_face, k, p);
        const BPCoeff ca = mask_bp_fp_terms(coeff[a], p);
        const BPCoeff cb = mask_bp_fp_terms(coeff[b], p);
        const BPEtaCoeff ea = to_eta_coeff(ca, L[static_cast<std::size_t>(k)]);
        const BPEtaCoeff eb = to_eta_coeff(cb, L[static_cast<std::size_t>(k)]);
        return 0.5 * (ea.D_weta * f[a] + eb.D_weta * f[b]);
    };

    auto eta_cell_weight = [&](int k) {
        return L[static_cast<std::size_t>(k)] * Lq[static_cast<std::size_t>(k)];
    };

    const int first_active_L = bp_first_active_collision_L_cell(L, Lq, p);

    auto cross_eta_divergence_at_w_face = [&](int i, int j_face, int k) {
        if (k < 0 || k >= p.NL) {
            return 0.0;
        }
        const double eta_weight = eta_cell_weight(k);
        if (!(eta_weight > 0.0) || k < first_active_L) {
            return 0.0;
        }
        const double q_left = (k == first_active_L)
            ? 0.0
            : 0.5 * (cross_at_w_face(i, j_face, k - 1) + cross_at_w_face(i, j_face, k));
        const double q_right = (k + 1 >= p.NL)
            ? 0.0
            : 0.5 * (cross_at_w_face(i, j_face, k) + cross_at_w_face(i, j_face, k + 1));
        return (q_right - q_left) / eta_weight;
    };

    auto cross_at_eta_face = [&](int i, int j, int k_face) {
        if (k_face <= first_active_L || k_face >= p.NL || j < 0 || j >= p.Nw) {
            return 0.0;
        }
        const std::size_t a = idx(i, j, k_face - 1, p);
        const std::size_t b = idx(i, j, k_face, p);
        const BPCoeff ca = mask_bp_fp_terms(coeff[a], p);
        const BPCoeff cb = mask_bp_fp_terms(coeff[b], p);
        const BPEtaCoeff ea = to_eta_coeff(ca, L[static_cast<std::size_t>(k_face - 1)]);
        const BPEtaCoeff eb = to_eta_coeff(cb, L[static_cast<std::size_t>(k_face)]);
        return 0.5 * (ea.D_weta * f[a] + eb.D_weta * f[b]);
    };

    auto cross_w_gradient_at_eta_face = [&](int i, int j, int k_face) {
        if (j <= 0) {
            const double dw = w[1] - w[0];
            return (cross_at_eta_face(i, 1, k_face) - cross_at_eta_face(i, 0, k_face))
                / std::max(dw, 1.0e-300);
        }
        if (j + 1 >= p.Nw) {
            const double dw = w[static_cast<std::size_t>(p.Nw - 1)] - w[static_cast<std::size_t>(p.Nw - 2)];
            return (cross_at_eta_face(i, p.Nw - 1, k_face) - cross_at_eta_face(i, p.Nw - 2, k_face))
                / std::max(dw, 1.0e-300);
        }
        const double dw = w[static_cast<std::size_t>(j + 1)] - w[static_cast<std::size_t>(j - 1)];
        return (cross_at_eta_face(i, j + 1, k_face) - cross_at_eta_face(i, j - 1, k_face))
            / std::max(dw, 1.0e-300);
    };

    // Radial-velocity fluxes are independent for each r and L.
    const int rw_work = p.Nr * p.NL;
    parallel_for_chunks(0, rw_work, effective_threads(p, rw_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / p.NL;
                const int k = q - i * p.NL;
                for (int j = 1; j < p.Nw; ++j) {
                    const std::size_t a = idx(i, j - 1, k, p);
                    const std::size_t b = idx(i, j, k, p);
                    const BPCoeff ca = mask_bp_fp_terms(coeff[a], p);
                    const BPCoeff cb = mask_bp_fp_terms(coeff[b], p);
                    const BPEtaCoeff ea = to_eta_coeff(ca, L[static_cast<std::size_t>(k)]);
                    const BPEtaCoeff eb = to_eta_coeff(cb, L[static_cast<std::size_t>(k)]);
                    const double drift = 0.5 * (ea.D_w + eb.D_w);
                    const double dw = w[static_cast<std::size_t>(j)] - w[static_cast<std::size_t>(j - 1)];
                    const double diag_flux = bp_diagonal_face_flux(
                        drift, 0.5 * ea.D_ww, 0.5 * eb.D_ww, f[a], f[b], dw, p);
                    const double cross_grad = cross_eta_divergence_at_w_face(i, j, k);
                    flux_w[(static_cast<std::size_t>(i) * static_cast<std::size_t>(p.Nw + 1)
                            + static_cast<std::size_t>(j))
                           * static_cast<std::size_t>(p.NL)
                           + static_cast<std::size_t>(k)] =
                        diag_flux - 0.5 * cross_grad;
                }
            }
        });

    // Angular-momentum fluxes include the cylindrical velocity-space geometry.
    const int rL_work = p.Nr * p.Nw;
    parallel_for_chunks(0, rL_work, effective_threads(p, rL_work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / p.Nw;
                const int j = q - i * p.Nw;
                for (int k = first_active_L + 1; k < p.NL; ++k) {
                    const std::size_t a = idx(i, j, k - 1, p);
                    const std::size_t b = idx(i, j, k, p);
                    const BPCoeff ca = mask_bp_fp_terms(coeff[a], p);
                    const BPCoeff cb = mask_bp_fp_terms(coeff[b], p);
                    const BPEtaCoeff ea = to_eta_coeff(ca, L[static_cast<std::size_t>(k - 1)]);
                    const BPEtaCoeff eb = to_eta_coeff(cb, L[static_cast<std::size_t>(k)]);
                    const double drift = 0.5 * (ea.D_eta + eb.D_eta);
                    const double deta = eta[static_cast<std::size_t>(k)] - eta[static_cast<std::size_t>(k - 1)];
                    const double diag_flux = bp_diagonal_face_flux(
                        drift, 0.5 * ea.D_etaeta, 0.5 * eb.D_etaeta, f[a], f[b], deta, p);
                    const double cross_grad = cross_w_gradient_at_eta_face(i, j, k);
                    flux_eta[(static_cast<std::size_t>(i) * static_cast<std::size_t>(p.Nw)
                              + static_cast<std::size_t>(j))
                             * static_cast<std::size_t>(p.NL + 1)
                             + static_cast<std::size_t>(k)] =
                        diag_flux - 0.5 * cross_grad;
                }
            }
        });

    std::vector<double> updated(f.size(), 0.0);
    const int work = p.Nr * p.Nw * p.NL;
    parallel_for_chunks(0, work, effective_threads(p, work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const int i = q / (p.Nw * p.NL);
                const int rem = q - i * p.Nw * p.NL;
                const int j = rem / p.NL;
                const int k = rem - j * p.NL;
                const double dw_cell =
                    w_edges[static_cast<std::size_t>(j + 1)] - w_edges[static_cast<std::size_t>(j)];
                const double eta_weight = eta_cell_weight(k);
                const double fw_right = flux_w[(static_cast<std::size_t>(i) * static_cast<std::size_t>(p.Nw + 1)
                                                + static_cast<std::size_t>(j + 1))
                                               * static_cast<std::size_t>(p.NL)
                                               + static_cast<std::size_t>(k)];
                const double fw_left = flux_w[(static_cast<std::size_t>(i) * static_cast<std::size_t>(p.Nw + 1)
                                               + static_cast<std::size_t>(j))
                                              * static_cast<std::size_t>(p.NL)
                                              + static_cast<std::size_t>(k)];
                const double f_eta_right = flux_eta[(static_cast<std::size_t>(i) * static_cast<std::size_t>(p.Nw)
                                                     + static_cast<std::size_t>(j))
                                                    * static_cast<std::size_t>(p.NL + 1)
                                                    + static_cast<std::size_t>(k + 1)];
                const double f_eta_left = flux_eta[(static_cast<std::size_t>(i) * static_cast<std::size_t>(p.Nw)
                                                    + static_cast<std::size_t>(j))
                                                   * static_cast<std::size_t>(p.NL + 1)
                                                   + static_cast<std::size_t>(k)];
                const double div_w = (fw_right - fw_left) / std::max(dw_cell, 1.0e-300);
                const double div_eta = (eta_weight > 0.0)
                    ? (f_eta_right - f_eta_left) / eta_weight
                    : 0.0;
                updated[static_cast<std::size_t>(q)] =
                    f[static_cast<std::size_t>(q)] - dt * (div_w + div_eta);
            }
        });
    f.swap(updated);
}

void enforce_bp_velocity_shell_positivity(
    std::vector<double>& f,
    const std::vector<double>& wq,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    const Params& p);

std::vector<double> bp_collision_rhs(
    const std::vector<double>& f,
    const std::vector<BPCoeff>& coeff,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    const Params& p) {
    std::vector<double> out = f;
    apply_bp_fixed_coefficients_once(out, coeff, w, L, Lq, 1.0, p);
    const int work = static_cast<int>(out.size());
    parallel_for_chunks(0, work, effective_threads(p, work),
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const std::size_t s = static_cast<std::size_t>(q);
                out[s] -= f[s];
            }
        });
    return out;
}

int bp_rkl2_stage_count(int explicit_substeps, const Params& p) {
    if (explicit_substeps <= 1) {
        return 1;
    }
    const double need = static_cast<double>(explicit_substeps);
    int stages = static_cast<int>(std::ceil(0.5 * (-1.0 + std::sqrt(9.0 + 16.0 * need))));
    stages = std::max(2, stages);
    return std::min(stages, std::max(2, p.bp_sts_max_stages));
}

double bp_rkl2_acceleration(int stages) {
    if (stages <= 1) {
        return 1.0;
    }
    return 0.25 * (static_cast<double>(stages) * static_cast<double>(stages)
                   + static_cast<double>(stages) - 2.0);
}

// For fixed coefficients the collision operator is linear. The Legendre
// stage recurrence extends its diffusive stability interval while keeping
// the first two time derivatives of the stability polynomial second order.
void apply_bp_rkl2_cycle(
    std::vector<double>& f,
    const std::vector<BPCoeff>& coeff,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    double dt,
    int stages,
    const Params& p) {
    if (stages <= 1) {
        apply_bp_fixed_coefficients_once(f, coeff, w, L, Lq, dt, p);
        return;
    }
    std::vector<double> y0 = f;
    std::vector<double> rhs0 = bp_collision_rhs(y0, coeff, w, L, Lq, p);
    std::vector<double> ym2 = y0;
    std::vector<double> ym1(y0.size(), 0.0);
    const int work = static_cast<int>(y0.size());
    const int nthreads = effective_threads(p, work);
    // RKL2 uses w1=4/(s^2+s-2), b0=b1=1/3. The first stage has
    // tilde(mu)_1=b1*w1; later stages use tilde(mu)_j=mu_j*w1.
    // Keeping these distinct preserves R'(0)=R''(0)=1 for the frozen
    // linear operator (Meyer, Balsara & Aslam 2014, JCP 257, 594).
    const double w1 = 4.0 / (static_cast<double>(stages) * static_cast<double>(stages)
                            + static_cast<double>(stages) - 2.0);
    const double mu1_tilde = w1 / 3.0;
    parallel_for_chunks(0, work, nthreads,
        [&](int lo, int hi, int) {
            for (int q = lo; q < hi; ++q) {
                const std::size_t s = static_cast<std::size_t>(q);
                ym1[s] = y0[s] + mu1_tilde * dt * rhs0[s];
            }
        });

    std::vector<double> b(static_cast<std::size_t>(stages + 1), 1.0 / 3.0);
    for (int j = 2; j <= stages; ++j) {
        b[static_cast<std::size_t>(j)] =
            (static_cast<double>(j) * static_cast<double>(j) + static_cast<double>(j) - 2.0)
            / (2.0 * static_cast<double>(j) * static_cast<double>(j + 1));
    }

    std::vector<double> yj(y0.size(), 0.0);
    for (int j = 2; j <= stages; ++j) {
        const double bj = b[static_cast<std::size_t>(j)];
        const double bjm1 = b[static_cast<std::size_t>(j - 1)];
        const double bjm2 = b[static_cast<std::size_t>(j - 2)];
        const double mu = (2.0 * static_cast<double>(j) - 1.0) / static_cast<double>(j) * bj / bjm1;
        const double nu = -(static_cast<double>(j) - 1.0) / static_cast<double>(j) * bj / bjm2;
        const double mu_tilde = mu * w1;
        const double a_jm1 = 1.0 - bjm1;
        const double gamma_tilde = -a_jm1 * mu_tilde;
        std::vector<double> rhsm1 = bp_collision_rhs(ym1, coeff, w, L, Lq, p);
        parallel_for_chunks(0, work, nthreads,
            [&](int lo, int hi, int) {
                for (int q = lo; q < hi; ++q) {
                    const std::size_t s = static_cast<std::size_t>(q);
                    yj[s] = mu * ym1[s] + nu * ym2[s] + (1.0 - mu - nu) * y0[s]
                        + mu_tilde * dt * rhsm1[s] + gamma_tilde * dt * rhs0[s];
                }
            });
        ym2.swap(ym1);
        ym1.swap(yj);
    }
    f.swap(ym1);
}

void apply_bp_collision_with_integrator(
    std::vector<double>& f,
    const std::vector<BPCoeff>& coeff,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    double total_dt,
    BPCollisionStepInfo& info,
    const Params& p) {
    if (p.bp_collision_integrator == "sts_rkl2") {
        const double stable_dt = std::max(info.explicit_dt_limit, 1.0e-300);
        const int max_stages = std::max(2, p.bp_sts_max_stages);
        const double max_cycle_dt = stable_dt * bp_rkl2_acceleration(max_stages);
        const int cycles = std::max(1, static_cast<int>(std::ceil(total_dt / std::max(max_cycle_dt, 1.0e-300))));
        info.sts_cycles = cycles;
        info.substeps = 0;
        for (int cycle = 0; cycle < cycles; ++cycle) {
            const double remaining = total_dt - static_cast<double>(cycle) * total_dt / static_cast<double>(cycles);
            const double dt_cycle = (cycle + 1 == cycles)
                ? remaining
                : total_dt / static_cast<double>(cycles);
            const int explicit_needed =
                std::max(1, static_cast<int>(std::ceil(dt_cycle / stable_dt)));
            const int stages = bp_rkl2_stage_count(explicit_needed, p);
            info.substeps += stages;
            apply_bp_rkl2_cycle(f, coeff, w, L, Lq, dt_cycle, stages, p);
            enforce_bp_velocity_shell_positivity(f, wq, L, Lq, p);
        }
        info.collision_dt = total_dt;
        return;
    }

    for (int sub = 0; sub < info.substeps; ++sub) {
        apply_bp_fixed_coefficients_once(f, coeff, w, L, Lq, info.collision_dt, p);
        enforce_bp_velocity_shell_positivity(f, wq, L, Lq, p);
    }
}

/// Floors negative velocity-space cells while preserving each radial shell's mass.
// Remove negative cell values and rescale positive quadrature mass to
// retain the shell total when it is positive. This correction controls
// positivity and has its own momentum/energy effects to measure in the ledger.
void enforce_bp_velocity_shell_positivity(
    std::vector<double>& f,
    const std::vector<double>& wq,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    const Params& p) {
    parallel_for_chunks(0, p.Nr, effective_threads(p, p.Nr),
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                double positive_mass = 0.0;
                double negative_mass = 0.0;
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        const std::size_t q = idx(i, j, k, p);
                        const double weight = wq[static_cast<std::size_t>(j)]
                            * Lq[static_cast<std::size_t>(k)]
                            * L[static_cast<std::size_t>(k)];
                        if (f[q] < 0.0) {
                            if (weight > 0.0) {
                                negative_mass += -f[q] * weight;
                            }
                            f[q] = 0.0;
                        } else if (weight > 0.0) {
                            positive_mass += f[q] * weight;
                        }
                    }
                }
                if (!(negative_mass > 0.0) || !(positive_mass > 0.0)) {
                    continue;
                }
                const double scale = std::max(0.0, (positive_mass - negative_mass) / positive_mass);
                for (int j = 0; j < p.Nw; ++j) {
                    for (int k = 0; k < p.NL; ++k) {
                        const double weight = wq[static_cast<std::size_t>(j)]
                            * Lq[static_cast<std::size_t>(k)]
                            * L[static_cast<std::size_t>(k)];
                        if (weight > 0.0) {
                            f[idx(i, j, k, p)] *= scale;
                        }
                    }
                }
            }
        });
}

/// Applies the local Boltzmann-Poisson collision source to total and optional tracer DFs.
BPCollisionStepInfo apply_boltzmann_poisson_collision_step(
    std::vector<double>& f,
    std::vector<double>* tracer_f,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& L,
    const std::vector<double>& wq,
    const std::vector<double>& Lq,
    double dt,
    const Params& p,
    const MpiContext& mpi,
    BPStreamedCollisionWorkspace* streamed_workspace = nullptr) {
    BPCollisionStepInfo info;
    if (using_bp_finite_angle_boltzmann(p)) {
        return apply_bp_finite_angle_boltzmann_collision_step(
            f, tracer_f, r, w, L, wq, Lq, dt, p, mpi);
    }
    if (using_bp_reversible_boltzmann(p)) {
        return apply_bp_reversible_boltzmann_collision_step(
            f, tracer_f, r, w, L, wq, Lq, dt, p, mpi);
    }
    if (using_bp_sampled_reversible_boltzmann(p)) {
        return apply_bp_sampled_reversible_boltzmann_collision_step(
            f, tracer_f, r, w, L, wq, Lq, dt, p, mpi);
    }
    if (using_bp_streamed_reversible_boltzmann(p)) {
        return apply_bp_streamed_reversible_boltzmann_collision_step(
            f, tracer_f, r, w, L, wq, Lq, dt, p, mpi,
            streamed_workspace);
    }
    if (!using_bp_fokker_planck(p) || dt <= 0.0 || p.bp_collision_strength == 0.0) {
        return info;
    }
    if (mpi.size > 1) {
        throw std::runtime_error(
            "Boltzmann-Poisson collisions diffuse in L and currently require a serial/OpenMP run. "
            "MPI L-slab support needs ghost fluxes across rank boundaries.");
    }

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
        double partner_mass_tol = 0.0;
        double temperature_floor = 0.0;
        double density_floor_abs = 0.0;
        double density_floor_rel = 0.0;
        double bath_speed_cutoff = 0.0;
        std::string collision_operator;
        std::string cross_section;
        std::string bath;
        std::string fp_terms;
    };
    // This application clock accumulates physical source time across skipped
    // calls. A changed physical/numerical parameter tuple starts a new cadence.
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
        && cadence.partner_mass_tol == p.bp_partner_mass_tol
        && cadence.temperature_floor == p.bp_temperature_floor
        && cadence.density_floor_abs == p.bp_density_floor_abs
        && cadence.density_floor_rel == p.bp_density_floor_rel
        && cadence.bath_speed_cutoff == p.bp_bath_speed_cutoff
        && cadence.collision_operator == p.bp_collision_operator
        && cadence.cross_section == p.bp_cross_section
        && cadence.bath == p.bp_bath
        && cadence.fp_terms == p.bp_fp_terms;
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
        cadence.partner_mass_tol = p.bp_partner_mass_tol;
        cadence.temperature_floor = p.bp_temperature_floor;
        cadence.density_floor_abs = p.bp_density_floor_abs;
        cadence.density_floor_rel = p.bp_density_floor_rel;
        cadence.bath_speed_cutoff = p.bp_bath_speed_cutoff;
        cadence.collision_operator = p.bp_collision_operator;
        cadence.cross_section = p.bp_cross_section;
        cadence.bath = p.bp_bath;
        cadence.fp_terms = p.bp_fp_terms;
    }
    cadence.dt_accum += dt;
    if (p.bp_collision_apply_every > 1 && cadence.calls_left > 0) {
        --cadence.calls_left;
        return info;
    }
    const double source_dt = cadence.dt_accum;
    cadence.dt_accum = 0.0;
    cadence.calls_left = std::max(0, p.bp_collision_apply_every - 1);

    struct CoeffCache {
        std::vector<BPCoeff> coeff;
        int calls_left = 0;
        int Nr = 0;
        int Nw = 0;
        int NL = 0;
        int table_points = 0;
        int speed_points = 0;
        int mu_points = 0;
        int refresh_every = 1;
        double time_left = 0.0;
        double last_t_coll = 0.0;
        double last_refresh_dt = 0.0;
        double refresh_chi = 0.0;
        double refresh_quantile = 0.0;
        double strength = 0.0;
        double sigma0 = 0.0;
        double sigma_u0 = 0.0;
        double sigma_n = 0.0;
        double subhalo_mass = 0.0;
        double subhalo_radius = 0.0;
        double coulomb_log = 0.0;
        double partner_mass_tol = 0.0;
        double temperature_floor = 0.0;
        double density_floor_abs = 0.0;
        double density_floor_rel = 0.0;
        double bath_speed_cutoff = 0.0;
        double yukawa_alpha = 0.0;
        double yukawa_mchi = 0.0;
        double yukawa_mphi = 0.0;
        double collision_l_min_rel = 0.0;
        std::string cross_section;
        std::string bath;
        std::string fp_terms;
        std::string refresh_mode;
    };
    // The coefficient cache has a distinct lifetime from the application clock.
    // Compare its complete parameter tuple, then refresh on either a call count
    // or the selected collision-time interval before advancing the source.
    static CoeffCache cache;

    const bool base_cache_matches =
        cache.coeff.size() == f.size()
        && cache.Nr == p.Nr
        && cache.Nw == p.Nw
        && cache.NL == p.NL
        && cache.table_points == p.bp_coeff_table_points
        && cache.speed_points == p.bp_coeff_speed_points
        && cache.mu_points == p.bp_coeff_mu_points
        && cache.refresh_every == p.bp_coeff_refresh_every
        && cache.refresh_chi == p.bp_coeff_refresh_chi
        && cache.refresh_quantile == p.bp_coeff_refresh_time_quantile
        && cache.strength == p.bp_collision_strength
        && cache.sigma0 == p.bp_sigma0
        && cache.sigma_u0 == p.bp_sigma_u0
        && cache.sigma_n == p.bp_sigma_n
        && cache.subhalo_mass == p.bp_subhalo_mass
        && cache.subhalo_radius == p.bp_subhalo_radius
        && cache.coulomb_log == p.bp_coulomb_log
        && cache.partner_mass_tol == p.bp_partner_mass_tol
        && cache.temperature_floor == p.bp_temperature_floor
        && cache.density_floor_abs == p.bp_density_floor_abs
        && cache.density_floor_rel == p.bp_density_floor_rel
        && cache.bath_speed_cutoff == p.bp_bath_speed_cutoff
        && cache.yukawa_alpha == p.bp_yukawa_alpha
        && cache.yukawa_mchi == p.bp_yukawa_mchi
        && cache.yukawa_mphi == p.bp_yukawa_mphi
        && cache.collision_l_min_rel == p.bp_collision_l_min_rel
        && cache.cross_section == p.bp_cross_section
        && cache.bath == p.bp_bath
        && cache.fp_terms == p.bp_fp_terms
        && cache.refresh_mode == p.bp_coeff_refresh_mode;
    const bool step_cache_matches =
        base_cache_matches
        && p.bp_coeff_refresh_mode == "steps"
        && p.bp_coeff_refresh_every > 1
        && cache.calls_left > 0;
    const bool time_cache_matches =
        base_cache_matches
        && p.bp_coeff_refresh_mode == "collision_time"
        && cache.time_left >= source_dt;
    const bool cache_matches =
        step_cache_matches || time_cache_matches;

    std::vector<BPCoeff> local_coeff;
    const std::vector<BPCoeff>* coeff = nullptr;
    if (cache_matches) {
        if (p.bp_coeff_refresh_mode == "steps") {
            --cache.calls_left;
        } else {
            cache.time_left = std::max(0.0, cache.time_left - source_dt);
        }
        coeff = &cache.coeff;
    } else if (p.bp_coeff_refresh_every > 1 || p.bp_coeff_refresh_mode == "collision_time") {
        cache.coeff = compute_bp_coefficients(f, r, w, L, wq, Lq, p);
        cache.calls_left = (p.bp_coeff_refresh_mode == "steps")
            ? p.bp_coeff_refresh_every - 1
            : 0;
        cache.Nr = p.Nr;
        cache.Nw = p.Nw;
        cache.NL = p.NL;
        cache.table_points = p.bp_coeff_table_points;
        cache.speed_points = p.bp_coeff_speed_points;
        cache.mu_points = p.bp_coeff_mu_points;
        cache.refresh_every = p.bp_coeff_refresh_every;
        cache.refresh_chi = p.bp_coeff_refresh_chi;
        cache.refresh_quantile = p.bp_coeff_refresh_time_quantile;
        cache.strength = p.bp_collision_strength;
        cache.sigma0 = p.bp_sigma0;
        cache.sigma_u0 = p.bp_sigma_u0;
        cache.sigma_n = p.bp_sigma_n;
        cache.subhalo_mass = p.bp_subhalo_mass;
        cache.subhalo_radius = p.bp_subhalo_radius;
        cache.coulomb_log = p.bp_coulomb_log;
        cache.partner_mass_tol = p.bp_partner_mass_tol;
        cache.temperature_floor = p.bp_temperature_floor;
        cache.density_floor_abs = p.bp_density_floor_abs;
        cache.density_floor_rel = p.bp_density_floor_rel;
        cache.bath_speed_cutoff = p.bp_bath_speed_cutoff;
        cache.yukawa_alpha = p.bp_yukawa_alpha;
        cache.yukawa_mchi = p.bp_yukawa_mchi;
        cache.yukawa_mphi = p.bp_yukawa_mphi;
        cache.collision_l_min_rel = p.bp_collision_l_min_rel;
        cache.cross_section = p.bp_cross_section;
        cache.bath = p.bp_bath;
        cache.fp_terms = p.bp_fp_terms;
        cache.refresh_mode = p.bp_coeff_refresh_mode;
        if (p.bp_coeff_refresh_mode == "collision_time") {
            cache.last_t_coll = estimate_bp_collision_time(f, cache.coeff, r, w, L, wq, Lq, p);
            cache.last_refresh_dt = std::isfinite(cache.last_t_coll)
                ? p.bp_coeff_refresh_chi * cache.last_t_coll
                : source_dt;
            cache.time_left = std::max(0.0, cache.last_refresh_dt - source_dt);
        } else {
            cache.last_t_coll = 0.0;
            cache.last_refresh_dt = 0.0;
            cache.time_left = 0.0;
        }
        coeff = &cache.coeff;
    } else {
        local_coeff = compute_bp_coefficients(f, r, w, L, wq, Lq, p);
        coeff = &local_coeff;
    }

    info = bp_collision_step_info(*coeff, w, L, Lq, source_dt, p);
    if (p.bp_coeff_refresh_mode == "collision_time" && cache.coeff.size() == f.size()) {
        info.collision_time_estimate = cache.last_t_coll;
        info.coeff_refresh_dt = cache.last_refresh_dt;
    }
    apply_bp_collision_with_integrator(f, *coeff, w, L, wq, Lq, source_dt, info, p);
    // The passive tracer receives the same total-distribution bath coefficients
    // and elapsed collision interval. Its values contribute no additional bath source.
    if (tracer_f != nullptr && !tracer_f->empty()) {
        BPCollisionStepInfo tracer_info = info;
        apply_bp_collision_with_integrator(*tracer_f, *coeff, w, L, wq, Lq, source_dt, tracer_info, p);
    }
    return info;
}

}  // namespace ev
