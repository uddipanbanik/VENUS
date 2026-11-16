#pragma once

// Interpolate an expanding Newtonian distribution at a characteristic
// footpoint while holding physical radial velocity P=w+H*r fixed across the
// two radial interpolation nodes. Inputs are the distribution, footpoint,
// mesh and endpoint background field. The output is an interpolated f value.
// This header is included inside namespace ev after the VP interpolation
// helpers; the caller first handles radial boundary regularization.

double interpolate_vp_expanding_physical_velocity_slice(
    const std::vector<double>& f,
    double rp,
    double wp,
    int k,
    const std::vector<double>& x_r,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const VlasovPoissonField& field,
    const Params& p) {
    regularize_radial_footpoint_for_interpolation(rp, wp, r, p);
    if (rp < r.front() || rp > r.back()) {
        return 0.0;
    }

    const double H = field.background.H;
    // Compute the physical radial velocity at the already regularized
    // footpoint using the same endpoint H as the field update.
    const double Pp = wp + H * rp;
    const double xp = radial_coordinate(rp, p);
    const int i0 = lower_cell(x_r, xp);
    const int i1 = i0 + 1;
    const double tr = (xp - x_r[i0]) / (x_r[i1] - x_r[i0]);

    auto sample_fixed_radius = [&](int i, double& out) {
        // At each radial node use w_i=P_footpoint-H*r_i. Thus the interpolation
        // connects the same physical velocity rather than two different Hubble
        // flows at equal peculiar velocity.
        const double wi = Pp - H * r[static_cast<std::size_t>(i)];
        if (wi < w.front() || wi > w.back()) {
            return false;
        }
        const int j = lower_cell(w, wi);
        const double tw = (wi - w[j]) / (w[j + 1] - w[j]);
        out = (1.0 - tw) * f[idx(i, j, k, p)] + tw * f[idx(i, j + 1, k, p)];
        return true;
    };

    double f0 = 0.0;
    double f1 = 0.0;
    const bool ok0 = sample_fixed_radius(i0, f0);
    const bool ok1 = sample_fixed_radius(i1, f1);
    // Use both radial samples when available. A single valid momentum
    // sample supplies the supported edge value; if neither is supported,
    // use the ordinary configured phase-space interpolation.
    if (ok0 && ok1) {
        return (1.0 - tr) * f0 + tr * f1;
    }
    if (ok0) {
        return f0;
    }
    if (ok1) {
        return f1;
    }
    return interpolate_slice(f, rp, wp, k, x_r, r, w, p);
}
