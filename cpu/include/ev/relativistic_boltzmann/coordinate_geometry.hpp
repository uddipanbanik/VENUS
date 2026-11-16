#pragma once

// Transform momenta and the local observer frame between coordinate charts at
// one spacetime event. The metric signature is (-,+,+,+), and c=1.
// Inputs are a metric, its orthonormal tetrad and a coordinate Jacobian. Outputs
// are transformed components or the corresponding on-shell momentum measure.
// A coordinate transformation changes the labels of the same physical event.

#include "relativistic_collision.hpp"

namespace rba {

// Components are ordered (time, first space, second space, third space).
// A tetrad matrix stores its four physical basis vectors in its columns.
using Vector = std::array<double,4>;
using Matrix = std::array<Vector,4>;
using Connection = std::array<Matrix,4>;

inline Matrix identity() {
    Matrix result{};
    for (int i = 0; i < 4; ++i) {
        result[i][i] = 1.0;
    }
    return result;
}

inline Vector vector(rb::Four p) { return {p.t,p.p.x,p.p.y,p.p.z}; }
inline rb::Four four(Vector p) { return {p[0],{p[1],p[2],p[3]}}; }

inline Vector multiply(const Matrix& a, const Vector& x) {
    Vector result{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i] += a[i][j]*x[j];
        }
    }
    return result;
}

inline Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                result[i][j] += a[i][k]*b[k][j];
            }
        }
    }
    return result;
}

inline Matrix transpose(const Matrix& a) {
    Matrix result{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i][j] = a[j][i];
        }
    }
    return result;
}

// Contract the coordinate vectors with the supplied metric.
// This computes their invariant scalar product, including the timelike sign.
inline double dot(const Matrix& g, const Vector& a, const Vector& b) {
    const Vector lowered = multiply(g,b);
    double result = 0.0;
    for (int i = 0; i < 4; ++i) {
        result += a[i]*lowered[i];
    }
    return result;
}

struct Inverse {
    Matrix value;
    double determinant = 0.0;
};

// Gauss--Jordan elimination applies the same row operations to the identity.
// Partial pivoting chooses the best-resolved entry in each column; a singular
// map stops the calculation before an undefined frame can be used.
inline Inverse inverse(Matrix a) {
    Matrix b = identity();
    double determinant = 1.0;
    for (int k = 0; k < 4; ++k) {
        int pivot = k;
        for (int i = k+1; i < 4; ++i) {
            if (std::abs(a[i][k]) > std::abs(a[pivot][k])) {
                pivot = i;
            }
        }
        rb::require(std::isfinite(a[pivot][k]) && std::abs(a[pivot][k]) > 1.0e-14,
                    "Singular or unresolved coordinate matrix.");
        if (pivot != k) {
            std::swap(a[pivot],a[k]);
            std::swap(b[pivot],b[k]);
            determinant = -determinant;
        }
        const double diagonal = a[k][k];
        determinant *= diagonal;
        for (int j = 0; j < 4; ++j) {
            a[k][j] /= diagonal;
            b[k][j] /= diagonal;
        }
        for (int i = 0; i < 4; ++i) {
            if (i == k) {
                continue;
            }
            const double coefficient = a[i][k];
            for (int j = 0; j < 4; ++j) {
                a[i][j] -= coefficient*a[k][j];
                b[i][j] -= coefficient*b[k][j];
            }
        }
    }
    rb::require(std::isfinite(determinant),"Coordinate determinant overflow.");
    return {b,determinant};
}

inline double determinant3(const std::array<std::array<double,3>,3>& a) {
    return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
           -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
           +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
}

struct Geometry {
    Vector position;
    Matrix metric;
    Matrix tetrad;  // Columns are the same physical e_hat_a in this chart.
};

// Check e_hat^T g e_hat=diag(-1,1,1,1). The same tetrad therefore supplies
// physical orthonormal components for all momenta in this local collision.
inline void validate_geometry(const Geometry& geometry) {
    Matrix eta{};
    for (int i = 0; i < 4; ++i) {
        eta[i][i] = i == 0 ? -1.0 : 1.0;
    }
    const Matrix gram = multiply(transpose(geometry.tetrad),multiply(geometry.metric,geometry.tetrad));
    for (int i = 0; i < 4; ++i) {
        rb::require(std::isfinite(geometry.position[i]),"Nonfinite event coordinate.");
        for (int j = 0; j < 4; ++j) {
            rb::require(std::isfinite(geometry.metric[i][j])
                        && std::abs(geometry.metric[i][j]-geometry.metric[j][i]) < 1.0e-12
                        && std::abs(gram[i][j]-eta[i][j]) < 2.0e-11,
                        "Metric and supplied tetrad are inconsistent.");
        }
    }
    rb::require(inverse(geometry.metric).determinant < 0.0,"The metric must have Lorentz signature.");
}

// Multiply physical frame components by the basis columns to obtain
// contravariant coordinate components at this event.
inline Vector from_tetrad(const Geometry& geometry, rb::Four p) {
    return multiply(geometry.tetrad,vector(p));
}

// Apply the inverse tetrad to recover the local energy and three-momentum.
// Collisions use these components because local four-momentum is conserved.
inline rb::Four to_tetrad(const Geometry& geometry, Vector p) {
    return four(multiply(inverse(geometry.tetrad).value,p));
}

// J is dx_new/dx_old at this event. Transform the metric covariantly and
// each physical tetrad vector contravariantly with the same Jacobian.
inline Geometry transform_geometry(const Geometry& old, Vector new_position, const Matrix& J) {
    // The metric has two covariant indices and uses the inverse Jacobian twice.
    // The tetrad vectors have one contravariant index and use the forward map.
    const Matrix K = inverse(J).value;
    Geometry result{new_position,multiply(transpose(K),multiply(old.metric,K)),
                    multiply(J,old.tetrad)};
    validate_geometry(result);
    return result;
}

// Invariant shell measure in covariant spatial momenta p_i. This chart needs
// spacelike x^0 slices (g^00<0) and the future branch p^0>0.
inline double covariant_shell_density(const Geometry& geometry, Vector contravariant_p) {
    const Inverse inv = inverse(geometry.metric);
    rb::require(inv.value[0][0] < 0.0 && contravariant_p[0] > 0.0,
                "The covariant spatial-momentum chart requires spacelike time slices.");
    // Eliminating the future energy from the four-dimensional mass-shell delta
    // function leaves d^3p_i/[sqrt(-g) p^0] in covariant spatial momenta.
    return 1.0/(std::sqrt(-inv.determinant)*contravariant_p[0]);
}

// Alternative p^i chart is restricted further: p_0 must stay negative on the
// selected future branch. Its singular points require another momentum chart.
inline double contravariant_shell_density(const Geometry& geometry, Vector p) {
    const Vector lowered = multiply(geometry.metric,p);
    rb::require(geometry.metric[0][0] < 0.0 && lowered[0] < 0.0,
                "The contravariant spatial-momentum chart is outside its domain.");
    // In contravariant spatial momenta the same measure is
    // sqrt(-g) d^3p^i/(-p_0). Its stated chart conditions keep this density positive.
    return std::sqrt(-inverse(geometry.metric).determinant)/(-lowered[0]);
}

// Supply covariant spatial components p_i and solve for the future p_0.
// The inverse metric and spacelike time slice fix the selected quadratic root.
inline Vector on_shell_covector(const Geometry& geometry, rb::Vec3 spatial, double mass) {
    rb::require(mass > 0.0 && std::isfinite(mass) && rb::finite(spatial),"Invalid shell data.");
    const Matrix g = inverse(geometry.metric).value;
    rb::require(g[0][0] < 0.0,"The time slices must be spacelike.");
    const Vector p{0.0,spatial.x,spatial.y,spatial.z};
    double b = 0.0;
    double c = mass*mass;
    for (int i = 1; i < 4; ++i) {
        b += g[0][i]*p[i];
        for (int j = 1; j < 4; ++j) {
            c += g[i][j]*p[i]*p[j];
        }
    }
    // The mass-shell equation g^{mu nu}p_mu p_nu=-m^2 is quadratic in p_0.
    // The selected root has p^0>0 on the admitted spacelike time slice.
    const double discriminant = b*b-g[0][0]*c;
    rb::require(discriminant > 0.0 && std::isfinite(discriminant),"Unresolved future mass shell.");
    return {(-b+std::sqrt(discriminant))/g[0][0],spatial.x,spatial.y,spatial.z};
}

// Differentiate the spatial covector transformation while constraining
// p_0 to the future mass shell. The determinant converts d^3p_i.
inline double covariant_shell_jacobian(const Matrix& inverse_J, Vector old_p) {
    rb::require(old_p[0] > 0.0,"A future branch is required.");
    std::array<std::array<double,3>,3> derivative{};
    for (int i = 1; i < 4; ++i) {
        for (int j = 1; j < 4; ++j) {
            derivative[i-1][j-1] = inverse_J[j][i]-inverse_J[0][i]*old_p[j]/old_p[0];
        }
    }
    return std::abs(determinant3(derivative));
}

// Here the future-shell relation eliminates p^0 while differentiating
// the spatial vector transformation. The absolute determinant is a measure.
inline double contravariant_shell_jacobian(const Matrix& J, Vector old_lowered) {
    rb::require(old_lowered[0] < 0.0,"A regular p^i momentum chart is required.");
    std::array<std::array<double,3>,3> derivative{};
    for (int i = 1; i < 4; ++i) {
        for (int j = 1; j < 4; ++j) {
            derivative[i-1][j-1] = J[i][j]-J[i][0]*old_lowered[j]/old_lowered[0];
        }
    }
    return std::abs(determinant3(derivative));
}

}  // namespace rba
