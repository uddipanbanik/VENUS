#pragma once

// Apply the same material elastic law on two spherical spacetime slices.
// A packet carries represented rest mass W and specific angular momentum
// eta=L^2/2. SC stores the local radial momentum per mass w; EF stores the
// positive canonical radial momentum per mass P. The physical material mass
// m_chi and cross-sectional area are model inputs, separate from W.
// The returned state includes every split remainder and its persistent RNG
// and optical-depth clock. Finite radial cells are explicitly approximate.

#include "invariant_amplitude.hpp"
#include <sstream>
#include <set>

namespace eb {

enum class Slice { schwarzschild, advanced_null };

struct Packet {
    double rest_weight = 0;  // Represented rest mass W; represented number is W/m_chi.
    double r = 0;
    double radial = 0;  // Specific w on SC; positive specific canonical P on EF.
    double eta = 0;
    std::uint64_t id = 0;
    std::uint64_t source_id = 0;  // Inherited source label, independent of the current cell.
    std::uint64_t ancestor_id = 0;  // Incoming record that supplies auxiliary properties.
};

struct Model {
    double material_mass = 1;
    double cross_section = 0;  // Constant isotropic material area in code units.
    std::string identity = "constant_material_elastic_v1";
};

struct Geometry {
    Slice slice = Slice::schwarzschild;
    double volume = 1;       // Proper spatial V on SC; coordinate shell V0 on EF.
    double time_factor = 1;  // e^mu on SC; b on EF.
    bool allow_radial_cell_approximation = false;
    double maximum_radial_separation = 0;
    std::size_t maximum_markers = 4096;
    std::uint64_t maximum_proposals = 1000000;
};

// Cell clocks survive transport and rebinding. The coupled driver refills
// packets at each accepted streaming endpoint and carries the other fields.
struct Cell {
    std::vector<Packet> packets;
    rb::Random random;
    double coordinate_time = 0;
    double optical_depth = -1;
    double number_majorant = 0;
    std::uint64_t next_id = 1;
    std::uint64_t proposals = 0;
    std::uint64_t collisions = 0;
    std::uint64_t radially_nonlocal_collisions = 0;
    double maximum_radial_distance = 0;
    double maximum_pair_energy_relative_residual = 0;
    double maximum_null_flux_factor = 0;
};

// Reconstruct the material four-momentum using a sampled tangential azimuth.
// The specific radial variable and eta=L^2/2 determine k^a/material_mass.
// Spherical symmetry leaves the tangential azimuth unrecorded. Sampling
// that angle reconstructs a physical three-momentum with |p_perp|/m_chi
// =sqrt(2eta)/r before the local elastic encounter.
inline rb::Four lift(const Packet& p, const Model& model, Slice slice, double beta) {
    rb::require(p.r > 0 && std::isfinite(p.r) && p.eta >= 0 && std::isfinite(p.eta)
                && std::isfinite(p.radial) && p.rest_weight > 0 && std::isfinite(p.rest_weight),
                "The spherical collision chart requires positive radius/weight and finite momentum.");
    const double transverse = std::sqrt(2*p.eta)/p.r;
    const double chi = 1+transverse*transverse;
    const double w = slice == Slice::schwarzschild ? p.radial
        : (rb::require(p.radial > 0,"EF future canonical momentum must be positive."),
           0.5*(p.radial-chi/p.radial));
    return rb::on_shell(model.material_mass,model.material_mass*rb::Vec3{
        w,transverse*std::cos(beta),transverse*std::sin(beta)});
}

// Replace momentum components at the unchanged physical radius. The inward
// EF branch uses the mass-shell identity to avoid cancellation in epsilon+w.
inline void project(Packet& p, rb::Four k, const Model& model, Slice slice) {
    rb::validate_momentum(model.material_mass,k);
    const double w = k.p.x/model.material_mass;
    const double transverse2 = (k.p.y*k.p.y+k.p.z*k.p.z)/(model.material_mass*model.material_mass);
    p.eta = 0.5*p.r*p.r*transverse2;
    if (slice == Slice::schwarzschild) {
        p.radial = w;
    } else {
        const double epsilon = k.t/model.material_mass;
        p.radial = w >= 0 ? epsilon+w : (1+transverse2)/(epsilon-w);
        rb::require(p.radial > 0 && std::isfinite(p.radial),"EF projection lost the future shell.");
    }
}

// D=E_U/(m_chi P) is the positive null-slice crossing factor. It converts
// the represented packet count on v=constant into the observer number
// measure used by the EF collision rate.
inline double null_D(const Packet& p, const Model& model) {
    return lift(p,model,Slice::advanced_null,0).t/(model.material_mass*p.radial);
}

// Check the material law, geometry, carried diagnostics and all packets
// before sampling. Separated radii require the declared finite-cell option
// and stay within its radial coordinate separation bound.
inline void validate(const Cell& s, const Model& model, const Geometry& g) {
    rb::require(g.slice == Slice::schwarzschild || g.slice == Slice::advanced_null,
                "Unknown spacetime slice convention.");
    rb::require(model.material_mass > 0 && std::isfinite(model.material_mass)
                && model.cross_section >= 0 && std::isfinite(model.cross_section)
                && model.identity == "constant_material_elastic_v1",
                "Unrecognized material collision model or invalid physical units.");
    rb::require(g.volume > 0 && std::isfinite(g.volume) && g.time_factor > 0
                && std::isfinite(g.time_factor) && g.maximum_radial_separation >= 0
                && std::isfinite(g.maximum_radial_separation) && g.maximum_markers >= s.packets.size()
                && g.maximum_proposals > 0 && std::isfinite(s.coordinate_time)
                && (s.optical_depth == -1 || (s.optical_depth >= 0 && std::isfinite(s.optical_depth)))
                && s.number_majorant >= 0 && std::isfinite(s.number_majorant),
                "Invalid collision geometry, clock, population or majorant.");
    for (double value : {s.maximum_radial_distance,s.maximum_pair_energy_relative_residual,
                         s.maximum_null_flux_factor}) {
        rb::require(value >= 0 && std::isfinite(value),"Invalid carried collision diagnostic.");
    }
    double lo = std::numeric_limits<double>::infinity();
    double hi = 0;
    double qmax = 0;
    std::set<std::uint64_t> ids;
    for (const auto& p : s.packets) {
        rb::validate_momentum(model.material_mass,lift(p,model,g.slice,0));
        rb::require(p.id < s.next_id && ids.insert(p.id).second,"Packet IDs must be unique and below next_id.");
        lo = std::min(lo,p.r);
        hi = std::max(hi,p.r);
        qmax = std::max(qmax,p.rest_weight/model.material_mass);
    }
    rb::require(s.number_majorant == 0 || s.number_majorant >= qmax,"The carried number majorant is too small.");
    if (!s.packets.empty()) {
        rb::require(g.allow_radial_cell_approximation ? hi-lo <= g.maximum_radial_separation : hi == lo,
                    "Separated radial populations require an explicitly approximate collision cell.");
    }
}

// Exact invariant pair factor for specified physical momenta at one event.
// SC uses F/(E1 E2), the Moller factor. EF uses F/(m_chi^2 P1 P2),
// which includes the two null-shell crossing Jacobians for the same
// invariant flux F. This is why the two slice rates differ.
inline double pair_factor(const Packet& a, const Packet& b, rb::Four ka, rb::Four kb,
                          const Model& model, const Geometry& g) {
    const double F = rb::invariant_flux(ka,kb);
    return g.slice == Slice::schwarzschild ? F/(ka.t*kb.t)
        : F/(model.material_mass*model.material_mass*a.radial*b.radial);
}

// Frozen local collision suboperator. Its internal clock is not a resolved
// coupled worldline. EF pairs at different r on one null slice have a
// two-way nonlocal exchange, even though their connecting slice is null.
inline void advance_cell(Cell& caller, double interval, const Model& model, const Geometry& g) {
    validate(caller,model,g);
    rb::require(interval >= 0 && std::isfinite(interval),"Invalid local collision interval.");
    if (interval == 0) return;
    // The local transaction contains both particle changes and random draws.
    // Any failed guard leaves the original caller state available for review.
    Cell s = caller;
    if (s.number_majorant == 0) {
        for (const auto& p : s.packets)
            s.number_majorant = std::max(s.number_majorant,p.rest_weight/model.material_mass);
    }
    const double end = s.coordinate_time+interval;
    rb::require(end > s.coordinate_time && std::isfinite(end),"Unresolved local event clock.");
    std::uint64_t proposals = 0;
    while (s.coordinate_time < end && s.packets.size() >= 2 && model.cross_section > 0) {
        // E_U/P is independent of the sampled tangential angle. Refresh after
        // every accepted state change; an old EF bound need not remain valid.
        double Dmax = 1;
        if (g.slice == Slice::advanced_null) {
            Dmax = 0;
            for (const auto& p : s.packets) Dmax = std::max(Dmax,null_D(p,model));
        }
        const double n = static_cast<double>(s.packets.size());
        // Unordered pair count n(n-1)/2 and the flux envelope 2 Dmax^2 produce
        // the n(n-1) factor. The time factor is the SC lapse or EF b, and volume
        // is respectively proper spatial volume or null-slice coordinate volume.
        const double rate = g.time_factor*model.cross_section/g.volume
            *s.number_majorant*n*(n-1)*Dmax*Dmax;
        rb::require(rate > 0 && std::isfinite(rate),"Unrepresentable slice collision majorant.");
        if (s.optical_depth < 0) s.optical_depth = -std::log(s.random.uniform());
        const double wait = s.optical_depth/rate;
        // Spend only the optical depth covered by this frozen interval. The
        // remaining depth survives transport and cell rebinding at the next step.
        if (wait > end-s.coordinate_time) {
            s.optical_depth -= rate*(end-s.coordinate_time);
            break;
        }
        rb::require(++proposals <= g.maximum_proposals,"Slice collision proposal cap exceeded.");
        rb::require(s.coordinate_time+wait > s.coordinate_time || wait == 0,"Event clock resolution exhausted.");
        s.coordinate_time += wait;
        ++s.proposals;
        const std::size_t i = rb::random_index(s.random,s.packets.size());
        std::size_t j = rb::random_index(s.random,s.packets.size()-1);
        if (j >= i) ++j;
        const Packet a = s.packets[i];
        const Packet b = s.packets[j];
        // Independent tangential azimuths sample the spherical distribution at
        // the representative event. Each lifted momentum still has material
        // mass m_chi, regardless of the packet rest weight.
        const rb::Four ka = lift(a,model,g.slice,2*rb::pi*s.random.uniform());
        const rb::Four kb = lift(b,model,g.slice,2*rb::pi*s.random.uniform());
        const double factor = pair_factor(a,b,ka,kb,model,g);
        s.maximum_null_flux_factor = std::max(s.maximum_null_flux_factor,
            g.slice == Slice::advanced_null ? factor : 0);
        // The number-weight ratio and exact slice flux remove the conservative
        // proposal envelope. The bound must hold before the random accept/reject
        // draw; an exceeded envelope stops the transaction.
        const double probability = std::max(a.rest_weight,b.rest_weight)/(model.material_mass*s.number_majorant)
            *factor/(2*Dmax*Dmax);
        rb::require(probability >= 0 && probability <= 1,"Slice flux envelope was exceeded.");
        if (s.random.uniform() < probability) {
            rb::require(s.packets.size()+(a.rest_weight != b.rest_weight) <= g.maximum_markers,
                        "Conservative split would exceed the packet cap.");
            const auto out = rb::scatter_isotropic(model.material_mass,ka,model.material_mass,kb,s.random);
            // Scatter the equal rest weight min(W1,W2) from both packets. A heavier
            // packet keeps a positive remainder with the incoming momentum and a new
            // ID. This preserves represented rest mass and weighted local four-momentum.
            const double scattered_weight = std::min(a.rest_weight,b.rest_weight);
            s.packets[i].rest_weight = scattered_weight;
            s.packets[j].rest_weight = scattered_weight;
            project(s.packets[i],out.first,model,g.slice);
            project(s.packets[j],out.second,model,g.slice);
            if (a.rest_weight != b.rest_weight) {
                rb::require(s.next_id < std::numeric_limits<std::uint64_t>::max(),"Packet ID overflow.");
                auto remainder = a.rest_weight > b.rest_weight ? a : b;
                remainder.rest_weight -= scattered_weight;
                remainder.id = s.next_id++;
                s.packets.push_back(remainder);
            }
            const auto residual = out.first+out.second-ka-kb;
            s.maximum_pair_energy_relative_residual = std::max(s.maximum_pair_energy_relative_residual,
                std::max(std::abs(residual.t),rb::norm(residual.p))/(ka.t+kb.t));
            // Record finite-radius exchanges explicitly. A null slice is not a
            // spacelike proper cell, and assigning a pair one frozen cell does not
            // resolve two particles meeting on their individual worldlines.
            const double separation = std::abs(a.r-b.r);
            s.radially_nonlocal_collisions += separation > 0;
            s.maximum_radial_distance = std::max(s.maximum_radial_distance,separation);
            ++s.collisions;
        }
        s.optical_depth = -std::log(s.random.uniform());
    }
    s.coordinate_time = end;
    validate(s,model,g);
    caller = std::move(s);
}

// These moments supply null-slice coordinate stresses and observer
// densities. number_crossing counts W/m_chi; rest_crossing sums W.
struct EFStress {
    double number_crossing = 0;
    double rest_crossing = 0;
    double t_rr = 0;
    double s = 0;
    double t_vr = 0;
    double t_vv = 0;
    double observer_number_density = 0;
    double observer_energy_density = 0;
};

// Coordinate stresses divide by V0. The observer moments include E_U/P,
// which converts the null-shell crossing count into the local number measure.
inline EFStress ef_stress(const Cell& cell, const Model& model, double a, double b, double V0) {
    rb::require(std::isfinite(a) && b > 0 && V0 > 0 && std::isfinite(b) && std::isfinite(V0),
                "Invalid EF stress geometry.");
    EFStress out;
    for (const auto& p : cell.packets) {
        const auto k = lift(p,model,Slice::advanced_null,0);
        const double W = p.rest_weight;
        const double q = W/model.material_mass;
        const double chi = 1+2*p.eta/(p.r*p.r);
        // The covariant energy per material mass is u_v=-b(aP+chi/P)/2.
        // Together with u_r=P it gives the coordinate stress components below.
        const double uv = -0.5*b*(a*p.radial+chi/p.radial);
        const double epsilon = k.t/model.material_mass;
        out.number_crossing += q;
        out.rest_crossing += W;
        out.t_rr += W*p.radial/V0;
        out.s += W*chi/(p.radial*V0);
        out.t_vr += W*uv/V0;
        out.t_vv += W*uv*uv/(p.radial*V0);
        // An EF packet contributes W E_U/(m_chi^2 P V0) to observer number
        // density and W E_U^2/(m_chi^2 P V0) to observer energy density.
        out.observer_number_density += q*epsilon/(p.radial*V0);
        out.observer_energy_density += W*epsilon*epsilon/(p.radial*V0);
    }
    return out;
}

}  // namespace eb
