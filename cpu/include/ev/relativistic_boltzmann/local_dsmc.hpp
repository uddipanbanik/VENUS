#pragma once

// Sample elastic material collisions in a frozen homogeneous proper cell.
// Inputs are positive statistical number weights, physical particle masses,
// a proper volume, a lapse and a coordinate-time interval. Outputs are the
// updated momenta, conservative weight splits and persistent collision clock.
// The local geometry is held fixed for this substep; the caller advances the
// field and particle orbits between collision substeps.

#include "relativistic_collision.hpp"
#include <cstring>
#include <set>

namespace rb {

// Point-local populations share one position. A finite cell instead
// represents a homogeneous spatial approximation with an explicit maximum
// separation. Its measured exchanges remain available to the caller.
enum class SpatialPolicy : std::uint64_t {
    point_local_population = 0,
    finite_cell_approximation = 1
};

// material_mass fixes the mass shell. number_weight counts represented
// physical particles, whose total rest mass is material_mass*number_weight.
struct Particle {
    double material_mass = 1.0;
    double number_weight = 1.0;
    Four momentum = on_shell(1.0,{});
    Vec3 position;
    std::uint64_t id = 0;
};

struct CellControls {
    double cross_section = 0.0;
    double proper_volume = 1.0;
    double lapse = 1.0;
    SpatialPolicy spatial_policy = SpatialPolicy::point_local_population;
    double maximum_pair_separation = 0.0;
    std::uint64_t maximum_proposals = 1000000;
    std::uint64_t maximum_markers = 100000;
};

struct CellState {
    std::vector<Particle> particles;
    Random random;
    double coordinate_time = 0.0;
    double remaining_optical_depth = -1.0;
    double number_weight_majorant = 0.0;
    std::uint64_t next_id = 1;
    std::uint64_t proposals = 0;
    std::uint64_t accepted = 0;
    std::uint64_t spacelike_exchanges = 0;
    double maximum_exchange_distance = 0.0;
};

// Multiply each physical particle four-momentum by its represented number.
// This weighted sum is the conservation budget for unequal marker weights.
inline Four represented_four_momentum(const std::vector<Particle>& particles) {
    Four result;
    for (const Particle& p : particles) {
        result = result+p.number_weight*p.momentum;
    }
    return result;
}

// The number budget uses statistical counts; multiplying by a common
// material mass gives represented rest mass for a one-species population.
inline double represented_number(const std::vector<Particle>& particles) {
    double result = 0.0;
    for (const Particle& p : particles) {
        result += p.number_weight;
    }
    return result;
}

inline void validate_cell(const CellState& state, const CellControls& c) {
    require(c.cross_section >= 0.0 && std::isfinite(c.cross_section)
            && c.proper_volume > 0.0 && std::isfinite(c.proper_volume)
            && c.lapse > 0.0 && std::isfinite(c.lapse)
            && c.maximum_markers > 0 && c.maximum_proposals > 0
            && c.maximum_pair_separation >= 0.0
            && std::isfinite(c.maximum_pair_separation),"Invalid cell controls.");
    require(c.spatial_policy == SpatialPolicy::point_local_population
            || c.spatial_policy == SpatialPolicy::finite_cell_approximation,
            "Unknown spatial collision policy.");
    require(std::isfinite(state.coordinate_time)
            && std::isfinite(state.remaining_optical_depth)
            && (state.remaining_optical_depth >= 0.0
                || state.remaining_optical_depth == -1.0)
            && state.number_weight_majorant >= 0.0
            && std::isfinite(state.number_weight_majorant)
            && state.particles.size() <= c.maximum_markers,"Invalid persistent cell state.");
    // Validate the whole incoming population before consuming random draws.
    // Unique IDs and positive weights make later conservative splits unambiguous.
    std::set<std::uint64_t> ids;
    Vec3 lower;
    Vec3 upper;
    const Vec3 first_position = state.particles.empty() ? Vec3{} : state.particles.front().position;
    lower = upper = first_position;
    for (const Particle& p : state.particles) {
        validate_momentum(p.material_mass,p.momentum);
        require(p.number_weight > 0.0 && std::isfinite(p.number_weight)
                && finite(p.position) && ids.insert(p.id).second
                && p.id < state.next_id,"Invalid marker weight, position or identity.");
        require(state.number_weight_majorant == 0.0 || p.number_weight <= state.number_weight_majorant,
                "A particle number weight exceeds the persistent majorant.");
        if (c.spatial_policy == SpatialPolicy::point_local_population) {
            require(p.position.x == first_position.x && p.position.y == first_position.y
                    && p.position.z == first_position.z,
                    "Point-local populations must occupy the same spatial node.");
        }
        lower = {std::min(lower.x,p.position.x),std::min(lower.y,p.position.y),std::min(lower.z,p.position.z)};
        upper = {std::max(upper.x,p.position.x),std::max(upper.y,p.position.y),std::max(upper.z,p.position.z)};
    }
    // O(N) conservative bound: every pair lies within the bounding-box diagonal.
    if (c.spatial_policy == SpatialPolicy::finite_cell_approximation) {
        require(norm(upper-lower) <= c.maximum_pair_separation,
                "The configured finite-cell separation error guard was exceeded.");
    }
}

// Reject the incomplete modulo block so every index has the same probability.
inline std::size_t random_index(Random& random, std::size_t n) {
    require(n > 0,"An index requires a nonempty population.");
    const std::uint64_t count = static_cast<std::uint64_t>(n);
    const std::uint64_t rejection = -count%count;
    std::uint64_t bits;
    do {
        bits = random.next();
    } while (bits < rejection);
    return static_cast<std::size_t>(bits%count);
}

// Each accepted pair scatters min(q1,q2) physical particles from each packet.
// The unscattered remainder keeps its incoming momentum. A rate proportional
// to max(q1,q2) gives the desired q1*q2 weak collision measure.
inline void scatter_weighted_pair(CellState& s, std::size_t i, std::size_t j) {
    const Particle first = s.particles[i];
    const Particle second = s.particles[j];
    const double q = std::min(first.number_weight,second.number_weight);
    const bool split_first = first.number_weight > q;
    const bool split_second = second.number_weight > q;
    const PairOutput result = scatter_isotropic(
        first.material_mass,first.momentum,second.material_mass,second.momentum,s.random);
    s.particles[i].number_weight = q;
    s.particles[j].number_weight = q;
    s.particles[i].momentum = result.first;
    s.particles[j].momentum = result.second;
    if (split_first || split_second) {
        require(s.next_id < std::numeric_limits<std::uint64_t>::max(),"Marker identity overflow.");
        Particle remainder = split_first ? first : second;
        remainder.number_weight -= q;
        remainder.id = s.next_id++;
        s.particles.push_back(remainder);
    }
}

// Exact thinning of a cellwise homogeneous Markov process. The majorant uses
// v_M <= 2. Normal proper time obeys d tau_n = alpha dt, so coordinate-time
// rates contain one lapse factor.
// The transaction preserves the caller's state on an exception.
inline void advance_cell(CellState& caller, double coordinate_dt, const CellControls& c) {
    validate_cell(caller,c);
    require(coordinate_dt >= 0.0 && std::isfinite(coordinate_dt),"Invalid collision interval.");
    if (coordinate_dt == 0.0) {
        return;
    }
    // Work on a copy. A failed majorant, unresolved clock or population cap
    // therefore preserves the complete incoming state and random stream.
    CellState s = caller;
    if (s.number_weight_majorant == 0.0) {
        for (const Particle& p : s.particles) {
            s.number_weight_majorant = std::max(s.number_weight_majorant,p.number_weight);
        }
    }
    const double end = s.coordinate_time+coordinate_dt;
    require(std::isfinite(end) && end > s.coordinate_time,"Collision clock cannot advance.");
    std::uint64_t interval_proposals = 0;
    while (s.coordinate_time < end && s.particles.size() >= 2 && c.cross_section > 0.0) {
        // Conservative splitting only decreases an individual weight. This
        // persistent majorant remains valid without an O(N) scan per proposal.
        const double qmax = s.number_weight_majorant;
        const double n = static_cast<double>(s.particles.size());
        // There are n(n-1)/2 unordered pairs. Combining this count with the flux
        // upper bound v_M<=2 gives n(n-1) in the proposal rate. The lapse converts
        // proper observer time into the caller coordinate-time interval.
        const double rate = c.lapse*c.cross_section/c.proper_volume*qmax*n*(n-1.0);
        require(rate > 0.0 && std::isfinite(rate),"Invalid collision majorant rate.");
        if (s.remaining_optical_depth < 0.0) {
            s.remaining_optical_depth = -std::log(s.random.uniform());
        }
        const double remaining = end-s.coordinate_time;
        const double wait = s.remaining_optical_depth/rate;
        // When the next proposal lies beyond this frozen interval, subtract only
        // the consumed optical depth. Carrying the remainder avoids redrawing a
        // new event time when the caller divides a physical interval.
        if (wait > remaining) {
            s.remaining_optical_depth -= rate*remaining;
            break;
        }
        require(++interval_proposals <= c.maximum_proposals,"Collision proposal budget exceeded.");
        const double next_time = s.coordinate_time+wait;
        require(next_time > s.coordinate_time || wait == 0.0,
                "Collision waiting time lies below coordinate clock resolution.");
        s.coordinate_time = next_time;
        ++s.proposals;
        // Choose one index from n and the other from n-1; shifting the latter
        // skips i. Thus every distinct pair has the same proposal probability.
        const std::size_t i = random_index(s.random,s.particles.size());
        std::size_t j = random_index(s.random,s.particles.size()-1);
        if (j >= i) {
            ++j;
        }
        const Particle& a = s.particles[i];
        const Particle& b = s.particles[j];
        // Thinning restores the actual Moller flux and the actual larger number
        // weight. Scattering the smaller common weight then yields the desired
        // q1*q2 weak collision measure while retaining a positive remainder.
        const double probability = std::max(a.number_weight,b.number_weight)/qmax
                                   *0.5*moller_flux_speed(a.momentum,b.momentum);
        require(probability >= 0.0 && probability <= 1.0+1.0e-14,
                "The Moller flux majorant was violated.");
        if (s.random.uniform() < probability) {
            require(s.particles.size()+(a.number_weight != b.number_weight)
                        <= c.maximum_markers,"Weighted splitting marker budget exceeded.");
            const double separation = norm(a.position-b.position);
            if (separation > 0.0) {
                ++s.spacelike_exchanges;
            }
            s.maximum_exchange_distance = std::max(s.maximum_exchange_distance,separation);
            scatter_weighted_pair(s,i,j);
            ++s.accepted;
        }
        // After this proposal, draw the next unit exponential optical depth.
        // Rejected proposals advance the proposal clock but preserve momenta.
        s.remaining_optical_depth = -std::log(s.random.uniform());
    }
    // Synchronize the returned cell with the complete frozen interval and
    // commit the whole transaction only after the loop has finished.
    s.coordinate_time = end;
    caller = std::move(s);
}

// Restart integers are written in explicit little-endian byte order.
// Double values are copied through their bit patterns, preserving clocks
// and momenta exactly rather than converting them through decimal text.
namespace restart_detail {
inline void put_u64(std::string& out, std::uint64_t x) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((x>>(8*i))&255));
    }
}
inline void put_double(std::string& out, double x) {
    std::uint64_t bits;
    std::memcpy(&bits,&x,8);
    put_u64(out,bits);
}
inline std::uint64_t checksum(const std::string& bytes) {
    std::uint64_t value = UINT64_C(14695981039346656037);
    for (unsigned char c : bytes) {
        value = (value^c)*UINT64_C(1099511628211);
    }
    return value;
}
struct Reader {
    const std::string& bytes;
    std::size_t offset = 0;
    std::uint64_t u64() {
        require(offset+8 <= bytes.size(),"Truncated collision restart.");
        std::uint64_t result = 0;
        for (int i = 0; i < 8; ++i) {
            result |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[offset++]))<<(8*i);
        }
        return result;
    }
    double real() {
        const std::uint64_t bits = u64();
        double result;
        std::memcpy(&result,&bits,8);
        return result;
    }
};
}  // namespace restart_detail

// Collision-cell state format, separate from the solver checkpoint schema.
// The checksum detects corruption; an external SHA256 authenticates provenance.
inline std::string serialize_cell(const CellState& s, const CellControls& c) {
    validate_cell(s,c);
    using namespace restart_detail;
    std::string out = "RBLOCAL2";
    for (double x : {c.cross_section,c.proper_volume,c.lapse,c.maximum_pair_separation}) {
        put_double(out,x);
    }
    for (std::uint64_t x : {static_cast<std::uint64_t>(c.spatial_policy),
                            c.maximum_proposals,c.maximum_markers,s.random.state,s.random.draws,
                            s.next_id,s.proposals,s.accepted,s.spacelike_exchanges}) {
        put_u64(out,x);
    }
    for (double x : {s.coordinate_time,s.remaining_optical_depth,s.number_weight_majorant,
                     s.maximum_exchange_distance}) {
        put_double(out,x);
    }
    put_u64(out,s.particles.size());
    for (const Particle& p : s.particles) {
        for (double x : {p.material_mass,p.number_weight,p.momentum.t,
                         p.momentum.p.x,p.momentum.p.y,p.momentum.p.z,
                         p.position.x,p.position.y,p.position.z}) {
            put_double(out,x);
        }
        put_u64(out,p.id);
    }
    put_u64(out,checksum(out));
    return out;
}

// Check the entire record and declared controls before returning a state.
// The local checksum detects corruption; the surrounding solver owns
// source identity and complete particle/field restart compatibility.
inline CellState deserialize_cell(const std::string& bytes, const CellControls& expected) {
    using namespace restart_detail;
    require(bytes.size() >= 8 && bytes.substr(0,8) == "RBLOCAL2","Unknown collision restart format.");
    Reader tail{bytes,bytes.size()-8};
    require(checksum(bytes.substr(0,bytes.size()-8)) == tail.u64(),"Collision restart checksum failed.");
    Reader r{bytes,8};
    CellControls c;
    c.cross_section = r.real();
    c.proper_volume = r.real();
    c.lapse = r.real();
    c.maximum_pair_separation = r.real();
    c.spatial_policy = static_cast<SpatialPolicy>(r.u64());
    c.maximum_proposals = r.u64();
    c.maximum_markers = r.u64();
    CellState s;
    s.random.state = r.u64();
    s.random.draws = r.u64();
    s.next_id = r.u64();
    s.proposals = r.u64();
    s.accepted = r.u64();
    s.spacelike_exchanges = r.u64();
    s.coordinate_time = r.real();
    s.remaining_optical_depth = r.real();
    s.number_weight_majorant = r.real();
    s.maximum_exchange_distance = r.real();
    const std::uint64_t count = r.u64();
    require(count <= expected.maximum_markers && count <= (bytes.size()-r.offset)/80,
            "Collision restart marker count exceeds bounds.");
    for (std::uint64_t i = 0; i < count; ++i) {
        Particle p;
        p.material_mass = r.real();
        p.number_weight = r.real();
        p.momentum.t = r.real();
        p.momentum.p = {r.real(),r.real(),r.real()};
        p.position = {r.real(),r.real(),r.real()};
        p.id = r.u64();
        s.particles.push_back(p);
    }
    require(r.offset+8 == bytes.size(),"Unexpected collision restart suffix.");
    validate_cell(s,c);
    require(serialize_cell(CellState{},c) == serialize_cell(CellState{},expected),
            "Collision restart controls differ from the requested model.");
    return s;
}

}  // namespace rb
