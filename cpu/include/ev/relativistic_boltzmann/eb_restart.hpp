#pragma once

// Save a complete spherical material collision cell in a binary record.
// The record binds the SC or EF slice, material mass, area, cell geometry and
// sampling limits to the packet population, random state and optical depth.
// The caller separately saves transported particles and the gravitational field;
// this cell record supplies the stochastic state needed for their restart.

#include "spherical_collisions.hpp"

namespace eb {

// This per-cell record accompanies the caller's transport and field state.
// Packet identity, ancestry and the residual random clock are serialized together.
inline std::string serialize_cell(const Cell& s, const Model& m, const Geometry& g) {
    validate(s,m,g);
    using namespace rb::restart_detail;
    std::string bytes = "EBSLICE1";
    put_u64(bytes,static_cast<std::uint64_t>(g.slice));
    // Write physical model and cell geometry before stochastic state so a
    // reader can compare their exact binary values with its requested model.
    for (double x : {m.material_mass,m.cross_section,g.volume,g.time_factor,g.maximum_radial_separation})
        put_double(bytes,x);
    put_u64(bytes,m.identity.size());
    bytes += m.identity;
    // The RNG state and draw count identify the actual future random stream.
    // The residual optical depth below retains an unfinished proposal wait.
    for (std::uint64_t x : {static_cast<std::uint64_t>(g.allow_radial_cell_approximation),
                            static_cast<std::uint64_t>(g.maximum_markers),g.maximum_proposals,
                            s.random.state,s.random.draws,s.next_id,s.proposals,s.collisions,
                            s.radially_nonlocal_collisions}) put_u64(bytes,x);
    for (double x : {s.coordinate_time,s.optical_depth,s.number_majorant,s.maximum_radial_distance,
                     s.maximum_pair_energy_relative_residual,s.maximum_null_flux_factor}) put_double(bytes,x);
    put_u64(bytes,s.packets.size());
    // Each packet preserves its radius, momentum, rest weight and identity.
    // source_id labels its origin; ancestor_id identifies the incoming record
    // whose auxiliary properties survive an unequal-weight split.
    for (const auto& p : s.packets) {
        for (double x : {p.rest_weight,p.r,p.radial,p.eta}) put_double(bytes,x);
        put_u64(bytes,p.id);
        put_u64(bytes,p.source_id);
        put_u64(bytes,p.ancestor_id);
    }
    put_u64(bytes,checksum(bytes));
    return bytes;
}

// Read into a new local state. Bounds, checksum and canonical
// re-serialization must all pass before the caller can use the result.
inline Cell deserialize_cell(const std::string& bytes, const Model& model, const Geometry& expected) {
    using namespace rb::restart_detail;
    rb::require(bytes.size() >= 16 && bytes.substr(0,8) == "EBSLICE1","Unknown EB slice restart.");
    Reader tail{bytes,bytes.size()-8};
    rb::require(checksum(bytes.substr(0,bytes.size()-8)) == tail.u64(),"EB slice restart checksum failed.");
    Reader r{bytes,8};
    Geometry g;
    Model m;
    g.slice = static_cast<Slice>(r.u64());
    m.material_mass = r.real();
    m.cross_section = r.real();
    g.volume = r.real();
    g.time_factor = r.real();
    g.maximum_radial_separation = r.real();
    const auto length = r.u64();
    rb::require(length <= 128 && length <= bytes.size()-r.offset,"Invalid EB model identity length.");
    m.identity = bytes.substr(r.offset,length);
    r.offset += length;
    const auto approximate = r.u64();
    rb::require(approximate <= 1,"Invalid spatial policy in EB restart.");
    g.allow_radial_cell_approximation = approximate;
    g.maximum_markers = r.u64();
    g.maximum_proposals = r.u64();
    Cell s;
    s.random.state = r.u64();
    s.random.draws = r.u64();
    s.next_id = r.u64();
    s.proposals = r.u64();
    s.collisions = r.u64();
    s.radially_nonlocal_collisions = r.u64();
    s.coordinate_time = r.real();
    s.optical_depth = r.real();
    s.number_majorant = r.real();
    s.maximum_radial_distance = r.real();
    s.maximum_pair_energy_relative_residual = r.real();
    s.maximum_null_flux_factor = r.real();
    const auto count = r.u64();
    rb::require(count <= expected.maximum_markers && count <= (bytes.size()-r.offset)/56,
                "EB slice restart population exceeds its authenticated bound.");
    for (std::uint64_t i = 0; i < count; ++i) {
        Packet p;
        p.rest_weight = r.real();
        p.r = r.real();
        p.radial = r.real();
        p.eta = r.real();
        p.id = r.u64();
        p.source_id = r.u64();
        p.ancestor_id = r.u64();
        s.packets.push_back(p);
    }
    rb::require(r.offset+8 == bytes.size(),"Unexpected EB slice restart suffix.");
    // Comparing empty records checks every model, slice and geometry field.
    // Comparing the full record then also checks packet and stochastic state
    // and rejects trailing or noncanonical bytes.
    rb::require(serialize_cell(Cell{},m,g) == serialize_cell(Cell{},model,expected),
                "EB material model/slice/geometry/algorithm controls changed.");
    rb::require(serialize_cell(s,model,expected) == bytes,"EB slice restart canonical correspondence failed.");
    return s;
}

}  // namespace eb
