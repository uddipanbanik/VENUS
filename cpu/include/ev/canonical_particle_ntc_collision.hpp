#pragma once

// Connect the canonical orbit distribution to local particle collision sampling.
// The orbit atlas is a table of energy rows and phase cells. An open carrier
// is a numerical mass element retaining a product's continuous orbit state.
// Inputs are atlas phase-cell masses/energy moments, physical phase-space
// geometry, a frozen potential, collision controls, and persistent sampler state.
// The equal-mass path samples disjoint represented-mass intervals as parcels,
// applies Bird no-time-counter elastic encounters, and transfers (lifts) each
// changed parcel's mass and energy into the atlas or an explicit open carrier.
// Outputs include updated atlas mass/energy moments and collision/conservation
// budgets. A parcel can scatter repeatedly, but its original ownership is
// withdrawn once. Parcel mass is represented sampling weight; the selected
// physical cross section retains the continuum normalization.
// The final wrapper exposes the separate weighted ownership-coupled NTC path.

#include "ev/canonical_orbit_phase_transport.hpp"

namespace ev {

namespace canonical_particle_ntc_detail {

// Ownership identifies where represented mass must be withdrawn when its
// sampled parcel changes: an atlas cell or a persistent open-particle record.
enum class OwnershipKind : std::uint8_t {
    canonical,
    open,
};

struct OwnershipTerm {
    OwnershipKind kind = OwnershipKind::canonical;
    std::size_t index = 0;
    double physical_mass = 0.0;
};

// One representative phase-space point carries an explicit list of mass
// ownership pieces. Its cumulative endpoint locates it on the global mass axis.
struct SourceAtom {
    WeightedPICMarker marker;
    OwnershipKind kind = OwnershipKind::canonical;
    std::size_t source_index = 0;
    std::size_t canonical_row = 0;
    std::vector<OwnershipTerm> ownership;
    long double cumulative_end = 0.0L;
};

// Parcel states and ownership lists use the same index. Sampling a point
// inside an interval chooses its orbit; the full interval determines its mass.
struct EqualMassParcelSet {
    std::vector<WeightedPICMarker> marker;
    std::vector<std::vector<OwnershipTerm>> ownership;
    std::vector<OwnershipKind> representative_kind;
    double physical_mass = 0.0;
    double active_open_mass = 0.0;
};

// Sum ownership in extended precision before comparing it with the
// represented particle weight used by the collision sampler.
inline long double ownership_mass(const std::vector<OwnershipTerm>& term) {
    long double result = 0.0L;
    for (const OwnershipTerm& item : term) {
        result += item.physical_mass;
    }
    return result;
}

// Retain positive mass pieces and coalesce consecutive pieces from the
// same owner, limiting list size while preserving the withdrawal ledger.
inline void append_ownership(
    std::vector<OwnershipTerm>& destination,
    OwnershipKind kind,
    std::size_t index,
    long double mass) {
    if (!(mass > 0.0L)) {
        return;
    }
    const double deposited = static_cast<double>(mass);
    if (!destination.empty()
        && destination.back().kind == kind
        && destination.back().index == index) {
        destination.back().physical_mass += deposited;
    } else {
        destination.push_back({kind, index, deposited});
    }
}

// Concatenate source ownership on one mass coordinate, divide it into
// equal parcels, and retain exact interval overlaps with every source owner.
inline EqualMassParcelSet make_equal_mass_parcels(
    const CanonicalNTCMarkerInitialization& canonical,
    const CanonicalOrbitAtlas& atlas,
    const std::vector<WeightedPICMarker>& open_marker,
    std::uint64_t seed,
    std::uint64_t epoch) {
    std::vector<SourceAtom> atom;
    atom.reserve(canonical.sample.markers.size() + open_marker.size());
    // Map each phase cell back to its energy/eta row so sorting keeps
    // nearby orbital states adjacent before parcel intervals are formed.
    std::vector<std::uint32_t> cell_to_row(
        atlas.column.size(), std::numeric_limits<std::uint32_t>::max());
    for (std::size_t row_index = 0; row_index < atlas.row.size(); ++row_index) {
        const CanonicalOrbitRow& row = atlas.row[row_index];
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            cell_to_row[row.first_cell + q] =
                static_cast<std::uint32_t>(row_index);
        }
    }
    for (std::size_t q = 0; q < canonical.sample.markers.size(); ++q) {
        SourceAtom source;
        source.marker = canonical.sample.markers[q];
        source.kind = OwnershipKind::canonical;
        source.source_index = q;
        source.ownership.reserve(canonical.ownership[q].size());
        canonical.ownership[q].for_each(
            [&](const CanonicalNTCOwnershipTerm& term) {
            append_ownership(
                source.ownership,
                OwnershipKind::canonical,
                term.canonical_cell,
                term.physical_mass);
        });
        if (source.ownership.empty()
            || source.ownership.front().index >= cell_to_row.size()
            || cell_to_row[source.ownership.front().index]
                == std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "Canonical equal-mass source has no orbit-row key.");
        }
        source.canonical_row =
            cell_to_row[source.ownership.front().index];
        atom.push_back(std::move(source));
    }
    double active_open_mass = 0.0;
    for (std::size_t q = 0; q < open_marker.size(); ++q) {
        const WeightedPICMarker& marker = open_marker[q];
        validate_weighted_pic_marker(
            marker, "Canonical equal-mass collision source");
        if (marker.state != WeightedPICMarkerState::active
            || !(marker.mass > 0.0)) {
            continue;
        }
        SourceAtom source;
        source.marker = marker;
        source.kind = OwnershipKind::open;
        source.source_index = q;
        source.canonical_row = atlas.row.size();
        source.ownership.push_back({
            OwnershipKind::open, q, marker.mass});
        atom.push_back(std::move(source));
        active_open_mass += marker.mass;
    }
    // A deterministic physical/identity ordering fixes the mass-coordinate
    // partition. It changes the sampling representation, not source masses.
    std::stable_sort(
        atom.begin(), atom.end(), [](const SourceAtom& first,
                                     const SourceAtom& second) {
            if (first.canonical_row != second.canonical_row) {
                return first.canonical_row < second.canonical_row;
            }
            if (first.marker.source_cell != second.marker.source_cell) {
                return first.marker.source_cell < second.marker.source_cell;
            }
            if (first.marker.eta != second.marker.eta) {
                return first.marker.eta < second.marker.eta;
            }
            if (first.marker.w != second.marker.w) {
                return first.marker.w < second.marker.w;
            }
            if (first.marker.r != second.marker.r) {
                return first.marker.r < second.marker.r;
            }
            if (first.marker.id != second.marker.id) {
                return first.marker.id < second.marker.id;
            }
            if (first.kind != second.kind) {
                return first.kind < second.kind;
            }
            return first.source_index < second.source_index;
        });
    long double total_mass = 0.0L;
    for (SourceAtom& source : atom) {
        const long double owned = ownership_mass(source.ownership);
        const long double scale = std::max(
            1.0L, static_cast<long double>(source.marker.mass));
        if (std::abs(owned - source.marker.mass)
            > 4096.0L * std::numeric_limits<double>::epsilon() * scale) {
            throw std::runtime_error(
                "Canonical equal-mass source ownership failed closure.");
        }
        total_mass += owned;
        source.cumulative_end = total_mass;
    }
    if (!(total_mass > 0.0L) || atom.empty()) {
        throw std::runtime_error(
            "Canonical equal-mass collision bridge found no active mass.");
    }
    const std::size_t parcel_count = canonical.sample.markers.size();
    if (parcel_count < 2) {
        throw std::runtime_error(
            "Canonical equal-mass collision bridge needs two parcels.");
    }

    EqualMassParcelSet result;
    result.marker.resize(parcel_count);
    result.ownership.resize(parcel_count);
    result.representative_kind.resize(parcel_count);
    result.physical_mass = static_cast<double>(total_mass);
    result.active_open_mass = active_open_mass;
    // Equal represented weights permit the conservative equal-mass DSMC
    // map. The final interval consumes the exact accumulated mass remainder.
    const long double parcel_mass = total_mass / parcel_count;
    for (std::size_t parcel = 0; parcel < parcel_count; ++parcel) {
        const long double interval_begin = parcel_mass * parcel;
        const long double interval_end = parcel + 1 == parcel_count
            ? total_mass : parcel_mass * (parcel + 1);
        // Select one quiet point within each parcel's own interval. The
        // representative velocity/radius and its full ownership remain distinct.
        const double quiet = weighted_pic_quiet_coordinate(
            parcel, 2, seed, epoch, 0, 0);
        const long double coordinate = interval_begin
            + quiet * (interval_end - interval_begin);
        const auto representative_iterator = std::upper_bound(
            atom.begin(), atom.end(), coordinate,
            [](long double value, const SourceAtom& source) {
                return value < source.cumulative_end;
            });
        if (representative_iterator == atom.end()) {
            throw std::runtime_error(
                "Canonical equal-mass representative left its mass interval.");
        }
        WeightedPICMarker marker = representative_iterator->marker;
        marker.mass = static_cast<double>(interval_end - interval_begin);
        marker.id = epoch + parcel;
        result.marker[parcel] = marker;
        result.representative_kind[parcel] = representative_iterator->kind;

        // Intersect this parcel interval with source atoms and their owned
        // pieces. These disjoint overlaps prevent double withdrawal later.
        std::size_t source_index = static_cast<std::size_t>(
            std::upper_bound(
                atom.begin(), atom.end(), interval_begin,
                [](long double value, const SourceAtom& source) {
                    return value < source.cumulative_end;
                }) - atom.begin());
        long double cursor = interval_begin;
        while (cursor < interval_end && source_index < atom.size()) {
            const long double source_begin = source_index == 0
                ? 0.0L : atom[source_index - 1].cumulative_end;
            const long double overlap_end = std::min(
                interval_end, atom[source_index].cumulative_end);
            const long double local_begin = std::max(cursor, source_begin)
                - source_begin;
            const long double local_end = overlap_end - source_begin;
            long double term_begin = 0.0L;
            for (const OwnershipTerm& term : atom[source_index].ownership) {
                const long double term_end = term_begin + term.physical_mass;
                const long double piece_begin = std::max(
                    local_begin, term_begin);
                const long double piece_end = std::min(local_end, term_end);
                append_ownership(
                    result.ownership[parcel],
                    term.kind,
                    term.index,
                    piece_end - piece_begin);
                term_begin = term_end;
                if (term_begin >= local_end) {
                    break;
                }
            }
            cursor = overlap_end;
            if (cursor >= atom[source_index].cumulative_end) {
                ++source_index;
            }
        }
        const long double owned = ownership_mass(result.ownership[parcel]);
        const long double expected = interval_end - interval_begin;
        if (std::abs(owned - expected)
            > 8192.0L * std::numeric_limits<double>::epsilon()
                * std::max(1.0L, expected)) {
            throw std::runtime_error(
                "Canonical equal-mass parcel ownership failed closure.");
        }
    }
    // Radius sorting supports the collision-shell builder. Apply the same
    // permutation to states, complete ownership lists, and representative tags.
    std::vector<std::size_t> order(parcel_count);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(
        order.begin(), order.end(), [&](std::size_t first, std::size_t second) {
            if (result.marker[first].r != result.marker[second].r) {
                return result.marker[first].r < result.marker[second].r;
            }
            return result.marker[first].id < result.marker[second].id;
        });
    EqualMassParcelSet sorted;
    sorted.marker.resize(parcel_count);
    sorted.ownership.resize(parcel_count);
    sorted.representative_kind.resize(parcel_count);
    sorted.physical_mass = result.physical_mass;
    sorted.active_open_mass = result.active_open_mass;
    for (std::size_t q = 0; q < parcel_count; ++q) {
        sorted.marker[q] = result.marker[order[q]];
        sorted.ownership[q] = std::move(result.ownership[order[q]]);
        sorted.representative_kind[q] =
            result.representative_kind[order[q]];
    }
    return sorted;
}

template <class PotentialAtRadius>
// Set each representative's eta and radial speed from the mass-weighted
// angular momentum and energy of its actual ownership. This aligns the
// sampled kinetic state with the source moments that will be withdrawn.
inline void align_parcels_with_owned_energy(
    EqualMassParcelSet& parcel,
    const CanonicalOrbitAtlas& atlas,
    const std::vector<WeightedPICMarker>& open_marker,
    PotentialAtRadius&& potential_at_radius) {
    std::vector<double> canonical_mean_energy(atlas.column.size(), 0.0);
    std::vector<double> canonical_eta(atlas.column.size(), 0.0);
    for (const CanonicalOrbitRow& row : atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t cell = row.first_cell + q;
            canonical_mean_energy[cell] = row.phase_mass[q] > 0.0
                ? row.phase_energy_mass[q] / row.phase_mass[q]
                : row.energy;
            canonical_eta[cell] = row.eta;
        }
    }
    std::vector<double> open_energy(open_marker.size(), 0.0);
    for (std::size_t q = 0; q < open_marker.size(); ++q) {
        const WeightedPICMarker& marker = open_marker[q];
        const double radius2 = std::max(
            marker.r * marker.r, std::numeric_limits<double>::min());
        open_energy[q] = potential_at_radius(marker.r)
            + 0.5 * marker.w * marker.w + marker.eta / radius2;
    }
    for (std::size_t q = 0; q < parcel.marker.size(); ++q) {
        long double mass = 0.0L;
        long double energy_mass = 0.0L;
        long double eta_mass = 0.0L;
        for (const OwnershipTerm& term : parcel.ownership[q]) {
            const double energy = term.kind == OwnershipKind::canonical
                ? canonical_mean_energy[term.index]
                : open_energy[term.index];
            const double eta = term.kind == OwnershipKind::canonical
                ? canonical_eta[term.index]
                : open_marker[term.index].eta;
            mass += term.physical_mass;
            energy_mass += static_cast<long double>(term.physical_mass)
                * energy;
            eta_mass += static_cast<long double>(term.physical_mass) * eta;
        }
        if (!(mass > 0.0L)) {
            throw std::runtime_error(
                "Canonical equal-mass parcel has no owned energy.");
        }
        WeightedPICMarker& marker = parcel.marker[q];
        const double target_energy = static_cast<double>(energy_mass / mass);
        marker.eta = std::max(0.0, static_cast<double>(eta_mass / mass));
        const double radius2 = std::max(
            marker.r * marker.r, std::numeric_limits<double>::min());
        // At the sampled radius, H-Phi-eta/r^2 must leave nonnegative radial
        // kinetic energy. A genuinely inaccessible mixture requires finer
        // sampling; only the stated rounding allowance permits zero clamping.
        double radial_speed2 = 2.0 * (
            target_energy - potential_at_radius(marker.r)
            - marker.eta / radius2);
        const double tolerance = 4096.0
            * std::numeric_limits<double>::epsilon()
            * std::max({
                1.0,
                std::abs(target_energy),
                std::abs(potential_at_radius(marker.r)),
                std::abs(marker.eta / radius2),
            });
        if (radial_speed2 < -tolerance) {
            std::ostringstream message;
            message
                << "Canonical equal-mass parcel ownership energy is inaccessible"
                << "; marker=" << q
                << ", r=" << marker.r
                << ", eta=" << marker.eta
                << ", H=" << target_energy
                << ", phi=" << potential_at_radius(marker.r)
                << ", radial_speed2=" << radial_speed2
                << ", ownership_terms=" << parcel.ownership[q].size()
                << ", source_cell=" << marker.source_cell
                << ". Increase the collision marker count.";
            throw std::runtime_error(message.str());
        }
        radial_speed2 = std::max(0.0, radial_speed2);
        marker.w = std::copysign(
            std::sqrt(radial_speed2), marker.w == 0.0 ? 1.0 : marker.w);
    }
}

// Classify outgoing mass by why the current atlas cannot represent its
// eta/energy/orbit. The explicit carrier retains that physical mass.
inline void classify_open_mass(
    CanonicalNTCCollisionDiagnostics& diagnostics,
    CanonicalNTCOpenReason reason,
    double mass) {
    diagnostics.physical_open_mass += mass;
    switch (reason) {
    case CanonicalNTCOpenReason::energy_above_atlas:
        diagnostics.physical_high_energy_open_mass += mass;
        break;
    case CanonicalNTCOpenReason::eta_outside_atlas:
        diagnostics.unrepresented_eta_mass += mass;
        break;
    case CanonicalNTCOpenReason::eta_row_missing:
        diagnostics.unrepresented_missing_eta_row_mass += mass;
        break;
    case CanonicalNTCOpenReason::energy_outside_row:
        diagnostics.unrepresented_energy_mass += mass;
        break;
    case CanonicalNTCOpenReason::orbit_geometry_failure:
        diagnostics.unrepresented_orbit_geometry_mass += mass;
        break;
    case CanonicalNTCOpenReason::none:
        throw std::runtime_error(
            "Canonical particle NTC open output has no reason.");
    }
}

}  // namespace canonical_particle_ntc_detail

/// Applies one local equal-mass Bird NTC collision interval to a quiet sample
/// of the instantaneous canonical distribution.  Unchanged parcels leave the
/// atlas untouched.  Every changed parcel is removed from its disjoint source
/// ownership exactly once and deposited once at its final continuous velocity.
template <class PotentialAtRadius>
inline CanonicalNTCCollisionDiagnostics apply_canonical_particle_ntc_collision(
    CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    const Params& p,
    const CanonicalNTCCollisionOptions& options,
    CanonicalNTCCollisionState& state,
    PotentialAtRadius&& potential_at_radius) {
    if (!(dt >= 0.0) || !std::isfinite(dt)
        || options.marker_budget < 2 || options.threads == 0) {
        throw std::runtime_error(
            "Invalid canonical equal-mass NTC collision request.");
    }
    CanonicalNTCCollisionDiagnostics diagnostics;
    if (dt == 0.0 || p.bp_collision_strength == 0.0) {
        return diagnostics;
    }
    Params pic_p = p;
    pic_p.physics_model = "vlasov_poisson";
    CanonicalNTCCollisionOptions source_options = options;
    // Equal collision weights are required for pairwise conservative DSMC.
    // The quiet source sample is therefore mass-proportional.
    source_options.core_importance_fraction = 0.0;
    const auto initialization_begin = std::chrono::steady_clock::now();
    CanonicalNTCMarkerInitialization canonical =
        initialize_canonical_ntc_markers(
            atlas,
            geometry,
            pic_p,
            source_options,
            state.marker_sampler,
            potential_at_radius,
            &state.source_orbit_cache);
    // This equal-mass entry samples atlas ownership alone. Existing open
    // carriers remain a separately bounded part of the event state.
    const std::vector<WeightedPICMarker> no_open_collision_source;
    canonical_particle_ntc_detail::EqualMassParcelSet parcel =
        canonical_particle_ntc_detail::make_equal_mass_parcels(
            canonical,
            atlas,
            no_open_collision_source,
            state.collision_sampler.seed,
            state.collision_sampler.next_counter);
    canonical_particle_ntc_detail::align_parcels_with_owned_energy(
        parcel, atlas, no_open_collision_source, potential_at_radius);
    diagnostics.marker_initialization_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initialization_begin).count();
    diagnostics.source_row_geometry_reuses =
        canonical.source_row_geometry_reuses;
    diagnostics.source_exact_orbit_uses =
        canonical.source_exact_orbit_uses;
    diagnostics.source_exact_orbit_builds =
        canonical.source_exact_orbit_builds;
    diagnostics.source_exact_orbit_cache_hits =
        canonical.source_exact_orbit_cache_hits;
    diagnostics.source_unique_exact_orbit_cells =
        canonical.source_unique_exact_orbit_cells;
    diagnostics.effective_marker_number =
        canonical.effective_marker_number;
    diagnostics.maximum_source_energy_row_fraction =
        canonical.maximum_source_energy_row_fraction;
    diagnostics.maximum_source_eta_cell_fraction =
        canonical.maximum_source_eta_cell_fraction;
    diagnostics.initialization = canonical.sample.diagnostics;
    diagnostics.initialization.marker_count = parcel.marker.size();
    diagnostics.initialization.source_mass = parcel.physical_mass;
    diagnostics.initialization.represented_mass =
        weighted_pic_total_marker_mass(parcel.marker);
    // Atlas-external carriers are bounded separately by the event scheduler.
    // They are not mixed into the equal-weight atlas sample.
    diagnostics.active_open_collision_sources = 0;

    // Observe initial sampling noise and physical moments before encounters.
    // Build the same radius-local shells used by the maintained PIC operator.
    const WeightedPICRadialMesh radial_mesh =
        make_weighted_pic_radial_mesh(geometry);
    const WeightedPICNoiseDiagnostics noise_before =
        diagnose_weighted_pic_noise(parcel.marker, radial_mesh);
    const WeightedPICDepositedMoments deposited_before =
        deposit_weighted_vlasov_pic(parcel.marker, radial_mesh);
    WeightedPICCollisionOptions collision_options = options.collision;
    // Repeated encounters update the continuous parcel velocity in place.
    // Its final state will be lifted once after the whole collision interval.
    collision_options.at_most_one_collision_per_marker = false;
    collision_options.input_ordered_by_radius = true;
    build_weighted_pic_collision_shells(
        parcel.marker,
        radial_mesh,
        p,
        collision_options,
        state.workspace);
    diagnostics.physical_collision_timescale =
        weighted_pic_sampled_collision_timescale(
            parcel.marker,
            p,
            collision_options,
            state.workspace,
            state.collision_sampler.seed
                ^ state.collision_sampler.next_counter);
    diagnostics.collision_majorant_timescale =
        weighted_pic_collision_majorant_timescale(state.workspace.shells);
    // Check the requested interval against the shell proposal-frequency
    // CFL before running the physical sampler; sampled rate and majorant
    // timescales are retained as separate diagnostics.
    const double collision_limit = weighted_pic_collision_timestep(
        state.workspace.shells, collision_options);
    if (std::isfinite(collision_limit)
        && dt > collision_limit
            * (1.0 + 32.0 * std::numeric_limits<double>::epsilon())) {
        throw std::runtime_error(
            "Canonical equal-mass NTC interval exceeds its collision CFL.");
    }

    // Preserve incoming parcel states to identify changed ownership and
    // compare its source energy with the final outgoing lift.
    const std::vector<WeightedPICMarker> marker_before = parcel.marker;
    std::vector<WeightedPICCollisionEvent> event;
    const auto collision_begin = std::chrono::steady_clock::now();
    diagnostics.collision = apply_weighted_pic_collision_interval(
        parcel.marker,
        radial_mesh,
        dt,
        p,
        collision_options,
        state.collision_sampler,
        state.workspace,
        options.threads,
        &event);
    diagnostics.collision_pair_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - collision_begin).count();
    diagnostics.virtual_collision_batches = 1;
    diagnostics.minimum_shell_virtual_batches = 1;
    diagnostics.maximum_shell_virtual_batches = 1;
    diagnostics.physical_radial_momentum_residual =
        diagnostics.collision.radial_momentum_error;
    diagnostics.physical_kinetic_energy_residual =
        diagnostics.collision.kinetic_energy_error;

    // Event records identify all touched parcels. A boolean union deduplicates
    // parcels that experienced several encounters during this interval.
    std::vector<unsigned char> changed(parcel.marker.size(), 0);
    for (const WeightedPICCollisionEvent& collision : event) {
        changed[collision.first_marker] = 1;
        changed[collision.second_marker] = 1;
        const auto first_kind =
            parcel.representative_kind[collision.first_marker];
        const auto second_kind =
            parcel.representative_kind[collision.second_marker];
        if (first_kind == canonical_particle_ntc_detail::OwnershipKind::canonical
            && second_kind
                == canonical_particle_ntc_detail::OwnershipKind::canonical) {
            ++diagnostics.canonical_canonical_events;
        } else if (
            first_kind == canonical_particle_ntc_detail::OwnershipKind::open
            && second_kind
                == canonical_particle_ntc_detail::OwnershipKind::open) {
            ++diagnostics.open_open_events;
        } else {
            ++diagnostics.canonical_open_events;
        }
    }
    std::vector<std::size_t> changed_index;
    changed_index.reserve(2 * event.size());
    for (std::size_t q = 0; q < changed.size(); ++q) {
        if (changed[q] != 0) {
            changed_index.push_back(q);
        }
    }

    // Share the frozen potential's radial search geometry for all outgoing
    // orbit lifts, including exact evaluation when cached phase error is large.
    std::vector<double> turning_scan;
    turning_scan.reserve(
        geometry.radial.edge.size() + geometry.radial.center.size());
    turning_scan.insert(
        turning_scan.end(),
        geometry.radial.edge.begin(), geometry.radial.edge.end());
    turning_scan.insert(
        turning_scan.end(),
        geometry.radial.center.begin(), geometry.radial.center.end());
    std::sort(turning_scan.begin(), turning_scan.end());
    turning_scan.erase(
        std::unique(
            turning_scan.begin(), turning_scan.end(),
            [](double first, double second) {
                return phase_space_nearly_equal(first, second);
            }),
        turning_scan.end());
    struct LiftedMarker {
        CanonicalNTCMarkerLift before;
        CanonicalNTCMarkerLift after;
        std::size_t marker = 0;
        std::size_t cached_phase_lifts = 0;
        std::size_t exact_phase_lifts = 0;
        double maximum_estimated_phase_cell_error = 0.0;
    };
    std::vector<LiftedMarker> lifted(changed_index.size());
    const auto lift_begin = std::chrono::steady_clock::now();
    // Each changed parcel has independent incoming/final orbit lifts, so
    // compute these conversions in disjoint thread ranges before atlas updates.
    if (!lifted.empty()) {
        const int threads = static_cast<int>(std::min<std::size_t>(
            options.threads, lifted.size()));
        parallel_for_chunks(
            0,
            static_cast<int>(lifted.size()),
            threads,
            [&](int begin, int end, int) {
                for (int number = begin; number < end; ++number) {
                    LiftedMarker& output = lifted[
                        static_cast<std::size_t>(number)];
                    output.marker = changed_index[
                        static_cast<std::size_t>(number)];
                    const WeightedPICMarker& source =
                        marker_before[output.marker];
                    const WeightedPICMarker& marker =
                        parcel.marker[output.marker];
                    const double source_tangential_speed =
                        std::sqrt(2.0 * source.eta)
                        / std::max(source.r, 1.0e-300);
                    const double tangential_speed = std::sqrt(2.0 * marker.eta)
                        / std::max(marker.r, 1.0e-300);
                    output.before = canonical_ntc_lift_marker(
                        atlas,
                        geometry,
                        p,
                        source.r,
                        BPFAVec3{source.w, source_tangential_speed, 0.0},
                        options.hidden_central_mass,
                        turning_scan,
                        options.maximum_lift_phase_interpolation_cell_error,
                        &output.cached_phase_lifts,
                        &output.exact_phase_lifts,
                        &output.maximum_estimated_phase_cell_error,
                        potential_at_radius);
                    output.after = canonical_ntc_lift_marker(
                        atlas,
                        geometry,
                        p,
                        marker.r,
                        BPFAVec3{marker.w, tangential_speed, 0.0},
                        options.hidden_central_mass,
                        turning_scan,
                        options.maximum_lift_phase_interpolation_cell_error,
                        &output.cached_phase_lifts,
                        &output.exact_phase_lifts,
                        &output.maximum_estimated_phase_cell_error,
                        potential_at_radius);
                }
            });
    }
    for (const LiftedMarker& marker : lifted) {
        diagnostics.outgoing_cached_phase_lifts += marker.cached_phase_lifts;
        diagnostics.outgoing_exact_phase_lifts += marker.exact_phase_lifts;
        diagnostics.maximum_estimated_lift_phase_cell_error = std::max(
            diagnostics.maximum_estimated_lift_phase_cell_error,
            marker.maximum_estimated_phase_cell_error);
    }
    diagnostics.outgoing_lift_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - lift_begin).count();

    // Stage atlas mass and its first two energy moments in extended
    // precision. Canonical reduced mass converts to physical mass by 8*pi^2.
    constexpr long double phase_factor =
        8.0L * static_cast<long double>(PI) * static_cast<long double>(PI);
    const double atlas_mass_before = canonical_orbit_atlas_mass(atlas);
    long double open_mass_before = 0.0L;
    for (const WeightedPICMarker& marker : state.open_marker) {
        open_mass_before += marker.mass;
    }
    diagnostics.physical_mass_before = static_cast<double>(
        phase_factor * atlas_mass_before + open_mass_before);
    std::vector<long double> cell_mass(atlas.column.size(), 0.0L);
    std::vector<long double> cell_energy(atlas.column.size(), 0.0L);
    std::vector<long double> cell_energy2(atlas.column.size(), 0.0L);
    for (const CanonicalOrbitRow& row : atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t cell = row.first_cell + q;
            cell_mass[cell] = row.phase_mass[q];
            cell_energy[cell] = row.phase_energy_mass[q];
            cell_energy2[cell] = row.phase_energy2_mass[q];
        }
    }
    std::vector<WeightedPICMarker> staged_open = state.open_marker;
    std::vector<WeightedPICMarker> new_open;
    new_open.reserve(lifted.size());
    std::uint64_t next_open_id = state.next_open_marker_id;
    // Destination lift weights divide one parcel among represented orbit
    // cells and carry the same energy into their first and second moments.
    auto deposit_lift = [&](const CanonicalNTCMarkerLift& lift,
                            long double physical_mass) {
        for (int q = 0; q < lift.terms; ++q) {
            const CanonicalNTCLiftTerm& term =
                lift.term[static_cast<std::size_t>(q)];
            const long double reduced = physical_mass / phase_factor
                * term.weight;
            const long double energy = term.energy;
            cell_mass[term.canonical_cell] += reduced;
            cell_energy[term.canonical_cell] += reduced * energy;
            cell_energy2[term.canonical_cell] +=
                reduced * energy * energy;
        }
    };
    long double replaced_mass = 0.0L;
    for (const LiftedMarker& replacement : lifted) {
        const std::size_t marker_index = replacement.marker;
        const WeightedPICMarker& source = marker_before[marker_index];
        const WeightedPICMarker& output = parcel.marker[marker_index];
        const long double packet_mass = output.mass;
        if (replacement.before.open) {
            throw std::runtime_error(
                "Canonical particle NTC could not lift a sampled source.");
        }
        long double withdrawn = 0.0L;
        long double withdrawn_energy = 0.0L;
        // Withdraw every source-owned piece once, in proportion to that
        // cell's current mean energy and second energy moment.
        for (const canonical_particle_ntc_detail::OwnershipTerm& term :
             parcel.ownership[marker_index]) {
            if (term.kind
                != canonical_particle_ntc_detail::OwnershipKind::canonical) {
                throw std::runtime_error(
                    "Canonical particle NTC received non-atlas ownership.");
            }
            if (!(cell_mass[term.index] > 0.0L)) {
                throw std::runtime_error(
                    "Canonical particle NTC withdrew an empty atlas cell.");
            }
            const long double physical_mass = term.physical_mass;
            const long double reduced = physical_mass / phase_factor;
            const long double mean_energy =
                cell_energy[term.index] / cell_mass[term.index];
            const long double mean_energy2 =
                cell_energy2[term.index] / cell_mass[term.index];
            cell_mass[term.index] -= reduced;
            cell_energy[term.index] -= reduced * mean_energy;
            cell_energy2[term.index] -= reduced * mean_energy2;
            withdrawn += physical_mass;
            withdrawn_energy += physical_mass * mean_energy;
        }
        const long double closure_tolerance = 8192.0L
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0L, packet_mass);
        if (std::abs(withdrawn - packet_mass) > closure_tolerance) {
            throw std::runtime_error(
                "Canonical particle NTC ownership mass failed closure.");
        }
        replaced_mass += packet_mass;
        const double source_energy = potential_at_radius(source.r)
            + 0.5 * source.w * source.w
            + source.eta / std::max(
                source.r * source.r, std::numeric_limits<double>::min());
        const double output_energy = potential_at_radius(output.r)
            + 0.5 * output.w * output.w
            + output.eta / std::max(
                output.r * output.r, std::numeric_limits<double>::min());
        // The continuous incoming parcel must carry the energy withdrawn
        // from its ownership. This check separates source sampling error
        // from any later destination-atlas energy quantization.
        const long double source_energy_mismatch =
            packet_mass * source_energy - withdrawn_energy;
        const long double energy_tolerance = 16384.0L
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0L, std::abs(withdrawn_energy));
        if (std::abs(source_energy_mismatch) > energy_tolerance) {
            throw std::runtime_error(
                "Canonical particle NTC source energy failed ownership closure.");
        }
        diagnostics.atlas_energy_quantization_error += static_cast<double>(
            packet_mass * output_energy - withdrawn_energy);
        // An outgoing orbit beyond the atlas becomes an explicit physical
        // carrier with a fresh ID. Representable mass is deposited directly.
        if (replacement.after.open) {
            WeightedPICMarker retained = output;
            retained.mass = static_cast<double>(packet_mass);
            retained.id = next_open_id++;
            retained.state = WeightedPICMarkerState::active;
            retained.source_cell = 0;
            retained.inner_boundary_speed = 0.0;
            retained.remaining_inner_time = 0.0;
            new_open.push_back(retained);
            canonical_particle_ntc_detail::classify_open_mass(
                diagnostics,
                replacement.after.open_reason,
                retained.mass);
        } else {
            deposit_lift(replacement.after, packet_mass);
        }
    }

    // Validate staged cell mass and H variance before installing atlas
    // moments. Remove only the declared roundoff-sized negative residues.
    const long double negative_tolerance =
        options.maximum_roundoff_negative_mass_fraction
        * std::max(1.0L, static_cast<long double>(atlas_mass_before));
    for (std::size_t cell = 0; cell < cell_mass.size(); ++cell) {
        if (cell_mass[cell] < -negative_tolerance) {
            throw std::runtime_error(
                "Canonical particle NTC produced negative atlas mass.");
        }
        if (cell_mass[cell] < 0.0L) {
            diagnostics.positivity_roundoff_mass += static_cast<double>(
                -phase_factor * cell_mass[cell]);
            cell_mass[cell] = 0.0L;
            cell_energy[cell] = 0.0L;
            cell_energy2[cell] = 0.0L;
        }
        if (cell_mass[cell] > 0.0L) {
            const long double variance_numerator =
                cell_energy2[cell] * cell_mass[cell]
                - cell_energy[cell] * cell_energy[cell];
            const long double tolerance = 4096.0L
                * std::numeric_limits<double>::epsilon()
                * std::max(1.0L, std::abs(cell_energy2[cell] * cell_mass[cell]));
            if (variance_numerator < -tolerance) {
                throw std::runtime_error(
                    "Canonical particle NTC produced negative H variance.");
            }
        }
    }
    for (WeightedPICMarker& marker : staged_open) {
        const double tolerance = 8192.0
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(marker.mass));
        if (marker.mass < -tolerance) {
            throw std::runtime_error(
                "Canonical particle NTC over-withdrew an open carrier.");
        }
        if (marker.mass < 0.0) {
            marker.mass = 0.0;
        }
    }
    staged_open.erase(
        std::remove_if(
            staged_open.begin(), staged_open.end(),
            [](const WeightedPICMarker& marker) {
                return !(marker.mass > 0.0);
            }),
        staged_open.end());
    staged_open.insert(
        staged_open.end(), new_open.begin(), new_open.end());
    long double atlas_mass_after = 0.0L;
    for (const long double mass : cell_mass) {
        atlas_mass_after += mass;
    }
    long double open_mass_after = 0.0L;
    for (const WeightedPICMarker& marker : staged_open) {
        open_mass_after += marker.mass;
    }
    diagnostics.physical_mass_after = static_cast<double>(
        phase_factor * atlas_mass_after + open_mass_after);
    diagnostics.physical_mass_residual =
        diagnostics.physical_mass_after - diagnostics.physical_mass_before;
    const double mass_scale = std::max(1.0, diagnostics.physical_mass_before);
    if (std::abs(diagnostics.physical_mass_residual)
        > 8192.0 * std::numeric_limits<double>::epsilon() * mass_scale) {
        throw std::runtime_error(
            "Canonical particle NTC changed total mass.");
    }
    // Explicit open mass retains conservation, while this separate budget
    // limits how much lies outside the atlas's intended energy/eta resolution.
    const double unrepresented_mass =
        diagnostics.unrepresented_eta_mass
        + diagnostics.unrepresented_missing_eta_row_mass
        + diagnostics.unrepresented_energy_mass
        + diagnostics.unrepresented_orbit_geometry_mass;
    if (unrepresented_mass
        > options.maximum_unrepresented_mass_fraction * mass_scale) {
        throw std::runtime_error(
            "Canonical particle NTC outputs exceed the atlas budget.");
    }

    // Commit the staged canonical moments after positivity, energy, and
    // total-mass checks. Open carriers and their ID sequence are installed too.
    for (CanonicalOrbitRow& row : atlas.row) {
        for (std::size_t q = 0; q < row.phase_mass.size(); ++q) {
            const std::size_t cell = row.first_cell + q;
            row.phase_mass[q] = static_cast<double>(cell_mass[cell]);
            row.phase_energy_mass[q] = static_cast<double>(cell_energy[cell]);
            row.phase_energy2_mass[q] = static_cast<double>(cell_energy2[cell]);
        }
    }
    state.open_marker = std::move(staged_open);
    state.next_open_marker_id = next_open_id;
    diagnostics.eulerian_to_atlas.source_mass = atlas_mass_before;
    diagnostics.eulerian_to_atlas.destination_mass =
        static_cast<double>(atlas_mass_after);
    diagnostics.eulerian_to_atlas.escaped_mass = static_cast<double>(
        (open_mass_after - open_mass_before) / phase_factor);
    diagnostics.eulerian_to_atlas.residual = atlas_mass_before
        - static_cast<double>(atlas_mass_after)
        - diagnostics.eulerian_to_atlas.escaped_mass;
    diagnostics.positivity_limited_pair_fraction = replaced_mass > 0.0L
        ? diagnostics.positivity_limited_pair_mass
            / static_cast<double>(replaced_mass)
        : 0.0;

    // Report density/energy-flux sampling noise on a bounded local scale
    // using the incoming parcel population, independently of collision errors.
    const double maximum_density = *std::max_element(
        deposited_before.density.begin(), deposited_before.density.end());
    double maximum_flux = 0.0;
    for (double value : deposited_before.radial_energy_flux_density) {
        maximum_flux = std::max(maximum_flux, std::abs(value));
    }
    for (std::size_t i = 0; i < noise_before.density_uncertainty.size(); ++i) {
        const double density_scale = std::max(
            std::abs(deposited_before.density[i]),
            0.01 * maximum_density);
        if (density_scale > 0.0) {
            diagnostics.maximum_relative_density_noise = std::max(
                diagnostics.maximum_relative_density_noise,
                noise_before.density_uncertainty[i] / density_scale);
        }
        const double flux_scale = std::max(
            std::abs(deposited_before.radial_energy_flux_density[i]),
            0.01 * maximum_flux);
        if (flux_scale > 0.0) {
            diagnostics.maximum_relative_energy_flux_noise = std::max(
                diagnostics.maximum_relative_energy_flux_noise,
                noise_before.radial_energy_flux_uncertainty[i] / flux_scale);
        }
    }
    return diagnostics;
}

/// Production ownership coupling for the canonical atlas.  The weighted NTC
/// sampler draws only physically realized pair events.  Each event withdraws
/// its finite packet from the source marker's disjoint atlas ownership and
/// lifts the two final continuous velocities directly into destination orbit
/// rows.  Representable products therefore never pass through a global
/// Eulerian-to-atlas remap, and no detached representable carrier is created.
template <class PotentialAtRadius>
inline CanonicalNTCCollisionDiagnostics apply_canonical_owned_ntc_collision(
    CanonicalOrbitAtlas& atlas,
    const PhaseSpaceCellGeometry& geometry,
    double dt,
    const Params& p,
    const CanonicalNTCCollisionOptions& options,
    CanonicalNTCCollisionState& state,
    PotentialAtRadius&& potential_at_radius) {
    // Select direct weighted source ownership for this entry: products
    // representable in the atlas are lifted there immediately by the
    // maintained canonical NTC implementation.
    CanonicalNTCCollisionOptions direct_options = options;
    direct_options.retain_collision_products_as_detached_carriers = false;
    return apply_canonical_ntc_collision(
        atlas,
        geometry,
        dt,
        p,
        direct_options,
        state,
        std::forward<PotentialAtRadius>(potential_at_radius));
}

}  // namespace ev
