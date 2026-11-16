#pragma once

// Science output and versioned restart of the spherical phase-grid state.
// Profiles retain their actual time and step; the phase projection is
//  g(r,w)=integral(L*f dL), with MPI angular slabs summed before publication.
// A grid restart contains the distribution, coordinate controls, optional tracer,
// inner-return carriers and Hamiltonian polynomial modes. Single-file and MPI
// formats declare their own versions and decomposition. Writers complete a
// staging file, preserve the previous restart and install the completed result.
// Readers restore only a compatible grid representation and physical clock.
// CSV/JSON columns and binary field order define the data formats used by analysis.

#include "ev/advection.hpp"
#include "ev/delayed_inner_orbit_carrier.hpp"
#include "ev/grids.hpp"
#include "ev/mpi.hpp"
#include "ev/phase_space_cells.hpp"

namespace ev {

/// Integrates L f over L to produce the two-dimensional g(r,w) phase projection.
// This projection integrates angular phase space only. Local density also
// requires radial velocity integration and its spherical geometric factor.
std::vector<double> phase_projection(
    const std::vector<double>& f,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    const Params& p) {
    std::vector<double> g(static_cast<std::size_t>(p.Nr) * p.Nw, 0.0);
    parallel_for_chunks(0, p.Nr, effective_threads(p, p.Nr),
        [&](int ilo, int ihi, int) {
            for (int i = ilo; i < ihi; ++i) {
                for (int j = 0; j < p.Nw; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < p.NL; ++k) {
                        sum += L[k] * f[idx(i, j, k, p)] * Lq[k];
                    }
                    g[static_cast<std::size_t>(i) * p.Nw + static_cast<std::size_t>(j)] = sum;
                }
            }
        });
    return g;
}

/// Sums local phase projections over MPI ranks to recover the global g(r,w).
std::vector<double> phase_projection_global(
    const std::vector<double>& f,
    const std::vector<double>& L,
    const std::vector<double>& Lq,
    const Params& local_p,
    const MpiContext& mpi) {
    std::vector<double> g = phase_projection(f, L, Lq, local_p);
    mpi_allreduce_sum_vector(g, mpi);
    return g;
}

/// Writes one full g(r,w) phase-space projection as a CSV table.
void write_phase_space(
    const std::filesystem::path& path,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& g,
    const Params& p) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open phase-space output file: " + path.string());
    }
    out << std::setprecision(17);
    out << "r,w,g\n";
    for (int i = 0; i < p.Nr; ++i) {
        for (int j = 0; j < p.Nw; ++j) {
            out << r[i] << ',' << w[j] << ','
                << g[static_cast<std::size_t>(i) * p.Nw + static_cast<std::size_t>(j)] << '\n';
        }
    }
    out.close();
    if (!out) {
        throw std::runtime_error("Failed to close phase-space output file after writing: " + path.string());
    }
}

/// Appends selected strided g(r,w) snapshot rows to the time-snapshot CSV stream.
void write_phase_snapshot_rows(
    std::ofstream& out,
    double t,
    int step,
    const std::vector<double>& r,
    const std::vector<double>& w,
    const std::vector<double>& g,
    const Params& p) {
    out << std::setprecision(17);
    for (int i = 0; i < p.Nr; i += p.phase_r_stride) {
        for (int j = 0; j < p.Nw; j += p.phase_w_stride) {
            out << t << ',' << step << ',' << i << ',' << j << ','
                << r[i] << ',' << w[j] << ','
                << g[static_cast<std::size_t>(i) * p.Nw + static_cast<std::size_t>(j)] << '\n';
        }
    }
}

/// Appends radial matter and metric profiles for one saved time.
void write_profile_rows(
    std::ofstream& out,
    double t,
    int step,
    const std::vector<double>& r,
    const Moments& moments,
    const Metric& metric,
    const Params& p) {
    out << std::setprecision(17);
    for (int i = 0; i < p.Nr; ++i) {
        out << t << ',' << step << ',' << r[i] << ','
            << moments.rho[i] << ',' << moments.p[i] << ',' << moments.pt[i] << ','
            << moments.j[i] << ','
            << metric.m[i] << ',' << metric.mu[i] << ',' << metric.lambda[i] << ','
            << metric.gap[i] << '\n';
    }
}

/// Appends radial moment profiles without metric columns, useful for passive components/tracers.
void write_moment_profile_rows(
    std::ofstream& out,
    double t,
    int step,
    const std::vector<double>& r,
    const Moments& moments,
    const Params& p) {
    out << std::setprecision(17);
    for (int i = 0; i < p.Nr; ++i) {
        out << t << ',' << step << ',' << r[i] << ','
            << moments.rho[i] << ',' << moments.p[i] << ',' << moments.pt[i] << ','
            << moments.j[i] << '\n';
    }
}

/// Writes one trivially copyable value to a binary output stream.
template <typename T>
void write_binary(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("Failed while writing restart file.");
    }
}

/// Reads one trivially copyable value from a binary restart stream.
template <typename T>
T read_binary(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Unexpected end of restart file.");
    }
    return value;
}

/// Writes one restart component vector with its explicit element count.
void write_restart_vector(std::ofstream& out, const std::vector<double>& values, const char* label) {
    write_binary(out, static_cast<std::uint64_t>(values.size()));
    out.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!out) {
        throw std::runtime_error(std::string("Failed while writing restart ") + label + " array.");
    }
}

/// Reads one restart component vector and verifies its element count.
std::vector<double> read_restart_vector(
    std::ifstream& in,
    std::uint64_t expected_values,
    const char* label) {
    const std::uint64_t n_values = read_binary<std::uint64_t>(in);
    if (n_values != expected_values) {
        throw std::runtime_error(std::string("Restart ") + label + " array has wrong size.");
    }
    std::vector<double> values(static_cast<std::size_t>(n_values));
    in.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!in) {
        throw std::runtime_error(std::string("Unexpected end of restart ") + label + " array.");
    }
    return values;
}

/// Reads a local L slab from a full single-rank restart component.
// Seek to the owned angular subarray in every (r,w) row. After reading the
// local slab, restore the stream position to the end of the global component
// so the next payload starts at its defined offset.
std::vector<double> read_restart_component_local_slab(
    std::ifstream& in,
    const Params& global_p,
    const LDecomposition& decomp,
    const char* label) {
    const std::uint64_t n_values = read_binary<std::uint64_t>(in);
    const std::uint64_t expected_values =
        static_cast<std::uint64_t>(global_p.Nr) * global_p.Nw * decomp.global_NL;
    if (n_values != expected_values) {
        throw std::runtime_error(std::string("Restart ") + label + " array has wrong size.");
    }
    const std::streamoff component0 = in.tellg();
    std::vector<double> values(
        static_cast<std::size_t>(global_p.Nr) * global_p.Nw * decomp.count,
        0.0);
    if (decomp.count > 0) {
        for (int i = 0; i < global_p.Nr; ++i) {
            for (int j = 0; j < global_p.Nw; ++j) {
                const std::uint64_t global_base =
                    (static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(global_p.Nw)
                     + static_cast<std::uint64_t>(j))
                        * static_cast<std::uint64_t>(decomp.global_NL)
                    + static_cast<std::uint64_t>(decomp.offset);
                const std::streamoff offset =
                    component0 + static_cast<std::streamoff>(
                        global_base * static_cast<std::uint64_t>(sizeof(double)));
                in.seekg(offset);
                if (!in) {
                    throw std::runtime_error(
                        std::string("Failed to seek local restart ") + label + " slab.");
                }
                double* dest = values.data()
                    + (static_cast<std::size_t>(i) * global_p.Nw + static_cast<std::size_t>(j))
                        * static_cast<std::size_t>(decomp.count);
                in.read(
                    reinterpret_cast<char*>(dest),
                    static_cast<std::streamsize>(decomp.count * sizeof(double)));
                if (!in) {
                    throw std::runtime_error(
                        std::string("Unexpected end of local restart ") + label + " slab.");
                }
            }
        }
    }
    const std::streamoff component_end =
        component0 + static_cast<std::streamoff>(n_values * static_cast<std::uint64_t>(sizeof(double)));
    in.seekg(component_end);
    if (!in) {
        throw std::runtime_error(std::string("Failed to seek past restart ") + label + " array.");
    }
    return values;
}

// The five P2 coefficients carry resolved within-cell structure beyond the
// cell average. Keeping them on restart preserves the next Hamiltonian remap.
inline void validate_restart_modal_coefficients(
    const std::vector<std::array<double, 5>>& modes,
    std::size_t expected_cells,
    const char* label) {
    if (modes.size() != expected_cells) {
        throw std::runtime_error(
            std::string(label) + " restart P2-mode array has the wrong size.");
    }
    for (const auto& cell : modes) {
        for (double value : cell) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    std::string(label) + " restart P2 mode is non-finite.");
            }
        }
    }
}

inline void write_restart_modal_coefficients(
    std::ofstream& out,
    const std::vector<std::array<double, 5>>& modes,
    const char* label) {
    write_binary(out, static_cast<std::uint64_t>(modes.size()));
    for (const auto& cell : modes) {
        out.write(
            reinterpret_cast<const char*>(cell.data()),
            static_cast<std::streamsize>(cell.size() * sizeof(double)));
    }
    if (!out) {
        throw std::runtime_error(
            std::string("Failed while writing restart ") + label
            + " P2 modes.");
    }
}

inline std::vector<std::array<double, 5>> read_restart_modal_coefficients(
    std::ifstream& in,
    std::uint64_t expected_cells,
    const char* label) {
    const std::uint64_t cells = read_binary<std::uint64_t>(in);
    if (cells != expected_cells) {
        throw std::runtime_error(
            std::string("Restart ") + label + " P2-mode array has wrong size.");
    }
    std::vector<std::array<double, 5>> modes(
        static_cast<std::size_t>(cells));
    for (auto& cell : modes) {
        in.read(
            reinterpret_cast<char*>(cell.data()),
            static_cast<std::streamsize>(cell.size() * sizeof(double)));
    }
    if (!in) {
        throw std::runtime_error(
            std::string("Unexpected end of restart ") + label
            + " P2 modes.");
    }
    validate_restart_modal_coefficients(
        modes, static_cast<std::size_t>(expected_cells), label);
    return modes;
}

inline std::vector<std::array<double, 5>>
read_restart_modal_coefficients_local_slab(
    std::ifstream& in,
    const Params& global_p,
    const LDecomposition& decomp,
    const char* label) {
    const std::uint64_t cells = read_binary<std::uint64_t>(in);
    const std::uint64_t expected_cells =
        static_cast<std::uint64_t>(global_p.Nr) * global_p.Nw
        * decomp.global_NL;
    if (cells != expected_cells) {
        throw std::runtime_error(
            std::string("Restart ") + label + " P2-mode array has wrong size.");
    }
    const std::streamoff component0 = in.tellg();
    std::vector<std::array<double, 5>> modes(
        static_cast<std::size_t>(global_p.Nr) * global_p.Nw
        * decomp.count);
    for (int i = 0; i < global_p.Nr; ++i) {
        for (int j = 0; j < global_p.Nw; ++j) {
            const std::uint64_t global_cell =
                (static_cast<std::uint64_t>(i) * global_p.Nw + j)
                    * decomp.global_NL
                + decomp.offset;
            in.seekg(
                component0 + static_cast<std::streamoff>(
                    global_cell * 5U * sizeof(double)));
            auto* destination = modes.data()
                + (static_cast<std::size_t>(i) * global_p.Nw + j)
                    * decomp.count;
            in.read(
                reinterpret_cast<char*>(destination),
                static_cast<std::streamsize>(
                    decomp.count * sizeof(std::array<double, 5>)));
        }
    }
    in.seekg(
        component0 + static_cast<std::streamoff>(
            cells * 5U * sizeof(double)));
    if (!in) {
        throw std::runtime_error(
            std::string("Unexpected end of restart ") + label
            + " P2-mode slab.");
    }
    validate_restart_modal_coefficients(modes, modes.size(), label);
    return modes;
}

/// Formats a restart time for filenames without using characters that invite shell/path trouble.
std::string restart_time_tag(double t) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9) << t;
    std::string tag = out.str();
    for (char& ch : tag) {
        if (ch == '-') {
            ch = 'm';
        } else if (ch == '.') {
            ch = 'p';
        }
    }
    return tag;
}

/// Builds a unique completed-restart backup path in the restart directory.
std::filesystem::path unique_restart_backup_path(
    const std::filesystem::path& path,
    int step,
    double t) {
    std::ostringstream base;
    base << "restart_backup_before_step"
         << std::setw(9) << std::setfill('0') << step
         << "_t" << restart_time_tag(t)
         << "_" << path.filename().string();
    std::filesystem::path candidate = path.parent_path() / base.str();
    int duplicate = 1;
    while (std::filesystem::exists(candidate)) {
        std::ostringstream name;
        name << base.str()
             << "_dup" << std::setw(3) << std::setfill('0') << duplicate++;
        candidate = path.parent_path() / name.str();
    }
    return candidate;
}

/// Copies a regular file through iostreams and verifies the byte count.
void copy_regular_file_binary_checked(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
    std::ifstream in(source, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open restart source for copy: " + source.string());
    }
    std::ofstream out(destination, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Could not open restart destination for copy: " + destination.string());
    }
    std::vector<char> buffer(1 << 20);
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = in.gcount();
        if (got > 0) {
            out.write(buffer.data(), got);
            if (!out) {
                throw std::runtime_error("Failed while writing restart copy: " + destination.string());
            }
        }
    }
    if (!in.eof()) {
        throw std::runtime_error("Failed while reading restart copy: " + source.string());
    }
    out.close();
    if (!out) {
        throw std::runtime_error("Failed to close restart copy: " + destination.string());
    }
    std::error_code size_error;
    const std::uintmax_t source_size = std::filesystem::file_size(source, size_error);
    if (size_error) {
        throw std::runtime_error("Could not stat restart source after copy: " + source.string());
    }
    const std::uintmax_t dest_size = std::filesystem::file_size(destination, size_error);
    if (size_error || dest_size != source_size) {
        throw std::runtime_error("Restart copy size mismatch: " + destination.string());
    }
}

/// Moves a completed staging file into place, falling back to a checked copy.
// Use a filesystem rename when supported. The fallback copies a completed
// file with checked length before removing the staging copy.
void install_staged_file_checked(
    const std::filesystem::path& staging_path,
    const std::filesystem::path& final_path) {
    std::error_code rename_error;
    std::filesystem::rename(staging_path, final_path, rename_error);
    if (!rename_error) {
        return;
    }
    copy_regular_file_binary_checked(staging_path, final_path);
    std::filesystem::remove(staging_path);
}

/// Preserves the previous completed restart before installing a newer completed restart.
std::filesystem::path backup_existing_restart_file(
    const std::filesystem::path& path,
    const std::filesystem::path& staging_dir,
    int step,
    double t) {
    if (!std::filesystem::exists(path)) {
        return {};
    }
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("Refusing to replace non-file restart artifact: " + path.string());
    }
    const std::filesystem::path backup = unique_restart_backup_path(path, step, t);
    std::filesystem::create_directories(staging_dir);
    const std::filesystem::path staged_backup = staging_dir / (backup.filename().string() + ".tmp");
    try {
        copy_regular_file_binary_checked(path, staged_backup);
        install_staged_file_checked(staged_backup, backup);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(staged_backup, ignored);
        std::filesystem::remove(backup, ignored);
        throw;
    }
    return backup;
}

/// Returns the staging path used to write a restart outside the protected restart directory.
std::filesystem::path restart_staging_path(
    const std::filesystem::path& final_path,
    const std::filesystem::path& staging_dir) {
    if (staging_dir.empty()) {
        throw std::runtime_error("Internal error: restart staging directory is empty.");
    }
    std::filesystem::create_directories(staging_dir);
    return staging_dir / (final_path.filename().string() + ".tmp");
}

/// Installs a completed restart after backing up the previous completed restart.
// Back up the last completed checkpoint before replacement. If cross-device
// installation fails, restore that saved checkpoint and propagate the I/O error.
void install_completed_restart_file(
    const std::filesystem::path& tmp,
    const std::filesystem::path& final_path,
    int step,
    double t) {
    std::filesystem::create_directories(final_path.parent_path());
    const std::filesystem::path backup =
        backup_existing_restart_file(final_path, std::filesystem::absolute(tmp).parent_path(), step, t);
    std::error_code rename_error;
    std::filesystem::rename(tmp, final_path, rename_error);
    if (!rename_error) {
        return;
    }
    std::error_code remove_error;
    std::filesystem::remove(final_path, remove_error);
    try {
        install_staged_file_checked(tmp, final_path);
    } catch (...) {
        if (!backup.empty() && std::filesystem::exists(backup)) {
            std::error_code ignored;
            std::filesystem::remove(final_path, ignored);
            copy_regular_file_binary_checked(backup, final_path);
        }
        throw;
    }
}

/// Compares floating-point restart metadata with a relative tolerance.
bool close_enough(double a, double b, double rel = 1.0e-12) {
    return std::isfinite(a) && std::isfinite(b) && std::isfinite(rel)
        && rel >= 0.0
        && std::abs(a - b)
            <= rel * std::max({1.0, std::abs(a), std::abs(b)});
}

/// Throws a clear error when restart metadata does not match requested parameters.
void validate_restart_double(const char* name, double got, double expected) {
    if (!close_enough(got, expected)) {
        throw std::runtime_error(std::string("Restart grid mismatch for ") + name);
    }
}

inline void validate_restart_distribution(
    const std::vector<double>& values,
    const Params& p,
    const char* label) {
    const std::size_t expected =
        static_cast<std::size_t>(p.Nr) * p.Nw * p.NL;
    validate_finite_nonnegative_distribution(values, expected, label);
}

/// Rejects corrupt or nonsensical restart clocks before they reach filenames,
/// loop bounds, or a narrowing conversion to the solver's int step counter.
inline void validate_restart_clock(double time, std::int64_t step) {
    if (!(time >= 0.0) || !std::isfinite(time)) {
        throw std::runtime_error("Restart time must be finite and nonnegative.");
    }
    if (step < 0
        || step > static_cast<std::int64_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("Restart step is outside int range.");
    }
}

inline void write_restart_inner_carrier(
    std::ofstream& out,
    const DelayedInnerCarrier& carrier,
    const Params& p,
    const char* label) {
    if (carrier.packets.size()
        > static_cast<std::size_t>(p.vp_inner_carrier_max_packets)) {
        throw std::runtime_error(
            std::string(label) + " exceeds vp_inner_carrier_max_packets.");
    }
    write_binary(
        out, static_cast<std::uint64_t>(carrier.packets.size()));
    for (const DelayedInnerCarrierPacket& packet : carrier.packets) {
        validate_delayed_inner_carrier_packet(packet, p.NL, label);
        write_binary(out, packet.reduced_mass);
        write_binary(out, static_cast<std::int32_t>(packet.eta_index));
        write_binary(out, packet.eta);
        write_binary(out, packet.boundary_speed);
        write_binary(out, packet.remaining_time);
    }
}

// An unresolved inner orbit retains its reduced mass, eta and remaining
// travel time. Older formats recover eta from their own declared angular cell
// centres; the newer format carries the particle value explicitly.
inline DelayedInnerCarrier read_restart_inner_carrier(
    std::ifstream& in,
    const Params& p,
    const char* label,
    bool packet_has_eta,
    const std::vector<double>* legacy_eta_center = nullptr) {
    const std::uint64_t count = read_binary<std::uint64_t>(in);
    if (count > static_cast<std::uint64_t>(
            p.vp_inner_carrier_max_packets)) {
        throw std::runtime_error(
            std::string(label) + " packet count exceeds the configured limit.");
    }
    DelayedInnerCarrier carrier;
    carrier.packets.reserve(static_cast<std::size_t>(count));
    std::vector<double> constructed_eta_center;
    if (!packet_has_eta && legacy_eta_center == nullptr) {
        constructed_eta_center = make_eta_axis(p).center;
        legacy_eta_center = &constructed_eta_center;
    }
    if (!packet_has_eta
        && (legacy_eta_center == nullptr
            || legacy_eta_center->size() != static_cast<std::size_t>(p.NL))) {
        throw std::runtime_error(
            std::string(label) + " lacks compatible legacy eta centres.");
    }
    for (std::uint64_t q = 0; q < count; ++q) {
        DelayedInnerCarrierPacket packet;
        packet.reduced_mass = read_binary<double>(in);
        packet.eta_index = read_binary<std::int32_t>(in);
        if (packet.eta_index < 0 || packet.eta_index >= p.NL) {
            throw std::runtime_error(
                std::string(label) + " contains an invalid eta index.");
        }
        packet.eta = packet_has_eta
            ? read_binary<double>(in)
            : (*legacy_eta_center)[static_cast<std::size_t>(packet.eta_index)];
        packet.boundary_speed = read_binary<double>(in);
        packet.remaining_time = read_binary<double>(in);
        validate_delayed_inner_carrier_packet(packet, p.NL, label);
        carrier.packets.push_back(packet);
    }
    return carrier;
}

/// Writes a single-rank binary restart containing metadata and the full f array.
// Validate all optional state components before writing the format header.
// The time/step and every flag describe one simultaneous accepted physical state.
void write_restart_state(
    const std::filesystem::path& path,
    const std::filesystem::path& staging_dir,
    const Params& p,
    double t,
    int step,
    const std::vector<double>& f,
    const std::vector<double>* tracer_f = nullptr,
    const DelayedInnerCarrier* carrier = nullptr,
    const DelayedInnerCarrier* tracer_carrier = nullptr,
    const std::vector<std::array<double, 5>>* modal_coefficients = nullptr,
    const std::vector<std::array<double, 5>>* tracer_modal_coefficients = nullptr) {
    const bool has_tracer = tracer_f != nullptr && !tracer_f->empty();
    const bool has_carrier = carrier != nullptr;
    const bool has_modal_coefficients =
        modal_coefficients != nullptr && !modal_coefficients->empty();
    const bool has_tracer_modal_coefficients =
        tracer_modal_coefficients != nullptr
        && !tracer_modal_coefficients->empty();
    validate_restart_clock(t, step);
    if (using_delayed_inner_return(p) != has_carrier) {
        throw std::runtime_error(
            "Delayed-return restart writes require exactly one persistent carrier payload.");
    }
    if (tracer_carrier != nullptr && !has_tracer) {
        throw std::runtime_error(
            "A tracer carrier cannot be written without a tracer distribution.");
    }
    if (has_tracer && tracer_f->size() != f.size()) {
        throw std::runtime_error("Tracer restart array size does not match total state array size.");
    }
    if (has_tracer_modal_coefficients && !has_tracer) {
        throw std::runtime_error(
            "Tracer P2 modes cannot be written without a tracer distribution.");
    }
    validate_restart_distribution(f, p, "state");
    if (has_tracer) {
        validate_restart_distribution(*tracer_f, p, "tracer");
    }
    if (has_modal_coefficients) {
        validate_restart_modal_coefficients(
            *modal_coefficients, f.size(), "state");
    }
    if (has_tracer_modal_coefficients) {
        validate_restart_modal_coefficients(
            *tracer_modal_coefficients, f.size(), "tracer");
    }
    const std::filesystem::path tmp = restart_staging_path(path, staging_dir);
    std::ofstream out(tmp, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open restart file for writing: " + tmp.string());
    }
    const char magic[16] = {'E', 'V', 'L', 'O', 'G', 'R', 'S', 'T', '1', '\0'};
    out.write(magic, sizeof(magic));
    // Version 9 stores the full angular value in each delayed carrier alongside
    // the grid representation and optional within-cell modes.
    const std::int32_t version = 9;
    write_binary(out, version);
    write_binary(out, static_cast<std::int32_t>(p.Nr));
    write_binary(out, static_cast<std::int32_t>(p.Nw));
    write_binary(out, static_cast<std::int32_t>(p.NL));
    write_binary(out, static_cast<std::int32_t>(radial_grid_code(p.radial_grid)));
    write_binary(out, static_cast<std::int32_t>(kinetic_grid_code(p.kinetic_grid)));
    write_binary(
        out,
        static_cast<std::int32_t>(kinetic_eta_grid_code(p.kinetic_eta_grid)));
    write_binary(out, p.kinetic_eta_power);
    if (p.kinetic_eta_grid == "composite_power") {
        write_binary(out, p.kinetic_eta_join);
        write_binary(
            out, static_cast<std::int32_t>(p.kinetic_eta_inner_cells));
    }
    write_binary(out, p.r_min);
    write_binary(out, p.r_max);
    write_binary(out, p.inner_refine_radius);
    write_binary(out, static_cast<std::int32_t>(p.inner_refine_cells));
    write_binary(out, p.w_max);
    write_binary(out, p.L_max);
    write_binary(out, p.w_stretch);
    write_binary(out, p.w_center);
    write_binary(out, p.central_mass);
    write_binary(out, t);
    write_binary(out, static_cast<std::int64_t>(step));
    write_binary(out, static_cast<std::int32_t>(has_tracer ? 1 : 0));
    write_binary(out, static_cast<std::int32_t>(has_carrier ? 1 : 0));
    write_binary(
        out, static_cast<std::int32_t>(has_modal_coefficients ? 1 : 0));
    write_binary(
        out,
        static_cast<std::int32_t>(has_tracer_modal_coefficients ? 1 : 0));
    write_restart_vector(out, f, "state");
    if (has_tracer) {
        write_restart_vector(out, *tracer_f, "tracer");
    }
    if (has_carrier) {
        write_restart_inner_carrier(out, *carrier, p, "state carrier");
        if (has_tracer) {
            const DelayedInnerCarrier empty;
            write_restart_inner_carrier(
                out,
                tracer_carrier != nullptr ? *tracer_carrier : empty,
                p,
                "tracer carrier");
        }
    }
    if (has_modal_coefficients) {
        write_restart_modal_coefficients(
            out, *modal_coefficients, "state");
    }
    if (has_tracer_modal_coefficients) {
        write_restart_modal_coefficients(
            out, *tracer_modal_coefficients, "tracer");
    }
    out.close();
    if (!out) {
        throw std::runtime_error("Failed to close restart file after writing: " + tmp.string());
    }
    install_completed_restart_file(tmp, path, step, t);
}

/// Reads a single-rank binary restart containing metadata and the full f array.
// Consume the supported version in its declared field order. Freshly clear
// optional output buffers, then populate only the components present in this file.
[[maybe_unused]] std::vector<double> read_restart_state(
    const std::filesystem::path& path,
    const Params& p,
    double& t,
    int& step,
    std::vector<double>* tracer_f = nullptr,
    bool* has_tracer = nullptr,
    DelayedInnerCarrier* carrier = nullptr,
    DelayedInnerCarrier* tracer_carrier = nullptr,
    std::vector<std::array<double, 5>>* modal_coefficients = nullptr,
    bool* has_modal_coefficients = nullptr,
    std::vector<std::array<double, 5>>* tracer_modal_coefficients = nullptr,
    bool* has_tracer_modal_coefficients = nullptr) {
    if (tracer_f != nullptr) {
        tracer_f->clear();
    }
    if (has_tracer != nullptr) {
        *has_tracer = false;
    }
    if (carrier != nullptr) {
        carrier->packets.clear();
    }
    if (tracer_carrier != nullptr) {
        tracer_carrier->packets.clear();
    }
    if (modal_coefficients != nullptr) {
        modal_coefficients->clear();
    }
    if (tracer_modal_coefficients != nullptr) {
        tracer_modal_coefficients->clear();
    }
    if (has_modal_coefficients != nullptr) {
        *has_modal_coefficients = false;
    }
    if (has_tracer_modal_coefficients != nullptr) {
        *has_tracer_modal_coefficients = false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open restart file: " + path.string());
    }

    char magic[16] = {};
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVLOGRST1")) {
        throw std::runtime_error("Restart file has wrong magic header.");
    }
    const std::int32_t version = read_binary<std::int32_t>(in);
    if (version < 1 || version > 9) {
        throw std::runtime_error("Unsupported restart file version.");
    }
    const std::int32_t nr = read_binary<std::int32_t>(in);
    const std::int32_t nw = read_binary<std::int32_t>(in);
    const std::int32_t nl = read_binary<std::int32_t>(in);
    if (nr != p.Nr || nw != p.Nw || nl != p.NL) {
        throw std::runtime_error("Restart file dimensions do not match requested grid.");
    }
    const std::int32_t saved_radial_grid_code = (version >= 2) ? read_binary<std::int32_t>(in) : 0;
    const std::string saved_radial_grid = radial_grid_name(saved_radial_grid_code);
    if (saved_radial_grid != p.radial_grid) {
        throw std::runtime_error("Restart radial grid does not match requested radial grid.");
    }
    if (version >= 5) {
        const std::string saved_kinetic_grid =
            kinetic_grid_name(read_binary<std::int32_t>(in));
        if (saved_kinetic_grid != p.kinetic_grid) {
            throw std::runtime_error(
                "Restart kinetic-grid representation does not match requested kinetic_grid.");
        }
    } else if (using_cell_average_eta_grid(p)) {
        throw std::runtime_error(
            "Legacy restart lacks kinetic-grid metadata and cannot be interpreted as "
            "cell_average_eta.");
    }
    if (version >= 6) {
        const std::string saved_eta_grid =
            kinetic_eta_grid_name(read_binary<std::int32_t>(in));
        if (saved_eta_grid != p.kinetic_eta_grid) {
            throw std::runtime_error(
                "Restart kinetic eta grid does not match requested kinetic_eta_grid.");
        }
        validate_restart_double(
            "kinetic_eta_power", read_binary<double>(in), p.kinetic_eta_power);
        if (saved_eta_grid == "composite_power") {
            validate_restart_double(
                "kinetic_eta_join",
                read_binary<double>(in),
                p.kinetic_eta_join);
            const int saved_inner_cells = read_binary<std::int32_t>(in);
            if (saved_inner_cells != p.kinetic_eta_inner_cells) {
                throw std::runtime_error(
                    "Restart kinetic_eta_inner_cells does not match the requested grid.");
            }
        }
    } else if (using_cell_average_eta_grid(p)) {
        throw std::runtime_error(
            "Pre-v6 cell-average restart used obsolete radial/eta cell geometry "
            "and cannot be interpreted by the positive-domain solver.");
    }
    validate_restart_double("r_min", read_binary<double>(in), p.r_min);
    validate_restart_double("r_max", read_binary<double>(in), p.r_max);
    if (version >= 3) {
        validate_restart_double("inner_refine_radius", read_binary<double>(in), p.inner_refine_radius);
        const std::int32_t saved_inner_cells = read_binary<std::int32_t>(in);
        if (saved_inner_cells != p.inner_refine_cells) {
            throw std::runtime_error("Restart grid mismatch for inner_refine_cells");
        }
    }
    validate_restart_double("w_max", read_binary<double>(in), p.w_max);
    validate_restart_double("L_max", read_binary<double>(in), p.L_max);
    validate_restart_double("w_stretch", read_binary<double>(in), p.w_stretch);
    validate_restart_double("w_center", read_binary<double>(in), p.w_center);
    validate_restart_double("central_mass", read_binary<double>(in), p.central_mass);
    t = read_binary<double>(in);
    const std::int64_t saved_step = read_binary<std::int64_t>(in);
    validate_restart_clock(t, saved_step);
    step = static_cast<int>(saved_step);
    const std::uint64_t expected_values = static_cast<std::uint64_t>(p.Nr) * p.Nw * p.NL;
    const bool saved_has_tracer = version >= 4 ? (read_binary<std::int32_t>(in) != 0) : false;
    const bool saved_has_carrier =
        version >= 7 ? (read_binary<std::int32_t>(in) != 0) : false;
    const bool saved_has_modal_coefficients =
        version >= 8 ? (read_binary<std::int32_t>(in) != 0) : false;
    const bool saved_has_tracer_modal_coefficients =
        version >= 8 ? (read_binary<std::int32_t>(in) != 0) : false;
    if (saved_has_carrier != using_delayed_inner_return(p)) {
        throw std::runtime_error(
            "Restart delayed-return carrier state does not match inner_boundary.");
    }
    if (has_tracer != nullptr) {
        *has_tracer = saved_has_tracer;
    }
    if (has_modal_coefficients != nullptr) {
        *has_modal_coefficients = saved_has_modal_coefficients;
    }
    if (has_tracer_modal_coefficients != nullptr) {
        *has_tracer_modal_coefficients =
            saved_has_tracer_modal_coefficients;
    }
    std::vector<double> f = read_restart_vector(in, expected_values, "state");
    validate_restart_distribution(f, p, "state");
    if (saved_has_tracer) {
        std::vector<double> loaded_tracer = read_restart_vector(in, expected_values, "tracer");
        validate_restart_distribution(loaded_tracer, p, "tracer");
        if (tracer_f != nullptr) {
            *tracer_f = std::move(loaded_tracer);
        }
    }
    if (saved_has_carrier) {
        DelayedInnerCarrier loaded =
            read_restart_inner_carrier(
                in, p, "state carrier", version >= 9);
        if (carrier != nullptr) {
            *carrier = std::move(loaded);
        }
        if (saved_has_tracer) {
            DelayedInnerCarrier loaded_tracer =
                read_restart_inner_carrier(
                    in, p, "tracer carrier", version >= 9);
            if (tracer_carrier != nullptr) {
                *tracer_carrier = std::move(loaded_tracer);
            }
        }
    }
    if (saved_has_modal_coefficients) {
        std::vector<std::array<double, 5>> loaded =
            read_restart_modal_coefficients(in, expected_values, "state");
        if (modal_coefficients != nullptr) {
            *modal_coefficients = std::move(loaded);
        }
    }
    if (saved_has_tracer_modal_coefficients) {
        if (!saved_has_tracer) {
            throw std::runtime_error(
                "Restart contains tracer P2 modes without a tracer distribution.");
        }
        std::vector<std::array<double, 5>> loaded =
            read_restart_modal_coefficients(in, expected_values, "tracer");
        if (tracer_modal_coefficients != nullptr) {
            *tracer_modal_coefficients = std::move(loaded);
        }
    }
    return f;
}

/// Builds the filename for one MPI rank-local restart payload.
std::filesystem::path rank_restart_path(const std::filesystem::path& manifest_path, int rank) {
    std::ostringstream suffix;
    suffix << ".rank" << std::setw(6) << std::setfill('0') << rank;
    return std::filesystem::path(manifest_path.string() + suffix.str());
}

/// Writes either a single-file restart or an MPI manifest plus rank-local payloads.
// Complete each rank-local checkpoint first. A barrier then permits the
// root to publish the common manifest, whose clock and population flags must
// match every rank payload.
void write_restart_state_distributed(
    const std::filesystem::path& path,
    const std::filesystem::path& staging_dir,
    const Params& global_p,
    const Params& local_p,
    const LDecomposition& decomp,
    const MpiContext& mpi,
    double t,
    int step,
    const std::vector<double>& f,
    const std::vector<double>* tracer_f = nullptr,
    const DelayedInnerCarrier* carrier = nullptr,
    const DelayedInnerCarrier* tracer_carrier = nullptr,
    const std::vector<std::array<double, 5>>* modal_coefficients = nullptr,
    const std::vector<std::array<double, 5>>* tracer_modal_coefficients = nullptr) {
    const bool has_tracer = tracer_f != nullptr && !tracer_f->empty();
    const bool has_carrier = carrier != nullptr;
    const bool has_modal_coefficients =
        modal_coefficients != nullptr && !modal_coefficients->empty();
    const bool has_tracer_modal_coefficients =
        tracer_modal_coefficients != nullptr
        && !tracer_modal_coefficients->empty();
    validate_restart_clock(t, step);
    if (using_delayed_inner_return(global_p) != has_carrier) {
        throw std::runtime_error(
            "Distributed delayed-return restart requires its carrier payload.");
    }
    if (has_tracer && tracer_f->size() != f.size()) {
        throw std::runtime_error("Tracer restart array size does not match total state array size.");
    }
    if (has_tracer_modal_coefficients && !has_tracer) {
        throw std::runtime_error(
            "Tracer P2 modes cannot be written without a tracer distribution.");
    }
    validate_restart_distribution(f, local_p, "state");
    if (has_tracer) {
        validate_restart_distribution(*tracer_f, local_p, "tracer");
    }
    if (has_modal_coefficients) {
        validate_restart_modal_coefficients(
            *modal_coefficients, f.size(), "state");
    }
    if (has_tracer_modal_coefficients) {
        validate_restart_modal_coefficients(
            *tracer_modal_coefficients, f.size(), "tracer");
    }
    if (mpi.size == 1) {
        write_restart_state(
            path,
            staging_dir,
            global_p,
            t,
            step,
            f,
            tracer_f,
            carrier,
            tracer_carrier,
            modal_coefficients,
            tracer_modal_coefficients);
        return;
    }

    const std::filesystem::path rank_path = rank_restart_path(path, mpi.rank);
    const std::filesystem::path rank_tmp = restart_staging_path(rank_path, staging_dir);
    {
        std::ofstream out(rank_tmp, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not open rank restart file for writing: " + rank_tmp.string());
        }
        const char rank_magic[16] = {'E', 'V', 'M', 'P', 'I', 'R', 'N', 'K', '1', '\0'};
        out.write(rank_magic, sizeof(rank_magic));
        const std::int32_t version = 7;
        write_binary(out, version);
        write_binary(out, static_cast<std::int32_t>(global_p.Nr));
        write_binary(out, static_cast<std::int32_t>(global_p.Nw));
        write_binary(out, static_cast<std::int32_t>(decomp.global_NL));
        write_binary(out, static_cast<std::int32_t>(radial_grid_code(global_p.radial_grid)));
        write_binary(
            out,
            static_cast<std::int32_t>(kinetic_grid_code(global_p.kinetic_grid)));
        write_binary(
            out,
            static_cast<std::int32_t>(
                kinetic_eta_grid_code(global_p.kinetic_eta_grid)));
        write_binary(out, global_p.kinetic_eta_power);
        if (global_p.kinetic_eta_grid == "composite_power") {
            write_binary(out, global_p.kinetic_eta_join);
            write_binary(
                out,
                static_cast<std::int32_t>(
                    global_p.kinetic_eta_inner_cells));
        }
        write_binary(out, global_p.r_min);
        write_binary(out, global_p.r_max);
        write_binary(out, global_p.inner_refine_radius);
        write_binary(out, static_cast<std::int32_t>(global_p.inner_refine_cells));
        write_binary(out, global_p.w_max);
        write_binary(out, global_p.L_max);
        write_binary(out, global_p.w_stretch);
        write_binary(out, global_p.w_center);
        write_binary(out, global_p.central_mass);
        write_binary(out, t);
        write_binary(out, static_cast<std::int64_t>(step));
        write_binary(out, static_cast<std::int32_t>(mpi.size));
        write_binary(out, static_cast<std::int32_t>(mpi.rank));
        write_binary(out, static_cast<std::int32_t>(decomp.offset));
        write_binary(out, static_cast<std::int32_t>(decomp.count));
        write_binary(out, static_cast<std::int32_t>(has_tracer ? 1 : 0));
        write_binary(out, static_cast<std::int32_t>(has_carrier ? 1 : 0));
        write_binary(
            out, static_cast<std::int32_t>(has_modal_coefficients ? 1 : 0));
        write_binary(
            out,
            static_cast<std::int32_t>(has_tracer_modal_coefficients ? 1 : 0));
        write_restart_vector(out, f, "state");
        if (has_tracer) {
            write_restart_vector(out, *tracer_f, "tracer");
        }
        if (has_carrier) {
            write_restart_inner_carrier(
                out, *carrier, local_p, "rank state carrier");
            if (has_tracer) {
                const DelayedInnerCarrier empty;
                write_restart_inner_carrier(
                    out,
                    tracer_carrier != nullptr ? *tracer_carrier : empty,
                    local_p,
                    "rank tracer carrier");
            }
        }
        if (has_modal_coefficients) {
            write_restart_modal_coefficients(
                out, *modal_coefficients, "rank state");
        }
        if (has_tracer_modal_coefficients) {
            write_restart_modal_coefficients(
                out, *tracer_modal_coefficients, "rank tracer");
        }
        out.close();
        if (!out) {
            throw std::runtime_error("Failed to close rank restart file after writing: " + rank_tmp.string());
        }
    }
    install_completed_restart_file(rank_tmp, rank_path, step, t);
    mpi_barrier(mpi);

    if (is_root(mpi)) {
        const std::filesystem::path tmp = restart_staging_path(path, staging_dir);
        std::ofstream out(tmp, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not open MPI restart manifest for writing: " + tmp.string());
        }
        const char magic[16] = {'E', 'V', 'M', 'P', 'I', 'R', 'S', 'T', '1', '\0'};
        out.write(magic, sizeof(magic));
        const std::int32_t version = 7;
        write_binary(out, version);
        write_binary(out, static_cast<std::int32_t>(global_p.Nr));
        write_binary(out, static_cast<std::int32_t>(global_p.Nw));
        write_binary(out, static_cast<std::int32_t>(decomp.global_NL));
        write_binary(out, static_cast<std::int32_t>(radial_grid_code(global_p.radial_grid)));
        write_binary(
            out,
            static_cast<std::int32_t>(kinetic_grid_code(global_p.kinetic_grid)));
        write_binary(
            out,
            static_cast<std::int32_t>(
                kinetic_eta_grid_code(global_p.kinetic_eta_grid)));
        write_binary(out, global_p.kinetic_eta_power);
        if (global_p.kinetic_eta_grid == "composite_power") {
            write_binary(out, global_p.kinetic_eta_join);
            write_binary(
                out,
                static_cast<std::int32_t>(
                    global_p.kinetic_eta_inner_cells));
        }
        write_binary(out, global_p.r_min);
        write_binary(out, global_p.r_max);
        write_binary(out, global_p.inner_refine_radius);
        write_binary(out, static_cast<std::int32_t>(global_p.inner_refine_cells));
        write_binary(out, global_p.w_max);
        write_binary(out, global_p.L_max);
        write_binary(out, global_p.w_stretch);
        write_binary(out, global_p.w_center);
        write_binary(out, global_p.central_mass);
        write_binary(out, t);
        write_binary(out, static_cast<std::int64_t>(step));
        write_binary(out, static_cast<std::int32_t>(mpi.size));
        write_binary(out, static_cast<std::int32_t>(has_tracer ? 1 : 0));
        write_binary(out, static_cast<std::int32_t>(has_carrier ? 1 : 0));
        write_binary(
            out, static_cast<std::int32_t>(has_modal_coefficients ? 1 : 0));
        write_binary(
            out,
            static_cast<std::int32_t>(has_tracer_modal_coefficients ? 1 : 0));
        out.close();
        if (!out) {
            throw std::runtime_error("Failed to close MPI restart manifest after writing: " + tmp.string());
        }
        install_completed_restart_file(tmp, path, step, t);
    }
    mpi_barrier(mpi);
    (void)local_p;
}

/// Validates restart-file grid metadata against requested global and local grids.
// Coordinates and their representation define what the stored f means.
// Compare radial/velocity/angular controls before interpreting any saved values.
void validate_restart_grid_header(
    std::ifstream& in,
    std::int32_t version,
    const Params& global_p,
    const LDecomposition& decomp,
    bool has_inner_refine) {
    if (version < 1) {
        throw std::runtime_error("Unsupported restart file version.");
    }
    const std::int32_t nr = read_binary<std::int32_t>(in);
    const std::int32_t nw = read_binary<std::int32_t>(in);
    const std::int32_t nl = read_binary<std::int32_t>(in);
    if (nr != global_p.Nr || nw != global_p.Nw || nl != decomp.global_NL) {
        throw std::runtime_error("Restart file dimensions do not match requested global grid.");
    }
    const std::int32_t saved_radial_grid_code = read_binary<std::int32_t>(in);
    const std::string saved_radial_grid = radial_grid_name(saved_radial_grid_code);
    if (saved_radial_grid != global_p.radial_grid) {
        throw std::runtime_error("Restart radial grid does not match requested radial grid.");
    }
    if (version >= 3) {
        const std::string saved_kinetic_grid =
            kinetic_grid_name(read_binary<std::int32_t>(in));
        if (saved_kinetic_grid != global_p.kinetic_grid) {
            throw std::runtime_error(
                "Restart kinetic-grid representation does not match requested kinetic_grid.");
        }
    } else if (using_cell_average_eta_grid(global_p)) {
        throw std::runtime_error(
            "Legacy MPI restart lacks kinetic-grid metadata and cannot be interpreted as "
            "cell_average_eta.");
    }
    if (version >= 4) {
        const std::string saved_eta_grid =
            kinetic_eta_grid_name(read_binary<std::int32_t>(in));
        if (saved_eta_grid != global_p.kinetic_eta_grid) {
            throw std::runtime_error(
                "Restart kinetic eta grid does not match requested kinetic_eta_grid.");
        }
        validate_restart_double(
            "kinetic_eta_power",
            read_binary<double>(in),
            global_p.kinetic_eta_power);
        if (saved_eta_grid == "composite_power") {
            validate_restart_double(
                "kinetic_eta_join",
                read_binary<double>(in),
                global_p.kinetic_eta_join);
            const int saved_inner_cells = read_binary<std::int32_t>(in);
            if (saved_inner_cells != global_p.kinetic_eta_inner_cells) {
                throw std::runtime_error(
                    "Initial data kinetic_eta_inner_cells does not match the requested grid.");
            }
        }
    } else if (using_cell_average_eta_grid(global_p)) {
        throw std::runtime_error(
            "Pre-v4 MPI cell-average restart used obsolete radial/eta geometry "
            "and cannot be interpreted by the positive-domain solver.");
    }
    validate_restart_double("r_min", read_binary<double>(in), global_p.r_min);
    validate_restart_double("r_max", read_binary<double>(in), global_p.r_max);
    if (has_inner_refine) {
        validate_restart_double("inner_refine_radius", read_binary<double>(in), global_p.inner_refine_radius);
        const std::int32_t saved_inner_cells = read_binary<std::int32_t>(in);
        if (saved_inner_cells != global_p.inner_refine_cells) {
            throw std::runtime_error("Restart grid mismatch for inner_refine_cells");
        }
    }
    validate_restart_double("w_max", read_binary<double>(in), global_p.w_max);
    validate_restart_double("L_max", read_binary<double>(in), global_p.L_max);
    validate_restart_double("w_stretch", read_binary<double>(in), global_p.w_stretch);
    validate_restart_double("w_center", read_binary<double>(in), global_p.w_center);
    validate_restart_double("central_mass", read_binary<double>(in), global_p.central_mass);
}

/// Reads this rank's L slab from a single-file restart written by a one-rank run.
// A single-file checkpoint can supply new MPI angular slabs. Distribute the
// cell arrays and assign hidden carriers by their saved eta ownership, converting
// global angular indices to local ones.
std::vector<double> read_single_restart_local_slab(
    const std::filesystem::path& path,
    const Params& global_p,
    const LDecomposition& decomp,
    double& t,
    int& step,
    std::vector<double>* tracer_f = nullptr,
    bool* has_tracer = nullptr,
    DelayedInnerCarrier* carrier = nullptr,
    DelayedInnerCarrier* tracer_carrier = nullptr,
    std::vector<std::array<double, 5>>* modal_coefficients = nullptr,
    bool* has_modal_coefficients = nullptr,
    std::vector<std::array<double, 5>>* tracer_modal_coefficients = nullptr,
    bool* has_tracer_modal_coefficients = nullptr) {
    if (tracer_f != nullptr) {
        tracer_f->clear();
    }
    if (has_tracer != nullptr) {
        *has_tracer = false;
    }
    if (carrier != nullptr) {
        carrier->packets.clear();
    }
    if (tracer_carrier != nullptr) {
        tracer_carrier->packets.clear();
    }
    if (modal_coefficients != nullptr) {
        modal_coefficients->clear();
    }
    if (tracer_modal_coefficients != nullptr) {
        tracer_modal_coefficients->clear();
    }
    if (has_modal_coefficients != nullptr) {
        *has_modal_coefficients = false;
    }
    if (has_tracer_modal_coefficients != nullptr) {
        *has_tracer_modal_coefficients = false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open restart file: " + path.string());
    }
    char magic[16] = {};
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVLOGRST1")) {
        throw std::runtime_error("Restart file has wrong magic header.");
    }
    const std::int32_t version = read_binary<std::int32_t>(in);
    if (version < 1 || version > 9) {
        throw std::runtime_error("Unsupported restart file version.");
    }
    const std::int32_t nr = read_binary<std::int32_t>(in);
    const std::int32_t nw = read_binary<std::int32_t>(in);
    const std::int32_t nl = read_binary<std::int32_t>(in);
    if (nr != global_p.Nr || nw != global_p.Nw || nl != decomp.global_NL) {
        throw std::runtime_error("Restart file dimensions do not match requested global grid.");
    }
    const std::int32_t saved_radial_grid_code = (version >= 2) ? read_binary<std::int32_t>(in) : 0;
    const std::string saved_radial_grid = radial_grid_name(saved_radial_grid_code);
    if (saved_radial_grid != global_p.radial_grid) {
        throw std::runtime_error("Restart radial grid does not match requested radial grid.");
    }
    if (version >= 5) {
        const std::string saved_kinetic_grid =
            kinetic_grid_name(read_binary<std::int32_t>(in));
        if (saved_kinetic_grid != global_p.kinetic_grid) {
            throw std::runtime_error(
                "Restart kinetic-grid representation does not match requested kinetic_grid.");
        }
    } else if (using_cell_average_eta_grid(global_p)) {
        throw std::runtime_error(
            "Legacy restart lacks kinetic-grid metadata and cannot be interpreted as "
            "cell_average_eta.");
    }
    if (version >= 6) {
        const std::string saved_eta_grid =
            kinetic_eta_grid_name(read_binary<std::int32_t>(in));
        if (saved_eta_grid != global_p.kinetic_eta_grid) {
            throw std::runtime_error(
                "Restart kinetic eta grid does not match requested kinetic_eta_grid.");
        }
        validate_restart_double(
            "kinetic_eta_power",
            read_binary<double>(in),
            global_p.kinetic_eta_power);
        if (saved_eta_grid == "composite_power") {
            validate_restart_double(
                "kinetic_eta_join",
                read_binary<double>(in),
                global_p.kinetic_eta_join);
            const int saved_inner_cells = read_binary<std::int32_t>(in);
            if (saved_inner_cells != global_p.kinetic_eta_inner_cells) {
                throw std::runtime_error(
                    "Initial data kinetic_eta_inner_cells does not match the requested grid.");
            }
        }
    } else if (using_cell_average_eta_grid(global_p)) {
        throw std::runtime_error(
            "Pre-v6 cell-average restart used obsolete radial/eta cell geometry.");
    }
    validate_restart_double("r_min", read_binary<double>(in), global_p.r_min);
    validate_restart_double("r_max", read_binary<double>(in), global_p.r_max);
    if (version >= 3) {
        validate_restart_double("inner_refine_radius", read_binary<double>(in), global_p.inner_refine_radius);
        const std::int32_t saved_inner_cells = read_binary<std::int32_t>(in);
        if (saved_inner_cells != global_p.inner_refine_cells) {
            throw std::runtime_error("Restart grid mismatch for inner_refine_cells");
        }
    }
    validate_restart_double("w_max", read_binary<double>(in), global_p.w_max);
    validate_restart_double("L_max", read_binary<double>(in), global_p.L_max);
    validate_restart_double("w_stretch", read_binary<double>(in), global_p.w_stretch);
    validate_restart_double("w_center", read_binary<double>(in), global_p.w_center);
    validate_restart_double("central_mass", read_binary<double>(in), global_p.central_mass);
    t = read_binary<double>(in);
    const std::int64_t saved_step = read_binary<std::int64_t>(in);
    validate_restart_clock(t, saved_step);
    step = static_cast<int>(saved_step);
    const bool saved_has_tracer = version >= 4 ? (read_binary<std::int32_t>(in) != 0) : false;
    const bool saved_has_carrier =
        version >= 7 ? (read_binary<std::int32_t>(in) != 0) : false;
    const bool saved_has_modal_coefficients =
        version >= 8 ? (read_binary<std::int32_t>(in) != 0) : false;
    const bool saved_has_tracer_modal_coefficients =
        version >= 8 ? (read_binary<std::int32_t>(in) != 0) : false;
    if (saved_has_carrier != using_delayed_inner_return(global_p)) {
        throw std::runtime_error(
            "Restart delayed-return carrier state does not match inner_boundary.");
    }
    if (has_tracer != nullptr) {
        *has_tracer = saved_has_tracer;
    }
    if (has_modal_coefficients != nullptr) {
        *has_modal_coefficients = saved_has_modal_coefficients;
    }
    if (has_tracer_modal_coefficients != nullptr) {
        *has_tracer_modal_coefficients =
            saved_has_tracer_modal_coefficients;
    }
    std::vector<double> f = read_restart_component_local_slab(in, global_p, decomp, "state");
    Params local_p = global_p;
    local_p.NL = decomp.count;
    validate_restart_distribution(f, local_p, "state");
    if (saved_has_tracer) {
        std::vector<double> loaded_tracer =
            read_restart_component_local_slab(in, global_p, decomp, "tracer");
        validate_restart_distribution(loaded_tracer, local_p, "tracer");
        if (tracer_f != nullptr) {
            *tracer_f = std::move(loaded_tracer);
        }
    }
    if (saved_has_carrier) {
        const DelayedInnerCarrier global_carrier =
            read_restart_inner_carrier(
                in,
                global_p,
                "single-file state carrier",
                version >= 9);
        if (carrier != nullptr) {
            for (DelayedInnerCarrierPacket packet : global_carrier.packets) {
                if (packet.eta_index >= decomp.offset
                    && packet.eta_index < decomp.offset + decomp.count) {
                    packet.eta_index -= decomp.offset;
                    carrier->packets.push_back(packet);
                }
            }
        }
        if (saved_has_tracer) {
            const DelayedInnerCarrier global_tracer_carrier =
                read_restart_inner_carrier(
                    in,
                    global_p,
                    "single-file tracer carrier",
                    version >= 9);
            if (tracer_carrier != nullptr) {
                for (DelayedInnerCarrierPacket packet :
                     global_tracer_carrier.packets) {
                    if (packet.eta_index >= decomp.offset
                        && packet.eta_index < decomp.offset + decomp.count) {
                        packet.eta_index -= decomp.offset;
                        tracer_carrier->packets.push_back(packet);
                    }
                }
            }
        }
    }
    if (saved_has_modal_coefficients) {
        std::vector<std::array<double, 5>> loaded =
            read_restart_modal_coefficients_local_slab(
                in, global_p, decomp, "state");
        if (modal_coefficients != nullptr) {
            *modal_coefficients = std::move(loaded);
        }
    }
    if (saved_has_tracer_modal_coefficients) {
        if (!saved_has_tracer) {
            throw std::runtime_error(
                "Restart contains tracer P2 modes without a tracer distribution.");
        }
        std::vector<std::array<double, 5>> loaded =
            read_restart_modal_coefficients_local_slab(
                in, global_p, decomp, "tracer");
        if (tracer_modal_coefficients != nullptr) {
            *tracer_modal_coefficients = std::move(loaded);
        }
    }
    return f;
}

/// Restores the local f slab from either a single-file restart or an MPI restart manifest.
// Inspect the identifier before choosing the reader. A distributed manifest
// requires the saved rank count, and each rank verifies its own clock, slab
// offset, count and optional component flags against that manifest.
std::vector<double> read_restart_state_distributed(
    const std::filesystem::path& path,
    const Params& global_p,
    const Params& local_p,
    const LDecomposition& decomp,
    const MpiContext& mpi,
    double& t,
    int& step,
    std::vector<double>* tracer_f = nullptr,
    bool* has_tracer = nullptr,
    DelayedInnerCarrier* carrier = nullptr,
    DelayedInnerCarrier* tracer_carrier = nullptr,
    std::vector<std::array<double, 5>>* modal_coefficients = nullptr,
    bool* has_modal_coefficients = nullptr,
    std::vector<std::array<double, 5>>* tracer_modal_coefficients = nullptr,
    bool* has_tracer_modal_coefficients = nullptr) {
    if (tracer_f != nullptr) {
        tracer_f->clear();
    }
    if (has_tracer != nullptr) {
        *has_tracer = false;
    }
    if (carrier != nullptr) {
        carrier->packets.clear();
    }
    if (tracer_carrier != nullptr) {
        tracer_carrier->packets.clear();
    }
    if (modal_coefficients != nullptr) {
        modal_coefficients->clear();
    }
    if (tracer_modal_coefficients != nullptr) {
        tracer_modal_coefficients->clear();
    }
    if (has_modal_coefficients != nullptr) {
        *has_modal_coefficients = false;
    }
    if (has_tracer_modal_coefficients != nullptr) {
        *has_tracer_modal_coefficients = false;
    }
    std::ifstream manifest(path, std::ios::binary);
    if (!manifest) {
        throw std::runtime_error("Could not open restart file: " + path.string());
    }
    char magic[16] = {};
    manifest.read(magic, sizeof(magic));
    const bool single_file_magic =
        manifest && fixed_binary_identifier_equals(magic, "EVLOGRST1");
    const bool mpi_manifest_magic =
        manifest && fixed_binary_identifier_equals(magic, "EVMPIRST1");
    manifest.close();

    if (single_file_magic) {
        return read_single_restart_local_slab(
            path,
            global_p,
            decomp,
            t,
            step,
            tracer_f,
            has_tracer,
            carrier,
            tracer_carrier,
            modal_coefficients,
            has_modal_coefficients,
            tracer_modal_coefficients,
            has_tracer_modal_coefficients);
    }
    if (!mpi_manifest_magic) {
        throw std::runtime_error("Restart file has wrong magic header.");
    }

    bool saved_has_tracer = false;
    bool saved_has_carrier = false;
    bool saved_has_modal_coefficients = false;
    bool saved_has_tracer_modal_coefficients = false;
    {
        std::ifstream in(path, std::ios::binary);
        in.read(magic, sizeof(magic));
        const std::int32_t version = read_binary<std::int32_t>(in);
        if (version < 1 || version > 7) {
            throw std::runtime_error("Unsupported MPI restart manifest version.");
        }
        validate_restart_grid_header(in, version, global_p, decomp, true);
        t = read_binary<double>(in);
        const std::int64_t saved_step = read_binary<std::int64_t>(in);
        validate_restart_clock(t, saved_step);
        step = static_cast<int>(saved_step);
        const std::int32_t saved_size = read_binary<std::int32_t>(in);
        if (saved_size != mpi.size) {
            throw std::runtime_error(
                "Distributed restart requires the same MPI rank count used when it was written.");
        }
        saved_has_tracer = version >= 2 ? (read_binary<std::int32_t>(in) != 0) : false;
        saved_has_carrier =
            version >= 5 ? (read_binary<std::int32_t>(in) != 0) : false;
        saved_has_modal_coefficients =
            version >= 6 ? (read_binary<std::int32_t>(in) != 0) : false;
        saved_has_tracer_modal_coefficients =
            version >= 6 ? (read_binary<std::int32_t>(in) != 0) : false;
        if (saved_has_carrier != using_delayed_inner_return(global_p)) {
            throw std::runtime_error(
                "MPI restart delayed-return carrier state does not match inner_boundary.");
        }
        if (has_tracer != nullptr) {
            *has_tracer = saved_has_tracer;
        }
        if (has_modal_coefficients != nullptr) {
            *has_modal_coefficients = saved_has_modal_coefficients;
        }
        if (has_tracer_modal_coefficients != nullptr) {
            *has_tracer_modal_coefficients =
                saved_has_tracer_modal_coefficients;
        }
    }

    const std::filesystem::path rank_path = rank_restart_path(path, mpi.rank);
    std::ifstream in(rank_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open rank restart file: " + rank_path.string());
    }
    in.read(magic, sizeof(magic));
    if (!in || !fixed_binary_identifier_equals(magic, "EVMPIRNK1")) {
        throw std::runtime_error("Rank restart file has wrong magic header.");
    }
    const std::int32_t version = read_binary<std::int32_t>(in);
    if (version < 1 || version > 7) {
        throw std::runtime_error("Unsupported rank restart file version.");
    }
    validate_restart_grid_header(in, version, global_p, decomp, true);
    const double rank_t = read_binary<double>(in);
    validate_restart_double("restart time", rank_t, t);
    const std::int64_t saved_step = read_binary<std::int64_t>(in);
    if (saved_step != step) {
        throw std::runtime_error("Rank restart step does not match manifest.");
    }
    const std::int32_t saved_size = read_binary<std::int32_t>(in);
    const std::int32_t saved_rank = read_binary<std::int32_t>(in);
    const std::int32_t saved_offset = read_binary<std::int32_t>(in);
    const std::int32_t saved_count = read_binary<std::int32_t>(in);
    if (saved_size != mpi.size || saved_rank != mpi.rank
        || saved_offset != decomp.offset || saved_count != decomp.count) {
        throw std::runtime_error("Rank restart decomposition does not match requested MPI layout.");
    }
    const std::uint64_t expected_values =
        static_cast<std::uint64_t>(local_p.Nr) * local_p.Nw * local_p.NL;
    const bool rank_has_tracer = version >= 2 ? (read_binary<std::int32_t>(in) != 0) : false;
    if (rank_has_tracer != saved_has_tracer) {
        throw std::runtime_error("Rank restart tracer flag does not match manifest.");
    }
    const bool rank_has_carrier =
        version >= 5 ? (read_binary<std::int32_t>(in) != 0) : false;
    if (rank_has_carrier != saved_has_carrier) {
        throw std::runtime_error(
            "Rank restart carrier flag does not match manifest.");
    }
    const bool rank_has_modal_coefficients =
        version >= 6 ? (read_binary<std::int32_t>(in) != 0) : false;
    const bool rank_has_tracer_modal_coefficients =
        version >= 6 ? (read_binary<std::int32_t>(in) != 0) : false;
    if (rank_has_modal_coefficients != saved_has_modal_coefficients
        || rank_has_tracer_modal_coefficients
            != saved_has_tracer_modal_coefficients) {
        throw std::runtime_error(
            "Rank restart P2-mode flags do not match manifest.");
    }
    std::vector<double> f = read_restart_vector(in, expected_values, "state");
    validate_restart_distribution(f, local_p, "state");
    if (rank_has_tracer) {
        std::vector<double> loaded_tracer = read_restart_vector(in, expected_values, "tracer");
        validate_restart_distribution(loaded_tracer, local_p, "tracer");
        if (tracer_f != nullptr) {
            *tracer_f = std::move(loaded_tracer);
        }
    }
    if (rank_has_carrier) {
        std::vector<double> legacy_local_eta_center;
        if (version < 7) {
            const std::vector<double> global_eta_center =
                make_eta_axis(global_p).center;
            legacy_local_eta_center.assign(
                global_eta_center.begin() + decomp.offset,
                global_eta_center.begin() + decomp.offset + decomp.count);
        }
        DelayedInnerCarrier loaded = read_restart_inner_carrier(
            in,
            local_p,
            "rank state carrier",
            version >= 7,
            version >= 7 ? nullptr : &legacy_local_eta_center);
        if (carrier != nullptr) {
            *carrier = std::move(loaded);
        }
        if (rank_has_tracer) {
            DelayedInnerCarrier loaded_tracer =
                read_restart_inner_carrier(
                    in,
                    local_p,
                    "rank tracer carrier",
                    version >= 7,
                    version >= 7 ? nullptr : &legacy_local_eta_center);
            if (tracer_carrier != nullptr) {
                *tracer_carrier = std::move(loaded_tracer);
            }
        }
    }
    if (rank_has_modal_coefficients) {
        std::vector<std::array<double, 5>> loaded =
            read_restart_modal_coefficients(
                in, expected_values, "rank state");
        if (modal_coefficients != nullptr) {
            *modal_coefficients = std::move(loaded);
        }
    }
    if (rank_has_tracer_modal_coefficients) {
        if (!rank_has_tracer) {
            throw std::runtime_error(
                "Rank restart contains tracer P2 modes without a tracer.");
        }
        std::vector<std::array<double, 5>> loaded =
            read_restart_modal_coefficients(
                in, expected_values, "rank tracer");
        if (tracer_modal_coefficients != nullptr) {
            *tracer_modal_coefficients = std::move(loaded);
        }
    }
    return f;
}

/// Names the effective characteristic/time integrator for metadata.
// Describe the actual transport dispatch in metadata. The persistent DSMC
// backend advances particles directly, while grid methods retain their own
// interpolation and characteristic integrators.
std::string time_integrator_description(const Params& p) {
    if (using_bp_spherical_dsmc(p)) {
        // SphDSMC bypasses the phase-grid semi-Lagrangian dispatch.  Its
        // persistent markers follow a self-gravitating kick--drift--kick
        // orbit step, followed by the local no-time-counter collision update.
        return "spherical_dsmc_self_gravitating_kick_drift_kick_local_ntc";
    }
    if (using_hamiltonian_sldg_transport(p)) {
        return "hamiltonian_aligned_cell_integrated_conservative_sldg_kdk_lc_continuity_midpoint_field";
    }
    if (using_conservative_sldg_transport(p)) {
        return "cell_integrated_conservative_sldg_kdk_lc_continuity_midpoint_field";
    }
    if (p.integrator == "conservative_finite_volume") {
        return "conservative_finite_volume_upwind_vlasov_poisson";
    }
    if (using_kick_drift_kick_integrator(p)) {
        return using_vlasov_poisson(p)
            ? std::string("semi_lagrangian_kick_drift_kick_peculiar_potential_frozen_") + p.vp_interpolation
            : "semi_lagrangian_kick_drift_kick_metric_frozen";
    }
    if (using_rk_characteristic_integrator(p) && p.time_integrator_order <= 4) {
        return std::string("semi_lagrangian_runge_kutta_characteristics_order_")
            + std::to_string(p.time_integrator_order);
    }
    if (using_rk_characteristic_integrator(p)) {
        return std::string("semi_lagrangian_extrapolated_midpoint_characteristics_requested_order_")
            + std::to_string(p.time_integrator_order)
            + "_effective_order_"
            + std::to_string(2 * ((p.time_integrator_order + 1) / 2));
    }
    return std::string("semi_lagrangian_adams_bashforth_characteristics_order_")
        + std::to_string(p.time_integrator_order);
}

/// Writes JSON metadata describing parameters, backend, grid decomposition, and output conventions.
// Record the physical model, collision law, numerical controls, data meaning
// and decomposition together. These values let an analysis reconstruct the
// equations and measures associated with the saved profiles.
void write_metadata(
    const std::filesystem::path& path,
    const Params& p,
    const Params& local_p,
    const LDecomposition& decomp,
    const MpiContext& mpi,
    std::size_t spherical_dsmc_marker_count = 0,
    double spherical_dsmc_resolved_total_mass =
        std::numeric_limits<double>::quiet_NaN(),
    double spherical_dsmc_initial_common_marker_mass =
        std::numeric_limits<double>::quiet_NaN(),
    std::uint32_t spherical_dsmc_restart_output_version = 0,
    std::uint32_t spherical_dsmc_restart_input_version = 0,
    bool spherical_dsmc_paper_exact = false,
    bool spherical_dsmc_force_consistent_orbit_audit = false,
    bool spherical_dsmc_force_radix_sort = false) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open metadata output file: " + path.string());
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"physics_model\": \"" << p.physics_model << "\",\n";
    out << "  \"vp_universe\": \"" << p.vp_universe << "\",\n";
    out << "  \"vp_alpha\": " << p.vp_alpha << ",\n";
    out << "  \"vp_time_offset\": " << p.vp_time_offset << ",\n";
    out << "  \"vp_gravity\": " << p.vp_gravity << ",\n";
    out << "  \"vp_poisson_source\": \"" << p.vp_poisson_source << "\",\n";
    out << "  \"vp_poisson_reconstruction\": \""
        << p.vp_poisson_reconstruction << "\",\n";
    out << "  \"vp_interpolation\": \"" << p.vp_interpolation << "\",\n";
    out << "  \"kinetic_grid\": \"" << p.kinetic_grid << "\",\n";
    out << "  \"kinetic_eta_grid\": \"" << p.kinetic_eta_grid << "\",\n";
    out << "  \"kinetic_eta_power\": " << p.kinetic_eta_power << ",\n";
    out << "  \"kinetic_eta_join\": " << p.kinetic_eta_join << ",\n";
    out << "  \"kinetic_eta_inner_cells\": "
        << p.kinetic_eta_inner_cells << ",\n";
    out << "  \"kinetic_ic_quadrature_points\": "
        << p.kinetic_ic_quadrature_points << ",\n";
    out << "  \"vp_timestep_mode\": \"" << p.vp_timestep_mode << "\",\n";
    out << "  \"vp_dynamical_eta\": " << p.vp_dynamical_eta << ",\n";
    out << "  \"vp_potential_variation_eta\": "
        << p.vp_potential_variation_eta << ",\n";
    out << "  \"vp_timestep_omitted_mass_fraction\": "
        << p.vp_timestep_omitted_mass_fraction << ",\n";
    out << "  \"vp_hamiltonian_reference\": \""
        << p.vp_hamiltonian_reference << "\",\n";
    out << "  \"vp_hamiltonian_reference_energy_bins\": "
        << p.vp_hamiltonian_reference_energy_bins << ",\n";
    out << "  \"vp_hamiltonian_reference_quadrature_points\": "
        << p.vp_hamiltonian_reference_quadrature_points << ",\n";
    out << "  \"vp_orbit_regularization\": \"" << p.vp_orbit_regularization << "\",\n";
    out << "  \"vp_orbit_energy_tolerance\": " << p.vp_orbit_energy_tolerance << ",\n";
    out << "  \"vp_orbit_lc_relative_tolerance\": "
        << p.vp_orbit_lc_relative_tolerance << ",\n";
    out << "  \"vp_orbit_lc_absolute_tolerance\": "
        << p.vp_orbit_lc_absolute_tolerance << ",\n";
    out << "  \"vp_orbit_lc_max_steps\": " << p.vp_orbit_lc_max_steps << ",\n";
    out << "  \"vp_orbit_lc_max_refinements\": "
        << p.vp_orbit_lc_max_refinements << ",\n";
    out << "  \"vp_transport_mass_conservative\": "
        << (using_conservative_phase_space_cells(p) ? "true" : "false") << ",\n";
    out << "  \"vp_radial_cell_inner_edge\": " << p.r_min << ",\n";
    out << "  \"vp_outer_inflow\": " << (using_vp_outer_inflow(p) ? "true" : "false") << ",\n";
    out << "  \"vp_inflow_start_time\": " << p.vp_inflow_start_time << ",\n";
    out << "  \"vp_inflow_period\": " << p.vp_inflow_period << ",\n";
    out << "  \"vp_inflow_time_sigma\": " << p.vp_inflow_time_sigma << ",\n";
    out << "  \"vp_inflow_time_cutoff\": " << p.vp_inflow_time_cutoff << ",\n";
    out << "  \"vp_inflow_count\": " << p.vp_inflow_count << ",\n";
    out << "  \"vp_inflow_low_l_count\": " << p.vp_inflow_low_l_count << ",\n";
    out << "  \"vp_inflow_high_l_delay\": " << p.vp_inflow_high_l_delay << ",\n";
    out << "  \"vp_inflow_shell_mass\": " << p.vp_inflow_shell_mass << ",\n";
    out << "  \"vp_inflow_high_shell_mass\": " << p.vp_inflow_high_shell_mass << ",\n";
    out << "  \"vp_inflow_sigma_u\": " << p.vp_inflow_sigma_u << ",\n";
    out << "  \"vp_inflow_high_sigma_u\": " << p.vp_inflow_high_sigma_u << ",\n";
    out << "  \"vp_inflow_low_sigma_t\": " << p.vp_inflow_low_sigma_t << ",\n";
    out << "  \"vp_inflow_high_sigma_t\": " << p.vp_inflow_high_sigma_t << ",\n";
    out << "  \"vp_inflow_low_l_center\": " << p.vp_inflow_low_l_center << ",\n";
    out << "  \"vp_inflow_high_l_center\": " << p.vp_inflow_high_l_center << ",\n";
    out << "  \"vp_inflow_freefall_mass\": " << p.vp_inflow_freefall_mass << ",\n";
    out << "  \"vp_inflow_vff_factor\": " << p.vp_inflow_vff_factor << ",\n";
    out << "  \"vp_inflow_coord_speed\": " << p.vp_inflow_coord_speed << ",\n";
    out << "  \"bp_collision_operator\": \"" << p.bp_collision_operator << "\",\n";
    // The collision update description follows the selected implementation,
    // including the distinction between velocity diffusion, finite-angle reactions
    // and persistent physical-particle encounters.
    const char* bp_collision_update = "none";
    if (using_bp_fokker_planck(p)) {
        bp_collision_update =
            "operator_split_explicit_conservative_spherical_fokker_planck";
    } else if (using_bp_finite_angle_boltzmann(p)) {
        bp_collision_update =
            "operator_split_deterministic_all_pair_finite_angle_boltzmann_quadrature";
    } else if (using_bp_reversible_boltzmann(p)) {
        bp_collision_update =
            "operator_split_entropy_dissipative_reversible_finite_angle_boltzmann";
    } else if (using_bp_sampled_reversible_boltzmann(p)) {
        bp_collision_update =
            "operator_split_hybrid_sampled_reversible_finite_angle_boltzmann";
    } else if (using_bp_streamed_reversible_boltzmann(p)) {
        bp_collision_update =
            "operator_split_compact_streamed_sampled_reversible_finite_angle_boltzmann";
    } else if (using_bp_spherical_dsmc(p)) {
        // A fresh sampler may assign one common mass, but the persistent state,
        // gravity, collision kernel, and restart retain each marker's mass.
        bp_collision_update =
            "persistent_mass_carrying_spherical_markers_unequal_mass_ntc";
    }
    out << "  \"bp_collision_update\": \""
        << bp_collision_update << "\",\n";
    out << "  \"bp_cross_section\": \"" << p.bp_cross_section << "\",\n";
    out << "  \"bp_boltzmann_kernel\": \"" << p.bp_boltzmann_kernel << "\",\n";
    out << "  \"bp_bath\": \"" << p.bp_bath << "\",\n";
    out << "  \"bp_fp_terms\": \"" << p.bp_fp_terms << "\",\n";
    out << "  \"bp_collision_flux\": \"" << p.bp_collision_flux << "\",\n";
    out << "  \"bp_collision_integrator\": \"" << p.bp_collision_integrator << "\",\n";
    out << "  \"bp_reversible_integrator\": \""
        << ((p.bp_collision_operator == "boltzmann_reversible"
                || p.bp_collision_operator == "boltzmann_reversible_sampled"
                || p.bp_collision_operator == "boltzmann_reversible_streamed")
                ? "symmetric_reaction_extent_heun_backward_euler"
                : "inactive")
        << "\",\n";
    out << "  \"bp_collision_splitting\": \"" << p.bp_collision_splitting << "\",\n";
    out << "  \"bp_coeff_refresh_mode\": \"" << p.bp_coeff_refresh_mode << "\",\n";
    out << "  \"bp_collision_strength\": " << p.bp_collision_strength << ",\n";
    out << "  \"bp_sigma0\": " << p.bp_sigma0 << ",\n";
    out << "  \"bp_sigma_u0\": " << p.bp_sigma_u0 << ",\n";
    out << "  \"bp_sigma_n\": " << p.bp_sigma_n << ",\n";
    out << "  \"bp_subhalo_mass\": " << p.bp_subhalo_mass << ",\n";
    out << "  \"bp_subhalo_radius\": " << p.bp_subhalo_radius << ",\n";
    out << "  \"bp_coulomb_log\": " << p.bp_coulomb_log << ",\n";
    out << "  \"bp_yukawa_alpha\": " << p.bp_yukawa_alpha << ",\n";
    out << "  \"bp_yukawa_mchi\": " << p.bp_yukawa_mchi << ",\n";
    out << "  \"bp_yukawa_mphi\": " << p.bp_yukawa_mphi << ",\n";
    out << "  \"bp_temperature_floor\": " << p.bp_temperature_floor << ",\n";
    out << "  \"bp_density_floor_abs\": " << p.bp_density_floor_abs << ",\n";
    out << "  \"bp_density_floor_rel\": " << p.bp_density_floor_rel << ",\n";
    out << "  \"bp_partner_mass_tol\": " << p.bp_partner_mass_tol << ",\n";
    out << "  \"bp_bath_speed_cutoff\": " << p.bp_bath_speed_cutoff << ",\n";
    out << "  \"bp_coeff_table_points\": " << p.bp_coeff_table_points << ",\n";
    out << "  \"bp_coeff_speed_points\": " << p.bp_coeff_speed_points << ",\n";
    out << "  \"bp_coeff_mu_points\": " << p.bp_coeff_mu_points << ",\n";
    out << "  \"bp_coeff_refresh_every\": " << p.bp_coeff_refresh_every << ",\n";
    out << "  \"bp_coeff_refresh_chi\": " << p.bp_coeff_refresh_chi << ",\n";
    out << "  \"bp_coeff_refresh_time_quantile\": " << p.bp_coeff_refresh_time_quantile << ",\n";
    out << "  \"bp_collision_apply_every\": " << p.bp_collision_apply_every << ",\n";
    out << "  \"bp_collision_cadence_semantics\": \""
        << ((p.bp_collision_splitting == "strang" && p.bp_collision_apply_every > 1)
                ? "vlasov_cfl_substeps_per_strang_macro"
                : "collision_calls")
        << "\",\n";
    out << "  \"bp_collision_cfl\": " << p.bp_collision_cfl << ",\n";
    out << "  \"bp_collision_max_substeps\": " << p.bp_collision_max_substeps << ",\n";
    out << "  \"bp_collision_locality_max_ratio\": "
        << p.bp_collision_locality_max_ratio << ",\n";
    out << "  \"bp_collision_fail_on_nonlocal_cells\": "
        << (p.bp_collision_fail_on_nonlocal_cells ? "true" : "false")
        << ",\n";
    out << "  \"bp_sts_max_stages\": " << p.bp_sts_max_stages << ",\n";
    out << "  \"bp_collision_l_min_rel\": " << p.bp_collision_l_min_rel << ",\n";
    out << "  \"bp_boltzmann_pair_phi_points\": " << p.bp_boltzmann_pair_phi_points << ",\n";
    out << "  \"bp_boltzmann_scatter_mu_points\": " << p.bp_boltzmann_scatter_mu_points << ",\n";
    out << "  \"bp_boltzmann_scatter_mu_quadrature\": \"gauss_legendre\",\n";
    out << "  \"bp_boltzmann_scatter_phi_points\": " << p.bp_boltzmann_scatter_phi_points << ",\n";
    out << "  \"bp_boltzmann_mass_tol\": " << p.bp_boltzmann_mass_tol << ",\n";
    out << "  \"bp_boltzmann_remap_max_skipped_fraction\": "
        << p.bp_boltzmann_remap_max_skipped_fraction << ",\n";
    out << "  \"bp_boltzmann_conserve_moments\": "
        << (p.bp_boltzmann_conserve_moments ? "true" : "false") << ",\n";
    out << "  \"bp_reversible_candidate_radius\": "
        << p.bp_reversible_candidate_radius << ",\n";
    out << "  \"bp_reversible_max_velocity_cells\": "
        << p.bp_reversible_max_velocity_cells << ",\n";
    out << "  \"bp_reversible_projection_tolerance\": "
        << p.bp_reversible_projection_tolerance << ",\n";
    out << "  \"bp_reversible_entropy_floor\": "
        << p.bp_reversible_entropy_floor << ",\n";
    out << "  \"bp_reversible_entropy_tolerance\": "
        << p.bp_reversible_entropy_tolerance << ",\n";
    out << "  \"bp_reversible_active_shell_floor_fraction\": "
        << p.bp_reversible_active_shell_floor_fraction << ",\n";
    out << "  \"bp_reversible_explicit_fast_path\": "
        << (p.bp_reversible_explicit_fast_path ? "true" : "false")
        << ",\n";
    out << "  \"bp_sampled_dense_cells\": "
        << p.bp_sampled_dense_cells << ",\n";
    out << "  \"bp_sampled_min_samples_per_batch\": "
        << p.bp_sampled_min_samples_per_batch << ",\n";
    out << "  \"bp_sampled_max_samples_per_batch\": "
        << p.bp_sampled_max_samples_per_batch << ",\n";
    out << "  \"bp_sampled_replicates_per_estimate\": "
        << p.bp_sampled_replicates_per_estimate << ",\n";
    out << "  \"bp_sampled_outgoing_angle_samples\": "
        << p.bp_sampled_outgoing_angle_samples << ",\n";
    out << "  \"bp_sampled_guided_pilot_samples\": "
        << p.bp_sampled_guided_pilot_samples << ",\n";
    out << "  \"bp_sampled_guided_pairs\": "
        << p.bp_sampled_guided_pairs << ",\n";
    out << "  \"bp_sampled_guided_fraction\": "
        << p.bp_sampled_guided_fraction << ",\n";
    out << "  \"bp_sampled_fixed_geometric_sequence\": "
        << (p.bp_sampled_fixed_geometric_sequence ? "true" : "false")
        << ",\n";
    out << "  \"bp_sampled_reuse_operator_graph\": "
        << (p.bp_sampled_reuse_operator_graph ? "true" : "false")
        << ",\n";
    out << "  \"bp_sampled_operator_refresh_l1_fraction\": "
        << p.bp_sampled_operator_refresh_l1_fraction << ",\n";
    out << "  \"bp_sampled_ntc_pair_thinning\": "
        << (p.bp_sampled_ntc_pair_thinning ? "true" : "false") << ",\n";
    out << "  \"bp_sampled_ntc_pilot_samples\": "
        << p.bp_sampled_ntc_pilot_samples << ",\n";
    out << "  \"bp_sampled_ntc_quantile\": "
        << p.bp_sampled_ntc_quantile << ",\n";
    out << "  \"bp_sampled_energy_strata\": "
        << p.bp_sampled_energy_strata << ",\n";
    out << "  \"bp_sampled_L_strata\": "
        << p.bp_sampled_L_strata << ",\n";
    out << "  \"bp_sampled_tail_fraction\": "
        << p.bp_sampled_tail_fraction << ",\n";
    out << "  \"bp_sampled_stratified_fraction\": "
        << p.bp_sampled_stratified_fraction << ",\n";
    out << "  \"bp_sampled_equilibrium_fraction\": "
        << p.bp_sampled_equilibrium_fraction << ",\n";
    out << "  \"bp_sampled_phase_density_fraction\": "
        << p.bp_sampled_phase_density_fraction << ",\n";
    out << "  \"bp_sampled_stratum_uniform_fraction\": "
        << p.bp_sampled_stratum_uniform_fraction << ",\n";
    out << "  \"bp_sampled_uniform_fraction\": "
        << p.bp_sampled_uniform_fraction << ",\n";
    out << "  \"bp_sampled_tail_power\": "
        << p.bp_sampled_tail_power << ",\n";
    out << "  \"bp_sampled_rhs_tolerance\": "
        << p.bp_sampled_rhs_tolerance << ",\n";
    out << "  \"bp_sampled_rhs_floor_fraction\": "
        << p.bp_sampled_rhs_floor_fraction << ",\n";
    out << "  \"bp_sampled_heat_tolerance\": "
        << p.bp_sampled_heat_tolerance << ",\n";
    out << "  \"bp_sampled_heat_floor_fraction\": "
        << p.bp_sampled_heat_floor_fraction << ",\n";
    out << "  \"bp_sampled_activity_boost_max\": "
        << p.bp_sampled_activity_boost_max << ",\n";
    out << "  \"bp_sampled_fail_on_unconverged\": "
        << (p.bp_sampled_fail_on_unconverged ? "true" : "false") << ",\n";
    out << "  \"bp_sampled_seed\": "
        << p.bp_sampled_seed << ",\n";
    out << "  \"bp_dsmc_particles\": " << p.bp_dsmc_particles << ",\n";
    out << "  \"bp_dsmc_total_mass\": " << p.bp_dsmc_total_mass << ",\n";
    out << "  \"bp_dsmc_shell_particles\": " << p.bp_dsmc_shell_particles << ",\n";
    out << "  \"bp_dsmc_min_shell_particles\": " << p.bp_dsmc_min_shell_particles << ",\n";
    out << "  \"bp_dsmc_collision_epsilon\": " << p.bp_dsmc_collision_epsilon << ",\n";
    out << "  \"bp_dsmc_gravity_eta\": " << p.bp_dsmc_gravity_eta << ",\n";
    out << "  \"bp_dsmc_seed\": " << p.bp_dsmc_seed << ",\n";
    if (using_bp_spherical_dsmc(p)) {
        const bool has_resolved_marker_state =
            spherical_dsmc_marker_count > 0
            && std::isfinite(spherical_dsmc_resolved_total_mass)
            && std::isfinite(spherical_dsmc_initial_common_marker_mass);
        out << "  \"bp_dsmc_resolved_marker_count\": ";
        if (has_resolved_marker_state) {
            out << spherical_dsmc_marker_count;
        } else {
            out << "null";
        }
        out << ",\n";
        out << "  \"bp_dsmc_resolved_total_mass\": ";
        if (has_resolved_marker_state) {
            out << spherical_dsmc_resolved_total_mass;
        } else {
            out << "null";
        }
        out << ",\n";
        out << "  \"bp_dsmc_initial_common_marker_mass\": ";
        if (has_resolved_marker_state) {
            out << spherical_dsmc_initial_common_marker_mass;
        } else {
            out << "null";
        }
        out << ",\n";
        out << "  \"bp_dsmc_restart_output_version\": "
            << spherical_dsmc_restart_output_version << ",\n";
        out << "  \"bp_dsmc_restart_input_version\": "
            << spherical_dsmc_restart_input_version << ",\n";
        out << "  \"bp_dsmc_collision_clock_source\": \""
            << (spherical_dsmc_restart_input_version >= 6
                    ? "restored_checkpoint_with_verified_collision_controls"
                    : "initial_or_legacy_particle_reconstruction")
            << "\",\n";
        out << "  \"bp_dsmc_paper_exact\": "
            << (spherical_dsmc_paper_exact ? "true" : "false") << ",\n";
        out << "  \"bp_dsmc_force_consistent_orbit_audit\": "
            << (spherical_dsmc_force_consistent_orbit_audit
                    ? "true"
                    : "false")
            << ",\n";
        out << "  \"bp_dsmc_force_radix_sort\": "
            << (spherical_dsmc_force_radix_sort ? "true" : "false")
            << ",\n";
        out << "  \"bp_dsmc_rng_state\": "
               "\"stateless_hash_of_seed_step_shell_and_trial\",\n";
    }
    out << "  \"Nr\": " << p.Nr << ",\n";
    out << "  \"Nw\": " << p.Nw << ",\n";
    out << "  \"NL\": " << p.NL << ",\n";
    out << "  \"r_min\": " << p.r_min << ",\n";
    out << "  \"r_max\": " << p.r_max << ",\n";
    out << "  \"w_max\": " << p.w_max << ",\n";
    out << "  \"L_max\": " << p.L_max << ",\n";
    out << "  \"radial_grid\": \"" << p.radial_grid << "\",\n";
    out << "  \"custom_radial_coordinate\": \"" << p.custom_radial_coordinate << "\",\n";
    out << "  \"inner_refine_radius\": " << p.inner_refine_radius << ",\n";
    out << "  \"inner_refine_cells\": " << p.inner_refine_cells << ",\n";
    out << "  \"w_stretch\": " << p.w_stretch << ",\n";
    out << "  \"w_center\": " << p.w_center << ",\n";
    out << "  \"adaptive_w\": " << (p.adaptive_w ? "true" : "false") << ",\n";
    out << "  \"adaptive_w_every\": " << p.adaptive_w_every << ",\n";
    out << "  \"adaptive_w_trigger\": " << p.adaptive_w_trigger << ",\n";
    out << "  \"adaptive_w_margin\": " << p.adaptive_w_margin << ",\n";
    out << "  \"adaptive_w_growth\": " << p.adaptive_w_growth << ",\n";
    out << "  \"adaptive_w_max\": " << p.adaptive_w_max << ",\n";
    out << "  \"integrator\": \"" << p.integrator << "\",\n";
    out << "  \"time_integrator_order\": " << p.time_integrator_order << ",\n";
    out << "  \"time_integrator_substeps\": " << p.time_integrator_substeps << ",\n";
    out << "  \"vp_sldg_subcells\": " << p.vp_sldg_subcells << ",\n";
    out << "  \"vp_sldg_eta_quadrature_points\": "
        << p.vp_sldg_eta_quadrature_points << ",\n";
    out << "  \"vp_sldg_geometry_tolerance\": "
        << p.vp_sldg_geometry_tolerance << ",\n";
    out << "  \"vp_sldg_max_area_defect\": "
        << p.vp_sldg_max_area_defect << ",\n";
    out << "  \"vp_sldg_max_map_defect\": "
        << p.vp_sldg_max_map_defect << ",\n";
    out << "  \"vp_sldg_max_refinement\": "
        << p.vp_sldg_max_refinement << ",\n";
    out << "  \"vp_sldg_max_work_per_subcell\": "
        << p.vp_sldg_max_work_per_subcell << ",\n";
    out << "  \"vp_sldg_fallback_mass_tolerance\": "
        << p.vp_sldg_fallback_mass_tolerance << ",\n";
    out << "  \"vp_sldg_initial_transport_dt\": "
        << p.vp_sldg_initial_transport_dt << ",\n";
    out << "  \"vp_sldg_target_fallback_fraction\": "
        << p.vp_sldg_target_fallback_fraction << ",\n";
    out << "  \"vp_sldg_timestep_growth\": "
        << p.vp_sldg_timestep_growth << ",\n";
    out << "  \"vp_sldg_timestep_shrink\": "
        << p.vp_sldg_timestep_shrink << ",\n";
    out << "  \"vp_inner_carrier_quadrature_points\": "
        << p.vp_inner_carrier_quadrature_points << ",\n";
    out << "  \"vp_inner_carrier_max_refinement\": "
        << p.vp_inner_carrier_max_refinement << ",\n";
    out << "  \"vp_inner_carrier_max_orbit_segments\": "
        << p.vp_inner_carrier_max_orbit_segments << ",\n";
    out << "  \"vp_inner_carrier_max_packets\": "
        << p.vp_inner_carrier_max_packets << ",\n";
    out << "  \"vp_inner_carrier_mass_tolerance\": "
        << p.vp_inner_carrier_mass_tolerance << ",\n";
    out << "  \"vp_inner_carrier_return_time_bin_width\": "
        << p.vp_inner_carrier_return_time_bin_width << ",\n";
    out << "  \"vp_inner_carrier_compress_min_packets\": "
        << p.vp_inner_carrier_compress_min_packets << ",\n";
    out << "  \"time_integrator\": \"" << time_integrator_description(p) << "\",\n";
    out << "  \"inner_boundary\": \""
        << (using_bp_spherical_dsmc(p)
                ? "origin_reflection_fixed"
                : p.inner_boundary)
        << "\",\n";
    out << "  \"outer_boundary\": \""
        << (using_bp_spherical_dsmc(p)
                ? "unbounded_marker_orbits_no_outer_wall"
                : p.outer_boundary)
        << "\",\n";
    if (using_bp_spherical_dsmc(p)) {
        out << "  \"params_inner_boundary_ignored\": \""
            << p.inner_boundary << "\",\n";
        out << "  \"params_outer_boundary_ignored\": \""
            << p.outer_boundary << "\",\n";
    }
    out << "  \"momentum_boundary\": \"" << p.momentum_boundary << "\",\n";
    out << "  \"inner_boundary_condition\": \""
        << (using_bp_spherical_dsmc(p)
                ? "marker_r_to_abs_r_and_vr_to_minus_vr_at_origin"
                : (using_delayed_inner_return(p)
                ? "positive_log_r_domain_with_delayed_pericentre_return_carrier_in_0_lt_r_lt_r_min"
                : (using_origin_parity_inner(p)
                ? "origin_parity_r_to_abs_r_and_vr_to_minus_vr_with_boundary_slice_symmetrization"
                : (using_conservative_phase_space_cells(p)
                       ? "finite_positive_domain_specular_reflection_at_r_min"
                       : "mirror_r_about_r_min_and_vr_about_zero_with_boundary_slice_symmetrization"))))
        << "\",\n";
    out << "  \"phase_projection\": \"g(r,w)=int_dL_L_f\",\n";
    out << "  \"output_dir\": \"" << p.output_dir << "\",\n";
    out << "  \"restart_dir\": \"" << p.restart_dir << "\",\n";
    out << "  \"t_end\": " << p.t_end << ",\n";
    out << "  \"cfl\": " << p.cfl << ",\n";
    out << "  \"dt_max\": " << p.dt_max << ",\n";
    out << "  \"max_steps\": " << p.max_steps << ",\n";
    out << "  \"save_every\": " << p.save_every << ",\n";
    out << "  \"phase_every\": " << p.phase_every << ",\n";
    out << "  \"phase_r_stride\": " << p.phase_r_stride << ",\n";
    out << "  \"phase_w_stride\": " << p.phase_w_stride << ",\n";
    out << "  \"restart_every\": " << p.restart_every << ",\n";
    out << "  \"restart_from\": \"" << p.restart_from << "\",\n";
    out << "  \"central_mass\": " << p.central_mass << ",\n";
    out << "  \"initial_data\": \"" << p.initial_data << "\",\n";
    out << "  \"initial_data_file\": \"" << p.initial_data_file << "\",\n";
    out << "  \"tracer_initial_data\": " << (p.tracer_initial_data ? "true" : "false") << ",\n";
    out << "  \"tracer_initial_data_file\": \"" << p.tracer_initial_data_file << "\",\n";
    out << "  \"custom_ic_normalize_mass\": "
        << (p.custom_ic_normalize_mass ? "true" : "false") << ",\n";
    out << "  \"custom_ic_target_mass\": " << p.custom_ic_target_mass << ",\n";
    out << "  \"shell_initial_df\": " << (p.shell_initial_df ? "true" : "false") << ",\n";
    out << "  \"R0\": " << p.R0 << ",\n";
    out << "  \"dR\": " << p.dR << ",\n";
    out << "  \"top_hat_shell\": " << (p.top_hat_shell ? "true" : "false") << ",\n";
    out << "  \"top_hat_density_shell\": " << (p.top_hat_density_shell ? "true" : "false") << ",\n";
    out << "  \"gaussian_density_shell\": " << (p.gaussian_density_shell ? "true" : "false") << ",\n";
    out << "  \"double_gaussian_density_shell\": "
        << (p.double_gaussian_density_shell ? "true" : "false") << ",\n";
    out << "  \"R0_2\": " << p.R0_2 << ",\n";
    out << "  \"dR_2\": " << p.dR_2 << ",\n";
    out << "  \"amp_2\": " << p.amp_2 << ",\n";
    out << "  \"w_in_2\": " << p.w_in_2 << ",\n";
    out << "  \"shell_radial_mode\": \""
        << (p.double_gaussian_density_shell
                ? "density_double_gaussian"
                : (p.top_hat_density_shell ? "density_tophat"
                                           : (p.gaussian_density_shell
                                                  ? "density_gaussian"
                                                  : (p.top_hat_shell ? "f_tophat" : "f_gaussian"))))
        << "\",\n";
    out << "  \"beta\": " << p.beta << ",\n";
    out << "  \"amp\": " << p.amp << ",\n";
    out << "  \"w_in\": " << p.w_in << ",\n";
    out << "  \"initial_df\": \""
        << (using_binary_initial_data(p)
                ? "binary_f_of_r_w_L"
                : (p.shell_initial_df ? "shell_boundary_seed_F0_with_R0_for_zeta_b"
                                      : "boundary_seed_F0_with_r_for_zeta"))
        << "\",\n";
    out << "  \"initial_metric_iters\": " << p.initial_metric_iters << ",\n";
    out << "  \"metric_floor\": " << p.metric_floor << ",\n";
    out << "  \"gap_stop\": " << p.gap_stop << ",\n";
    out << "  \"active_floor_rel\": " << p.active_floor_rel << ",\n";
    out << "  \"active_floor_abs\": " << p.active_floor_abs << ",\n";
    out << "  \"f_floor\": " << p.f_floor << ",\n";
    out << "  \"threads\": " << p.threads << ",\n";
    out << "  \"parallel_backend\": \"" << parallel_backend_name() << "\",\n";
    out << "  \"mpi_enabled\": " << (mpi.enabled ? "true" : "false") << ",\n";
    out << "  \"mpi_size\": " << mpi.size << ",\n";
    out << "  \"detected_multi_rank_size\": " << detected_multi_rank_size() << ",\n";
    out << "  \"allow_multi_rank_launch\": " << (p.allow_multi_rank_launch ? "true" : "false") << ",\n";
    out << "  \"decomposition\": \""
        << (using_bp_spherical_dsmc(p)
                ? "single_process_persistent_markers_openmp"
                : "L_slab")
        << "\",\n";
    out << "  \"global_NL\": " << decomp.global_NL << ",\n";
    out << "  \"rank0_local_L_offset\": " << decomp.offset << ",\n";
    out << "  \"rank0_local_NL\": " << local_p.NL << ",\n";
    out << "  \"restart_layout\": \""
        << ((mpi.size > 1) ? "mpi_manifest_plus_rank_files" : "single_file") << "\"\n";
    out << "}\n";
    out.close();
    if (!out) {
        throw std::runtime_error("Failed to close metadata output file after writing: " + path.string());
    }
}
}  // namespace ev
