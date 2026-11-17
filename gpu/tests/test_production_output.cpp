#include "venus_gpu/production_output.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

template <class Action>
void expect_runtime_error(Action action) {
    bool threw = false;
    try {
        action();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    const fs::path directory = "venus_gpu_profile_contract_test";
    fs::create_directories(directory);

    venus_gpu::RadialMesh mesh;
    mesh.radius_center = {1.0, 2.0};
    mesh.shell_volume = {2.0, 4.0};
    venus_gpu::FieldHost field;
    field.shell_mass = {4.0, 0.0};
    field.shell_radial_momentum = {2.0, 0.0};
    field.shell_radial_second_moment = {5.0, 0.0};
    field.shell_tangential_second_moment = {3.0, 0.0};
    field.effective_marker_number = {8.0, 0.0};

    venus_gpu::driver::write_profile_csv(
        directory.string(), 7, 1.25, field, mesh);
    const fs::path profile = directory / "profile_00007.csv";
    assert(fs::is_regular_file(profile));
    assert(!fs::exists(profile.string() + ".tmp"));
    std::ifstream input(profile);
    std::ostringstream content;
    content << input.rdbuf();
    const std::string text = content.str();
    assert(text.find(
        "time,radius,density,radial_mean,radial_variance,"
        "tangential_second,neff\n") == 0);
    // Cell zero: rho=2, mean=1/2, variance=5/4-1/4=1,
    // tangential second moment=3/4.
    assert(text.find("1.25,1,2,0.5,1,0.75,8\n") != std::string::npos);
    assert(text.find("1.25,2,0,0,0,0,0\n") != std::string::npos);

    venus_gpu::FieldHost bad = field;
    bad.shell_mass.pop_back();
    expect_runtime_error([&] {
        venus_gpu::driver::write_profile_csv(
            directory.string(), 8, 1.25, bad, mesh);
    });

    fs::remove(profile);

    const auto times = venus_gpu::driver::parse_phase_space_snapshot_times("0.3,0,0.1");
    assert((times == std::vector<double>{0.0, 0.1, 0.3}));
    venus_gpu::driver::validate_phase_space_snapshot_times(times, 0.1);
    assert(venus_gpu::driver::phase_space_snapshot_index(times, 0.3, 0.1) == 3);
    assert(venus_gpu::driver::phase_space_snapshot_index(times, 0.2, 0.1) == -1);
    assert(venus_gpu::driver::parse_phase_space_snapshot_times("").empty());
    for (const auto* invalid : {"0,,1", "0,", "nan", "inf", "-1", "1,1", "1x"}) {
        expect_runtime_error([&] {
            venus_gpu::driver::parse_phase_space_snapshot_times(invalid);
        });
    }
    expect_runtime_error([&] {
        venus_gpu::driver::validate_phase_space_snapshot_times({0.15}, 0.1);
    });
    expect_runtime_error([&] {
        venus_gpu::driver::validate_phase_space_snapshot_times({1e300}, 0.1);
    });

    mesh.edge = {0.5, 1.5, 2.5};
    field.shell_radial_energy_flux = {0.0, 0.0};
    field.shell_mass_square = {2.0, 0.0};
    field.source_mass_edge = {0.0, 4.0, 4.0};
    field.source_line_density = {4.0, 0.0};
    field.phi = {-3.0, -2.0};
    field.phi_r = {1.0, 1.0};
    venus_gpu::VlasovPoissonOptions options;
    options.expansion.enabled = true;
    options.expansion.alpha = 2.0 / 3.0;
    options.expansion.time_offset = 12.0;
    const auto field_path = directory / "field.json";
    venus_gpu::driver::write_phase_space_field_json(
        field_path.string(), 0.0, field, mesh, options, true);
    std::ifstream field_input(field_path);
    std::ostringstream field_content;
    field_content << field_input.rdbuf();
    assert(field_content.str().find("\"phase_volume\": \"8*pi^2*dr*du*deta\"")
        != std::string::npos);
    assert(field_content.str().find("\"phi\": [-3,-2]") != std::string::npos);
    assert(field_content.str().find("\"scale_factor\": 1") != std::string::npos);
    bad = field;
    bad.phi.pop_back();
    expect_runtime_error([&] {
        venus_gpu::driver::write_phase_space_field_json(
            field_path.string(), 0.0, bad, mesh, options, true);
    });
    fs::remove(field_path);
    fs::remove(directory);
    return 0;
}
