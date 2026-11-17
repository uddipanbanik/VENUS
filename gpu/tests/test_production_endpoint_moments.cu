// Run the actual production driver and compare its endpoint profile with
// a fresh deposition of the saved final particle checkpoint.
#include "venus_gpu/production_driver.hpp"
#include "venus_gpu/production_output.hpp"
#include "venus_gpu/restart.hpp"
#include "venus_gpu/vlasov_poisson.hpp"
#include "venus_gpu/wpic.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path directory = fs::temp_directory_path()
        / ("venus_gpu_endpoint_moments_" + std::to_string(unique));
    fs::create_directories(directory);
    venus_gpu::RestartState initial;
    for (std::uint64_t i = 0; i < 64; ++i) {
        venus_gpu::Marker marker;
        marker.mass = 1.0 / 64.0;
        marker.r = 1.0 + 0.01 * static_cast<double>(i);
        marker.w = -0.2 + 0.006 * static_cast<double>(i);
        marker.eta = 0.005;
        marker.id = i;
        initial.markers.push_back(marker);
    }
    const fs::path input = directory / "initial.bin";
    const fs::path output = directory / "output";
    venus_gpu::write_restart(input.string(), initial);
    std::vector<std::string> arguments = {
        "endpoint-moments", "--restart", input.string(),
        "--output-directory", output.string(), "--end-time", "0.1",
        "--maximum-timestep", "0.1", "--output-interval", "0.1",
        "--radial-cells", "16", "--radial-minimum", "0.1",
        "--radial-maximum", "10", "--history-every", "1",
        "--restart-every", "1",
    };
    std::vector<char*> argv;
    for (std::string& value : arguments) argv.push_back(value.data());
    const int status = venus_gpu::driver::run_production_task(
        static_cast<int>(argv.size()), argv.data(),
        venus_gpu::driver::ProductionTask::vlasov_poisson);
    assert(status == 0);
    const auto endpoint = venus_gpu::read_restart(
        (output / "restart_state.bin").string());
    assert(endpoint.time == 0.1 && endpoint.step == 1);

    std::vector<double> edge(17);
    for (int i = 0; i <= 16; ++i) {
        edge[i] = std::exp(std::log(0.1)
            + static_cast<double>(i) / 16 * (std::log(10.0) - std::log(0.1)));
    }
    edge.front() = 0.1;
    edge.back() = 10.0;
    const auto mesh = venus_gpu::RadialMesh::from_edges(edge);
    venus_gpu::WpicState state(endpoint.markers);
    venus_gpu::VlasovPoissonModule vp(mesh);
    venus_gpu::VlasovPoissonOptions options;
    options.gravity = 1.0;
    options.particle_shape = venus_gpu::ParticleShape::tsc;
    options.inner_boundary = venus_gpu::InnerBoundary::origin_parity;
    options.outer_boundary = venus_gpu::OuterBoundary::reflecting_wall;
    options.expansion.enabled = true;
    options.expansion.alpha = 2.0 / 3.0;
    options.expansion.time_offset = 12.0;
    options.subtract_homogeneous_background = false;
    options.carry_active_mass_inside_inner_face = true;
    vp.deposit_and_solve(state, endpoint.time, options);
    const auto field = vp.download_field();
    std::ifstream profile(output / "profile_00001.csv");
    assert(profile);
    std::string line;
    std::getline(profile, line);
    double maximum_mean_error = 0.0;
    double maximum_variance_error = 0.0;
    std::size_t cell = 0;
    while (std::getline(profile, line)) {
        std::stringstream row(line);
        std::string token;
        std::vector<double> values;
        while (std::getline(row, token, ',')) values.push_back(std::stod(token));
        assert(values.size() == 7 && cell < mesh.cells());
        const double mass = field.shell_mass[cell];
        const double mean = mass > 0.0 ? field.shell_radial_momentum[cell] / mass : 0.0;
        const double second = mass > 0.0 ? field.shell_radial_second_moment[cell] / mass : 0.0;
        const double variance = std::max(0.0, second - mean * mean);
        maximum_mean_error = std::max(maximum_mean_error, std::abs(values[3] - mean));
        maximum_variance_error = std::max(maximum_variance_error, std::abs(values[4] - variance));
        ++cell;
    }
    std::cout << "Endpoint profile versus saved particles: mean error="
              << maximum_mean_error << " variance error="
              << maximum_variance_error << '\n';
    assert(cell == mesh.cells());
    assert(maximum_mean_error < 1e-12);
    assert(maximum_variance_error < 1e-12);

    const auto read_bytes = [](const fs::path& path) {
        std::ifstream input(path, std::ios::binary);
        assert(input);
        return std::string(std::istreambuf_iterator<char>(input), {});
    };
    const auto set_argument = [&](const std::string& name,
                                  const std::string& value) {
        const auto found = std::find(arguments.begin(), arguments.end(), name);
        assert(found != arguments.end() && found + 1 != arguments.end());
        *(found + 1) = value;
    };
    const auto run_again = [&]() {
        argv.clear();
        for (std::string& value : arguments) argv.push_back(value.data());
        return venus_gpu::driver::run_production_task(
            static_cast<int>(argv.size()), argv.data(),
            venus_gpu::driver::ProductionTask::vlasov_poisson);
    };
    // Emulate an accepted orbit step followed by a walltime checkpoint
    // between scheduled outputs.  The saved profile remains at t=0.1.
    const auto saved_profile = read_bytes(output / "profile_00001.csv");
    const auto saved_history = read_bytes(output / "history_gpu.csv");
    vp.advance_kdk(state, endpoint.time, 0.05, options);
    auto interrupted = endpoint;
    interrupted.markers = state.download();
    interrupted.time = 0.15;
    interrupted.step = 2;
    const auto interrupted_path = directory / "interrupted.bin";
    venus_gpu::write_restart(interrupted_path.string(), interrupted);
    set_argument("--restart", interrupted_path.string());
    set_argument("--end-time", "0.2");
    assert(run_again() == 0);
    assert(read_bytes(output / "profile_00001.csv") == saved_profile);
    const auto continued_history = read_bytes(output / "history_gpu.csv");
    assert(continued_history.size() > saved_history.size());
    assert(continued_history.compare(0, saved_history.size(), saved_history) == 0);
    const auto continued_profile = read_bytes(output / "profile_00002.csv");
    const auto continued = venus_gpu::read_restart(
        (output / "restart_state.bin").string());
    assert(continued.time == 0.2 && continued.step == 3);

    const auto expect_stale_rejection = [&](const std::string& filename) {
        std::ostringstream diagnostic;
        auto* previous = std::cerr.rdbuf(diagnostic.rdbuf());
        const int result = run_again();
        std::cerr.rdbuf(previous);
        assert(result == 2);
        assert(diagnostic.str().find("later than restart time")
            != std::string::npos);
        assert(diagnostic.str().find(filename) != std::string::npos);
        std::cout << diagnostic.str();
    };
    // Isolate a profile ahead of the supplied checkpoint by retaining only
    // the earlier history while testing the future t=0.2 profile.
    std::ofstream(output / "history_gpu.csv") << saved_history;
    expect_stale_rejection("profile_00002.csv");
    assert(read_bytes(output / "profile_00001.csv") == saved_profile);
    assert(read_bytes(output / "profile_00002.csv") == continued_profile);
    assert(read_bytes(output / "history_gpu.csv") == saved_history);
    std::ofstream(output / "history_gpu.csv") << continued_history;

    set_argument("--restart", (output / "restart_state.bin").string());
    set_argument("--end-time", "0.3");
    for (const auto* filename : {"history_gpu.csv", "collision_history_gpu.csv"}) {
        const fs::path path = output / filename;
        const bool existed = fs::exists(path);
        const std::string original = existed ? read_bytes(path) : "step,time\n";
        const std::string future = original + "999,0.21\n";
        std::ofstream(path) << future;
        expect_stale_rejection(filename);
        assert(read_bytes(path) == future);
        assert(read_bytes(output / "profile_00002.csv") == continued_profile);
        if (existed) std::ofstream(path) << original;
        else fs::remove(path);
    }

    // One marker makes deposition ordering exact, so this test isolates every
    // state change caused by the optional output path. The ordinary evolution
    // and its selected-snapshot run must produce byte-identical checkpoints,
    // profiles, and all physical history columns.
    auto snapshot_initial = initial;
    snapshot_initial.markers.resize(1);
    snapshot_initial.markers[0].mass = 1.0;
    const auto snapshot_input = directory / "snapshot_initial.bin";
    venus_gpu::write_restart(snapshot_input.string(), snapshot_initial);
    const auto unsaved_output = directory / "unsaved";
    const auto saved_output = directory / "saved";
    set_argument("--restart", snapshot_input.string());
    set_argument("--output-directory", unsaved_output.string());
    set_argument("--end-time", "0.3");
    set_argument("--maximum-timestep", "0.05");
    assert(run_again() == 0);
    set_argument("--output-directory", saved_output.string());
    arguments.push_back("--phase-space-snapshot-times");
    arguments.push_back("0,0.1,0.3");
    assert(run_again() == 0);
    assert(!fs::exists(unsaved_output / "phase_space"));
    assert(read_bytes(unsaved_output / "restart_state.bin")
        == read_bytes(saved_output / "restart_state.bin"));
    for (int index = 0; index <= 3; ++index) {
        std::ostringstream filename;
        filename << "profile_" << std::setw(5) << std::setfill('0') << index << ".csv";
        assert(read_bytes(unsaved_output / filename.str())
            == read_bytes(saved_output / filename.str()));
    }
    const auto physical_history = [&](const fs::path& path) {
        std::stringstream input(read_bytes(path));
        std::string line, physical;
        while (std::getline(input, line)) {
            physical += line.substr(0, line.rfind(',')) + '\n';
        }
        return physical;
    };
    assert(physical_history(unsaved_output / "history_gpu.csv")
        == physical_history(saved_output / "history_gpu.csv"));
    assert(!fs::exists(saved_output / "phase_space" / "snapshot_00002"));
    const auto final_snapshot = saved_output / "phase_space" / "snapshot_00003";
    assert(read_bytes(final_snapshot / "particles.bin")
        == read_bytes(saved_output / "restart_state.bin"));
    const auto snapshot_endpoint = venus_gpu::read_restart(
        (final_snapshot / "particles.bin").string());
    venus_gpu::WpicState snapshot_state(snapshot_endpoint.markers);
    vp.deposit_and_solve(snapshot_state, snapshot_endpoint.time, options);
    const auto expected_field = directory / "expected_field.json";
    venus_gpu::driver::write_phase_space_field_json(expected_field.string(),
        snapshot_endpoint.time, vp.download_field(), mesh, options, true);
    assert(read_bytes(expected_field) == read_bytes(final_snapshot / "field.json"));
    const auto retained_snapshot = read_bytes(
        saved_output / "phase_space" / "snapshot_00001" / "particles.bin");
    const auto fresh_continuation = directory / "snapshot_continuation";
    set_argument("--restart", (saved_output / "phase_space"
        / "snapshot_00001" / "particles.bin").string());
    set_argument("--output-directory", fresh_continuation.string());
    assert(run_again() == 0);
    assert(read_bytes(fresh_continuation / "restart_state.bin")
        == read_bytes(saved_output / "restart_state.bin"));
    assert(!fs::exists(fresh_continuation / "phase_space" / "snapshot_00000"));
    assert(read_bytes(fresh_continuation / "phase_space"
        / "snapshot_00001" / "particles.bin") == retained_snapshot);
    // Exercise same-directory continuation from a selected endpoint. Remove
    // only the later ordinary diagnostics, leaving a future snapshot receipt.
    fs::remove(saved_output / "profile_00002.csv");
    fs::remove(saved_output / "profile_00003.csv");
    fs::remove(saved_output / "history_gpu.csv");
    set_argument("--output-directory", saved_output.string());
    std::ostringstream stale_snapshot_error;
    auto* previous_error = std::cerr.rdbuf(stale_snapshot_error.rdbuf());
    assert(run_again() == 2);
    std::cerr.rdbuf(previous_error);
    assert(stale_snapshot_error.str().find("phase-space snapshot") != std::string::npos);
    assert(stale_snapshot_error.str().find("later than restart time") != std::string::npos);
    fs::remove_all(final_snapshot);
    assert(run_again() == 0);
    assert(read_bytes(saved_output / "phase_space"
        / "snapshot_00001" / "particles.bin") == retained_snapshot);
    assert(read_bytes(saved_output / "restart_state.bin")
        == read_bytes(unsaved_output / "restart_state.bin"));
    std::cout << "Selected snapshots: byte-identical endpoint particles, profiles, "
                 "physical history; actual field and restart continuation agree.\n";
    fs::remove_all(directory);
    std::cout << "GPU production endpoint and restart-output tests passed.\n";
}
