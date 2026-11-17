// Exercise the actual restart-tool entry points with physical state fixtures.
// These checks guard state invalidation and extensive internal-energy safety.
#define main refine_weighted_pic_restart_main
#include "../tools/refine_weighted_pic_restart.cpp"
#undef main
#define main inspect_weighted_pic_restart_main
#include "../tools/inspect_weighted_pic_restart.cpp"
#undef main

#include <cassert>
#include <chrono>
#include <fstream>
#include <iterator>
#include <sstream>

namespace {

int invoke(int (*entry)(int, char**), std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    std::ostringstream messages;
    auto* old_out = std::cout.rdbuf(messages.rdbuf());
    auto* old_error = std::cerr.rdbuf(messages.rdbuf());
    const int result = entry(static_cast<int>(argv.size()), argv.data());
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_error);
    return result;
}

std::string bytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::uint32_t version(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    stream.seekg(16);
    return ev::weighted_pic_detail::read_scalar<std::uint32_t>(stream);
}

ev::WeightedPICRestartState fixture() {
    ev::WeightedPICRestartState state;
    state.time = 1.25;
    state.step = 37;
    state.sampler = {1234, 41};
    state.collision_sampler = {5678, 83};
    state.potential_event_age = 0.03;
    state.collision_timescale_estimate = 0.02;
    state.potential_timescale_estimate = 0.5;
    for (std::uint64_t i = 0; i < 3; ++i) {
        ev::WeightedPICMarker marker;
        marker.mass = 0.125 * (i + 1);
        marker.r = 0.7 + 0.2 * i;
        marker.w = -0.1 + 0.15 * i;
        marker.eta = 0.01 * i;
        marker.id = 11 + 7 * i;
        state.markers.push_back(marker);
    }
    state.has_frozen_field = true;
    state.frozen_field_contract = "restart-tools-fixture";
    auto& field = state.frozen_field;
    field.source_mass = {0.75};
    field.phi = {-1.0};
    field.phi_r = {0.5};
    field.radial_edge = {0.5, 1.5};
    field.source_mass_edge = {0.0, 0.75};
    field.source_line_density = {0.75};
    field.source_line_density_linear = {0.0};
    field.source_line_density_quadratic = {0.0};
    field.phi_edge = {-1.5, -0.5};
    field.background.tau = field.background.a = 1.0;
    return state;
}

void assert_fresh_field(const ev::WeightedPICRestartState& state) {
    assert(!state.has_frozen_field);
    assert(state.frozen_field_contract.empty());
    assert(state.potential_event_age == 0.0);
    assert(state.collision_event_age == 0.0);
    assert(std::isinf(state.potential_timescale_estimate));
    assert(std::isinf(state.collision_timescale_estimate));
    assert(state.time == 1.25 && state.step == 37);
    assert(state.sampler.seed == 1234 && state.sampler.next_counter == 41);
    assert(state.collision_sampler.seed == 5678
           && state.collision_sampler.next_counter == 83);
}

}  // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path()
        / ("venus-restart-tools-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    const auto input = root / "input.bin";
    const auto output = root / "output.bin";
    auto state = fixture();
    ev::write_weighted_pic_restart(input, state);
    const auto original = bytes(input);
    auto refine = [&](bool reinitialize = true) {
        std::vector<std::string> args{
            "refine", input.string(), output.string(), "3"};
        if (reinitialize) args.push_back("--reinitialize-field");
        return invoke(refine_weighted_pic_restart_main, args);
    };
    auto export_particles = [&]() {
        return invoke(inspect_weighted_pic_restart_main,
            {"inspect", input.string(), "--export-particles", output.string()});
    };
    assert(refine(false) != 0 && !std::filesystem::exists(output));
    assert(refine() == 0);
    auto refined = ev::read_weighted_pic_restart(output);
    assert(version(output) == 6 && refined.markers.size() == 9);
    assert_fresh_field(refined);
    // Each group preserves mass and both radial moments at its phase-space point.
    for (std::size_t i = 0; i < state.markers.size(); ++i) {
        double mass = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
            const auto& marker = refined.markers[3 * i + j];
            mass += marker.mass;
            assert(marker.r == state.markers[i].r);
            assert(marker.w == state.markers[i].w);
            assert(marker.eta == state.markers[i].eta);
        }
        assert(std::abs(mass - state.markers[i].mass) < 1.0e-15);
    }
    assert(bytes(input) == original);
    std::filesystem::remove(output);
    state.collision_event_age = 0.01;
    ev::write_weighted_pic_restart(input, state);
    assert(refine() != 0 && export_particles() != 0);
    assert(!std::filesystem::exists(output));
    state.collision_event_age = 0.0;
    for (int internal_property = 0; internal_property < 2; ++internal_property) {
        state.markers[0].internal_energy = internal_property == 0 ? 0.03 : 0.0;
        state.markers[0].internal_velocity_dispersion =
            internal_property == 1 ? 0.2 : 0.0;
        ev::write_weighted_pic_restart(input, state);
        assert(refine() != 0 && !std::filesystem::exists(output));
        // Particle export retains whole-object internal state without splitting.
        assert(export_particles() == 0);
        const auto exported = ev::read_weighted_pic_restart(output);
        assert(version(output) == 6);
        assert_fresh_field(exported);
        assert(exported.markers[0].mass == state.markers[0].mass);
        assert(exported.markers[0].internal_energy
               == state.markers[0].internal_energy);
        assert(exported.markers[0].internal_velocity_dispersion
               == state.markers[0].internal_velocity_dispersion);
        std::filesystem::remove(output);
    }
    const auto before_self_export = bytes(input);
    assert(invoke(inspect_weighted_pic_restart_main,
        {"inspect", input.string(), "--export-particles", input.string()}) != 0);
    assert(bytes(input) == before_self_export);
    state = fixture();
    state.has_frozen_field = false;
    ev::write_weighted_pic_restart(input, state);
    assert(refine(false) == 0);
    assert_fresh_field(ev::read_weighted_pic_restart(output));
    std::filesystem::remove_all(root);
    std::cout << "WPIC restart tools preserve particle moments and guard field/internal state\n";
    return 0;
}
