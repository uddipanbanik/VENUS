#include "venus_gpu/wpic.hpp"

#include "ev/weighted_vlasov_pic_collisions.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void compare_model(const std::string& cpu_name,
                   venus_gpu::CrossSectionModel gpu_model) {
    ev::Params cpu;
    cpu.bp_cross_section = cpu_name;
    cpu.bp_boltzmann_kernel =
        cpu_name == "yukawa_born" ? "yukawa_born" : "isotropic";
    cpu.bp_collision_strength = 0.73;
    cpu.bp_sigma0 = 1.31;
    cpu.bp_sigma_u0 = 0.27;
    cpu.bp_sigma_n = 3.4;
    cpu.bp_yukawa_alpha = 0.031;
    cpu.bp_yukawa_mchi = 1.8;
    cpu.bp_yukawa_mphi = 0.42;
    ev::WeightedPICCollisionOptions cpu_options;

    venus_gpu::CollisionOptions gpu;
    gpu.cross_section = gpu_model;
    gpu.collision_strength = cpu.bp_collision_strength;
    gpu.sigma0 = cpu.bp_sigma0;
    gpu.sigma_u0 = cpu.bp_sigma_u0;
    gpu.sigma_power = cpu.bp_sigma_n;
    gpu.yukawa_alpha = cpu.bp_yukawa_alpha;
    gpu.yukawa_mchi = cpu.bp_yukawa_mchi;
    gpu.yukawa_mphi = cpu.bp_yukawa_mphi;

    const std::vector<double> speeds = {
        0.0, 1.0e-14, 3.3e-8, 1.0e-7, 1.0e-6, 1.0e-5,
        0.03, 0.2, 0.73, 2.0, 30.0};
    for (double speed : speeds) {
        const double expected = ev::bp_fa_total_cross_section(speed, cpu);
        const double measured =
            venus_gpu::material_cross_section_per_mass(speed, gpu);
        const double tolerance = 16.0
            * std::numeric_limits<double>::epsilon()
            * std::max(1.0, std::abs(expected));
        assert(std::abs(measured - expected) <= tolerance);
        const double first_mass = 0.017;
        const double second_mass = 2.3;
        const double cpu_pair =
            ev::weighted_pic_collision_detail::pair_total_cross_section(
                speed,
                first_mass,
                second_mass,
                cpu,
                cpu_options);
        const double gpu_pair =
            0.5 * (first_mass + second_mass) * measured;
        assert(std::abs(gpu_pair - cpu_pair)
            <= tolerance * 0.5 * (first_mass + second_mass));

        for (double unit : {0.0, 0.01, 0.25, 0.5, 0.9, 1.0}) {
            const double expected_mu =
                ev::weighted_pic_collision_detail::sample_scatter_mu_from_unit(
                    speed,
                    first_mass,
                    second_mass,
                    cpu,
                    cpu_options,
                    unit);
            const double measured_mu =
                venus_gpu::material_scatter_cosine(speed, unit, gpu);
            assert(std::abs(measured_mu - expected_mu)
                <= 32.0 * std::numeric_limits<double>::epsilon());
        }
    }
}

}  // namespace

int main() {
    compare_model("constant", venus_gpu::CrossSectionModel::constant);
    compare_model("powerlaw", venus_gpu::CrossSectionModel::power_law);
    compare_model("yukawa_born", venus_gpu::CrossSectionModel::yukawa_born);
    std::cout << "Current CPU/GPU material SIDM formulas match.\n";
    return 0;
}
