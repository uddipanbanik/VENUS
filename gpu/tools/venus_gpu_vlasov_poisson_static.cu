// Static-background front end for the maintained collisionless CUDA kernel.
#include "venus_gpu/production_driver.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--universe"
            && (i + 1 >= argc || std::string(argv[i + 1]) != "static")) {
            std::cerr << "Static Vlasov-Poisson requires --universe static.\n";
            return 2;
        }
    }
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Static Vlasov-Poisson WPIC: H=0; --universe static is fixed.\n";
            break;
        }
    }
    std::vector<char*> arguments(argv, argv + argc);
    char option[] = "--universe";
    char background[] = "static";
    arguments.push_back(option);
    arguments.push_back(background);
    arguments.push_back(nullptr);
    return venus_gpu::driver::run_production_task(
        argc + 2, arguments.data(),
        venus_gpu::driver::ProductionTask::vlasov_poisson);
}
