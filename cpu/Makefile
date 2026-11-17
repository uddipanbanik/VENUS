# Build CPU solvers, task-specific launchers and analysis utilities.
# Override CXX/MPICXX and flags for the local compiler. Targets share headers from include/ev.
# Compilation creates binaries; only the explicitly named run/check-setup targets start executables.

CXX ?= g++
MPICXX ?= mpicxx
PYTHON ?= python3

# Section: executable names and source files.
# Task-specific venus_cpu_* binaries below are the preferred interfaces.  The
# standalone EV source and venus_modular_solver remain generic dispatchers for
# configurations that intentionally select the physics model at run time.
EV_TARGET := venus_einstein_vlasov
EV_MPI_TARGET := venus_einstein_vlasov_mpi
EV_SOURCE := src/venus_einstein_vlasov.cpp
MOD_TARGET := venus_modular_solver
MOD_MPI_TARGET := venus_modular_solver_mpi
MOD_SOURCE := src/venus_modular_solver.cpp
SIDM_WPIC_COMPAT_TARGET := run_weighted_vlasov_pic_frank_collisional
SIDM_WPIC_COMPAT_SOURCE := tools/run_weighted_vlasov_pic_frank_collisional.cpp
SIDM_WPIC_DRIVER_SOURCE := src/drivers/sidm_wpic.cpp
EV_WPIC_DRIVER_SOURCE := src/drivers/einstein_vlasov_wpic.cpp
EXPANDING_WPIC_COMPAT_TARGET := run_weighted_vlasov_pic_expanding_powerlaw
EXPANDING_WPIC_COMPAT_SOURCE := tools/run_weighted_vlasov_pic_expanding_powerlaw.cpp
EXPANDING_WPIC_DRIVER_SOURCE := src/drivers/expanding_wpic.cpp
# Track shared headers as prerequisites so changing a numerical definition rebuilds its applications.
MODULE_HEADERS := $(wildcard include/ev/*.hpp include/ev/apps/*.hpp include/ev/relativistic_boltzmann/*.hpp)
CPU_TASK_SOLVER_TARGETS := \
	venus_cpu_einstein_vlasov_eulerian \
	venus_cpu_vlasov_poisson_eulerian \
	venus_cpu_boltzmann_poisson_fokker_planck \
	venus_cpu_boltzmann_poisson_landau \
	venus_cpu_boltzmann_poisson_finite_angle \
	venus_cpu_boltzmann_poisson_exact \
	venus_cpu_boltzmann_poisson_reversible \
	venus_cpu_boltzmann_poisson_reversible_sampled \
	venus_cpu_boltzmann_poisson_reversible_streamed \
	venus_cpu_boltzmann_poisson_canonical_orbit \
	venus_cpu_boltzmann_poisson_expanding_prompt_cusp \
	venus_cpu_sidm_sphdsmc_constant \
	venus_cpu_sidm_sphdsmc_velocity_dependent
# Create explicit OpenMP names. The MPI list admits the collisionless Eulerian EV and VP applications.
CPU_TASK_SOLVER_OPENMP_TARGETS := $(addsuffix _openmp,$(CPU_TASK_SOLVER_TARGETS))
CPU_TASK_SOLVER_MPI_TARGETS := \
	venus_cpu_einstein_vlasov_eulerian_mpi \
	venus_cpu_vlasov_poisson_eulerian_mpi
WPIC_TASK_RUNNER_TARGETS := \
	venus_cpu_einstein_vlasov_wpic_low_compactness \
	venus_cpu_einstein_vlasov_wpic_black_hole \
	venus_cpu_vlasov_poisson_wpic_static \
	venus_cpu_vlasov_poisson_wpic_expanding \
	venus_cpu_subhalo_wpic \
	venus_cpu_sidm_wpic_constant \
	venus_cpu_sidm_wpic_velocity_dependent
# Discover maintained standalone C++ tools and derive their executable/link-check names.
TOOL_SOURCES := $(sort $(wildcard tools/*.cpp))
TOOL_TARGETS := $(patsubst tools/%.cpp,%,$(TOOL_SOURCES))
TOOL_CHECK_DIR := build/toolcheck
TOOL_CHECK_TARGETS := $(patsubst tools/%.cpp,$(TOOL_CHECK_DIR)/%,$(TOOL_SOURCES))
DRIVER_TOOL_TARGETS := \
	$(SIDM_WPIC_COMPAT_TARGET) \
	$(EXPANDING_WPIC_COMPAT_TARGET)
DRIVER_TOOL_CHECK_TARGETS := $(addprefix $(TOOL_CHECK_DIR)/,$(DRIVER_TOOL_TARGETS))
GENERIC_TOOL_CHECK_TARGETS := $(filter-out $(DRIVER_TOOL_CHECK_TARGETS),$(TOOL_CHECK_TARGETS))
SPECIAL_TOOL_TARGETS := \
	$(SIDM_WPIC_COMPAT_TARGET) \
	$(EXPANDING_WPIC_COMPAT_TARGET)
GENERIC_TOOL_TARGETS := $(filter-out $(SPECIAL_TOOL_TARGETS),$(TOOL_TARGETS))
TOOL_IN_TREE_BINARIES := $(addprefix tools/,$(TOOL_TARGETS))
# Section: default example configurations.
# Override CONFIG or MOD_CONFIG for local zero-time setup checks.
CONFIG ?= examples/einstein_vlasov_high_compactness_horizon_formation.ini
MOD_CONFIG ?= examples/modular/vp_expanding_nfw_gamma2_collisionless.ini
# Paths are relative to cpu/, matching the release-wide output layout.
SETUP_OUTPUT_ROOT ?= ../output/cpu

# Section: compiler flags.
# OPENMP_FLAGS is intentionally separate so clusters can override it when their
# compiler wrapper needs a site-specific OpenMP flag or runtime library.
BASE_CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic -ffp-contract=off
DEBUG_CXXFLAGS ?= -std=c++17 -O0 -g -Wall -Wextra -Wpedantic
OPENMP_FLAGS ?= -fopenmp
PTHREAD_FLAGS ?= -pthread
EXTRA_LDLIBS ?=

.PHONY: all help mpi openmp serial debug debug-openmp modular modular-mpi tools all-tools weighted-pic-frank cpu-task-solvers cpu-task-solvers-openmp cpu-task-solvers-mpi cpu-task-solvers-all task-solvers wpic-task-runners organized-solvers run run-mpi run-modular check-setup check-setup-modular clean

# Section: default and MPI builds.
all: serial organized-solvers

mpi: $(EV_MPI_TARGET) $(MOD_MPI_TARGET) $(CPU_TASK_SOLVER_MPI_TARGETS)

$(EV_MPI_TARGET): $(EV_SOURCE)
	$(MPICXX) $(BASE_CXXFLAGS) $(OPENMP_FLAGS) -DEV_USE_MPI $< -o $@ $(OPENMP_FLAGS) $(EXTRA_LDLIBS)

$(MOD_MPI_TARGET): $(MOD_SOURCE) $(wildcard include/ev/*.hpp)
	$(MPICXX) $(BASE_CXXFLAGS) $(OPENMP_FLAGS) -DEV_USE_MPI -Iinclude $< -o $@ $(OPENMP_FLAGS) $(EXTRA_LDLIBS)

openmp: $(EV_TARGET) $(MOD_TARGET) $(CPU_TASK_SOLVER_OPENMP_TARGETS)

help:
	@echo "Preferred portable task builds: make cpu-task-solvers"
	@echo "Preferred WPIC task builds:     make wpic-task-runners"
	@echo "All portable task fronts:       make organized-solvers"
	@echo "OpenMP grid/SphDSMC fronts:     make cpu-task-solvers-openmp"
	@echo "MPI+OpenMP EV/VP fronts only:   make cpu-task-solvers-mpi"
	@echo "All supported variants:         make cpu-task-solvers-all"
	@echo "Compatibility dispatchers:      make serial | openmp | mpi"
	@echo "BP and SphDSMC intentionally have no MPI task targets."

$(EV_TARGET): $(EV_SOURCE)
	$(CXX) $(BASE_CXXFLAGS) $(OPENMP_FLAGS) $< -o $@ $(OPENMP_FLAGS) $(EXTRA_LDLIBS)

modular: $(MOD_TARGET)

modular-mpi: $(MOD_MPI_TARGET)

$(MOD_TARGET): $(MOD_SOURCE) $(wildcard include/ev/*.hpp)
	$(CXX) $(BASE_CXXFLAGS) $(OPENMP_FLAGS) -Iinclude $< -o $@ $(OPENMP_FLAGS) $(EXTRA_LDLIBS)

# Section: fail-closed task-specific CPU applications.
# These tiny mains share every numerical kernel with venus_modular_solver but
# reject a mismatched equation, collision operator, or cross section before
# state/output construction.  The generic targets above remain available for
# configurations that intentionally select a model dynamically.
cpu-task-solvers task-solvers: $(CPU_TASK_SOLVER_TARGETS)

# Unsuffixed task binaries use the portable std::thread backend.  Every task
# also has an explicit _openmp variant.  Only collisionless Eulerian EV and VP
# receive _mpi variants because BP/SphDSMC collision state is not decomposed.
cpu-task-solvers-openmp: $(CPU_TASK_SOLVER_OPENMP_TARGETS)

cpu-task-solvers-mpi: $(CPU_TASK_SOLVER_MPI_TARGETS)

cpu-task-solvers-all: cpu-task-solvers cpu-task-solvers-openmp cpu-task-solvers-mpi

# Generate portable-thread and OpenMP compile rules from the same application source.
define define_cpu_task_solver
$(1): $(2) $$(MODULE_HEADERS)
	$$(CXX) $$(BASE_CXXFLAGS) $$(PTHREAD_FLAGS) -Iinclude $$< -o $$@ $$(EXTRA_LDLIBS)

$(1)_openmp: $(2) $$(MODULE_HEADERS)
	$$(CXX) $$(BASE_CXXFLAGS) $$(OPENMP_FLAGS) -Iinclude $$< -o $$@ $$(OPENMP_FLAGS) $$(EXTRA_LDLIBS)
endef

$(eval $(call define_cpu_task_solver,venus_cpu_einstein_vlasov_eulerian,src/apps/einstein_vlasov/eulerian.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_vlasov_poisson_eulerian,src/apps/vlasov_poisson/eulerian.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_boltzmann_poisson_fokker_planck,src/apps/boltzmann_poisson/fokker_planck.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_boltzmann_poisson_landau,src/apps/boltzmann_poisson/landau.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_boltzmann_poisson_finite_angle,src/apps/boltzmann_poisson/finite_angle.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_boltzmann_poisson_exact,src/apps/boltzmann_poisson/exact.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_boltzmann_poisson_reversible,src/apps/boltzmann_poisson/reversible.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_boltzmann_poisson_reversible_sampled,src/apps/boltzmann_poisson/reversible_sampled.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_boltzmann_poisson_reversible_streamed,src/apps/boltzmann_poisson/reversible_streamed.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_boltzmann_poisson_canonical_orbit,src/apps/boltzmann_poisson/canonical_orbit.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_boltzmann_poisson_expanding_prompt_cusp,src/apps/boltzmann_poisson/expanding_prompt_cusp.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_sidm_sphdsmc_constant,src/apps/sidm/sphdsmc_constant.cpp))
$(eval $(call define_cpu_task_solver,venus_cpu_sidm_sphdsmc_velocity_dependent,src/apps/sidm/sphdsmc_velocity_dependent.cpp))

venus_cpu_einstein_vlasov_eulerian_mpi: src/apps/einstein_vlasov/eulerian.cpp $(MODULE_HEADERS)
	$(MPICXX) $(BASE_CXXFLAGS) $(OPENMP_FLAGS) -DEV_USE_MPI -Iinclude $< -o $@ $(OPENMP_FLAGS) $(EXTRA_LDLIBS)

venus_cpu_vlasov_poisson_eulerian_mpi: src/apps/vlasov_poisson/eulerian.cpp $(MODULE_HEADERS)
	$(MPICXX) $(BASE_CXXFLAGS) $(OPENMP_FLAGS) -DEV_USE_MPI -Iinclude $< -o $@ $(OPENMP_FLAGS) $(EXTRA_LDLIBS)

# Task-specific WPIC binaries compile the shared numerical drivers with a
# lightweight profile contract.  No solver source or numerical kernel is
# copied.  Compatibility run_* commands retain their combined interfaces.
wpic-task-runners: $(WPIC_TASK_RUNNER_TARGETS)

organized-solvers: cpu-task-solvers wpic-task-runners

# Each weighted-particle application links a small model-specific main to its shared driver.
venus_cpu_einstein_vlasov_wpic_low_compactness: src/apps/einstein_vlasov/wpic_low_compactness.cpp $(EV_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -DVENUS_CPU_EV_WPIC_LOW_PROFILE -Iinclude src/apps/einstein_vlasov/wpic_low_compactness.cpp $(EV_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

venus_cpu_einstein_vlasov_wpic_black_hole: src/apps/einstein_vlasov/wpic_black_hole.cpp $(EV_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -DVENUS_CPU_EV_WPIC_BLACK_HOLE_PROFILE -Iinclude src/apps/einstein_vlasov/wpic_black_hole.cpp $(EV_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

venus_cpu_vlasov_poisson_wpic_static: src/apps/vlasov_poisson/static_wpic.cpp $(EXPANDING_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -DVENUS_CPU_STATIC_WPIC_VP_PROFILE -DVENUS_CPU_EXPANDING_WPIC_VP_PROFILE -Iinclude src/apps/vlasov_poisson/static_wpic.cpp $(EXPANDING_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

venus_cpu_vlasov_poisson_wpic_expanding: src/apps/vlasov_poisson/expanding_wpic.cpp $(EXPANDING_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -DVENUS_CPU_EXPANDING_WPIC_VP_PROFILE -Iinclude src/apps/vlasov_poisson/expanding_wpic.cpp $(EXPANDING_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

venus_cpu_subhalo_wpic: src/apps/subhalo/expanding_wpic.cpp $(EXPANDING_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -DVENUS_CPU_EXPANDING_WPIC_SUBHALO_PROFILE -Iinclude src/apps/subhalo/expanding_wpic.cpp $(EXPANDING_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

venus_cpu_sidm_wpic_constant: src/apps/sidm/wpic_constant.cpp $(SIDM_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -DVENUS_CPU_SIDM_WPIC_CONSTANT_PROFILE -Iinclude src/apps/sidm/wpic_constant.cpp $(SIDM_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

venus_cpu_sidm_wpic_velocity_dependent: src/apps/sidm/wpic_velocity_dependent.cpp $(SIDM_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -DVENUS_CPU_SIDM_WPIC_VELOCITY_DEPENDENT_PROFILE -Iinclude src/apps/sidm/wpic_velocity_dependent.cpp $(SIDM_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

weighted-pic-frank: $(SIDM_WPIC_COMPAT_TARGET)

$(SIDM_WPIC_COMPAT_TARGET): $(SIDM_WPIC_COMPAT_SOURCE) $(SIDM_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(OPENMP_FLAGS) -Iinclude $(SIDM_WPIC_COMPAT_SOURCE) $(SIDM_WPIC_DRIVER_SOURCE) -o $@ $(OPENMP_FLAGS) $(EXTRA_LDLIBS)

$(EXPANDING_WPIC_COMPAT_TARGET): $(EXPANDING_WPIC_COMPAT_SOURCE) $(EXPANDING_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -Iinclude $(EXPANDING_WPIC_COMPAT_SOURCE) $(EXPANDING_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

# Compile and link every maintained C++ utility into an isolated link-check
# directory.  This portable pthread check also works on Apple Clang hosts that
# do not provide OpenMP; the dedicated performance targets above retain their
# original OpenMP or pthread flags.
tools all-tools: $(TOOL_CHECK_TARGETS)

# Compile utility executables into build/toolcheck, keeping link checks separate from source files.
$(GENERIC_TOOL_CHECK_TARGETS): $(TOOL_CHECK_DIR)/%: tools/%.cpp $(MODULE_HEADERS)
	@mkdir -p $(TOOL_CHECK_DIR)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -Iinclude $< -o $@ $(EXTRA_LDLIBS)

$(TOOL_CHECK_DIR)/$(SIDM_WPIC_COMPAT_TARGET): $(SIDM_WPIC_COMPAT_SOURCE) $(SIDM_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	@mkdir -p $(TOOL_CHECK_DIR)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -Iinclude $(SIDM_WPIC_COMPAT_SOURCE) $(SIDM_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

$(TOOL_CHECK_DIR)/$(EXPANDING_WPIC_COMPAT_TARGET): $(EXPANDING_WPIC_COMPAT_SOURCE) $(EXPANDING_WPIC_DRIVER_SOURCE) $(MODULE_HEADERS)
	@mkdir -p $(TOOL_CHECK_DIR)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -Iinclude $(EXPANDING_WPIC_COMPAT_SOURCE) $(EXPANDING_WPIC_DRIVER_SOURCE) -o $@ $(EXTRA_LDLIBS)

$(GENERIC_TOOL_TARGETS): %: tools/%.cpp $(MODULE_HEADERS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -Iinclude $< -o $@ $(EXTRA_LDLIBS)

# Section: serial/debug builds.
# The serial targets use std::thread fallback parallelism and are useful on
# laptops or login nodes where OpenMP is not available.
serial: $(EV_SOURCE) $(MOD_SOURCE)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) $(EV_SOURCE) -o $(EV_TARGET)_serial $(EXTRA_LDLIBS)
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -Iinclude $(MOD_SOURCE) -o $(MOD_TARGET)_serial $(EXTRA_LDLIBS)

debug: $(EV_SOURCE) $(MOD_SOURCE)
	$(CXX) $(DEBUG_CXXFLAGS) $(PTHREAD_FLAGS) $(EV_SOURCE) -o $(EV_TARGET)_debug $(EXTRA_LDLIBS)
	$(CXX) $(DEBUG_CXXFLAGS) $(PTHREAD_FLAGS) -Iinclude $(MOD_SOURCE) -o $(MOD_TARGET)_debug $(EXTRA_LDLIBS)

debug-openmp: $(EV_SOURCE) $(MOD_SOURCE)
	$(CXX) $(DEBUG_CXXFLAGS) $(OPENMP_FLAGS) $(EV_SOURCE) -o $(EV_TARGET)_debug $(OPENMP_FLAGS) $(EXTRA_LDLIBS)
	$(CXX) $(DEBUG_CXXFLAGS) $(OPENMP_FLAGS) -Iinclude $(MOD_SOURCE) -o $(MOD_TARGET)_debug $(OPENMP_FLAGS) $(EXTRA_LDLIBS)

# Section: local run helpers.
run: $(EV_TARGET)
	./$(EV_TARGET) --config $(CONFIG)

run-mpi: $(EV_MPI_TARGET)
	mpirun -np 2 ./$(EV_MPI_TARGET) --config $(CONFIG)

# The zero-time setup runs validate initialization and output construction for two sample configurations.
check-setup: serial
	./$(EV_TARGET)_serial --config $(CONFIG) --t-end 0 --output-dir "$(SETUP_OUTPUT_ROOT)/setup_serial" --threads 2
	./$(MOD_TARGET)_serial --config $(MOD_CONFIG) --t-end 0 --output-dir "$(SETUP_OUTPUT_ROOT)/setup_modular" --threads 2

run-modular: $(MOD_TARGET)
	./$(MOD_TARGET) --config $(MOD_CONFIG)

check-setup-modular: serial
	./$(MOD_TARGET)_serial --config $(MOD_CONFIG) --t-end 0 --output-dir "$(SETUP_OUTPUT_ROOT)/setup_modular" --threads 2

# Section: cleanup.
clean:
	rm -f $(EV_TARGET) $(EV_MPI_TARGET) $(EV_TARGET)_serial $(EV_TARGET)_debug
	rm -f $(MOD_TARGET) $(MOD_MPI_TARGET) $(MOD_TARGET)_serial $(MOD_TARGET)_debug
	rm -f $(CPU_TASK_SOLVER_TARGETS)
	rm -f $(CPU_TASK_SOLVER_OPENMP_TARGETS)
	rm -f $(CPU_TASK_SOLVER_MPI_TARGETS)
	rm -f $(WPIC_TASK_RUNNER_TARGETS)
	rm -f $(TOOL_TARGETS)
	rm -f $(TOOL_IN_TREE_BINARIES)
	rm -f $(TOOL_CHECK_TARGETS)
	rm -rf $(addsuffix .dSYM,$(TOOL_TARGETS))
	rm -rf $(addsuffix .dSYM,$(CPU_TASK_SOLVER_TARGETS))
	rm -rf $(addsuffix .dSYM,$(CPU_TASK_SOLVER_OPENMP_TARGETS))
	rm -rf $(addsuffix .dSYM,$(CPU_TASK_SOLVER_MPI_TARGETS))
	rm -rf $(addsuffix .dSYM,$(WPIC_TASK_RUNNER_TARGETS))
	rm -rf $(EV_TARGET).dSYM $(EV_MPI_TARGET).dSYM $(EV_TARGET)_serial.dSYM $(EV_TARGET)_debug.dSYM
	rm -rf $(MOD_TARGET).dSYM $(MOD_MPI_TARGET).dSYM $(MOD_TARGET)_serial.dSYM $(MOD_TARGET)_debug.dSYM
	find python tools -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	find python tools -type d -name __pycache__ -empty -delete
	-rmdir $(TOOL_CHECK_DIR) build 2>/dev/null

# Reproducible finite numerical checks, compiled independently of production binaries.
TEST_SOURCES := $(wildcard tests/test_*.cpp)
TEST_BINARIES := $(patsubst tests/%.cpp,build/tests/%,$(TEST_SOURCES))
.PHONY: check
build/tests/%: tests/%.cpp $(MODULE_HEADERS)
	@mkdir -p build/tests
	$(CXX) $(BASE_CXXFLAGS) $(PTHREAD_FLAGS) -Iinclude $< -o $@ $(EXTRA_LDLIBS)
check: $(TEST_BINARIES)
	@set -e; for test in $(TEST_BINARIES); do echo "Running $$test"; "$$test"; done
