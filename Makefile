# Build entry points for the combined CPU/GPU release and manual.
# Inputs are the bundled source trees and optional compiler/architecture overrides.
# Targets create executables under cpu/ or build/, and the manual under notes/.

CPU_DIR := cpu
GPU_DIR := gpu
# Keep generated GPU build trees separate from the source. Command-line values override these defaults.
BUILD_DIR ?= build
GPU_ARCH ?= 80
BUILD_JOBS ?= 8
PYTHON ?= python3

.PHONY: help cpu cpu-openmp cpu-mpi cpu-all cpu-compat cpu-tools cpu-check-setup \
        python-check script-check gpu-configure gpu \
        gpu-mpi-configure gpu-mpi manual clean

help:
	@printf '%s\n' \
	  'VENUS 1.0 targets:' \
	  '  cpu                         build the portable task-specific CPU solvers' \
	  '  cpu-openmp / cpu-mpi        build supported task-specific parallel variants' \
	  '  cpu-all                     build every supported CPU task variant' \
	  '  cpu-compat                  build the generic compatibility dispatchers' \
	  '  cpu-tools                   compile every maintained C++ CPU tool' \
	  '  cpu-check-setup             run packaged zero-time CPU setup checks' \
	  '  gpu                         configure and build one-GPU targets' \
	  '  gpu-mpi                     build one-rank-per-GPU targets' \
	  '  python-check                syntax-check the maintained Python tools' \
	  '  script-check                syntax-check shell, Slurm, and Node scripts' \
	  '  manual                      compile the unified PDF manual'

# Delegate CPU builds to the task-specific application targets in cpu/Makefile.
cpu:
	$(MAKE) -C $(CPU_DIR) organized-solvers

cpu-openmp:
	$(MAKE) -C $(CPU_DIR) cpu-task-solvers-openmp

cpu-mpi:
	$(MAKE) -C $(CPU_DIR) cpu-task-solvers-mpi

cpu-all:
	$(MAKE) -C $(CPU_DIR) cpu-task-solvers-all
	$(MAKE) -C $(CPU_DIR) wpic-task-runners

cpu-compat:
	$(MAKE) -C $(CPU_DIR) serial

cpu-tools:
	$(MAKE) -C $(CPU_DIR) all-tools

cpu-check-setup:
	$(MAKE) -C $(CPU_DIR) check-setup

# Parse the Python tools and shell launchers independently of any scientific run.
python-check:
	$(PYTHON) -m compileall -q $(CPU_DIR)/python $(CPU_DIR)/tools $(GPU_DIR)/scripts $(GPU_DIR)/tools

script-check:
	bash scripts/check_shell_syntax.sh

# Pass the GPU architecture and CPU reference headers into an out-of-source Release configuration.
gpu-configure:
	cmake -S $(GPU_DIR) -B "$(BUILD_DIR)/gpu" \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_CUDA_ARCHITECTURES="$(GPU_ARCH)" \
	  -DVENUS_CPU_REFERENCE_INCLUDE="$(CURDIR)/$(CPU_DIR)/include"

gpu: gpu-configure
	cmake --build "$(BUILD_DIR)/gpu" -j $(BUILD_JOBS)

# The distributed build adds one MPI process per GPU while using the same numerical source.
gpu-mpi-configure:
	cmake -S $(GPU_DIR) -B "$(BUILD_DIR)/gpu-mpi" \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_CUDA_ARCHITECTURES="$(GPU_ARCH)" \
	  -DVENUS_GPU_ENABLE_MPI=ON \
	  -DVENUS_CPU_REFERENCE_INCLUDE="$(CURDIR)/$(CPU_DIR)/include"

gpu-mpi: gpu-mpi-configure
	cmake --build "$(BUILD_DIR)/gpu-mpi" -j $(BUILD_JOBS)

# Compile the pedagogical manual with its own reference and build-log checks.
manual:
	$(MAKE) -C notes manual

# Remove named generated binaries and build artifacts from these configured output locations.
clean:
	$(MAKE) -C $(CPU_DIR) clean
	rm -rf $(BUILD_DIR)/gpu $(BUILD_DIR)/gpu-mpi
	rm -f notes/venus_manual.aux notes/venus_manual.fdb_latexmk
	rm -f notes/venus_manual.fls notes/venus_manual.log
	rm -f notes/venus_manual.out notes/venus_manual.toc
