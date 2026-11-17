# Configure CUDA libraries, model-specific applications and optional MPI/benchmark targets.
# Inputs are this GPU source tree, CUDA, Python and the CPU reference include directory.
# Generated physics-table headers and compiled targets are written to the chosen CMake build directory.

cmake_minimum_required(VERSION 3.18)

project(VENUS_GPU VERSION 1.0.0 LANGUAGES CXX CUDA)

# This directory builds the CUDA module inside the combined VENUS release.
# The GPU solver retains its own device state and implementation, while the
# bundled CPU include tree supplies the authoritative unequal-Plummer lookup
# table and the reference implementations used by parity tests/benchmarks.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Apollo's A100 nodes are sm_80.  This is a cache value so another cluster
  # can select its native architecture at configure time.
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA GPU architectures")
endif()

option(VENUS_GPU_BUILD_TESTS "Build numerical and driver tests" ON)
option(VENUS_GPU_BUILD_BENCHMARKS "Build the standalone GPU benchmark" ON)
option(VENUS_GPU_ENABLE_MPI "Build the one-MPI-rank-per-GPU layer" OFF)
set(VENUS_BUNDLED_CPU_REFERENCE_INCLUDE
  "${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include")
set(VENUS_CPU_REFERENCE_INCLUDE
  "${VENUS_BUNDLED_CPU_REFERENCE_INCLUDE}" CACHE PATH
  "VENUS CPU include path for parity targets and the authoritative subhalo table")
if(VENUS_CPU_REFERENCE_INCLUDE STREQUAL "")
  # An empty include setting selects the bundled CPU reference tree, which
  # supplies the Plummer encounter table and host parity implementations.
  set(VENUS_CPU_REFERENCE_INCLUDE
    "${VENUS_BUNDLED_CPU_REFERENCE_INCLUDE}" CACHE PATH
    "VENUS CPU include path for parity targets and the authoritative subhalo table"
    FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# The core library owns device state, restart I/O, gravity, collisions and receiver reductions.
add_library(venus_gpu STATIC
  src/wpic_restart.cpp
  src/wpic_state.cu
  src/wpic_vlasov_poisson.cu
  src/wpic_collisions.cu
  src/receiver_sums_kernel.cu
)
add_library(VENUS::gpu ALIAS venus_gpu)
target_include_directories(venus_gpu PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${VENUS_BUNDLED_CPU_REFERENCE_INCLUDE})
target_link_libraries(venus_gpu PUBLIC CUDA::cudart)
target_compile_options(venus_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
# Separate CUDA compilation resolves device symbols across these source files at link time.
set_target_properties(venus_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

# Host-only task output contract.  It is a separate target so CSV behavior can
# be tested on a machine with no visible CUDA device.
add_library(venus_gpu_production_output STATIC
  src/wpic_production_output.cpp)
target_include_directories(venus_gpu_production_output PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(venus_gpu_production_output PUBLIC cxx_std_17)

# The table generator uses the release-wide Python >=3.8 contract.  State the
# floor here so CMake fails during configuration with a clear diagnostic instead
# of selecting an older interpreter and failing later in the custom command.
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
# Use the CPU unequal-Plummer table as the numerical source for the generated flat CUDA arrays.
set(VENUS_SUBHALO_TABLE_SOURCE
  "${VENUS_CPU_REFERENCE_INCLUDE}/ev/weighted_vlasov_pic_subhalo_kernel_table.hpp")
if(NOT EXISTS "${VENUS_SUBHALO_TABLE_SOURCE}")
  message(FATAL_ERROR
    "Missing the VENUS CPU subhalo table: ${VENUS_SUBHALO_TABLE_SOURCE}\n"
    "Set VENUS_CPU_REFERENCE_INCLUDE to a compatible VENUS CPU include tree.")
endif()
set(VENUS_GENERATED_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(VENUS_GENERATED_SUBHALO_TABLE
  "${VENUS_GENERATED_INCLUDE}/venus_gpu_generated_subhalo_table.hpp")
set(VENUS_SUBHALO_TABLE_GENERATOR
  "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cuda_subhalo_table.py")

# Generate at build time, not configure time.  Changes to either the CPU table
# or the converter invalidate the output and rebuild the CUDA source.
add_custom_command(
  OUTPUT "${VENUS_GENERATED_SUBHALO_TABLE}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${VENUS_GENERATED_INCLUDE}"
  COMMAND ${Python3_EXECUTABLE}
    "${VENUS_SUBHALO_TABLE_GENERATOR}"
    "${VENUS_SUBHALO_TABLE_SOURCE}"
    "${VENUS_GENERATED_SUBHALO_TABLE}"
  DEPENDS
    "${VENUS_SUBHALO_TABLE_GENERATOR}"
    "${VENUS_SUBHALO_TABLE_SOURCE}"
  COMMENT "Generating the VENUS CUDA unequal-subhalo table"
  VERBATIM)
add_custom_target(venus_gpu_generate_subhalo_table
  DEPENDS "${VENUS_GENERATED_SUBHALO_TABLE}")
# Finish table generation before compiling the CUDA source that includes it.
add_dependencies(venus_gpu venus_gpu_generate_subhalo_table)
set_source_files_properties("${VENUS_GENERATED_SUBHALO_TABLE}"
  PROPERTIES GENERATED TRUE)
target_sources(venus_gpu PRIVATE
  src/wpic_subhalo_collisions.cu
  "${VENUS_GENERATED_SUBHALO_TABLE}")
target_include_directories(venus_gpu PRIVATE "${VENUS_GENERATED_INCLUDE}")
target_compile_definitions(venus_gpu PUBLIC VENUS_GPU_HAS_SUBHALO=1)

# Add MPI communication around the same GPU library when the option is enabled.
if(VENUS_GPU_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_library(venus_gpu_mpi STATIC src/wpic_mpi.cpp)
  add_library(VENUS::gpu_mpi ALIAS venus_gpu_mpi)
  target_include_directories(venus_gpu_mpi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(venus_gpu_mpi PUBLIC venus_gpu MPI::MPI_CXX)
  target_compile_features(venus_gpu_mpi PUBLIC cxx_std_14)
endif()


if(VENUS_GPU_BUILD_TESTS)
  enable_testing()

  # Pure host/device pair-space arithmetic is testable without a GPU.
  add_executable(test_collision_matching tests/test_collision_matching.cpp)
  target_include_directories(test_collision_matching PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME collision_matching COMMAND test_collision_matching)

  # Restart serialization is host-only, so this regression can run even when
  # the configured test node has no visible GPU.
  add_executable(test_wpic_restart
    tests/test_wpic_restart.cpp
    src/wpic_restart.cpp)
  target_include_directories(test_wpic_restart PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME wpic_restart COMMAND test_wpic_restart)

  add_executable(test_production_output tests/test_production_output.cpp)
  target_link_libraries(test_production_output PRIVATE
    venus_gpu_production_output stdc++fs)
  target_compile_features(test_production_output PRIVATE cxx_std_17)
  add_test(NAME production_output COMMAND test_production_output)

  add_executable(test_wpic_vlasov_poisson tests/test_wpic_vlasov_poisson.cu)
  target_include_directories(test_wpic_vlasov_poisson PRIVATE
    ${VENUS_BUNDLED_CPU_REFERENCE_INCLUDE})
  target_link_libraries(test_wpic_vlasov_poisson PRIVATE venus_gpu)
  add_test(NAME wpic_vlasov_poisson COMMAND test_wpic_vlasov_poisson)

  add_executable(test_wpic_collisions tests/test_wpic_collisions.cu)
  target_link_libraries(test_wpic_collisions PRIVATE venus_gpu)
  add_test(NAME wpic_collisions COMMAND test_wpic_collisions)

  if(VENUS_CPU_REFERENCE_INCLUDE)
    add_executable(test_wpic_vlasov_poisson_cpu_parity
      tests/test_wpic_vlasov_poisson_cpu_parity.cu)
    target_include_directories(test_wpic_vlasov_poisson_cpu_parity PRIVATE
      ${VENUS_CPU_REFERENCE_INCLUDE})
    target_link_libraries(test_wpic_vlasov_poisson_cpu_parity PRIVATE
      venus_gpu stdc++fs)
    target_compile_features(test_wpic_vlasov_poisson_cpu_parity PRIVATE
      cxx_std_17)
    set_target_properties(test_wpic_vlasov_poisson_cpu_parity PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME wpic_vlasov_poisson_cpu_parity
      COMMAND test_wpic_vlasov_poisson_cpu_parity)

    add_executable(test_wpic_subhalo_collisions
      tests/test_wpic_subhalo_collisions.cu)
    target_include_directories(test_wpic_subhalo_collisions PRIVATE
      ${VENUS_CPU_REFERENCE_INCLUDE})
    target_link_libraries(test_wpic_subhalo_collisions PRIVATE venus_gpu)
    target_compile_features(test_wpic_subhalo_collisions PRIVATE cxx_std_17)
    set_target_properties(test_wpic_subhalo_collisions PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
    add_test(NAME wpic_subhalo_collisions COMMAND test_wpic_subhalo_collisions)

    add_executable(test_material_physics_cpu_parity
      tests/test_material_physics_cpu_parity.cpp)
    target_include_directories(test_material_physics_cpu_parity PRIVATE
      ${VENUS_CPU_REFERENCE_INCLUDE})
    target_link_libraries(test_material_physics_cpu_parity PRIVATE venus_gpu)
    target_compile_features(test_material_physics_cpu_parity PRIVATE cxx_std_17)
    add_test(NAME material_physics_cpu_parity
      COMMAND test_material_physics_cpu_parity)
  endif()

  if(VENUS_GPU_ENABLE_MPI)
    add_executable(test_wpic_mpi tests/test_wpic_mpi.cu)
    target_link_libraries(test_wpic_mpi PRIVATE venus_gpu_mpi)
    add_test(NAME wpic_mpi_2rank
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_wpic_mpi>
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(wpic_mpi_2rank PROPERTIES PROCESSORS 2)
  endif()
endif()


# The restart comparison tool links only restart I/O and can inspect stored states on the host.
add_executable(compare_wpic_restarts
  tools/compare_wpic_restarts.cpp
  src/wpic_restart.cpp)
target_include_directories(compare_wpic_restarts PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(VENUS_GPU_BUILD_BENCHMARKS)
  add_executable(benchmark_vlasov_poisson_gpu
    benchmarks/benchmark_vlasov_poisson_gpu.cu)
  target_link_libraries(benchmark_vlasov_poisson_gpu PRIVATE venus_gpu)
  add_executable(benchmark_material_sidm_gpu
    benchmarks/benchmark_material_sidm_gpu.cu)
  target_link_libraries(benchmark_material_sidm_gpu PRIVATE venus_gpu)
  # Compatibility aggregate: runs the two focused workloads sequentially.
  add_executable(benchmark_wpic benchmarks/benchmark_wpic.cu)
  target_link_libraries(benchmark_wpic PRIVATE venus_gpu)
  if(VENUS_GPU_ENABLE_MPI)
    add_executable(benchmark_wpic_mpi benchmarks/benchmark_wpic_mpi.cu)
    target_link_libraries(benchmark_wpic_mpi PRIVATE venus_gpu_mpi)
  endif()
  if(VENUS_CPU_REFERENCE_INCLUDE)
    find_package(Threads REQUIRED)
    add_executable(benchmark_wpic_cpu benchmarks/benchmark_wpic_cpu.cpp)
    target_include_directories(benchmark_wpic_cpu PRIVATE
      ${VENUS_CPU_REFERENCE_INCLUDE})
    target_link_libraries(benchmark_wpic_cpu PRIVATE Threads::Threads)
    target_compile_features(benchmark_wpic_cpu PRIVATE cxx_std_17)
  endif()
endif()

if(VENUS_CPU_REFERENCE_INCLUDE)
  # One numerical engine backs every task-specific executable.  The
  # tiny mains select a fail-closed parser contract; physics kernels are never
  # copied into task runners.
  add_library(venus_gpu_production_driver STATIC
    src/wpic_production_driver.cu)
  target_include_directories(venus_gpu_production_driver PRIVATE
    ${VENUS_BUNDLED_CPU_REFERENCE_INCLUDE})
  target_link_libraries(venus_gpu_production_driver PUBLIC
    venus_gpu venus_gpu_production_output stdc++fs)
  target_compile_features(venus_gpu_production_driver PUBLIC cxx_std_17)
  set_target_properties(venus_gpu_production_driver PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON)


  if(VENUS_GPU_BUILD_TESTS)
    add_executable(test_production_endpoint_moments tests/test_production_endpoint_moments.cu)
    target_link_libraries(test_production_endpoint_moments PRIVATE venus_gpu_production_driver)
    target_compile_features(test_production_endpoint_moments PRIVATE cxx_std_17)
    set_target_properties(test_production_endpoint_moments PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME production_endpoint_moments COMMAND test_production_endpoint_moments)
  endif()

  # Each task-specific main links the common driver, preserving a single implementation of evolution.
  function(venus_add_gpu_task target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE venus_gpu_production_driver)
    target_compile_features(${target} PRIVATE cxx_std_17)
    set_target_properties(${target} PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
  endfunction()

  venus_add_gpu_task(venus_gpu_vlasov_poisson_static
    tools/venus_gpu_vlasov_poisson_static.cu)
  venus_add_gpu_task(venus_gpu_vlasov_poisson
    tools/venus_gpu_vlasov_poisson.cu)
  venus_add_gpu_task(venus_gpu_subhalo_boltzmann_poisson
    tools/venus_gpu_subhalo_boltzmann_poisson.cu)
  venus_add_gpu_task(venus_gpu_sidm_boltzmann_poisson
    tools/venus_gpu_sidm_boltzmann_poisson.cu)
  venus_add_gpu_task(venus_gpu_sidm_constant
    tools/venus_gpu_sidm_constant.cu)
  venus_add_gpu_task(venus_gpu_sidm_power_law
    tools/venus_gpu_sidm_power_law.cu)
  venus_add_gpu_task(venus_gpu_sidm_yukawa_born
    tools/venus_gpu_sidm_yukawa_born.cu)

  if(VENUS_GPU_BUILD_TESTS)
    add_test(NAME production_driver_cli
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_production_driver_cli.py
        $<TARGET_FILE:venus_gpu_vlasov_poisson> $<TARGET_FILE:venus_gpu_subhalo_boltzmann_poisson>
        $<TARGET_FILE:venus_gpu_sidm_boltzmann_poisson> $<TARGET_FILE:venus_gpu_sidm_constant>
        $<TARGET_FILE:venus_gpu_vlasov_poisson_static>)
  endif()

  # Compatibility launchers retain their executable name and auto-select
  # collisionless/subhalo behavior from strength.  Task launchers use a
  # task-specific executable above.
  venus_add_gpu_task(run_wpic_subhalo_gpu
    tools/run_wpic_subhalo_gpu_compat.cu)

  if(VENUS_GPU_ENABLE_MPI)
    add_library(venus_gpu_production_driver_mpi STATIC
      src/wpic_production_driver.cu)
    target_include_directories(venus_gpu_production_driver_mpi PRIVATE
      ${VENUS_BUNDLED_CPU_REFERENCE_INCLUDE})
    target_compile_definitions(venus_gpu_production_driver_mpi PRIVATE
      VENUS_GPU_MPI_RUNNER=1)
    target_link_libraries(venus_gpu_production_driver_mpi PUBLIC
      venus_gpu_mpi venus_gpu_production_output stdc++fs)
    target_compile_features(venus_gpu_production_driver_mpi PUBLIC cxx_std_17)
    set_target_properties(venus_gpu_production_driver_mpi PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_SEPARABLE_COMPILATION ON)

    # Build the distributed application variants against the driver compiled with MPI state exchange.
    function(venus_add_gpu_mpi_task target source)
      add_executable(${target} ${source})
      target_link_libraries(${target} PRIVATE venus_gpu_production_driver_mpi)
      target_compile_features(${target} PRIVATE cxx_std_17)
      set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
    endfunction()

    # All three underlying MPI adapters exist: globally reduced VP,
    # radius-slab material collisions, and exact subhalo migration/cadence.
    venus_add_gpu_mpi_task(venus_gpu_vlasov_poisson_mpi
      tools/venus_gpu_vlasov_poisson.cu)
    venus_add_gpu_mpi_task(venus_gpu_subhalo_boltzmann_poisson_mpi
      tools/venus_gpu_subhalo_boltzmann_poisson.cu)
    venus_add_gpu_mpi_task(venus_gpu_sidm_boltzmann_poisson_mpi
      tools/venus_gpu_sidm_boltzmann_poisson.cu)
    venus_add_gpu_mpi_task(venus_gpu_sidm_constant_mpi
      tools/venus_gpu_sidm_constant.cu)
    venus_add_gpu_mpi_task(venus_gpu_sidm_power_law_mpi
      tools/venus_gpu_sidm_power_law.cu)
    venus_add_gpu_mpi_task(venus_gpu_sidm_yukawa_born_mpi
      tools/venus_gpu_sidm_yukawa_born.cu)
    venus_add_gpu_mpi_task(run_wpic_subhalo_gpu_mpi
      tools/run_wpic_subhalo_gpu_compat.cu)
  endif()

endif()
