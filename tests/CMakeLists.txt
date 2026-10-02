# ========== #
# Unit Tests #
# ========== #

enable_testing()

setup_test(test_eos)
setup_test(test_implicit_cfl)
setup_test(test_radiating_boundary)
setup_test(test_plm)
setup_test(test_weno)
setup_test(test_reconstruct)
setup_test(test_riemann)
setup_test(test_weno5_cuda_line)
setup_test(test_backward_substitution)
setup_test(test_scalar)
setup_test(test_internal_boundary)
setup_test(test_rectify)
setup_test(test_flip_zero_count)
setup_test(test_pull_2d)
setup_test(test_pull_3d)
setup_test(test_slab)
setup_test(test_cubed)
setup_test(test_cubed_sphere)
setup_test(test_cubed_sphere_exchange)
setup_test(test_coordinate)
setup_test(test_hydrostatic)
setup_test(test_hydro_options)
setup_test(test_cycle_diagnostics)
setup_test(test_flux_positivity_carry)
setup_test(test_yaml_keys)
setup_test(test_sedimentation_guards)
setup_test(test_balance_column)
setup_test(test_hydro_ref_x1)
setup_test(test_uranus_cycle1_abort)
setup_test(test_pref_local_seam)
setup_test(test_face_floor)
setup_test(test_output_dir)
setup_test(test_netcdf_utils)
setup_test(test_user_output)
setup_test(test_process_group)
setup_test(test_refine)
setup_test(test_diffusion)
setup_test(test_diffusion_moist)
setup_test(test_diffusion_x1_scale)
setup_test(test_condensate_conservation)
setup_test(test_parentless_cloud)
setup_test(test_fix_vapor_volume)
setup_test(test_parentless_cloud_nb1)
setup_test(test_vapor_column_nb1)
setup_test(test_wall_saturation)
setup_test(test_cloud_parent_slots)
setup_test(test_two_cards_species)
setup_test(test_forcing)
setup_parallel_test(test_exchange 2)
setup_parallel_test(test_output_barrier 2)
setup_parallel_test(test_check_redo_parallel 2)
if(APPLE)  # Gloo is the Mac backend; Linux runs UCX
  setup_parallel_test(test_gloo_one_tensor 2)
endif()
setup_parallel_test(test_sedimentation_cubed_seam 2)
# The same two ranks on Gloo, which Linux CI (UCX by default) would not reach:
# the limiter exchanges more than one variable, and a Gloo send takes one
# tensor. With UCX built, the Gloo result must equal the UCX one bit for bit.
set(seam_compare "")
if(UCX_FOUND)
  set(seam_compare --compare-ucx)
endif()
add_test(NAME test_sedimentation_cubed_seam_gloo
         COMMAND ${Python3_EXECUTABLE}
                 ${CMAKE_CURRENT_SOURCE_DIR}/run_seam_backends.py
                 --exe $<TARGET_FILE:test_sedimentation_cubed_seam.${buildl}>
                 ${seam_compare})
set_tests_properties(test_sedimentation_cubed_seam_gloo PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  SKIP_RETURN_CODE 125
  LABELS "exchange")

setup_parallel_test(test_parentless_cloud_nb1_mp 2)
if(UCX_FOUND)  # the same two ranks on Gloo, which Linux (UCX) would not reach
  add_test(NAME test_parentless_cloud_nb1_mp_gloo
           COMMAND torchrun --no-python --nproc-per-node=2
                   $<TARGET_FILE:test_parentless_cloud_nb1_mp.${buildl}>)
  set_tests_properties(test_parentless_cloud_nb1_mp_gloo PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    ENVIRONMENT "BACKEND=gloo")
endif()

if(UCX_FOUND)
  add_test(
    NAME test_exchange_ucx
    COMMAND torchrun
            --no-python
            --nproc-per-node=2
            $<TARGET_FILE:test_exchange.${buildl}>
  )
  set_tests_properties(test_exchange_ucx PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    ENVIRONMENT
      "BACKEND=ucx;BLOCKS_PER_PROCESS=3;EXPECT_LOCAL_NEIGHBOR=1;EXPECT_REMOTE_NEIGHBOR=1"
    LABELS "ucx;exchange"
  )

  if(CUDA)
    add_test(
      NAME test_exchange_ucx_cuda
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_exchange_backend.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type ${buildl}
              --backend ucx
              --device cuda
    )
    set_tests_properties(test_exchange_ucx_cuda PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      SKIP_RETURN_CODE 125
      LABELS "ucx;exchange;gpu"
    )
  endif()
endif()
# Built so they keep compiling against the current API, but DISABLED: each needs
# something the repo does not provide (see issue #224).
setup_test(test_read_topo)  # topography .pt file; no assertions
setup_test(test_aneos)      # real ANEOSThermoImpl (only a weak stub here) and example.aneos
set_tests_properties(test_read_topo.${buildl} test_aneos.${buildl} PROPERTIES DISABLED TRUE)

# Not registered, because they no longer compile or have nothing to run against:
#   test_thomas_solver.cu         TridiagSolverOptions, alloc_eigen5, forward_sweep_cuda are gone
#   test_transform.cpp            athena/, exo3/ headers
#   test_cs_velocity_rotation.cpp application/, athena/, exo3/ headers
#   test_implicit.cpp             globals.h, fvm/, namespace canoe
#   test_hydro.cpp                namespace canoe, flux_divergence
#   test_crater.py                canoe module, h5py and a local 256^3 HDF5 dump
#   test_robert.py                checker for two netCDF files passed on argv; no robert case

set_tests_properties(test_exchange.release PROPERTIES
  ENVIRONMENT
    "BLOCKS_PER_PROCESS=3;EXPECT_LOCAL_NEIGHBOR=1;EXPECT_REMOTE_NEIGHBOR=1"
)

add_test(
  NAME test_radiating_boundary_python
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/test_radiating_boundary_python.py
)
set_tests_properties(test_radiating_boundary_python PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  LABELS "python;boundary"
)
set_property(GLOBAL APPEND PROPERTY SNAPY_PYTHON_TESTS test_radiating_boundary_python)

# Every python ctest that imports snapy goes through this, so it is enrolled for
# SNAPY_TEST_PYTHONPATH below; the raw add_test ones drive a built binary and import none.
function(snapy_add_python_test name)
  cmake_parse_arguments(ARG "" "TIMEOUT" "LABELS;ARGS" ${ARGN})
  add_test(
    NAME ${name}_python
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py ${ARG_ARGS}
  )
  set_tests_properties(${name}_python PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    LABELS "${ARG_LABELS}"
  )
  if(ARG_TIMEOUT)
    set_tests_properties(${name}_python PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()
  set_property(GLOBAL APPEND PROPERTY SNAPY_PYTHON_TESTS ${name}_python)
endfunction()

snapy_add_python_test(test_mesh_exchange LABELS "python;exchange")
snapy_add_python_test(test_cubed_sphere_vertical_velocity_exchange
                      LABELS "python;exchange;cubed-sphere;diagnostic")
snapy_add_python_test(test_jit_user_forcing TIMEOUT 30 LABELS "python;forcing")
snapy_add_python_test(test_fix_vapor_reports_failure TIMEOUT 60 LABELS "python;eos;positivity")
snapy_add_python_test(test_fix_vapor_reports_failure_cuda TIMEOUT 60 LABELS "python;eos;positivity;cuda")
set_tests_properties(test_fix_vapor_reports_failure_cuda_python PROPERTIES SKIP_RETURN_CODE 125)
snapy_add_python_test(test_fix_vapor_counts_every_column TIMEOUT 120 LABELS "python;eos;positivity")
snapy_add_python_test(test_implicit_advection_cfl TIMEOUT 60 LABELS "python;hydro;implicit")
snapy_add_python_test(test_implicit_options_type TIMEOUT 60 LABELS "python;implicit")
snapy_add_python_test(test_shear_cfl TIMEOUT 60 LABELS "python;hydro;implicit")
snapy_add_python_test(test_check_redo_floor TIMEOUT 60 LABELS "python;mesh;redo")
snapy_add_python_test(test_check_redo_saturation TIMEOUT 60 LABELS "python;mesh;redo;eos")
snapy_add_python_test(test_check_redo_saturation_cuda TIMEOUT 60 LABELS "python;mesh;redo;eos;cuda")
set_tests_properties(test_check_redo_saturation_cuda_python PROPERTIES SKIP_RETURN_CODE 125)
snapy_add_python_test(test_eos_temp2inteng TIMEOUT 60 LABELS "python;eos")
snapy_add_python_test(test_flux_positivity TIMEOUT 300 LABELS "python;positivity")
snapy_add_python_test(test_flux_positivity_cubedsphere TIMEOUT 300
                      LABELS "python;positivity;cubed-sphere")
snapy_add_python_test(test_flux_positivity_cubedsphere_cuda TIMEOUT 300
                      LABELS "python;positivity;cubed-sphere;cuda")
set_tests_properties(test_flux_positivity_cubedsphere_cuda_python PROPERTIES
                     SKIP_RETURN_CODE 125)
snapy_add_python_test(test_flux_positivity_cubedsphere_moist TIMEOUT 300
                      LABELS "python;positivity;cubed-sphere")
snapy_add_python_test(test_flux_positivity_cubedsphere_moist_cuda TIMEOUT 300
                      LABELS "python;positivity;cubed-sphere;cuda")
set_tests_properties(test_flux_positivity_cubedsphere_moist_cuda_python PROPERTIES
                     SKIP_RETURN_CODE 125)
snapy_add_python_test(test_vic_moist_device TIMEOUT 120
                      LABELS "python;implicit;cuda")
set_tests_properties(test_vic_moist_device_python PROPERTIES SKIP_RETURN_CODE 125)
snapy_add_python_test(test_tracer_dry_convention TIMEOUT 120 LABELS "python;scalar;implicit;forcing")
snapy_add_python_test(test_eos_species_registry TIMEOUT 120 LABELS "python;eos")
snapy_add_python_test(test_stage_forcing_dry_tracer TIMEOUT 60 LABELS "python;forcing;scalar")
snapy_add_python_test(test_local_horizontal_cells TIMEOUT 60 LABELS "python;mesh;coordinate")
snapy_add_python_test(test_diffusion_x1_scale TIMEOUT 120 LABELS "python;forcing")

# DISABLED: snapy.exchange.init_dist still calls the removed SlabLayout(px, py, ...)
# constructor (see issue #224).
add_test(
  NAME test_exchange_python
  COMMAND torchrun --nproc-per-node=4 ${CMAKE_CURRENT_SOURCE_DIR}/test_exchange.py
)
set_tests_properties(test_exchange_python PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
  LABELS "python;exchange"
  DISABLED TRUE
)
set_property(GLOBAL APPEND PROPERTY SNAPY_PYTHON_TESTS test_exchange_python)

# Checks the SNAPY_TEST_PYTHONPATH wiring below. Its own PYTHONPATH stands in for the
# caller's and must survive behind SNAPY_TEST_PYTHONPATH (only checkable with CMake >= 3.22).
set(_marker "${CMAKE_CURRENT_BINARY_DIR}/caller_pythonpath_marker")
if(SNAPY_TEST_PYTHONPATH AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
  set(_import_path_args "${SNAPY_TEST_PYTHONPATH}" "${_marker}")
elseif(SNAPY_TEST_PYTHONPATH)
  set(_import_path_args "${SNAPY_TEST_PYTHONPATH}")
endif()
snapy_add_python_test(test_python_import_path LABELS "python" ARGS ${_import_path_args})
set_tests_properties(test_python_import_path_python PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_marker}"
  SKIP_RETURN_CODE 125
)

file(GLOB inputs *.yaml *.py)
# create tests folder if does not exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
foreach(input ${inputs})
  get_filename_component(input_name ${input} NAME)
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/tests/${input_name})
endforeach()

if (DEFINED KINTERA_DATA_DIR AND EXISTS "${KINTERA_DATA_DIR}/nasa9.dat")
  execute_process(
    COMMAND ln -sf ${KINTERA_DATA_DIR}/nasa9.dat
            ${CMAKE_BINARY_DIR}/tests/nasa9.dat)
endif()

# ---- Python ctests: say which snapy they test -------------------------------------------
# `COMMAND ${Python3_EXECUTABLE} <test>.py` imports `snapy` through sys.path, so with nothing
# set these exercise whatever snapy is installed, not the tree they were configured from --
# a false red when the branch is fine and the install is stale, and a false green when the
# branch is broken and the install is fine. cmake builds only the C++ libraries here (the
# importable package comes from `pip install`), so there is nothing in the build tree to point
# at by default; give the directory that holds this build's installed snapy:
#     cmake -B build -DSNAPY_TEST_PYTHONPATH=/path/to/site-packages
set(SNAPY_TEST_PYTHONPATH "" CACHE PATH
    "PYTHONPATH for the *_python ctests; empty means they test the installed snapy")
get_property(SNAPY_PYTHON_TESTS GLOBAL PROPERTY SNAPY_PYTHON_TESTS)
if(SNAPY_TEST_PYTHONPATH)
  # Prepend, do not replace: the caller's PYTHONPATH may be what selects the kintera/pyharp
  # this tree was configured against, and a test's own ENVIRONMENT must survive.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
    set_tests_properties(${SNAPY_PYTHON_TESTS} PROPERTIES ENVIRONMENT_MODIFICATION
                         "PYTHONPATH=path_list_prepend:${SNAPY_TEST_PYTHONPATH}")
  else()
    foreach(test IN LISTS SNAPY_PYTHON_TESTS)
      set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT
                   "PYTHONPATH=${SNAPY_TEST_PYTHONPATH}:$ENV{PYTHONPATH}")
    endforeach()
  endif()
  message(STATUS "python ctests import snapy from ${SNAPY_TEST_PYTHONPATH}")
else()
  message(WARNING
    "SNAPY_TEST_PYTHONPATH is unset: the *_python ctests will import the installed snapy, "
    "not this tree. Their verdicts do not constrain this branch.")
endif()

# ============= #
# Example Tests #
# ============= #
add_test(
  NAME test_straka
  COMMAND ${CMAKE_COMMAND}
          -Dbuildl=release
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_straka.cmake
)

set_tests_properties(test_straka PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

add_test(
  NAME test_shallow_xy
  COMMAND ${CMAKE_COMMAND}
          -Dbuildl=release
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_xy.cmake
)

set_tests_properties(test_shallow_xy PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

add_test(
  NAME test_shallow_splash
  COMMAND ${CMAKE_COMMAND}
          -Dbuildl=release
          -P ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_splash.cmake
)

set_tests_properties(test_shallow_splash PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

if (NOT APPLE)
    add_test(
      NAME test_restart_multiblock
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_multiblock.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )

    set_tests_properties(test_restart_multiblock PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      TIMEOUT 600
      LABELS "restart;examples"
    )

    add_test(
      NAME test_restart_cycle_limit
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_cycle_limit.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )

    set_tests_properties(test_restart_cycle_limit PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      SKIP_RETURN_CODE 125
      TIMEOUT 600
      LABELS "restart;examples"
    )

    add_test(
      NAME test_restart_new_output
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_new_output.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )

    set_tests_properties(test_restart_new_output PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      TIMEOUT 600
      SKIP_RETURN_CODE 125
      LABELS "restart;examples"
    )

    add_test(
      NAME test_restart_insert_output
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_insert_output.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )

    set_tests_properties(test_restart_insert_output PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      TIMEOUT 600
      SKIP_RETURN_CODE 125
      LABELS "restart;examples"
    )

    add_test(
      NAME test_restart_key_collision
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_key_collision.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )

    set_tests_properties(test_restart_key_collision PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      TIMEOUT 600
      SKIP_RETURN_CODE 125
      LABELS "restart;examples"
    )

    add_test(
      NAME test_restart_inplace_reorder
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/run_restart_inplace_reorder.py
              --build-dir ${CMAKE_BINARY_DIR}
              --build-type release
    )

    set_tests_properties(test_restart_inplace_reorder PROPERTIES
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
      TIMEOUT 600
      SKIP_RETURN_CODE 125
      LABELS "restart;examples"
    )
endif()

if (FULL_TESTS)
  setup_parallel_test(test_mesh_multi_block 2)

  add_test(
    NAME test_exchange_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_exchange_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
  )

  set_tests_properties(test_exchange_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROCESSORS 6
    RUN_SERIAL TRUE
    TIMEOUT 1200
    LABELS "decomp;gloo"
  )

  add_test(
    NAME test_shallow_xy_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_xy_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
  )

  set_tests_properties(test_shallow_xy_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROCESSORS 4
    RUN_SERIAL TRUE
    TIMEOUT 1200
    LABELS "decomp;gloo"
  )

  add_test(
    NAME test_shallow_splash_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_splash_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
            --backend gloo
  )

  set_tests_properties(test_shallow_splash_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    PROCESSORS 6
    RUN_SERIAL TRUE
    TIMEOUT 2400
    LABELS "decomp;gloo"
  )

  add_test(
    NAME test_shallow_splash_ucx_cuda_decomp
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_shallow_splash_decomp.py
            --build-dir ${CMAKE_BINARY_DIR}
            --build-type release
            --backend ucx
            --device cuda
  )

  set_tests_properties(test_shallow_splash_ucx_cuda_decomp PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    SKIP_RETURN_CODE 125
    PROCESSORS 2
    RUN_SERIAL TRUE
    TIMEOUT 2400
    LABELS "gpu;ucx;long"
  )
endif()

# A build without CUDA skips every CUDA test, even when a GPU is visible.
# Registration is the gate; torch.cuda.is_available() is not.
if(NOT CUDA)
  get_property(_snapy_tests DIRECTORY PROPERTY TESTS)
  foreach(_t IN LISTS _snapy_tests)
    set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT_MODIFICATION
                 "SNAPY_BUILD_CUDA=set:0")
    set(_disable FALSE)
    if(_t MATCHES "cuda")
      set(_disable TRUE)
    endif()
    get_property(_labels TEST ${_t} PROPERTY LABELS)
    foreach(_l IN LISTS _labels)
      if(_l STREQUAL "cuda" OR _l STREQUAL "gpu")
        set(_disable TRUE)
      endif()
    endforeach()
    if(_disable)
      set_tests_properties(${_t} PROPERTIES DISABLED TRUE)
    endif()
  endforeach()
endif()
