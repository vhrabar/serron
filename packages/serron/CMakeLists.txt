cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CUDA_HOME})
        set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc" CACHE FILEPATH "" FORCE)
    else()
        find_program(_NVCC nvcc)
        if(_NVCC)
            set(CMAKE_CUDA_COMPILER "${_NVCC}" CACHE FILEPATH "" FORCE)
        endif()
    endif()
endif()

# CXX is always enabled; CUDA only when the backend is requested.
project(morphottention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Python_FIND_VIRTUALENV FIRST)

# Interpreter only. csrc/ includes no Python.h and defines no PyInit_, so the libraries are plain
# shared objects that torch.ops.load_library dlopens; nothing needs the CPython headers or ABI.
find_package(Python REQUIRED COMPONENTS Interpreter)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(_serron_cuda_default ON)
else()
    set(_serron_cuda_default OFF)
endif()
option(SERRON_CUDA "Build the CUDA backend (requires nvcc)" ${_serron_cuda_default})

if(SERRON_CUDA)
    enable_language(CUDA)
    #  20 device
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 100-real 120-real 120-virtual)
    endif()
    if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
        set(TORCH_CUDA_ARCH_LIST "7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX")
    endif()
    message(STATUS "serron: building with CUDA backend (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "serron: building C++ only (no CUDA backend)")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED CONFIG)


execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')"
        OUTPUT_VARIABLE TORCH_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT EXISTS "${TORCH_LIB_DIR}")
    message(FATAL_ERROR "Could not locate torch lib dir at: ${TORCH_LIB_DIR}")
endif()

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__version__.split('+')[0])"
        OUTPUT_VARIABLE SERRON_TORCH_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(SERRON_CUDA)
    set(SERRON_WITH_CUDA_PY True)
else()
    set(SERRON_WITH_CUDA_PY False)
endif()
# Restrict the extension to the torch stable ABI
option(SERRON_STABLE_ABI "Build against the torch stable ABI only" ON)

# Lowest torch release the stable ABI is targeted at
set(SERRON_TORCH_TARGET "2.13" CACHE STRING "Minimum torch version to target, major.minor")

if(SERRON_STABLE_ABI)
    # TORCH_TARGET_VERSION is (major << 56) | (minor << 48);
    string(REPLACE "." ";" _serron_target_parts "${SERRON_TORCH_TARGET}")
    list(LENGTH _serron_target_parts _serron_target_len)
    if(_serron_target_len LESS 2)
        message(FATAL_ERROR "SERRON_TORCH_TARGET must be major.minor, got '${SERRON_TORCH_TARGET}'")
    endif()
    list(GET _serron_target_parts 0 _serron_target_major)
    list(GET _serron_target_parts 1 _serron_target_minor)
    math(EXPR SERRON_TORCH_TARGET_HEX
            "(${_serron_target_major} << 56) | (${_serron_target_minor} << 48)" OUTPUT_FORMAT HEXADECIMAL)
    message(STATUS "serron: stable ABI only, targeting torch ${SERRON_TORCH_TARGET} "
            "(TORCH_TARGET_VERSION=${SERRON_TORCH_TARGET_HEX}ULL)")
endif()

if(SERRON_STABLE_ABI)
    set(SERRON_STABLE_ABI_PY True)
else()
    set(SERRON_STABLE_ABI_PY False)
endif()

configure_file(cmake/_build_info.py.in "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" @ONLY)

if(SERRON_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# _C holds the op schemas and the CPU kernels and links only torch_cpu/c10, so it loads on CPU-only torch.
# _C_cuda holds the CUDA kernels and is loaded on top of it only when torch has CUDA.
add_library(_C SHARED
        csrc/bindings.cpp
        csrc/cpu/morphology/operators.cpp
        csrc/cpu/morphology/backward.cpp
)
set(_serron_targets _C)

if(SERRON_CUDA)
    add_library(_C_cuda SHARED
            csrc/cuda/bindings.cpp
            csrc/cuda/morphology/operators.cu
            csrc/cuda/morphology/backward.cu
    )
    list(APPEND _serron_targets _C_cuda)
endif()

# Headers are listed only so IDEs index them; they are not compiled on their own.
set(_serron_headers
        csrc/ops.h
        csrc/compat/acc_type.h
        csrc/compat/check.h
        csrc/compat/flatness.h
        csrc/compat/grain_size.h
        csrc/compat/registration.h
        csrc/cpu/utils/boundaries.h
        csrc/cpu/morphology/enums.h
        csrc/cpu/morphology/ops_policy.h
        csrc/cpu/morphology/separable.h
)
set_source_files_properties(${_serron_headers} PROPERTIES HEADER_FILE_ONLY TRUE)
target_sources(_C PRIVATE ${_serron_headers})

if(SERRON_CUDA)
    set(_serron_cuda_headers
            csrc/cuda/utils/boundaries.cuh
            csrc/cuda/utils/declarations.cuh
            csrc/compat/cuda_launch_check.cuh
            csrc/cuda/utils/smem.cuh
            csrc/cuda/morphology/entry_points.cuh
            csrc/cuda/morphology/enums.cuh
            csrc/cuda/morphology/ops_policy.cuh
            csrc/cuda/morphology/scan.cuh
            csrc/cuda/morphology/separable.cuh
    )
    set_source_files_properties(${_serron_cuda_headers} PROPERTIES HEADER_FILE_ONLY TRUE)
    target_sources(_C_cuda PRIVATE ${_serron_cuda_headers})
endif()

set(SERRON_CUDA_NVCC_THREADS 1 CACHE STRING "nvcc --threads (per-.cu arch parallelism)")

# -lineinfo on Debug; full device debug (-G) only when explicitly sanitizing.
option(CUDA_SANITIZE "Build device code with -G for compute-sanitizer" OFF)


foreach(_target IN LISTS _serron_targets)
    # No "lib" prefix: the loader looks for _C / _C_cuda by those exact stems. No CPython tag either, which is
    # what makes one wheel usable on every interpreter.
    set_target_properties(${_target} PROPERTIES PREFIX "")

    target_compile_options(${_target} PRIVATE
            $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--threads=${SERRON_CUDA_NVCC_THREADS}>
            $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<CONFIG:Debug>>:-lineinfo>
            $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<CONFIG:Debug>,$<BOOL:${CUDA_SANITIZE}>>:-G>
    )

    if(SERRON_STABLE_ABI)
        target_compile_definitions(${_target} PRIVATE
                TORCH_STABLE_ONLY
                TORCH_TARGET_VERSION=${SERRON_TORCH_TARGET_HEX}ULL
        )
    endif()

    if(MSVC)
        target_compile_options(${_target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        )
    endif()

    target_include_directories(${_target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/csrc
            ${CMAKE_CURRENT_SOURCE_DIR}/csrc/compat
            ${TORCH_INCLUDE_DIRS}
    )

    if(NOT WIN32)
        set_target_properties(${_target} PROPERTIES
                BUILD_RPATH   "${TORCH_LIB_DIR}"
                INSTALL_RPATH "$ORIGIN/../torch/lib"
        )
    endif()
endforeach()

target_link_libraries(_C PRIVATE "$<TARGET_FILE:torch_cpu>")

if(SERRON_CUDA)
    target_include_directories(_C_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(_C_cuda PRIVATE "$<TARGET_FILE:torch_cpu>" CUDA::cudart)
endif()

install(TARGETS ${_serron_targets} LIBRARY DESTINATION serron)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" DESTINATION serron)
