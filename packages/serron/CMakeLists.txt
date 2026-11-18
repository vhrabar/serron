cmake_minimum_required(VERSION 3.27)

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
    if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
        set(_serron_cuda_archs 89-real 100-real 120-real 120-virtual)
    else()
        set(_serron_cuda_archs 75-real 80-real 86-real 89-real 90-real 100-real 120-real 120-virtual)
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
        set(CMAKE_CUDA_ARCHITECTURES ${_serron_cuda_archs})
    endif()

    enable_language(CUDA)
    #  20 device
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
    message(STATUS "serron: building with CUDA backend (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "serron: building C++ only (no CUDA backend)")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent.as_posix())"
        OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_LIB_DIR "${TORCH_INSTALL_PREFIX}/lib")
if(NOT EXISTS "${TORCH_LIB_DIR}")
    message(FATAL_ERROR "Could not locate torch lib dir at: ${TORCH_LIB_DIR}")
endif()

set(TORCH_INCLUDE_DIRS
        "${TORCH_INSTALL_PREFIX}/include"
        "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include")
find_library(SERRON_TORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

if(SERRON_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()


option(SERRON_ROCM "Build the ROCm backend" OFF)
set(SERRON_ROCM_ROOT "" CACHE PATH "ROCm install path; else ROCM_PATH, HIP_PATH or /opt/rocm")
if(SERRON_ROCM)
    if(SERRON_ROCM_ROOT)
        find_path(SERRON_HIP_INCLUDE_DIR hip/hip_runtime_api.h HINTS "${SERRON_ROCM_ROOT}" PATH_SUFFIXES include
                NO_DEFAULT_PATH)
    else()
        find_path(SERRON_HIP_INCLUDE_DIR hip/hip_runtime_api.h HINTS ENV ROCM_PATH ENV HIP_PATH PATHS /opt/rocm
                PATH_SUFFIXES include)
    endif()
    if(NOT SERRON_HIP_INCLUDE_DIR)
        message(FATAL_ERROR "serron: SERRON_ROCM is ON but hip/hip_runtime_api.h was not found!")
    endif()
    message(STATUS "serron: building the ROCm backend (HIP headers in ${SERRON_HIP_INCLUDE_DIR})")

    # ROCm kernels are built for common archs + portable AMD GCN SPIR-V
    set(SERRON_ROCM_ARCHS "gfx90a;gfx942;gfx950;gfx1100;gfx1201" CACHE STRING
            "GPU architectures to build native ROCm kernel images for; the portable image covers the rest")
    if(SERRON_ROCM_ROOT)
        find_program(SERRON_HIPCC hipcc HINTS "${SERRON_ROCM_ROOT}/bin" NO_DEFAULT_PATH)
    else()
        find_program(SERRON_HIPCC hipcc HINTS ENV ROCM_PATH ENV HIP_PATH PATHS /opt/rocm PATH_SUFFIXES bin)
    endif()
    if(NOT SERRON_HIPCC)
        message(FATAL_ERROR "serron: SERRON_ROCM is ON but hipcc was not found.")
    endif()
    get_filename_component(SERRON_HIPCC_ROOT "${SERRON_HIPCC}" REALPATH)
    get_filename_component(SERRON_HIPCC_ROOT "${SERRON_HIPCC_ROOT}" DIRECTORY)
    get_filename_component(SERRON_HIPCC_ROOT "${SERRON_HIPCC_ROOT}" DIRECTORY)
    execute_process(COMMAND "${SERRON_HIPCC}" "--rocm-path=${SERRON_HIPCC_ROOT}" --version
            OUTPUT_VARIABLE _serron_hipcc_version OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    string(REGEX MATCH "[^\n]*clang version[^\n]*" _serron_hipcc_version "${_serron_hipcc_version}")

    find_path(SERRON_ROCM_DEVICE_LIBS ocml.bc HINTS "${SERRON_HIPCC_ROOT}" NO_DEFAULT_PATH
            PATH_SUFFIXES amdgcn/bitcode lib/amdgcn/bitcode lib/llvm/amdgcn/bitcode)
    set(SERRON_HIPCC_FLAGS "--rocm-path=${SERRON_HIPCC_ROOT}")
    if(SERRON_ROCM_DEVICE_LIBS)
        list(APPEND SERRON_HIPCC_FLAGS "--rocm-device-lib-path=${SERRON_ROCM_DEVICE_LIBS}")
    endif()
    message(STATUS "serron: ROCm kernel images built by ${SERRON_HIPCC} (${_serron_hipcc_version}) for ${SERRON_ROCM_ARCHS} and AMD GCN SPIR-V")
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
if(SERRON_ROCM)
    set(SERRON_WITH_ROCM_PY True)
else()
    set(SERRON_WITH_ROCM_PY False)
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

# _C holds the op schemas and the CPU kernels and links only torch_cpu/c10, so it loads on CPU-only torch.
# _C_cuda holds the CUDA kernels and is loaded on top of it only when torch has CUDA.
add_library(_C SHARED
        csrc/bindings.cpp
        csrc/cpu/morphology/operators.cpp
        csrc/cpu/morphology/backward.cpp
)
set(_serron_targets _C)

set(_serron_compiled_targets _C)

if(SERRON_CUDA)
    find_program(SERRON_BIN2C bin2c HINTS "${CUDAToolkit_BIN_DIR}" REQUIRED)
    set(_serron_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

    set(_serron_fatbin_headers)
    foreach(_kernels IN ITEMS forward backward)
        set(_fatbin_target _serron_${_kernels}_fatbin)
        add_library(${_fatbin_target} OBJECT csrc/gpu/morphology/${_kernels}_kernels.cu)
        set_target_properties(${_fatbin_target} PROPERTIES CUDA_FATBIN_COMPILATION ON)
        if(WIN32)
            target_compile_definitions(${_fatbin_target} PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)
        endif()
        list(APPEND _serron_compiled_targets ${_fatbin_target})

        set(_header "${_serron_generated_dir}/${_kernels}_fatbin.h")
        add_custom_command(
                OUTPUT "${_header}"
                COMMAND "${CMAKE_COMMAND}"
                        -DBIN2C=${SERRON_BIN2C}
                        -DINPUT=$<TARGET_OBJECTS:${_fatbin_target}>
                        -DOUTPUT=${_header}
                        -DNAME=serron_${_kernels}_fatbin
                        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_fatbin.cmake"
                DEPENDS ${_fatbin_target} $<TARGET_OBJECTS:${_fatbin_target}>
                        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_fatbin.cmake"
                COMMENT "Embedding the ${_kernels} kernels fatbin"
                VERBATIM
        )
        list(APPEND _serron_fatbin_headers "${_header}")
    endforeach()

    add_library(_C_cuda SHARED
            csrc/gpu/bindings.cpp
            csrc/gpu/morphology/operators.cpp
            csrc/gpu/morphology/backward.cpp
            csrc/gpu/morphology/dispatch_info.cpp
            csrc/cuda/utils/backend.cpp
            csrc/cuda/utils/driver.cpp
            ${_serron_fatbin_headers}
    )
    target_include_directories(_C_cuda PRIVATE "${_serron_generated_dir}")
    list(APPEND _serron_targets _C_cuda)
    list(APPEND _serron_compiled_targets _C_cuda)
endif()

if(SERRON_ROCM)
    add_library(_C_rocm SHARED
            csrc/gpu/bindings.cpp
            csrc/gpu/morphology/operators.cpp
            csrc/gpu/morphology/backward.cpp
            csrc/gpu/morphology/dispatch_info.cpp
            csrc/rocm/utils/backend.cpp
            csrc/rocm/utils/runtime.cpp
    )
    target_include_directories(_C_rocm PRIVATE "${SERRON_HIP_INCLUDE_DIR}")

    # One code object per architecture and one portable image per kernel file
    set(_serron_image_dir "${CMAKE_CURRENT_BINARY_DIR}/rocm_images")
    set(_serron_image_includes "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc" "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc/compat")
    foreach(_dir IN LISTS TORCH_INCLUDE_DIRS)
        list(APPEND _serron_image_includes "-I${_dir}")
    endforeach()
    set(_serron_rocm_images)
    foreach(_kernels IN ITEMS forward backward)
        set(_source "${CMAKE_CURRENT_SOURCE_DIR}/csrc/gpu/morphology/${_kernels}_kernels.cu")
        foreach(_target IN LISTS SERRON_ROCM_ARCHS ITEMS amdgcnspirv)
            set(_image "${_serron_image_dir}/${_kernels}_${_target}.img")
            add_custom_command(
                    OUTPUT "${_image}"
                    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_serron_image_dir}"
                    COMMAND "${SERRON_HIPCC}" ${SERRON_HIPCC_FLAGS} -x hip -std=c++20 "--offload-arch=${_target}"
                            -DTORCH_STABLE_ONLY -DTORCH_TARGET_VERSION=${SERRON_TORCH_TARGET_HEX}ULL
                            ${_serron_image_includes} -M -MT "${_image}" -MF "${_image}.d" "${_source}"
                    COMMAND "${SERRON_HIPCC}" ${SERRON_HIPCC_FLAGS} -x hip -O3 -std=c++20
                            "--offload-arch=${_target}" --offload-device-only -c
                            -DTORCH_STABLE_ONLY -DTORCH_TARGET_VERSION=${SERRON_TORCH_TARGET_HEX}ULL
                            ${_serron_image_includes} -o "${_image}" "${_source}"
                    DEPFILE "${_image}.d"
                    MAIN_DEPENDENCY "${_source}"
                    COMMENT "Building the ${_kernels} kernels for ${_target}"
                    VERBATIM
            )
            list(APPEND _serron_rocm_images "${_image}")
        endforeach()
    endforeach()
    add_custom_target(_serron_rocm_images ALL DEPENDS ${_serron_rocm_images})
    list(APPEND _serron_targets _C_rocm)
    list(APPEND _serron_compiled_targets _C_rocm)
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
            csrc/compat/atomic.cuh
            csrc/gpu/utils/border_mode.h
            csrc/gpu/utils/boundaries.cuh
            csrc/gpu/utils/declarations.h
            csrc/cuda/utils/device_runtime.h
            csrc/cuda/utils/driver.h
            csrc/cuda/utils/shuffle.cuh
            csrc/gpu/utils/backend.h
            csrc/gpu/utils/device_runtime.h
            csrc/gpu/utils/launch.h
            csrc/gpu/utils/shuffle.cuh
            csrc/gpu/utils/smem.h
            csrc/rocm/utils/device_runtime.hip.h
            csrc/rocm/utils/runtime.hip.h
            csrc/rocm/utils/shuffle.hip.h
            csrc/gpu/morphology/arg_tap.h
            csrc/gpu/morphology/entry_points.h
            csrc/gpu/morphology/enums.h
            csrc/gpu/morphology/kernel_tables.h
            csrc/gpu/morphology/ops_policy.cuh
            csrc/gpu/morphology/scan.cuh
            csrc/gpu/morphology/separable.h
    )
    set_source_files_properties(${_serron_cuda_headers} PROPERTIES HEADER_FILE_ONLY TRUE)
    target_sources(_C_cuda PRIVATE ${_serron_cuda_headers})
endif()

if(SERRON_ROCM)
    set(_serron_rocm_headers
            csrc/gpu/utils/backend.h
            csrc/gpu/utils/launch.h
            csrc/gpu/utils/smem.h
            csrc/rocm/utils/runtime.hip.h
    )
    set_source_files_properties(${_serron_rocm_headers} PROPERTIES HEADER_FILE_ONLY TRUE)
    target_sources(_C_rocm PRIVATE ${_serron_rocm_headers})
endif()

set(SERRON_CUDA_NVCC_THREADS 1 CACHE STRING "nvcc --threads (per-.cu arch parallelism)")

# -lineinfo on Debug; full device debug (-G) only when explicitly sanitizing.
option(CUDA_SANITIZE "Build device code with -G for compute-sanitizer" OFF)


foreach(_target IN LISTS _serron_compiled_targets)
    # The nvcc flags torch's CMake config used to add to CMAKE_CUDA_FLAGS; the torch headers need relaxed constexpr.
    target_compile_options(${_target} PRIVATE
            "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl>"
            $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda>
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
endforeach()

foreach(_target IN LISTS _serron_targets)
    set_target_properties(${_target} PROPERTIES PREFIX "")

    if(NOT WIN32)
        set_target_properties(${_target} PROPERTIES
                BUILD_RPATH   "${TORCH_LIB_DIR}"
                INSTALL_RPATH "$ORIGIN/../torch/lib"
        )
    endif()
endforeach()

target_link_libraries(_C PRIVATE "${SERRON_TORCH_CPU_LIBRARY}")

if(SERRON_CUDA)
    target_include_directories(_C_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(_C_cuda PRIVATE "${SERRON_TORCH_CPU_LIBRARY}" ${CMAKE_DL_LIBS})
endif()

if(SERRON_ROCM)
    target_link_libraries(_C_rocm PRIVATE "${SERRON_TORCH_CPU_LIBRARY}" ${CMAKE_DL_LIBS})
endif()

# On Windows the .dll is a RUNTIME artifact, which would otherwise land in bin/ outside the package.
install(TARGETS ${_serron_targets} LIBRARY DESTINATION serron RUNTIME DESTINATION serron)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.py" DESTINATION serron)
