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


project(morphottention LANGUAGES CXX CUDA)

#  23 host / 20 device
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

set(CMAKE_CUDA_ARCHITECTURES 90 100 120)
set(TORCH_CUDA_ARCH_LIST "9.0;10.0;12.0")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Python_FIND_VIRTUALENV FIRST)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

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

find_package(CUDAToolkit REQUIRED)

Python_add_library(_C MODULE WITH_SOABI
        csrc/bindings.cpp
        csrc/cuda/morphology.cu
)

# -lineinfo on Debug; full device debug (-G) only when explicitly sanitizing.
option(CUDA_SANITIZE "Build device code with -G for compute-sanitizer" OFF)
target_compile_options(_C PRIVATE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--threads=0>
        $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<CONFIG:Debug>>:-lineinfo>
        $<$<AND:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>,$<CONFIG:Debug>,$<BOOL:${CUDA_SANITIZE}>>:-G>
)


if(MSVC)
    target_compile_options(_C PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

target_include_directories(_C PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/compat
        ${TORCH_INCLUDE_DIRS}
        ${Python_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(_C PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::cublas
)

if(NOT WIN32)
    set_target_properties(_C PROPERTIES
            BUILD_RPATH   "${TORCH_LIB_DIR}"
            INSTALL_RPATH "$ORIGIN/../torch/lib"
    )
endif()

install(TARGETS _C LIBRARY DESTINATION serron)