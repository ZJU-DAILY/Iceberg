cmake_minimum_required(VERSION 3.9)
project(mips-benchmark VERSION 0.1.0 LANGUAGES C CXX)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the C standard to C11
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0074 NEW)

# #####################################################################################################################
# DEPENDENCIES
# #####################################################################################################################


# clang-format

# attempt to find the binary if user did not specify
find_program(CLANG_FORMAT_BIN
        NAMES clang-format clang-format-14)

if("${CLANG_FORMAT_BIN}" STREQUAL "CLANG_FORMAT_BIN-NOTFOUND")
        message(WARNING "couldn't find clang-format.")
else()
        message(STATUS "found clang-format at ${CLANG_FORMAT_BIN}")
endif()

# attempt to find the binary if user did not specify
find_program(CLANG_TIDY_BIN
        NAMES clang-tidy clang-tidy-14)

if("${CLANG_TIDY_BIN}" STREQUAL "CLANG_TIDY_BIN-NOTFOUND")
        message(WARNING "couldn't find clang-tidy.")
else()
        set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
        message(STATUS "found clang-tidy at ${CLANG_TIDY_BIN}")
endif()

# #####################################################################################################################
# COMPILER SETUP
# #####################################################################################################################

# Includes.
set(MIPS_BENCHMARK_INCLUDE_DIR 
    ${PROJECT_SOURCE_DIR}/baselines
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
)

# Define the MKL_ROOT variable. The user must provide this path during configuration
# if it's not found automatically. Example: cmake -DMKL_ROOT=/path/to/intel/mkl ..
if(NOT DEFINED MKL_ROOT)
    set(MKL_ROOT "" CACHE PATH "Intel MKL root directory")
endif()

set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")

# Manually set link libraries (example, modify as needed)
set(MKL_LIBRARIES
    ${MKL_ROOT}/lib/libmkl_intel_ilp64.so
    ${MKL_ROOT}/lib/libmkl_core.so
    ${MKL_ROOT}/lib/libmkl_sequential.so
    pthread
    m
    dl
)


find_package(MKL REQUIRED)
if(MKL_FOUND)
    message(STATUS "Found MKL: ${MKL_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "MKL not found, please set MKL_ROOT")
endif()

include_directories(${MIPS_BENCHMARK_INCLUDE_DIR})
message(STATUS "MIPS_BENCHMARK_INCLUDE_DIR: ${MIPS_BENCHMARK_INCLUDE_DIR}")

set(THRID_PARTY_INCLUDE_DIR 
    ${PROJECT_SOURCE_DIR}/third_party/efanna_graph/include
)
include_directories(${THRID_PARTY_INCLUDE_DIR})
message(STATUS "THRID_PARTY_INCLUDE_DIR: ${THRID_PARTY_INCLUDE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops -lprofiler -mavx")


# #####################################################################################################################
# Other CMake modules
# #####################################################################################################################

find_package(Boost REQUIRED)
if (Boost_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Boost_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Boost_CXX_FLAGS}")
else()
    message(FATAL_ERROR "Boost dynamic-bitset is required")
endif()

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP is required")
endif()


add_subdirectory(test)
add_subdirectory(third_party/efanna_graph)


# #####################################################################################################################
# MAKE TARGETS
# #####################################################################################################################

# #########################################
# "make format"
# "make check-format"
# #########################################
set(MIPS_BENCHMARK_BUILD_SUPPORT_DIR "${CMAKE_SOURCE_DIR}/build_support")

string(CONCAT MIPS_BENCHMARK_FORMAT_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/baselines,"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests,"
)

# Runs clang format and updates files in place.
add_custom_target(format ${MIPS_BENCHMARK_BUILD_SUPPORT_DIR}/run_clang_format.py
        ${CLANG_FORMAT_BIN}
        ${MIPS_BENCHMARK_BUILD_SUPPORT_DIR}/clang_format_exclusions.txt
        --source_dirs
        ${MIPS_BENCHMARK_FORMAT_DIRS}
        --fix
        --quiet
)

# Runs clang format and exits with a non-zero exit code if any files need to be reformatted
add_custom_target(check-format ${MIPS_BENCHMARK_BUILD_SUPPORT_DIR}/run_clang_format.py
        ${CLANG_FORMAT_BIN}
        ${MIPS_BENCHMARK_BUILD_SUPPORT_DIR}/clang_format_exclusions.txt
        --source_dirs
        ${MIPS_BENCHMARK_FORMAT_DIRS}
        --quiet
)

# ##########################################################
# "make check-clang-tidy" target
# ##########################################################
# runs clang-tidy and exits with a non-zero exit code if any errors are found.
# note that clang-tidy automatically looks for a .clang-tidy file in parent directories
add_custom_target(check-clang-tidy
        ${MIPS_BENCHMARK_BUILD_SUPPORT_DIR}/run_clang_tidy.py # run LLVM's clang-tidy script
        -clang-tidy-binary ${CLANG_TIDY_BIN} # using our clang-tidy binary
        -p ${CMAKE_BINARY_DIR} # using cmake's generated compile commands
)
add_custom_target(fix-clang-tidy
        ${MIPS_BENCHMARK_BUILD_SUPPORT_DIR}/run_clang_tidy.py # run LLVM's clang-tidy script
        -clang-tidy-binary ${CLANG_TIDY_BIN} # using our clang-tidy binary
        -p ${CMAKE_BINARY_DIR} # using cmake's generated compile commands
        -clang-apply-replacements-binary ${CLANG_APPLY_REPLACEMENTS_BIN} # using our clang-apply-replacements binary
        -fix # apply suggested changes generated by clang-tidy
)
add_custom_target(check-clang-tidy-diff
        ${MIPS_BENCHMARK_BUILD_SUPPORT_DIR}/run_clang_tidy.py # run LLVM's clang-tidy script
        -clang-tidy-binary ${CLANG_TIDY_BIN} # using our clang-tidy binary
        -p ${CMAKE_BINARY_DIR} # using cmake's generated compile commands
        -only-diff # only check diff files to master
)
add_custom_target(fix-clang-tidy-diff
        ${MIPS_BENCHMARK_BUILD_SUPPORT_DIR}/run_clang_tidy.py # run LLVM's clang-tidy script
        -clang-tidy-binary ${CLANG_TIDY_BIN} # using our clang-tidy binary
        -p ${CMAKE_BINARY_DIR} # using cmake's generated compile commands
        -clang-apply-replacements-binary ${CLANG_APPLY_REPLACEMENTS_BIN} # using our clang-apply-replacements binary
        -fix # apply suggested changes generated by clang-tidy
        -only-diff # only check diff files to master
)