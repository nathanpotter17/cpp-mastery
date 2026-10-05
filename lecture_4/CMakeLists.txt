cmake_minimum_required(VERSION 3.20)

project(MyApp
    VERSION 1.0.0
    LANGUAGES CXX
)

# Set a default before configuring targets.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug
    Release
    RelWithDebInfo
    MinSizeRel
)

add_executable(myapp main.cpp)

target_compile_features(myapp PRIVATE cxx_std_23)

set_target_properties(myapp PROPERTIES
    CXX_EXTENSIONS OFF
)

target_compile_options(myapp PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# Release-only IPO/LTO
include(CheckIPOSupported)

check_ipo_supported(
    RESULT ipo_supported
    OUTPUT ipo_error
    LANGUAGES CXX
)

if(ipo_supported)
    message(STATUS "IPO/LTO is supported")

    set_property(TARGET myapp PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
else()
    message(WARNING
        "IPO/LTO is not supported:\n"
        "${ipo_error}"
    )
endif()

option(
    ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
    OFF
)

# Always define the interface target.
add_library(sanitizers INTERFACE)

if(ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR
            "ENABLE_SANITIZERS requires Clang or GCC"
        )
    endif()

    target_compile_options(sanitizers INTERFACE
        -g
        -O1
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -fsanitize=address,undefined
    )

    target_link_options(sanitizers INTERFACE
        -fsanitize=address,undefined
    )
endif()

target_link_libraries(myapp PRIVATE sanitizers)

