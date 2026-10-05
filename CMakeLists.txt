cmake_minimum_required(VERSION 3.20)

project(CppMastery
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

# Always define the interface target.
add_library(sanitizers INTERFACE)

option(
    ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
    OFF
)

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

# Release-only IPO/LTO
include(CheckIPOSupported)

check_ipo_supported(
    RESULT ipo_supported
    OUTPUT ipo_error
    LANGUAGES CXX
)

if(ipo_supported)
    message(STATUS "IPO/LTO is supported")
else()
    message(WARNING
        "IPO/LTO is not supported:\n"
        "${ipo_error}"
    )
endif()

# One executable per lecture_N/ directory, named after the directory.
function(add_lecture lecture_dir)
    get_filename_component(name "${lecture_dir}" NAME)

    add_executable(${name} ${lecture_dir}/main.cpp)

    target_compile_features(${name} PRIVATE cxx_std_23)

    set_target_properties(${name} PROPERTIES
        CXX_EXTENSIONS OFF
    )

    target_compile_options(${name} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

    if(ipo_supported)
        set_property(TARGET ${name} PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
        )
    endif()

    target_link_libraries(${name} PRIVATE sanitizers)
endfunction()

# The build scripts configure one lecture per build tree (lecture_N/build/...).
# Leave LECTURE empty to get every lecture in one tree, e.g. for an IDE.
set(LECTURE "" CACHE STRING "Lecture directory to build (e.g. lecture_3); empty builds all")

if(LECTURE)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LECTURE}/main.cpp)
        message(FATAL_ERROR "No such lecture: ${LECTURE}/main.cpp")
    endif()

    add_lecture(${LECTURE})
else()
    # CONFIGURE_DEPENDS re-runs the glob on build, so new lectures are picked up.
    file(GLOB lecture_mains CONFIGURE_DEPENDS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        lecture_*/main.cpp
    )

    foreach(lecture_main IN LISTS lecture_mains)
        get_filename_component(lecture_dir "${lecture_main}" DIRECTORY)
        add_lecture(${lecture_dir})
    endforeach()
endif()
