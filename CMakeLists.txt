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

# Always define the interface target.
add_library(static_runtime INTERFACE)

option(
    STATIC_RUNTIME
    "Link the C++ runtime into the executable (fully static on MinGW)"
    OFF
)

if(STATIC_RUNTIME)
    if(ENABLE_SANITIZERS)
        message(FATAL_ERROR
            "STATIC_RUNTIME can't be combined with ENABLE_SANITIZERS"
        )
    endif()

    if(MINGW)
        # No libstdc++-6.dll, libgcc_s_seh-1.dll or libwinpthread-1.dll needed
        # next to the .exe. Windows' own DLLs are still linked dynamically.
        target_link_options(static_runtime INTERFACE -static)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        # glibc stays dynamic: it isn't meant to be linked statically, and
        # dlopen (Vulkan drivers, Wayland) needs it.
        target_link_options(static_runtime INTERFACE
            -static-libstdc++
            -static-libgcc
        )
    endif()
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

# vendor/<lib>/ with a CMakeLists.txt is built and linked as <lib>::<lib> or
# <lib>, whichever exists (other target names: link them in lecture.cmake).
# Header-only libs go on the include path as SYSTEM, so -Wall and friends
# don't warn about them: vendor/<lib>/include/ if it exists, else vendor/<lib>/.
function(add_vendored_lib target lib_dir)
    get_filename_component(lib "${lib_dir}" NAME)

    if(NOT EXISTS ${lib_dir}/CMakeLists.txt)
        if(IS_DIRECTORY ${lib_dir}/include)
            target_include_directories(${target} SYSTEM PRIVATE ${lib_dir}/include)
        else()
            target_include_directories(${target} SYSTEM PRIVATE ${lib_dir})
        endif()
        return()
    endif()

    # When every lecture shares one build tree, the first lecture to vendor
    # <lib> wins, since two copies would define the same target names.
    get_property(already_added GLOBAL PROPERTY vendored_${lib})

    if(NOT already_added)
        set_property(GLOBAL PROPERTY vendored_${lib} TRUE)

        # The default binary dir, <build>/lecture_N/vendor/<lib>, would put a
        # lecture_N/ directory where the lecture_N executable goes.
        set(lib_binary_dir ${CMAKE_BINARY_DIR}/vendor/${lib})

        # SYSTEM (CMake 3.25+) also silences warnings from the lib's headers.
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
            add_subdirectory(${lib_dir} ${lib_binary_dir} SYSTEM)
        else()
            add_subdirectory(${lib_dir} ${lib_binary_dir})
        endif()
    endif()

    if(TARGET ${lib}::${lib})
        target_link_libraries(${target} PRIVATE ${lib}::${lib})
    elseif(TARGET ${lib})
        target_link_libraries(${target} PRIVATE ${lib})
    else()
        message(STATUS "${target}: vendor/${lib} has no ${lib} target, link it in lecture.cmake")
    endif()
endfunction()

# Each lecture_N/ is its own project, built as an executable named lecture_N:
#   main.cpp or src/main.cpp  entry point
#   src/**/*.cpp              other sources
#   src/includes/*.h          headers, #include "includes/foo.h"
#   vendor/<lib>/             third-party code, see add_vendored_lib()
#   lecture.cmake             optional extra setup, ${name} is the target
function(add_lecture lecture_dir)
    get_filename_component(name "${lecture_dir}" NAME)
    set(dir ${CMAKE_CURRENT_SOURCE_DIR}/${lecture_dir})

    if(EXISTS ${dir}/main.cpp AND EXISTS ${dir}/src/main.cpp)
        message(FATAL_ERROR
            "${lecture_dir} has both main.cpp and src/main.cpp, keep one"
        )
    endif()

    file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${dir}/src/*.cpp)

    if(EXISTS ${dir}/main.cpp)
        list(APPEND sources ${dir}/main.cpp)
    endif()

    add_executable(${name} ${sources})

    # src/ is the include root, so headers are included by their path under it.
    target_include_directories(${name} PRIVATE ${dir}/src)

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

    target_link_libraries(${name} PRIVATE sanitizers static_runtime)

    # Runs before the vendored libs are added, so it can set their options.
    if(EXISTS ${dir}/lecture.cmake)
        include(${dir}/lecture.cmake)
    endif()

    file(GLOB vendored CONFIGURE_DEPENDS LIST_DIRECTORIES true ${dir}/vendor/*)

    foreach(lib_dir IN LISTS vendored)
        if(IS_DIRECTORY ${lib_dir})
            add_vendored_lib(${name} ${lib_dir})
        endif()
    endforeach()
endfunction()

# The build scripts configure one lecture per build tree (lecture_N/build/...).
# Leave LECTURE empty to get every lecture in one tree, e.g. for an IDE.
set(LECTURE "" CACHE STRING "Lecture directory to build (e.g. lecture_3); empty builds all")

if(LECTURE)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LECTURE}/main.cpp
        AND NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LECTURE}/src/main.cpp)
        message(FATAL_ERROR "No such lecture: ${LECTURE}/main.cpp or ${LECTURE}/src/main.cpp")
    endif()

    add_lecture(${LECTURE})
else()
    # CONFIGURE_DEPENDS re-runs the glob on build, so new lectures are picked up.
    file(GLOB lecture_mains CONFIGURE_DEPENDS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        lecture_*/main.cpp
        lecture_*/src/main.cpp
    )

    set(lecture_dirs "")
    foreach(lecture_main IN LISTS lecture_mains)
        string(REGEX REPLACE "/(src/)?main\\.cpp$" "" lecture_dir "${lecture_main}")
        list(APPEND lecture_dirs ${lecture_dir})
    endforeach()
    list(REMOVE_DUPLICATES lecture_dirs)

    foreach(lecture_dir IN LISTS lecture_dirs)
        add_lecture(${lecture_dir})
    endforeach()
endif()
