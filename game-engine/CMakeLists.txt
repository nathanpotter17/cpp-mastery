cmake_minimum_required(VERSION 3.25)

# A standalone project, separate from the lecture_* build at the repo root.
# Configure and build it with ./build.bash.
project(GameEngine LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# --- vendor/SDL3 -------------------------------------------------------------
# Normal variables set before add_subdirectory() act as SDL's options. We build
# a static library with only what a Vulkan window needs, and list every
# subsystem so turning one on later (SDL_AUDIO) is a one-word edit.

set(SDL_SHARED OFF)
set(SDL_STATIC ON)
set(SDL_TEST_LIBRARY OFF)
set(SDL_TESTS OFF)
set(SDL_EXAMPLES OFF)
set(SDL_INSTALL OFF)

set(SDL_VIDEO ON)       # windows, input events
set(SDL_AUDIO OFF)
set(SDL_GPU OFF)        # SDL's own GPU API; we use Vulkan directly
set(SDL_RENDER OFF)     # SDL's 2D renderer
set(SDL_CAMERA OFF)
set(SDL_JOYSTICK OFF)
set(SDL_HAPTIC OFF)
set(SDL_HIDAPI OFF)
set(SDL_POWER OFF)
set(SDL_SENSOR OFF)
set(SDL_DIALOG OFF)
set(SDL_TRAY OFF)

# Video backends: Vulkan surfaces on Wayland (with libdecor for title bars on
# GNOME), or Windows' own backend there. No OpenGL, X11, KMS/DRM or headless.
set(SDL_VULKAN ON)
set(SDL_OPENGL OFF)
set(SDL_OPENGLES OFF)
set(SDL_WAYLAND ON)
set(SDL_WAYLAND_LIBDECOR ON)
set(SDL_X11 OFF)
set(SDL_KMSDRM OFF)
set(SDL_OFFSCREEN OFF)
set(SDL_DUMMYVIDEO OFF)

# Linux desktop integration we don't use: D-Bus (screensaver, portals), IBus
# (input methods), udev (device hotplug), io_uring, PipeWire (audio/camera).
set(SDL_DBUS OFF)
set(SDL_IBUS OFF)
set(SDL_LIBUDEV OFF)
set(SDL_LIBURING OFF)
set(SDL_PIPEWIRE OFF)

# SYSTEM: -Wall and friends don't warn about third-party headers.
add_subdirectory(vendor/SDL3 SYSTEM)

# --- vendor/Vulkan-Headers ---------------------------------------------------
# vulkan.h and vulkan.hpp only. There is nothing to link: vk::raii::Context
# loads the Vulkan loader (libvulkan) at runtime.

add_subdirectory(vendor/Vulkan-Headers SYSTEM)

# --- game-engine -------------------------------------------------------------

# compile_commands.json for clangd. Turned on only now, so it lists our sources
# and not SDL's: clangd gives a header the flags of the closest-named file in
# it, and would parse includes/sdl.h as C, borrowing SDL's SDL.c.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(game-engine)

# src/ is the include root: headers are included as "includes/foo.h".
file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp)
target_sources(game-engine PRIVATE ${sources})
target_include_directories(game-engine PRIVATE src)

target_compile_features(game-engine PRIVATE cxx_std_23)
set_target_properties(game-engine PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(game-engine PRIVATE -Wall -Wextra -Wpedantic)

# vulkan.hpp:
# - NO_CONSTRUCTORS: plain aggregate structs, so designated initializers work.
# - HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS: acquireNextImage/presentKHR return
#   eErrorOutOfDateKHR (e.g. after a resize) instead of throwing it.
target_compile_definitions(game-engine PRIVATE
    VULKAN_HPP_NO_CONSTRUCTORS
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
)

target_link_libraries(game-engine PRIVATE SDL3::SDL3 Vulkan::Headers)

# --- build options -----------------------------------------------------------

if(ENABLE_SANITIZERS)
    target_compile_options(game-engine PRIVATE
        -g
        -O1
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -fsanitize=address,undefined
    )
    target_link_options(game-engine PRIVATE -fsanitize=address,undefined)
endif()

# Link-time optimization for Release builds, when the toolchain supports it.
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported LANGUAGES CXX)

if(ipo_supported)
    set_property(TARGET game-engine PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
