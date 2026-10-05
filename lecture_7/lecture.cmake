# Included by add_lecture() before vendor/ is added, so normal variables set
# here act as options for the vendored projects (and stay out of the cache,
# so they don't leak into other lectures). ${name} is this lecture's target.
#
# vendor/glfw            -> target glfw            (linked automatically)
# vendor/Vulkan-Headers  -> target Vulkan-Headers  (linked automatically)

# GLFW: we only want the library, not its docs or install rules.
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)

# GLFW's X11 backend refuses to configure without the Xinerama headers.
# Build Wayland-only when they're missing rather than failing outright;
# `sudo apt install libxinerama-dev` brings X11 back on the next configure.
if(UNIX AND NOT APPLE)
    find_path(XINERAMA_INCLUDE_DIR X11/extensions/Xinerama.h)

    if(XINERAMA_INCLUDE_DIR)
        set(GLFW_BUILD_X11 ON)
    else()
        message(STATUS "${name}: no Xinerama headers, building GLFW without X11")
        set(GLFW_BUILD_X11 OFF)
    endif()
endif()

# vulkan.hpp: plain aggregate structs, so C++20 designated initializers work.
target_compile_definitions(${name} PRIVATE VULKAN_HPP_NO_CONSTRUCTORS)
