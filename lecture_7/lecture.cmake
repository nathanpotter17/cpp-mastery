# Included by add_lecture() before vendor/ is added, so normal variables set
# here act as options for the vendored projects (and stay out of the cache,
# so they don't leak into other lectures). ${name} is this lecture's target.
#
# vendor/SDL3            -> target SDL3::SDL3      (linked automatically)
# vendor/Vulkan-Headers  -> target Vulkan-Headers  (linked automatically)

# SDL3: a static library with only what a Vulkan window needs. Every
# subsystem is listed, so turning one on later (SDL_AUDIO) is a one-word edit.
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

# vulkan.hpp:
# - NO_CONSTRUCTORS: plain aggregate structs, so designated initializers work.
# - HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS: acquireNextImage/presentKHR return
#   eErrorOutOfDateKHR (e.g. after a resize) instead of throwing it.
target_compile_definitions(${name} PRIVATE
    VULKAN_HPP_NO_CONSTRUCTORS
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
)
