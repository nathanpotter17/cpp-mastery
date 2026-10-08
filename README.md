# CPP Mastery - C++ 23

## Goals

- Demonstrate modern C++ mastery through various C++ 23 projects.
- Demonstrate comprehensive understanding of CMake 3+.
- Demonstrate mastery of both g++ and Clang/LLVM toolchains.

## Requirements

- VS Code (VSCodium) with the clangd, CMake, and Slang extensions
- Ubuntu Linux (Wayland session) or Windows 11 (MSYS2 / MinGW-w64)
- C++23 toolchain:
    - CMake 3.25+ and Ninja
    - g++ 14+
    - Clang 19+ and lld
- Vulkan (lecture 7 and beyond only):
    - A Vulkan loader and a GPU driver with Vulkan 1.4
    - The Vulkan validation layers for debug builds(`vulkan-validationlayers` on Ubuntu)
    - On Linux, SDL3's Wayland backend needs `pkg-config`, `libwayland-dev`, `libxkbcommon-dev` and `libdecor-0-dev`
    - game-engine only: `slangc` 2026.19
    - game-engine only: zlib development files, for PNG decoding (`zlib1g-dev` on Ubuntu)

## Build System

Run `./build.bash`, choose a lecture, then a configuration: 1–6 build, 7–12 build and run (Enter picks 7, Debug Clang). "All lectures" offers the builds only. `game-engine/build.bash` has the same menu for the engine.
