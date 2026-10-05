// stb's implementations, compiled once here instead of in main.cpp.
// Headers in src/includes get our full warning set, so quiet the one
// warning stb's implementation trips (GCC and Clang both read these pragmas).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "includes/stb_image.h"
#include "includes/stb_image_write.h"
#pragma GCC diagnostic pop
