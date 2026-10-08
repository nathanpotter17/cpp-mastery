#include "includes/arena.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

// --- Acquire in the constructor, release in the destructor -------------------

// The member initializer list (after the :) initializes the members before
// the body runs.
Arena::Arena(std::size_t capacity) : memory_{new std::byte[capacity]}, capacity_{capacity} {}

// delete[] matches new[]. Deleting nullptr does nothing, so a moved-from
// arena is destroyed safely.
Arena::~Arena() {
    delete[] memory_;
}

// --- Moving ------------------------------------------------------------------

// std::exchange(x, v) sets x to v and returns x's old value: we take other's
// block and leave other empty, so only one arena ever owns it.
Arena::Arena(Arena&& other) noexcept
    : memory_{std::exchange(other.memory_, nullptr)},
      capacity_{std::exchange(other.capacity_, 0)},
      used_{std::exchange(other.used_, 0)} {}

Arena& Arena::operator=(Arena&& other) noexcept {
    // Moving an arena into itself must not free its own block.
    if (this != &other) {
        delete[] memory_;
        memory_ = std::exchange(other.memory_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
        used_ = std::exchange(other.used_, 0);
    }
    return *this;
}

// --- Allocating --------------------------------------------------------------

void* Arena::allocate(std::size_t size, std::size_t alignment) {
    // A bad alignment is a bug in the caller, so we assert. A power of two
    // has one bit set, and x & (x - 1) clears the lowest set bit.
    assert(alignment != 0 && (alignment & (alignment - 1)) == 0 && "alignment must be a power of two");
    assert(memory_ != nullptr && "allocating from a moved-from arena");

    // Round the next free address up to a multiple of alignment: add
    // alignment - 1, then clear the low bits with the mask ~(alignment - 1).
    const auto next = reinterpret_cast<std::uintptr_t>(memory_ + used_);
    const std::uintptr_t aligned = (next + alignment - 1) & ~(alignment - 1);
    const std::size_t padding = aligned - next;

    // Running out of space is a normal outcome, not a bug: report it.
    if (padding + size > capacity_ - used_) {
        return nullptr;
    }

    std::byte* result = memory_ + used_ + padding;
    used_ += padding + size;
    return result;
}

std::size_t Arena::offset_of(const void* pointer) const {
    // Subtracting two pointers into the same block counts the bytes between them.
    return static_cast<std::size_t>(static_cast<const std::byte*>(pointer) - memory_);
}
