#include "includes/arena.h"
#include "includes/tracer.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <print>
#include <type_traits>
#include <utility>

// --- Three ways to pass an argument ------------------------------------------

// By value: the parameter is a new object, copied from the argument.
void inspect_copy(Tracer tracer) {
    std::println("    inspecting a copy of {}", tracer.name());
}

// By reference: the parameter is another name for the caller's object.
// const: we promise not to change it.
void inspect_reference(const Tracer& tracer) {
    std::println("    inspecting {} through a reference", tracer.name());
}

// By pointer: the parameter holds the object's address, or nullptr for "none".
void inspect_pointer(const Tracer* tracer) {
    if (tracer == nullptr) {
        std::println("    nothing to inspect");
        return;
    }
    std::println("    inspecting {} through a pointer", tracer->name());
}

// --- The lifetimes demo ------------------------------------------------------

void demo_lifetimes() {
    std::println("Scopes:");
    {
        Tracer a{'a'};
        {
            Tracer b{'b'};
            std::println("    end of the inner scope");
        }
        std::println("    end of the outer scope");
    }

    std::println("References and pointers:");
    int count = 3;

    int& alias = count; // a reference: another name for count
    alias += 1;

    int* pointer = &count; // a pointer: holds count's address (&count)
    *pointer += 1;         // *pointer is the object it points to

    std::println("    count = {}, alias = {}, *pointer = {}", count, alias, *pointer);
    std::println("    pointer points at alias: {}", pointer == &alias);

    // A pointer can be re-pointed; a reference is bound once, for good.
    int other = 10;
    pointer = &other;
    *pointer = 11;
    std::println("    after re-pointing: count = {}, other = {}", count, other);

    // const on the left of * protects the value; on the right, the pointer.
    const int* read_only = &count; // *read_only = 1; won't compile
    int* const fixed = &count;     // fixed = &other;  won't compile
    *fixed = 6;
    std::println("    through read_only: {}", *read_only);

    std::println("Passing arguments:");
    const Tracer c{'c'};
    inspect_copy(c);
    inspect_reference(c);
    inspect_pointer(&c);
    inspect_pointer(nullptr);
    std::println("    end of demo_lifetimes");
}

// --- Functions that transfer ownership ---------------------------------------

// Creates a Tracer on the heap. The caller becomes its owner.
std::unique_ptr<Tracer> make_tracer(char name) {
    return std::make_unique<Tracer>(name);
}

// Takes ownership by value: the Tracer is destroyed when this function returns.
void consume(std::unique_ptr<Tracer> tracer) {
    std::println("    consuming {}", tracer->name());
}

// --- RAII for a C resource ---------------------------------------------------

// A function object: closer(file) calls this operator(), closing the file.
struct FileCloser {
    void operator()(std::FILE* file) const {
        std::println("    closing the file");
        std::fclose(file);
    }
};

// --- The rule of zero --------------------------------------------------------

// No special member functions written: the compiler generates them from the
// members. A unique_ptr can't be copied, so neither can an Owner, and moving
// an Owner moves its unique_ptr.
struct Owner {
    std::unique_ptr<Tracer> tracer;
};

static_assert(!std::is_copy_constructible_v<Owner>);
static_assert(std::is_move_constructible_v<Owner>);

// --- The ownership demo ------------------------------------------------------

void demo_ownership() {
    std::println("Raw new and delete:");
    Tracer* raw = new Tracer{'r'};
    std::println("    {} lives on the heap", raw->name());
    delete raw; // without this line, r is never destroyed: a leak

    std::println("Moving:");
    {
        Tracer m{'m'};
        // std::move doesn't move anything: it casts m to Tracer&&, which
        // selects the move constructor, and that does the work.
        const Tracer taken = std::move(m);
        std::println("    m is now {}, taken is {}", m.name(), taken.name());
    }

    std::println("unique_ptr:");
    {
        std::unique_ptr<Tracer> first = make_tracer('u');

        // Ownership moves from first to second. The Tracer itself stays put.
        std::unique_ptr<Tracer> second = std::move(first);
        std::println("    first is empty: {}, second owns {}", first == nullptr, second->name());

        consume(std::move(second));
        std::println("    back from consume, second is empty: {}", second == nullptr);

        const auto kept = make_tracer('k');
        std::println("    end of scope");
    }

    std::println("A custom deleter:");
    {
        // std::tmpfile opens a temporary file that is deleted when it's closed.
        const std::unique_ptr<std::FILE, FileCloser> file{std::tmpfile()};
        if (file) {
            // get() lends the raw pointer to C functions; file still owns it.
            std::fputs("hello", file.get());
            std::println("    wrote {} bytes", std::ftell(file.get()));
        }
    }

    std::println("The rule of zero:");
    {
        Owner owner{std::make_unique<Tracer>('o')};
        const Owner new_owner = std::move(owner);
        std::println("    owner is empty: {}", owner.tracer == nullptr);
    }

    std::println("shared_ptr and weak_ptr:");
    std::weak_ptr<Tracer> observer;
    {
        auto shared = std::make_shared<Tracer>('s');
        const auto another = shared; // copying a shared_ptr shares ownership
        observer = shared;           // a weak_ptr watches without owning
        std::println("    owners: {}", shared.use_count());

        shared.reset();
        std::println("    owners after one reset: {}", another.use_count());

        // lock() returns a shared_ptr: empty if the object is already gone.
        if (const auto locked = observer.lock()) {
            std::println("    observer can still reach {}", locked->name());
        }
    }
    std::println("    observer expired: {}", observer.expired());
}

// --- The same three members, in two orders -----------------------------------

// Each member must start at a multiple of its own alignment, so the compiler
// inserts padding: 7 bytes after tag, and 7 after flag to round the size up
// to a multiple of the struct's alignment (8, from double).
struct Loose {
    char tag;
    double value;
    char flag;
};

// Largest member first: the two chars share one padded tail.
struct Tight {
    double value;
    char tag;
    char flag;
};

static_assert(sizeof(Tight) < sizeof(Loose));

// --- Raising the alignment ---------------------------------------------------

// alignas(16) makes every Vec4 start at a multiple of 16, the alignment SIMD
// instructions want for four floats loaded at once.
struct alignas(16) Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

static_assert(alignof(Vec4) == 16 && sizeof(Vec4) == 16);

// --- Reading raw memory ------------------------------------------------------

// Any object may be read as a sequence of unsigned chars: its bytes.
void print_bytes(const char* label, const void* object, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(object);

    std::print("    {:<12}", label);
    for (std::size_t i = 0; i < size; ++i) {
        // bytes[i] is shorthand for *(bytes + i): pointer arithmetic moves
        // in steps of the pointed-to type, here 1 byte.
        std::print(" {:02x}", bytes[i]);
    }
    std::println();
}

// An address is a number, so alignment is divisibility.
bool is_aligned(const void* address, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(address) % alignment == 0;
}

// --- The alignment demo ------------------------------------------------------

void demo_alignment() {
    std::println("Bytes in memory:");
    const int number = 0x11223344;
    const float one = 1.0f;
    print_bytes("0x11223344", &number, sizeof(number));
    print_bytes("1.0f", &one, sizeof(one));

    std::println("Size, alignment and padding:");
    std::println("    {:<6} {:>4} {:>5}  {}", "type", "size", "align", "member offsets");
    std::println("    {:<6} {:>4} {:>5}  tag {}, value {}, flag {}", "Loose", sizeof(Loose), alignof(Loose),
        offsetof(Loose, tag), offsetof(Loose, value), offsetof(Loose, flag));
    std::println("    {:<6} {:>4} {:>5}  value {}, tag {}, flag {}", "Tight", sizeof(Tight), alignof(Tight),
        offsetof(Tight, value), offsetof(Tight, tag), offsetof(Tight, flag));
    std::println("    {:<6} {:>4} {:>5}  x {}, y {}, z {}, w {}", "Vec4", sizeof(Vec4), alignof(Vec4),
        offsetof(Vec4, x), offsetof(Vec4, y), offsetof(Vec4, z), offsetof(Vec4, w));

    std::println("Aligned addresses:");
    // The largest alignment a plain new or a local variable is guaranteed.
    std::println("    alignof(std::max_align_t) = {}", alignof(std::max_align_t));

    const Vec4 on_stack{};
    const auto on_heap = std::make_unique<Vec4>();
    std::println("    Vec4 on the stack is 16-aligned: {}", is_aligned(&on_stack, alignof(Vec4)));
    std::println("    Vec4 on the heap is 16-aligned:  {}", is_aligned(on_heap.get(), alignof(Vec4)));

    // assert stops the program if its condition is false: a check for bugs
    // in our own code. The string is always true; it only labels the failure.
    assert(is_aligned(on_heap.get(), alignof(Vec4)) && "new must respect alignas");
}

// --- The arena in use --------------------------------------------------------

void demo_arena() {
    std::println("Arena:");
    Arena arena{64};

    char* letter = arena.create<char>();
    Vec4* vector = arena.create<Vec4>();
    double* number = arena.create<double>();
    Tight* tight = arena.create<Tight>();

    *letter = 'x';
    vector->w = 1.0f;
    *number = 2.5;
    tight->tag = 't';

    std::println("    char   at offset {:>2}", arena.offset_of(letter));
    std::println("    Vec4   at offset {:>2}  (15 bytes of padding first)", arena.offset_of(vector));
    std::println("    double at offset {:>2}", arena.offset_of(number));
    std::println("    Tight  at offset {:>2}", arena.offset_of(tight));
    std::println("    used {} of {} bytes", arena.used(), arena.capacity());

    // The next Vec4 would start at 64, the end of the block.
    const Vec4* overflow = arena.create<Vec4>();
    std::println("    another Vec4 fits: {}", overflow != nullptr);

    // Moving hands over the block; the objects in it don't move at all.
    const Arena moved = std::move(arena);
    std::println("    after the move: arena holds {} bytes, moved holds {}", arena.capacity(), moved.capacity());
    std::println("    the Vec4 is still at offset {}, w = {}", moved.offset_of(vector), vector->w);
}

// --- The program -------------------------------------------------------------

int main() {
    // The preprocessor runs before the compiler. CMake's release builds
    // define NDEBUG, which turns every assert() into nothing.
#ifdef NDEBUG
    std::println("Release build: assert() is disabled\n");
#else
    std::println("Debug build: assert() is enabled\n");
#endif

    demo_lifetimes();
    std::println();
    demo_ownership();
    std::println();
    demo_alignment();
    std::println();
    demo_arena();

    return EXIT_SUCCESS;
}
