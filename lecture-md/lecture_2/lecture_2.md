# Lecture 2: Memory, ownership and RAII

Every object in a C++ program lives somewhere in memory, for some stretch of time, and something has to end that time and give the memory back. This lecture is about those three things: where objects live, how we refer to them, and who cleans up.

By the end we'll have a program that shows each idea as it happens. A class announces its own construction, copying, moving and destruction, so the output is a trace of every object's life. Along the way we cover:
- scopes and lifetimes, the stack and the heap;
- references and pointers, `const` with both, and `nullptr`;
- passing arguments by value, by reference and by pointer;
- raw `new` and `delete`, and why we stop using them;
- **RAII**, move semantics, and the rules of five and zero;
- `std::unique_ptr` (with a custom deleter), `std::shared_ptr` and `std::weak_ptr`;
- reading raw bytes, `sizeof`, `alignof`, padding and `alignas`;
- `assert`, and what it's for.

We finish by writing a memory arena, the kind of allocator game engines use, which puts ownership, alignment and `assert` to work together.

## 2.1 Project layout: headers and source files

### Why
This lecture writes two types that deserve their own files: a class that traces its own lifetime, and a memory arena. Splitting them out of `main.cpp` keeps each one readable and reusable. It also shows how C++ is really built, from separately compiled files joined at the end. Everything else, the demos that exercise the two types, stays in `main.cpp`.

### How
- **The layout:** from this lecture on, a lecture's code lives under `lecture_N/src/`.
  - The root CMake compiles `src/main.cpp` and every other `.cpp` file under `src/`.
  - It puts `src/` on the include path, so our headers in `src/includes/` are included as `"includes/arena.h"`. Quotes, rather than angle brackets, mark our own headers.
- **The files:**
  - `src/includes/tracer.h`: the `Tracer` class, entirely in the header.
  - `src/includes/arena.h` and `src/arena.cpp`: the `Arena` class, declared in the header and defined in the source file.
  - `src/main.cpp`: every demo, and `main`.
- **Translation units:** the compiler turns each `.cpp` file, with everything it `#include`s pasted in, into an object file. It sees one file at a time. The *linker* then joins the object files into the executable, matching each call to the one definition of that function, wherever it was compiled.
- **Headers hold declarations:** a header tells every file that includes it what exists and how to use it. `arena.h` declares `Arena`'s member functions; `arena.cpp` defines them, once.
- **What may be defined in a header:** a function defined in a header ends up in every translation unit that includes it, so the linker would find several definitions. Functions defined *inside* a class body, `constexpr` functions, templates, and anything marked `inline` are exempt from that rule, which is why `tracer.h` needs no `.cpp` file.
- **`#pragma once`** makes sure a header's text is pasted only once per translation unit, even if several headers include it.

### Code
From the repo root:
```bash
mkdir -p lecture_2/src/includes
```

`lecture_2/.clangd`:
```yaml
# clangd reads compile flags from the clang debug build (build-debug-clang.bash).
CompileFlags:
  CompilationDatabase: build/debug-clang
```

## 2.2 Watching objects live and die: `src/includes/tracer.h`

### Why
Object lifetimes are invisible: a copy here, a destructor there, and nothing tells us. To *see* them, we write a class that prints a line from each of its special member functions. Every later demo uses it.

### How
- **`class` vs `struct`:** they're the same thing, except that a `class`'s members are private by default. `public:` members can be used by anyone; `private:` ones only by the class's own member functions. `Tracer` keeps its name private, and offers read access through `name()`.
- **Constructors** run when an object is created. `explicit Tracer(char name)` takes the name. The *member initializer list*, `: name_{name}`, initializes the members before the constructor's body runs. `explicit` stops a lone `char` converting silently into a `Tracer`.
- **The destructor**, `~Tracer()`, runs automatically when the object's lifetime ends. We never call it ourselves. This is the most important function in C++: everything else in this lecture is built on it.
- **The copy constructor**, `Tracer(const Tracer& other)`, creates a new object as a copy of an existing one.
- **The move constructor**, `Tracer(Tracer&& other)`, creates a new object by *taking over* `other`'s contents. `Tracer&&` is an *rvalue reference*: it binds to objects that are about to disappear, or that we've explicitly given up with `std::move` (2.4). A moved-from object is still alive, and its destructor still runs, so the move constructor must leave it in a safe state. Ours marks it with the name `_`.
- **`noexcept`** promises the move constructor never throws an exception. When a `std::vector` grows (lecture 3), it moves its elements only if their move constructor promises this, and copies them otherwise. A type that can't be copied is moved anyway.
- **`= delete`** removes a function. We don't need assignment between `Tracer`s, so we delete both assignment operators; code that tries one won't compile.
- **The special member functions:** the destructor, the copy constructor, copy assignment, the move constructor and move assignment. If we don't declare them, the compiler writes them for us, member by member. Declaring some affects which others it writes, which is why `Tracer` lists all of them, either defined or deleted.

### Code
`lecture_2/src/includes/tracer.h`:
```cpp
#pragma once

#include <print>

// --- A class that reports its own lifetime -----------------------------------

// Every special member function prints a line, so the program's output shows
// exactly when objects are created, copied, moved and destroyed.
class Tracer {
public:
    // explicit: a char can't silently turn into a Tracer.
    explicit Tracer(char name) : name_{name} {
        std::println("    [{}] constructed", name_);
    }

    // Copy constructor: a new object with the same value as `other`.
    Tracer(const Tracer& other) : name_{other.name_} {
        std::println("    [{}] copied", name_);
    }

    // Move constructor: takes over `other`'s contents. `other` stays alive and
    // will still be destroyed, so we mark it as emptied.
    Tracer(Tracer&& other) noexcept : name_{other.name_} {
        other.name_ = '_';
        std::println("    [{}] moved", name_);
    }

    // We never assign one Tracer to another, so we forbid it.
    Tracer& operator=(const Tracer&) = delete;
    Tracer& operator=(Tracer&&) = delete;

    // Destructor: runs automatically when the object's lifetime ends.
    ~Tracer() {
        std::println("    [{}] destroyed", name_);
    }

    char name() const {
        return name_;
    }

private:
    char name_;
};
```

## 2.3 Scopes, references and pointers

### Why
Before we manage memory ourselves, we need to know how C++ manages it for us, and the two ways of referring to an object without copying it.

### How
- **`main.cpp` begins** with its includes: our two headers first, then the standard ones. `arena.h` is written in 2.6.
- **Automatic storage ("the stack"):** a variable declared in a function lives until the end of its enclosing block, the closing `}`. Then its destructor runs. Objects in the same block are destroyed in the reverse order of their construction, so the newest object goes first. A bare `{ ... }` block is the simplest way to end lifetimes early.
- **References:** `int& alias = count;` makes `alias` another name for `count`. Anything done to `alias` happens to `count`. A reference must be initialized, can't be null, and can never be re-bound to another object.
- **Pointers:** `int* pointer = &count;` stores the address of `count`. `&` takes an address, and `*pointer` (dereferencing) is the object at that address. Unlike a reference, a pointer can be re-pointed at another object, or set to `nullptr`, meaning "points at nothing". Dereferencing `nullptr` is undefined behavior, so code that receives a pointer checks it first.
- **`const` and pointers:** read the declaration right to left.
  - `const int* read_only` is a pointer to a `const int`: we can't change the value through it, but we can re-point it.
  - `int* const fixed` is a `const` pointer to an `int`: we can change the value, but not where it points.
- **`->`:** `tracer->name()` is shorthand for `(*tracer).name()`, calling a member through a pointer.
- **Passing arguments:**
  - **By value** (`Tracer tracer`): the parameter is a copy. The output shows `[c] copied` on the way in and `[c] destroyed` on the way out.
  - **By `const` reference** (`const Tracer& tracer`): no copy, and no changes. This is the default for passing anything bigger than a few numbers.
  - **By pointer** (`const Tracer* tracer`): no copy, and the caller may pass `nullptr` for "no object". Use it when "nothing" is a valid argument.
- **Lifetimes end at `}` regardless of how the block is left.** `c` is destroyed after the demo's last line, at the closing brace.
- **Dangling:** a reference or pointer to an object whose lifetime has ended *dangles*, and using it is undefined behavior. The classic mistake is returning a reference to a local variable, which is destroyed as the function returns. Compilers warn about the simple cases, and the sanitizer build (`build-debug-clang-ub.bash`) catches many of the rest at run time.

### Code
`lecture_2/src/main.cpp`, part 1:
```cpp
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
```

## 2.4 Ownership

### Why
Automatic storage ends lifetimes at the end of a scope, but sometimes an object must outlive the function that creates it, or must be big, or its count must be decided at run time. Those objects go on the *heap* (the free store), and their lifetime is ours to end. The question every C++ program must answer is: who *owns* each heap object, and so is responsible for destroying it?

### How
- **`new` and `delete`:**
  - `new Tracer{'r'}` allocates memory on the heap, constructs a `Tracer` in it, and returns a pointer.
  - `delete raw` runs the destructor and frees the memory.
  - **The problem:** nothing makes us write the `delete`. Forget it and the object leaks. An early `return`, or an exception, between the `new` and the `delete` leaks it too. And deleting twice is undefined behavior. Modern C++ code almost never writes `delete`.
- **RAII ("Resource Acquisition Is Initialization"):** tie each resource to an object whose destructor releases it. Because destructors run automatically at the end of a lifetime, the release can't be forgotten, skipped, or done twice. The resource can be memory, a file, a lock or a GPU buffer; the pattern is the same.
- **Move semantics:**
  - **What `std::move` does:** it moves nothing. It casts its argument to an rvalue reference (`Tracer&&`), and that selects the move constructor, which does the work.
  - **After a move:** `m` still exists, with whatever contents the move left it. Our move constructor leaves the name `_`, so the output shows `m is now _`.
  - **Moved-from objects:** they can be destroyed or assigned a new value. Don't rely on what's in them otherwise.
- **`std::unique_ptr<T>`:** the standard RAII owner of a single heap object.
  - **Creating:** `std::make_unique<Tracer>('u')` allocates and constructs in one step, and returns the owning `unique_ptr`.
  - **Freeing:** the destructor of the `unique_ptr` deletes the object. There's no `delete` anywhere in our code.
  - **Only moves:** it can't be copied, since two owners would both delete. It can be moved, which transfers ownership: after `second = std::move(first)`, `first` is empty and compares equal to `nullptr`. The `Tracer` itself never moves: only the pointer does, so the output has no `[u] moved` line.
- **Ownership in function signatures:**
  - **Returning** a `unique_ptr` (`make_tracer`) hands ownership to the caller.
  - **Taking one by value** (`consume`) takes ownership from the caller, who must `std::move` it in. The object is destroyed when `consume` returns, so `[u] destroyed` appears before `back from consume`.
  - **A raw pointer or a reference** says "use it, don't own it". `get()` produces a raw pointer for code that only borrows.
- **A custom deleter:**
  - **Why:** C libraries hand out resources through pointers with their own release functions. `std::tmpfile` returns a `std::FILE*` that must be passed to `std::fclose`.
  - **How:** `std::unique_ptr<std::FILE, FileCloser>` calls `FileCloser{}(pointer)` instead of `delete`.
  - **The deleter:** `FileCloser` is a *function object*: a struct with an `operator()`, so its objects can be called like functions.
- **The rule of five and the rule of zero:**
  - **Five:** a class that directly manages a resource must define or delete all five special member functions, because the compiler's versions would just copy the raw pointer. We write one in 2.6 and 2.7.
  - **Zero:** every other class should define *none* of them, and leave resource management to members that already do it right. `Owner` holds a `unique_ptr` and declares nothing, so it's automatically movable and not copyable, as the `static_assert`s confirm. Those checks use type traits from `<type_traits>`, compile-time questions about types, which lecture 4 explores.
- **`std::shared_ptr` and `std::weak_ptr`:**
  - **`shared_ptr`:** for the rare object with several owners and no obvious last one. Copying a `shared_ptr` adds an owner, and `use_count()` counts them. The object is destroyed when the last owner lets go, here when `another` goes out of scope.
  - **`weak_ptr`:** observes a shared object without keeping it alive. `lock()` gives a temporary `shared_ptr` if the object still exists, and an empty one otherwise.
  - **The cost:** this extra machinery and bookkeeping is why `unique_ptr` is the default and `shared_ptr` the exception.

### Code
`lecture_2/src/main.cpp`, part 2:
```cpp
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
```

## 2.5 Memory layout and alignment

### Why
Objects are bytes at addresses, and the hardware has opinions about which addresses each type may use. A `double` should start at a multiple of 8, and SIMD instructions that load four floats at once want a multiple of 16. The compiler follows those rules for us, and the rules change how big our structs are.

### How
- **Alignment:** every type has an alignment, a power of two that each object's address must be a multiple of. `alignof(T)` reports it. A `char` can go anywhere (1), and a `double` needs a multiple of 8.
- **Padding:** a struct's members are laid out in declaration order, each at the next offset that suits its alignment. The compiler fills the gaps with unused *padding* bytes. The struct's own alignment is its largest member's, and its size is rounded up to a multiple of it, so in an array, every element is aligned.
  - **`Loose`** puts a `char`, a `double` and a `char` in that order: 1 byte, 7 padding bytes, 8 bytes, 1 byte, then 7 more padding bytes, 24 in total.
  - **`Tight`** has the same members, largest first. The two `char`s share one padded tail, so it's 16.
  - **The general rule:** ordering members from largest to smallest alignment minimizes padding.
- **`alignas(16)`** raises a type's alignment. `Vec4` is four floats (16 bytes, alignment 4), but with `alignas(16)` every `Vec4` starts at a multiple of 16.
- **`static_assert`** checks each of these claims during compilation, so the code documents its own layout.
- **Seeing it:** we've claimed that objects are bytes and addresses are numbers. The alignment demo looks at both directly: an object's raw bytes, each member's offset, and whether real addresses honor the alignment rules.
- **Reading bytes:**
  - **The pointer:** `print_bytes` takes `const void*`, a pointer to anything, and converts it to `const unsigned char*`. C++ allows reading any object's bytes through an `unsigned char` pointer.
  - **Indexing:** `bytes[i]` is shorthand for `*(bytes + i)`. Pointer arithmetic moves in steps of the pointed-to type, here one byte.
  - **What we see:** `0x11223344` is stored as `44 33 22 11`, lowest byte first. That's *little-endian* order, which x86 and ARM machines use. `1.0f` is `00 00 80 3f`: floats are stored as a sign, an exponent and a fraction, not as anything resembling `1`.
- **`offsetof(Type, member)`** gives a member's distance from the start of the struct in bytes. The offsets show `Loose`'s padding exactly where we said it would be.
- **Addresses are numbers.** `reinterpret_cast<std::uintptr_t>(address)` reinterprets a pointer as an unsigned integer type big enough to hold any address. Then "aligned to 16" is just "divisible by 16". `reinterpret_cast` is the bluntest cast there is, used only for low-level code like this.
- **`std::max_align_t`** has the largest alignment any built-in type needs: 16 on our platforms. Plain `new` guarantees at least that much for any type. Local variables honor any `alignas`, and since C++17 so does `new`, so a `Vec4` is 16-aligned on the stack and on the heap.
- **`assert(condition)`** from `<cassert>` stops the program with an error message, naming the file, line and condition, if the condition is false.
  - **What it's for:** it checks something that can only be false if *our own code* has a bug. Bad input or a full disk is a normal event to handle, not a bug.
  - **The message:** `&& "message"` adds a message to the condition. A string literal converts to `true`, so it doesn't change the result, but it's printed with the failure.
  - **Release builds:** CMake's release builds define `NDEBUG`, which turns every `assert` into nothing, so asserts cost nothing in release builds. Never put code with a needed side effect inside one.
  - **Compared with `static_assert`:** `static_assert` checks at compile time, and `assert` at run time, in debug builds.

### Code
`lecture_2/src/main.cpp`, part 3:
```cpp
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
```

## 2.6 A memory arena: `src/includes/arena.h`

### Why
A general-purpose `new` must handle any size, in any order, freed at any time, and that flexibility has a cost. Lots of data in a game engine is simpler than that: it's created during one frame and all thrown away at the end of it. An *arena* (or bump allocator) serves that pattern almost for free. It owns one block of memory, hands out pieces of it front to back, and frees everything at once. Writing one uses every idea in this lecture.

### How
- **The class's job:** `Arena` owns a raw block of memory. That makes it a resource-managing class, so the rule of five applies:
  - **No copying:** both copy operations are deleted. Two arenas sharing a block would both `delete[]` it.
  - **Moving:** both move operations are declared, to transfer the block.
  - **Freeing:** the destructor frees the block.
- **The interface:**
  - `allocate(size, alignment)` returns raw memory, or `nullptr` when the arena is full.
  - `used()` and `capacity()` report progress.
  - `offset_of()` turns an address back into a position in the block. Real addresses change from run to run, but offsets don't, so our output is the same every time.
- **A member function template:**
  - **The parameter:** `template <typename T>` before `create` makes `T` a parameter chosen at the call: `arena.create<Vec4>()`. The compiler stamps out one version of `create` per type used. Templates go in headers, because the compiler needs the body to stamp them out.
  - **What it does:** `create<T>()` allocates `sizeof(T)` bytes at `alignof(T)`, then constructs a `T` there.
- **Placement new:** `new (memory) T{}` constructs an object in memory we already have, instead of allocating. Lecture 3's containers do exactly this internally.
- **Destructors:**
  - **The problem:** the arena never runs the destructors of the objects in it, which is only correct for types whose destructors do nothing.
  - **The guard:** `std::is_trivially_destructible_v<T>` asks the compiler whether `T`'s destructor does nothing, and the `static_assert` turns "this would leak" into a compile error. `create<std::string>()` wouldn't build.

### Code
`lecture_2/src/includes/arena.h`:
```cpp
#pragma once

#include <cstddef>
#include <new>
#include <type_traits>

// --- A bump allocator --------------------------------------------------------

// An arena owns one block of memory and hands out pieces of it from front to
// back. Pieces are never freed one by one: the whole block is freed when the
// arena is destroyed. Game engines use arenas for short-lived data, such as
// everything one frame needs.
class Arena {
public:
    explicit Arena(std::size_t capacity);

    // The rule of five: a class that owns a raw resource writes (or deletes)
    // all five special member functions, because the generated ones would
    // just copy the pointer.
    Arena(const Arena&) = delete; // two owners would both delete[] the block
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;
    ~Arena();

    // `size` bytes starting at a multiple of `alignment`, or nullptr if the
    // arena is full. `alignment` must be a power of two.
    void* allocate(std::size_t size, std::size_t alignment);

    // A member function template: the caller picks T, as in create<Vec4>().
    template <typename T>
    T* create() {
        // The arena never runs destructors, so T must not need one.
        static_assert(std::is_trivially_destructible_v<T>);

        void* memory = allocate(sizeof(T), alignof(T));
        if (memory == nullptr) {
            return nullptr;
        }

        // Placement new constructs a T in memory we already have.
        return new (memory) T{};
    }

    std::size_t used() const {
        return used_;
    }

    std::size_t capacity() const {
        return capacity_;
    }

    // Where a pointer from this arena lies, in bytes from the block's start.
    std::size_t offset_of(const void* pointer) const;

private:
    std::byte* memory_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t used_ = 0;
};
```

## 2.7 The arena's implementation: `src/arena.cpp`

### Why
The non-template member functions are defined in a `.cpp` file, compiled once (2.1). This is where the rule of five is written out, and where alignment becomes arithmetic.

### How
- **Defining members outside the class:** `arena.cpp` includes its own header, so the compiler checks every definition against its declaration. `Arena::allocate` names the class it belongs to, and inside the body, members are used directly.
- **Acquire and release:**
  - **Acquire:** the constructor allocates the block with `new std::byte[capacity]`. `std::byte` is a byte that isn't a character or a number, just raw memory.
  - **Release:** the destructor frees it with `delete[]`, which matches `new[]`. Mixing up `delete` and `delete[]` is undefined behavior.
- **Moving with `std::exchange`:**
  - **`std::exchange(x, v)`** from `<utility>` sets `x` to `v` and returns `x`'s old value.
  - **The move constructor** takes all three members from `other`, and leaves it empty in the same step.
  - **The moved-from arena** has a null block, and `delete[]` of `nullptr` does nothing, so destroying it is safe.
  - **Move assignment** must first free the block this arena already owns. It must also not free its own block when an arena is moved into itself, hence `this != &other`. `this` is a pointer to the object the member function was called on.
- **Aligning:**
  - **Rounding up:** take the address of the next free byte, add `alignment - 1`, and clear the low bits with the mask `~(alignment - 1)`. That rounds up to the next multiple of `alignment`. It only works for powers of two, whose `alignment - 1` is all low bits set.
  - **The padding** is the difference between the rounded and the original address.
  - **The bit trick in the assert:** a power of two has exactly one bit set, and `x & (x - 1)` clears the lowest set bit, so it's zero only for powers of two.
- **Two kinds of failure:**
  - **Bugs: assert.** A bad alignment, or allocating from a moved-from arena, can only come from a bug in the calling code, so `allocate` asserts.
  - **Normal outcomes: report them.** Running out of space is a normal condition the caller must handle, so `allocate` returns `nullptr`. Asserting there would be wrong: the check would vanish in release builds, exactly when real data might fill the arena.
- **Pointer subtraction:** subtracting two pointers into the same block gives the number of elements between them, here bytes. That's how `offset_of` works.

### Code
`lecture_2/src/arena.cpp`:
```cpp
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
```

## 2.8 The program

### Why
The last part of `main.cpp` puts the arena to work, and `main` runs every demo in order. Creating objects of different alignments shows the padding the arena inserts. The final steps show it filling up, and being moved.

### How
- **Declare before use:** each demo is defined above `main`, so `main` can call it without any extra declarations, just as in lecture 1.
- **The preprocessor:** `#ifdef NDEBUG ... #else ... #endif` is handled by the preprocessor, before compilation. Only one of the two branches is compiled, depending on whether `NDEBUG` is defined. The first line of output tells us whether this build's `assert`s are active.
- **The arena demo:**
  - **Filling it:** a `char`, a `Vec4`, a `double` and a `Tight` go into a 64-byte arena. The `Vec4` needs a 16-aligned address, so 15 bytes after the `char` are skipped.
  - **Using the objects:** they're written through the returned pointers, like any other objects.
  - **Running out:** after 56 bytes, the next `Vec4` would start at 64, the end of the block, so `create` returns `nullptr`, and we check for it.
  - **Moving it:** moving the arena transfers the block. The `Vec4` is still at the same offset, and its value is unchanged: no object in the arena moved, only the owner changed.
- **Offsets are the same on every run:** `new std::byte[64]` returns memory aligned for any type of up to 64 bytes with an ordinary alignment, so the block starts 16-aligned.

### Code
`lecture_2/src/main.cpp`, part 4:
```cpp
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
```

## 2.9 Build and run

```bash
./build-scripts-linux/build-debug-clang.bash lecture_2 && ./lecture_2/build/debug-clang/lecture_2
```

```text
Debug build: assert() is enabled

Scopes:
    [a] constructed
    [b] constructed
    end of the inner scope
    [b] destroyed
    end of the outer scope
    [a] destroyed
References and pointers:
    count = 5, alias = 5, *pointer = 5
    pointer points at alias: true
    after re-pointing: count = 5, other = 11
    through read_only: 6
Passing arguments:
    [c] constructed
    [c] copied
    inspecting a copy of c
    [c] destroyed
    inspecting c through a reference
    inspecting c through a pointer
    nothing to inspect
    end of demo_lifetimes
    [c] destroyed

Raw new and delete:
    [r] constructed
    r lives on the heap
    [r] destroyed
Moving:
    [m] constructed
    [m] moved
    m is now _, taken is m
    [m] destroyed
    [_] destroyed
unique_ptr:
    [u] constructed
    first is empty: true, second owns u
    consuming u
    [u] destroyed
    back from consume, second is empty: true
    [k] constructed
    end of scope
    [k] destroyed
A custom deleter:
    wrote 5 bytes
    closing the file
The rule of zero:
    [o] constructed
    owner is empty: true
    [o] destroyed
shared_ptr and weak_ptr:
    [s] constructed
    owners: 2
    owners after one reset: 1
    observer can still reach s
    [s] destroyed
    observer expired: true

Bytes in memory:
    0x11223344   44 33 22 11
    1.0f         00 00 80 3f
Size, alignment and padding:
    type   size align  member offsets
    Loose    24     8  tag 0, value 8, flag 16
    Tight    16     8  value 0, tag 8, flag 9
    Vec4     16    16  x 0, y 4, z 8, w 12
Aligned addresses:
    alignof(std::max_align_t) = 16
    Vec4 on the stack is 16-aligned: true
    Vec4 on the heap is 16-aligned:  true

Arena:
    char   at offset  0
    Vec4   at offset 16  (15 bytes of padding first)
    double at offset 32
    Tight  at offset 40
    used 56 of 64 bytes
    another Vec4 fits: false
    after the move: arena holds 0 bytes, moved holds 64
    the Vec4 is still at offset 16, w = 1
```

Reading the output:
- **Scopes:** `b` is destroyed at the inner `}` and `a` at the outer one, newest first.
- **Passing arguments:** only the by-value call produces a `copied` and a matching `destroyed`.
- **Moving:** `m` is moved into `taken`, and both are destroyed at the end of the block: `taken` first, as `[m]`, then the emptied `m` as `[_]`.
- **unique_ptr:** ownership moved twice without a single `moved` line from `Tracer`, because only the pointer moved.
- **shared_ptr:** `s` survived the first `reset()`, because `another` still owned it, and was destroyed when `another` went out of scope. The `weak_ptr` then reported it expired.
- **Bytes:** little-endian integers, and padding exactly where we predicted it.
- **Arena:** the `char` at 0, the `Vec4` at 16 after 15 bytes of padding, then the `double` and the `Tight` packed behind it, 56 of 64 bytes used. The extra `Vec4` doesn't fit.

A release build prints the same, except for the first line:
```bash
./build-scripts-linux/build-release-clang.bash lecture_2 && ./lecture_2/build/release-clang/lecture_2 | head -1
```

```text
Release build: assert() is disabled
```

In lecture 3 we move from single objects to many: arrays, vectors, spans and strings, all of which are contiguous memory with an owner.
