// Vulkanizer's device heap: what kernels' malloc and free become (src/ptx/heap.cpp appends these
// functions to modules that call them, the inliner calls them in their place). The heap is one
// buffer per device (cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE), 8 MiB by default), whose address the
// driver puts into __vulkanizer_heap when it loads the module.
//
// Blocks are powers of two from 32 bytes on, with a 16-byte header (its size class, and the next
// free block while it is free), so the pointers are 16-byte aligned as CUDA's. free puts a block on
// its class's free list, malloc takes one from there or else cuts a new one from the top. The list
// heads are 64-bit: a 32-bit offset into the heap and a counter, changed with compare-and-swap, so
// a head that was taken and put back meanwhile (ABA) does not match. Memory is never given back to
// the top; a request that does not fit returns NULL.
//
// Compiled to PTX (heap_lib.ptx, which the build embeds) with tools/gen_heap_lib.sh:
//   .cuda/bin/nvcc -ptx -rdc=true -arch=compute_61 -O3 src/ptx/heap_lib.cu -o src/ptx/heap_lib.ptx
// (-rdc keeps the functions although nothing here calls them).

namespace {

constexpr unsigned kClasses = 32;  // 2^0 .. 2^31 bytes (from 2^5 on in use)

// At the start of the buffer; the driver sets size, and top to the first block (after this header).
struct Heap {
    unsigned long long size;              // bytes in the buffer
    unsigned long long top;               // offset of the first never used byte
    unsigned long long free[kClasses];    // list heads: counter << 32 | offset (0: empty)
};

__device__ __forceinline__ unsigned sizeClass(unsigned long long bytes) {
    unsigned c = 5;
    while (c < kClasses - 1 && (1ull << c) < bytes) ++c;
    return c;
}

}  // namespace

extern "C" {

__device__ unsigned long long __vulkanizer_heap;  // set by the driver (0: no heap)

__device__ void* __vulkanizer_malloc(unsigned long long bytes) {
    Heap* h = reinterpret_cast<Heap*>(*(volatile unsigned long long*)&__vulkanizer_heap);
    if (!h || bytes > (1ull << 30)) return nullptr;
    char* base = reinterpret_cast<char*>(h);
    const unsigned c = sizeClass(bytes + 16);
    unsigned long long head = *(volatile unsigned long long*)&h->free[c];
    while (unsigned(head)) {
        const unsigned off = unsigned(head);
        const unsigned next = *(volatile unsigned*)(base + off + 4);
        const unsigned long long taken = (((head >> 32) + 1) << 32) | next;
        const unsigned long long seen = atomicCAS(&h->free[c], head, taken);
        if (seen == head) return base + off + 16;
        head = seen;
    }
    const unsigned long long off = atomicAdd(&h->top, 1ull << c);
    if (off + (1ull << c) > h->size) return nullptr;
    *(unsigned*)(base + off) = c;
    return base + off + 16;
}

__device__ void __vulkanizer_free(void* p) {
    Heap* h = reinterpret_cast<Heap*>(*(volatile unsigned long long*)&__vulkanizer_heap);
    if (!h || !p) return;
    char* base = reinterpret_cast<char*>(h);
    const unsigned off = unsigned(static_cast<char*>(p) - 16 - base);
    const unsigned c = *(volatile unsigned*)(base + off);
    unsigned long long head = *(volatile unsigned long long*)&h->free[c];
    for (;;) {
        *(volatile unsigned*)(base + off + 4) = unsigned(head);
        __threadfence();  // the link before the block is on the list
        const unsigned long long put = (((head >> 32) + 1) << 32) | off;
        const unsigned long long seen = atomicCAS(&h->free[c], head, put);
        if (seen == head) return;
        head = seen;
    }
}

}  // extern "C"
