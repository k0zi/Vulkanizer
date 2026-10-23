#pragma once

#include <string>

namespace vulkanizer::ptx {

// Device-side malloc and free: appends Vulkanizer's heap functions (heap_lib.cu) to a module that
// declares them as .extern functions; the inliner calls __vulkanizer_malloc / __vulkanizer_free in
// their place. The driver points the module's __vulkanizer_heap at the device's heap. Other modules
// are left as they are.
void appendDeviceHeap(std::string& source);

// The heap's header (heap_lib.cu's struct Heap) the driver initializes: the buffer's size in bytes
// at 0, the first block's offset at 8 (the free lists after it start empty).
constexpr unsigned long long kHeapHeaderBytes = 512;

}  // namespace vulkanizer::ptx
