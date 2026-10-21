// Shared by nvidia.cpp and vulkanizer.cpp: the list evaluated against whichever cuda.h is included.
std::vector<long long> abiValues() {
#define VALUE(e) static_cast<long long>(e),
    return {VULKANIZER_CU_VALUES(VALUE) VULKANIZER_CU_LAYOUTS(VALUE)};
#undef VALUE
}
