// Checks that the Driver API enums and structures of Vulkanizer's clean-room cuda.h match the
// official header (built only when a CUDA toolkit is present in .cuda/).
#include "abi_list.h"

#include <cstdio>

namespace nvidia { std::vector<long long> abiValues(); }
namespace vulkanizer { std::vector<long long> abiValues(); }

int main() {
    static const char* const kNames[] = {
#define NAME(e) #e,
        VULKANIZER_CU_VALUES(NAME) VULKANIZER_CU_LAYOUTS(NAME)
#undef NAME
    };
    const std::vector<long long> a = nvidia::abiValues(), b = vulkanizer::abiValues();
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) std::printf("MISMATCH %-60s nvidia %lld, vulkanizer %lld\n", kNames[i], a[i], b[i]), ++bad;
    std::printf(bad ? "%d mismatch(es)\n" : "OK\n", bad);
    return bad ? 1 : 0;
}
