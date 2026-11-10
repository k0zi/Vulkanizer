#pragma once

#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

inline const char* cuErrorName(CUresult r) {
    const char* name = nullptr;
    cuGetErrorName(r, &name);
    return name ? name : "<unknown>";
}

#define CHECK_CU(call)                                                                            \
    do {                                                                                          \
        CUresult r_ = (call);                                                                     \
        if (r_ != CUDA_SUCCESS) {                                                                 \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, cuErrorName(r_)); \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

#define EXPECT_CU(expected, call)                                                                 \
    do {                                                                                          \
        CUresult r_ = (call);                                                                     \
        if (r_ != (expected)) {                                                                   \
            std::fprintf(stderr, "%s:%d: %s returned %s, expected %s\n", __FILE__, __LINE__, #call, \
                         cuErrorName(r_), cuErrorName(expected));                                 \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

#define EXPECT(cond)                                                                              \
    do {                                                                                          \
        if (!(cond)) {                                                                            \
            std::fprintf(stderr, "%s:%d: expectation failed: %s\n", __FILE__, __LINE__, #cond);   \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

inline std::string readTextFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
