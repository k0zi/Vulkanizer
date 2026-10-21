#pragma once
#include <cstddef>
#include <vector>

struct Layout {
    size_t propSize, funcAttrSize, launchConfigSize;
    std::vector<size_t> propOffsets, funcAttrOffsets, launchConfigOffsets;
    std::vector<long> errors;
};
