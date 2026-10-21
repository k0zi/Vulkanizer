#include "driver/printf.h"

#include <cctype>
#include <cstdio>
#include <cstring>

namespace vulkanizer::driver {

std::string formatDevicePrintf(const std::string& format, const uint8_t* args, size_t argBytes,
                               const std::function<std::string(uint64_t)>& readString) {
    std::string out;
    size_t at = 0;  // offset in the argument block
    auto take = [&](size_t size, void* dst) {
        at = (at + size - 1) / size * size;
        if (at + size > argBytes) return false;
        std::memcpy(dst, args + at, size);
        at += size;
        return true;
    };
    char buf[512];
    for (size_t i = 0; i < format.size(); ++i) {
        if (format[i] != '%') {
            out += format[i];
            continue;
        }
        // %[flags][width][.precision][length]conversion
        size_t j = i + 1;
        while (j < format.size() && std::strchr("-+ #0", format[j])) ++j;
        while (j < format.size() && (std::isdigit(static_cast<unsigned char>(format[j])) || format[j] == '.')) ++j;
        bool wide = false;
        size_t lengthStart = j;
        while (j < format.size() && std::strchr("hlLqjzt", format[j])) {
            if (format[j] == 'l' || format[j] == 'L' || format[j] == 'q' || format[j] == 'j' || format[j] == 'z' ||
                format[j] == 't')
                wide = true;
            ++j;
        }
        if (j >= format.size()) {
            out += format.substr(i);
            break;
        }
        const char conv = format[j];
        // the spec without CUDA's length modifiers, plus the one the host value needs
        std::string spec = format.substr(i, lengthStart - i);
        const std::string rest(1, conv);
        bool ok = true;
        if (conv == '%') {
            out += '%';
        } else if (std::strchr("diouxXc", conv)) {
            if (wide) {
                uint64_t v = 0;
                ok = take(8, &v);
                std::snprintf(buf, sizeof buf, (spec + "ll" + rest).c_str(), (unsigned long long)v);
            } else {
                uint32_t v = 0;
                ok = take(4, &v);
                std::snprintf(buf, sizeof buf, (spec + rest).c_str(), v);
            }
            out += buf;
        } else if (std::strchr("eEfFgGaA", conv)) {
            double v = 0;
            ok = take(8, &v);
            std::snprintf(buf, sizeof buf, (spec + rest).c_str(), v);
            out += buf;
        } else if (conv == 's') {
            uint64_t p = 0;
            ok = take(8, &p);
            std::snprintf(buf, sizeof buf, (spec + "s").c_str(), p ? readString(p).c_str() : "(null)");
            out += buf;
        } else if (conv == 'p') {
            uint64_t p = 0;
            ok = take(8, &p);
            std::snprintf(buf, sizeof buf, "0x%llx", (unsigned long long)p);
            out += buf;
        } else {
            out += format.substr(i, j - i + 1);
        }
        if (!ok) {
            out += "<missing argument>";
            break;
        }
        i = j;
    }
    return out;
}

}  // namespace vulkanizer::driver
