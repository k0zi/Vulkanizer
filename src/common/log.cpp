#include "common/log.h"

#include <algorithm>
#include <cstdarg>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace vulkanizer {

static LogLevel parseLevel(const char* s) {
    if (!s || !*s) return LogLevel::Warn;
    if (s[0] >= '0' && s[0] <= '5' && s[1] == '\0') return static_cast<LogLevel>(s[0] - '0');
    struct { const char* name; LogLevel level; } names[] = {
        {"off", LogLevel::Off},     {"error", LogLevel::Error}, {"warn", LogLevel::Warn},
        {"info", LogLevel::Info},   {"debug", LogLevel::Debug}, {"trace", LogLevel::Trace},
    };
    for (auto& n : names)
        if (strcasecmp(s, n.name) == 0) return n.level;
    return LogLevel::Warn;
}

LogLevel logLevel() {
    static const LogLevel level = parseLevel(std::getenv("VULKANIZER_LOG"));
    return level;
}

void logMessage(LogLevel level, const char* fmt, ...) {
    static const char* const tags[] = {"", "error", "warn", "info", "debug", "trace"};
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[vulkanizer:%s] %s\n", tags[static_cast<int>(level)], buf);
}

namespace {

std::atomic<ApiStats*> g_stats{nullptr};

void printApiProfile() {
    std::vector<const ApiStats*> rows;
    for (const ApiStats* s = g_stats.load(); s; s = s->next)
        if (s->calls) rows.push_back(s);
    if (rows.empty()) return;
    std::sort(rows.begin(), rows.end(), [](const ApiStats* a, const ApiStats* b) { return a->ns > b->ns; });
    Dl_info info{};
    dladdr(reinterpret_cast<void*>(&printApiProfile), &info);
    const char* lib = info.dli_fname ? std::strrchr(info.dli_fname, '/') : nullptr;
    std::fprintf(stderr, "\n[vulkanizer] host time per API call (%s):\n", lib ? lib + 1 : "?");
    std::fprintf(stderr, "  %10s %10s %9s  %s\n", "total ms", "calls", "avg us", "function");
    for (size_t i = 0; i < rows.size() && i < 25; ++i)
        std::fprintf(stderr, "  %10.3f %10llu %9.2f  %s\n", double(rows[i]->ns) * 1e-6,
                     (unsigned long long)rows[i]->calls, double(rows[i]->ns) * 1e-3 / double(rows[i]->calls),
                     rows[i]->name);
}

}  // namespace

bool profilingEnabled() {
    static const bool on = [] {
        const char* env = std::getenv("VULKANIZER_PROFILE");
        const bool enabled = env && *env && std::strcmp(env, "0") != 0;
        if (enabled) std::atexit(printApiProfile);
        return enabled;
    }();
    return on;
}

ApiStats::ApiStats(const char* n) : name(n) {
    next = g_stats.load();
    while (!g_stats.compare_exchange_weak(next, this)) {
    }
}

}  // namespace vulkanizer
