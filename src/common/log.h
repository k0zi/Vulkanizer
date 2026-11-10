#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace vulkanizer {

enum class LogLevel : int { Off = 0, Error, Warn, Info, Debug, Trace };

// Parsed once from VULKANIZER_LOG (off|error|warn|info|debug|trace or 0-5). Default: warn.
LogLevel logLevel();
void logMessage(LogLevel level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

// VULKANIZER_PROFILE=1: host time spent in each API entry point, printed at exit.
bool profilingEnabled();

struct ApiStats {
    const char* name;
    std::atomic<uint64_t> calls{0}, ns{0};
    ApiStats* next = nullptr;
    explicit ApiStats(const char* n);  // registers itself
};

class ApiTimer {
public:
    explicit ApiTimer(ApiStats& s) : stats_(profilingEnabled() ? &s : nullptr) {
        if (stats_) start_ = std::chrono::steady_clock::now();
    }
    ~ApiTimer() {
        if (!stats_) return;
        stats_->calls.fetch_add(1, std::memory_order_relaxed);
        stats_->ns.fetch_add(uint64_t((std::chrono::steady_clock::now() - start_).count()), std::memory_order_relaxed);
    }

private:
    ApiStats* stats_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace vulkanizer

#define VULKANIZER_LOG(level, ...)                                         \
    do {                                                              \
        if (::vulkanizer::logLevel() >= (level)) ::vulkanizer::logMessage((level), __VA_ARGS__); \
    } while (0)

#define VULKANIZER_ERROR(...) VULKANIZER_LOG(::vulkanizer::LogLevel::Error, __VA_ARGS__)
#define VULKANIZER_WARN(...) VULKANIZER_LOG(::vulkanizer::LogLevel::Warn, __VA_ARGS__)
#define VULKANIZER_INFO(...) VULKANIZER_LOG(::vulkanizer::LogLevel::Info, __VA_ARGS__)
#define VULKANIZER_DEBUG(...) VULKANIZER_LOG(::vulkanizer::LogLevel::Debug, __VA_ARGS__)

// Logs every API call at trace level and times it with VULKANIZER_PROFILE=1.
#define VULKANIZER_TRACE_API()                                          \
    static ::vulkanizer::ApiStats vulkanizer_api_stats_(__func__);           \
    ::vulkanizer::ApiTimer vulkanizer_api_timer_(vulkanizer_api_stats_);          \
    VULKANIZER_LOG(::vulkanizer::LogLevel::Trace, "%s", __func__)

// Warns once per call site that a feature is not implemented yet.
#define VULKANIZER_WARN_ONCE(...)                                          \
    do {                                                              \
        static std::atomic<bool> warned_{false};                      \
        if (!warned_.exchange(true)) VULKANIZER_WARN(__VA_ARGS__);         \
    } while (0)
