#include "ctrk/log.hpp"

#include <cstdio>
#include <mutex>

namespace ctrk {

namespace {

void default_sink(LogLevel level, std::string_view msg) {
  static constexpr const char* kNames[] = {"DEBUG", "INFO", "WARN", "ERROR"};
  std::fprintf(stderr, "[ctrk %s] %.*s\n", kNames[static_cast<int>(level)],
               static_cast<int>(msg.size()), msg.data());
}

std::mutex g_mutex;
LogSink g_sink;  // empty => default_sink

}  // namespace

void set_log_sink(LogSink sink) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_sink = std::move(sink);
}

void log(LogLevel level, std::string_view msg) {
  LogSink sink;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    sink = g_sink;
  }
  if (sink) {
    sink(level, msg);
  } else {
    default_sink(level, msg);
  }
}

}  // namespace ctrk
