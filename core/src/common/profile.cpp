#include "ctrk/profile.hpp"

#include <atomic>
#include <mutex>
#include <utility>

namespace ctrk {

namespace {

std::mutex g_mutex;
ProfileSink g_sink;
std::atomic<bool> g_enabled{false};

}  // namespace

void set_profile_sink(ProfileSink sink) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_sink = std::move(sink);
  g_enabled.store(static_cast<bool>(g_sink), std::memory_order_relaxed);
}

namespace detail {

bool profile_enabled() {
  return g_enabled.load(std::memory_order_relaxed);
}

void profile_emit(std::string_view stage, double ms) {
  ProfileSink sink;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    sink = g_sink;
  }
  if (sink) sink(stage, ms);
}

}  // namespace detail

}  // namespace ctrk
