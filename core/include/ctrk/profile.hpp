#pragma once

#include <chrono>
#include <functional>
#include <string_view>

namespace ctrk {

using ProfileSink = std::function<void(std::string_view stage, double ms)>;

// Replace the process-wide per-stage latency sink (empty disables profiling,
// the default). Called synchronously from pipeline threads; keep it cheap and
// thread-safe if several pipelines run concurrently.
void set_profile_sink(ProfileSink sink);

namespace detail {
bool profile_enabled();
void profile_emit(std::string_view stage, double ms);
}  // namespace detail

// RAII wall-clock scope reporting to the profile sink. Near-zero cost while
// no sink is installed (one relaxed atomic load, no clock reads).
class ProfileScope {
 public:
  explicit ProfileScope(std::string_view stage)
      : stage_(stage), active_(detail::profile_enabled()) {
    if (active_) t0_ = std::chrono::steady_clock::now();
  }
  ~ProfileScope() {
    if (active_) {
      const auto dt = std::chrono::steady_clock::now() - t0_;
      detail::profile_emit(stage_, std::chrono::duration<double, std::milli>(dt).count());
    }
  }
  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;

 private:
  std::string_view stage_;
  bool active_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace ctrk
