#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

namespace ctrk_ros {

// Route ctrk's process-wide log sink to an rclcpp logger named "ctrk" and fan
// its process-wide profile sink out to registered collectors. Idempotent;
// nodes call it in on_configure before constructing core objects. One bridge
// per process regardless of how many components a container composes — the
// core sinks are the one piece of ctrk global state (ctrk/log.hpp).
void install_process_sinks();

// RAII registration of a profile collector. Every collector sees every stage
// from every ctrk pipeline in the process (stage names are pipeline-prefixed,
// e.g. "det.infer" / "sot.head"); run one tracker per process when exact
// attribution matters — scripts/ros_bench.sh does. Callbacks run under the
// registry lock on the emitting pipeline's thread: keep them cheap.
class ProfileCollector {
 public:
  using Fn = std::function<void(std::string_view stage, double ms)>;
  explicit ProfileCollector(Fn fn);
  ~ProfileCollector();
  ProfileCollector(const ProfileCollector&) = delete;
  ProfileCollector& operator=(const ProfileCollector&) = delete;

 private:
  uint64_t id_;
};

}  // namespace ctrk_ros
