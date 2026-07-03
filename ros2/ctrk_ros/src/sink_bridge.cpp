#include "sink_bridge.hpp"

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include <ctrk/log.hpp>
#include <ctrk/profile.hpp>
#include <rclcpp/rclcpp.hpp>

namespace ctrk_ros {

namespace {

struct Registry {
  std::mutex mutex;
  std::vector<std::pair<uint64_t, ProfileCollector::Fn>> collectors;
  uint64_t next_id = 1;
};

Registry& registry() {
  static Registry r;
  return r;
}

}  // namespace

void install_process_sinks() {
  static std::once_flag once;
  std::call_once(once, [] {
    ctrk::set_log_sink([](ctrk::LogLevel level, std::string_view msg) {
      const auto logger = rclcpp::get_logger("ctrk");
      const int len = static_cast<int>(msg.size());
      switch (level) {
        case ctrk::LogLevel::Debug:
          RCLCPP_DEBUG(logger, "%.*s", len, msg.data());
          break;
        case ctrk::LogLevel::Info:
          RCLCPP_INFO(logger, "%.*s", len, msg.data());
          break;
        case ctrk::LogLevel::Warn:
          RCLCPP_WARN(logger, "%.*s", len, msg.data());
          break;
        case ctrk::LogLevel::Error:
          RCLCPP_ERROR(logger, "%.*s", len, msg.data());
          break;
      }
    });
    ctrk::set_profile_sink([](std::string_view stage, double ms) {
      Registry& r = registry();
      const std::lock_guard<std::mutex> lock(r.mutex);
      for (auto& [id, fn] : r.collectors) fn(stage, ms);
    });
  });
}

ProfileCollector::ProfileCollector(Fn fn) {
  Registry& r = registry();
  const std::lock_guard<std::mutex> lock(r.mutex);
  id_ = r.next_id++;
  r.collectors.emplace_back(id_, std::move(fn));
}

ProfileCollector::~ProfileCollector() {
  Registry& r = registry();
  const std::lock_guard<std::mutex> lock(r.mutex);
  std::erase_if(r.collectors, [this](const auto& e) { return e.first == id_; });
}

}  // namespace ctrk_ros
