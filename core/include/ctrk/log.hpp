#pragma once

#include <functional>
#include <string_view>

namespace ctrk {

enum class LogLevel : int { Debug = 0, Info, Warn, Error };

using LogSink = std::function<void(LogLevel, std::string_view)>;

// Replace the process-wide log sink (e.g. with an rclcpp logger, as ctrk_ros does).
// Passing an empty function restores the default stderr sink. This is the one
// deliberate piece of global state in ctrk; sinks must be thread-safe.
void set_log_sink(LogSink sink);

void log(LogLevel level, std::string_view msg);

}  // namespace ctrk
