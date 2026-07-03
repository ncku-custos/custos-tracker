#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ctrk {

// Latency samples for one pipeline stage. Percentiles are nearest-rank.
class StageStats {
 public:
  void add_ns(int64_t ns) { samples_.push_back(ns); }

  size_t count() const { return samples_.size(); }

  double mean_ms() const {
    if (samples_.empty()) return 0.0;
    double sum = 0.0;
    for (int64_t s : samples_) sum += static_cast<double>(s);
    return sum / static_cast<double>(samples_.size()) / 1e6;
  }

  double p50_ms() const { return percentile_ms(50.0); }
  double p95_ms() const { return percentile_ms(95.0); }

  double percentile_ms(double p) const {
    if (samples_.empty()) return 0.0;
    std::vector<int64_t> sorted = samples_;
    std::sort(sorted.begin(), sorted.end());
    const size_t rank = static_cast<size_t>(
        std::ceil(p / 100.0 * static_cast<double>(sorted.size())));
    return static_cast<double>(sorted[std::max<size_t>(rank, 1) - 1]) / 1e6;
  }

 private:
  std::vector<int64_t> samples_;
};

// Named-stage timing registry. Single-threaded, like the core objects it
// instruments.
class StageTimer {
 public:
  class Scope {
   public:
    Scope(StageStats& stats) : stats_(stats), t0_(clock::now()) {}
    ~Scope() {
      stats_.add_ns(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0_)
                        .count());
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    using clock = std::chrono::steady_clock;
    StageStats& stats_;
    clock::time_point t0_;
  };

  Scope scope(const std::string& stage) { return Scope(stats_[stage]); }
  const std::map<std::string, StageStats>& stats() const { return stats_; }

 private:
  std::map<std::string, StageStats> stats_;
};

}  // namespace ctrk
