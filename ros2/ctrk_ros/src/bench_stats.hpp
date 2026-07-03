#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace ctrk_ros {

// Minimal per-stage latency accumulator for the `bench` param (core's
// StageTimer lives in a private header; nearest-rank percentiles match it).
// Not locked: all add() calls come from the node's mutually-exclusive
// callback group, table() from a lifecycle transition on the same executor.
class BenchStats {
 public:
  void add(std::string_view stage, double ms) { samples_[std::string(stage)].push_back(ms); }
  bool empty() const { return samples_.empty(); }

  // "stage n mean_ms p50_ms p95_ms" rows, one per stage.
  std::string table() const {
    std::string out = "stage n mean_ms p50_ms p95_ms";
    for (const auto& [stage, samples] : samples_) {
      std::vector<double> sorted = samples;
      std::sort(sorted.begin(), sorted.end());
      double sum = 0;
      for (double v : sorted) sum += v;
      char row[160];
      std::snprintf(row, sizeof(row), "\n%s %zu %.2f %.2f %.2f", stage.c_str(), sorted.size(),
                    sum / static_cast<double>(sorted.size()), rank(sorted, 50), rank(sorted, 95));
      out += row;
    }
    return out;
  }

 private:
  static double rank(const std::vector<double>& sorted, int pct) {
    const auto n = static_cast<double>(sorted.size());
    const auto idx = static_cast<size_t>(std::ceil(pct / 100.0 * n)) - 1;
    return sorted[std::min(idx, sorted.size() - 1)];
  }

  std::map<std::string, std::vector<double>> samples_;
};

}  // namespace ctrk_ros
