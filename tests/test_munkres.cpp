#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "common/munkres.hpp"

namespace ctrk {
namespace {

// Exhaustive optimal assignment cost. If rows > cols, exactly rows-cols rows
// stay unassigned (mirroring Munkres' zero-cost dummy padding).
double brute_force_cost(const std::vector<std::vector<float>>& cost) {
  const int rows = static_cast<int>(cost.size());
  const int cols = static_cast<int>(cost[0].size());
  const int skips_allowed = std::max(0, rows - cols);
  double best = std::numeric_limits<double>::max();

  std::vector<bool> used(cols, false);
  auto rec = [&](auto&& self, int row, int skips, double acc) -> void {
    if (acc >= best) return;
    if (row == rows) {
      best = std::min(best, acc);
      return;
    }
    if (skips < skips_allowed) self(self, row + 1, skips + 1, acc);
    for (int j = 0; j < cols; ++j)
      if (!used[j]) {
        used[j] = true;
        self(self, row + 1, skips, acc + cost[row][j]);
        used[j] = false;
      }
  };
  rec(rec, 0, 0, 0.0);
  return best;
}

double assignment_cost(const std::vector<std::vector<float>>& cost,
                       const std::vector<int>& row_to_col) {
  double total = 0.0;
  for (size_t i = 0; i < row_to_col.size(); ++i)
    if (row_to_col[i] >= 0) total += cost[i][row_to_col[i]];
  return total;
}

void expect_valid(const std::vector<int>& row_to_col, int cols) {
  std::vector<bool> used(cols, false);
  for (int c : row_to_col) {
    if (c < 0) continue;
    ASSERT_LT(c, cols);
    EXPECT_FALSE(used[c]) << "column assigned twice";
    used[c] = true;
  }
}

TEST(Munkres, KnownThreeByThree) {
  // Classic example: optimal cost 5 (0->1? no: rows 0,1,2 -> cols 2,1,0 etc.)
  const std::vector<std::vector<float>> cost = {{1, 2, 3}, {2, 4, 6}, {3, 6, 9}};
  const auto r = munkres_solve(cost);
  expect_valid(r, 3);
  EXPECT_DOUBLE_EQ(assignment_cost(cost, r), brute_force_cost(cost));
}

TEST(Munkres, MatchesBruteForceOnRandomSquare) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.f, 10.f);
  for (int n = 1; n <= 7; ++n) {
    for (int trial = 0; trial < 25; ++trial) {
      std::vector<std::vector<float>> cost(n, std::vector<float>(n));
      for (auto& row : cost)
        for (auto& v : row) v = dist(rng);
      const auto r = munkres_solve(cost);
      expect_valid(r, n);
      EXPECT_NEAR(assignment_cost(cost, r), brute_force_cost(cost), 1e-6)
          << "n=" << n << " trial=" << trial;
    }
  }
}

TEST(Munkres, MatchesBruteForceOnRectangular) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(0.f, 10.f);
  for (auto [rows, cols] : std::vector<std::pair<int, int>>{{3, 6}, {6, 3}, {1, 5}, {5, 1}}) {
    for (int trial = 0; trial < 25; ++trial) {
      std::vector<std::vector<float>> cost(rows, std::vector<float>(cols));
      for (auto& row : cost)
        for (auto& v : row) v = dist(rng);
      const auto r = munkres_solve(cost);
      ASSERT_EQ(r.size(), static_cast<size_t>(rows));
      expect_valid(r, cols);
      EXPECT_NEAR(assignment_cost(cost, r), brute_force_cost(cost), 1e-6)
          << rows << "x" << cols << " trial=" << trial;
    }
  }
}

TEST(Munkres, HandlesTiesAndZeros) {
  const std::vector<std::vector<float>> cost = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  const auto r = munkres_solve(cost);
  expect_valid(r, 3);
  EXPECT_EQ(std::count_if(r.begin(), r.end(), [](int c) { return c >= 0; }), 3);
}

TEST(Munkres, EmptyInput) {
  EXPECT_TRUE(munkres_solve({}).empty());
}

}  // namespace
}  // namespace ctrk
