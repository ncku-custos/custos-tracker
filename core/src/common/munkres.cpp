#include "common/munkres.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ctrk {

namespace {

constexpr double kZeroEps = 1e-9;

// Classic star/prime Munkres on an n x n matrix (Bourgeois & Lassalle form).
class Munkres {
 public:
  explicit Munkres(std::vector<std::vector<double>> c)
      : n_(static_cast<int>(c.size())),
        c_(std::move(c)),
        star_(n_, std::vector<bool>(n_, false)),
        prime_(n_, std::vector<bool>(n_, false)),
        row_cov_(n_, false),
        col_cov_(n_, false) {}

  // Returns star matrix: star_[i][j] == true iff row i assigned to col j.
  const std::vector<std::vector<bool>>& solve() {
    reduce();
    star_initial();
    // Step 3: cover starred columns; done when all n are covered. The inner
    // loop (steps 4-6) must NOT recompute covers — it mutates them.
    while (!cover_star_columns()) {
      for (;;) {
        int zr, zc;
        while (!find_uncovered_zero(zr, zc)) adjust();
        prime_[zr][zc] = true;
        const int star_col = star_in_row(zr);
        if (star_col < 0) {  // augmenting path found -> back to step 3
          augment(zr, zc);
          clear_covers_and_primes();
          break;
        }
        row_cov_[zr] = true;
        col_cov_[star_col] = false;
      }
    }
    return star_;
  }

 private:
  static bool is_zero(double v) { return std::abs(v) < kZeroEps; }

  void reduce() {
    for (auto& row : c_) {
      const double m = *std::min_element(row.begin(), row.end());
      for (auto& v : row) v -= m;
    }
    for (int j = 0; j < n_; ++j) {
      double m = std::numeric_limits<double>::max();
      for (int i = 0; i < n_; ++i) m = std::min(m, c_[i][j]);
      for (int i = 0; i < n_; ++i) c_[i][j] -= m;
    }
  }

  void star_initial() {
    std::vector<bool> row_used(n_, false), col_used(n_, false);
    for (int i = 0; i < n_; ++i)
      for (int j = 0; j < n_; ++j)
        if (!row_used[i] && !col_used[j] && is_zero(c_[i][j])) {
          star_[i][j] = true;
          row_used[i] = col_used[j] = true;
        }
  }

  // Covers columns with stars; true when all n columns are covered.
  bool cover_star_columns() {
    int covered = 0;
    for (int j = 0; j < n_; ++j) {
      col_cov_[j] = false;
      for (int i = 0; i < n_; ++i) col_cov_[j] = col_cov_[j] || star_[i][j];
      covered += col_cov_[j] ? 1 : 0;
    }
    return covered == n_;
  }

  bool find_uncovered_zero(int& zr, int& zc) const {
    for (int i = 0; i < n_; ++i) {
      if (row_cov_[i]) continue;
      for (int j = 0; j < n_; ++j)
        if (!col_cov_[j] && is_zero(c_[i][j])) {
          zr = i;
          zc = j;
          return true;
        }
    }
    return false;
  }

  int star_in_row(int row) const {
    for (int j = 0; j < n_; ++j)
      if (star_[row][j]) return j;
    return -1;
  }

  int star_in_col(int col) const {
    for (int i = 0; i < n_; ++i)
      if (star_[i][col]) return i;
    return -1;
  }

  int prime_in_row(int row) const {
    for (int j = 0; j < n_; ++j)
      if (prime_[row][j]) return j;
    return -1;
  }

  // Alternating primed/starred path starting at primed (zr, zc): stars along
  // the path are removed, primes become stars.
  void augment(int zr, int zc) {
    std::vector<std::pair<int, int>> path{{zr, zc}};
    for (;;) {
      const int r = star_in_col(path.back().second);
      if (r < 0) break;
      path.emplace_back(r, path.back().second);
      path.emplace_back(r, prime_in_row(r));
    }
    for (size_t k = 0; k < path.size(); ++k) {
      auto [i, j] = path[k];
      star_[i][j] = (k % 2 == 0);
    }
  }

  void clear_covers_and_primes() {
    std::fill(row_cov_.begin(), row_cov_.end(), false);
    std::fill(col_cov_.begin(), col_cov_.end(), false);
    for (auto& row : prime_) std::fill(row.begin(), row.end(), false);
  }

  // No uncovered zero: create one by shifting the minimum uncovered value.
  void adjust() {
    double e = std::numeric_limits<double>::max();
    for (int i = 0; i < n_; ++i)
      if (!row_cov_[i])
        for (int j = 0; j < n_; ++j)
          if (!col_cov_[j]) e = std::min(e, c_[i][j]);
    for (int i = 0; i < n_; ++i)
      for (int j = 0; j < n_; ++j) {
        if (row_cov_[i]) c_[i][j] += e;
        if (!col_cov_[j]) c_[i][j] -= e;
      }
  }

  int n_;
  std::vector<std::vector<double>> c_;
  std::vector<std::vector<bool>> star_, prime_;
  std::vector<bool> row_cov_, col_cov_;
};

}  // namespace

std::vector<int> munkres_solve(const std::vector<std::vector<float>>& cost) {
  const int rows = static_cast<int>(cost.size());
  const int cols = rows > 0 ? static_cast<int>(cost[0].size()) : 0;
  if (rows == 0 || cols == 0) return std::vector<int>(rows, -1);

  const int n = std::max(rows, cols);
  std::vector<std::vector<double>> padded(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) padded[i][j] = static_cast<double>(cost[i][j]);

  Munkres solver(std::move(padded));
  const auto& star = solver.solve();

  std::vector<int> row_to_col(rows, -1);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      if (star[i][j]) row_to_col[i] = j;
  return row_to_col;
}

}  // namespace ctrk
