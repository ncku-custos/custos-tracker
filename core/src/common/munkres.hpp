#pragma once

#include <vector>

namespace ctrk {

// Minimum-cost assignment (Munkres / Hungarian, O(n^3)). `cost` is row-major
// and may be rectangular; missing cells are treated as free dummy assignments.
// Returns row_to_col: for each row the assigned column, or -1 if unassigned.
//
// Gating note: pass large-but-finite costs (e.g. 1e6) for forbidden pairs and
// post-filter matches whose cost exceeds the gate — infinities are not valid
// inputs.
std::vector<int> munkres_solve(const std::vector<std::vector<float>>& cost);

}  // namespace ctrk
