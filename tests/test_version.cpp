#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "ctrk/version.hpp"

TEST(Version, IsSemver) {
  const std::string v = ctrk::version();
  EXPECT_EQ(std::count(v.begin(), v.end(), '.'), 2) << v;
}
