#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "tbd/byte_tracker.hpp"

namespace ctrk {
namespace {

Detection det(float x, float y, float w, float h, float score, int cls = 0) {
  return {{x, y, w, h}, score, cls};
}

std::optional<Track> find_confirmed_near(const std::vector<Track>& tracks, float x, float y) {
  for (const auto& t : tracks)
    if (t.state == TrackState::Confirmed && std::abs(t.box.cx() - x) < 25.f &&
        std::abs(t.box.cy() - y) < 25.f)
      return t;
  return std::nullopt;
}

TEST(Coast, BoxesAdvanceWithMotionAndStateIsUntouched) {
  AssocConfig cfg;
  cfg.use_byte = false;
  ByteTracker tracker(cfg);
  int id = -1;
  for (int i = 0; i < 4; ++i) {
    const auto out = tracker.update({det(100.f + 5 * static_cast<float>(i), 100, 40, 80, 0.9f)});
    if (!out.empty()) id = out[0].id;
  }
  // Constant velocity established (~5 px/frame). Coast two frames.
  const auto c1 = tracker.coast();
  ASSERT_EQ(c1.size(), 1u);
  EXPECT_EQ(c1[0].id, id);
  EXPECT_EQ(c1[0].state, TrackState::Confirmed);
  const auto c2 = tracker.coast();
  EXPECT_GT(c2[0].box.cx(), c1[0].box.cx());
  EXPECT_NEAR(c2[0].box.cx() - c1[0].box.cx(), 5.f, 2.f);
}

TEST(Coast, DoesNotAgeTracksOutAndKeepsIdAcrossTheGap) {
  AssocConfig cfg;
  cfg.use_byte = false;
  cfg.max_age = 3;  // tight, in DETECT-frame units
  ByteTracker tracker(cfg);
  int id = -1;
  for (int i = 0; i < 3; ++i) {
    const auto out = tracker.update({det(100.f + 2 * static_cast<float>(i), 100, 40, 80, 0.9f)});
    if (!out.empty()) id = out[0].id;
  }
  // Coast far past max_age: lifecycle must not decay on detector-free frames.
  for (int i = 0; i < 20; ++i) {
    const auto out = tracker.coast();
    ASSERT_EQ(out.size(), 1u) << "coast frame " << i;
    EXPECT_EQ(out[0].state, TrackState::Confirmed);
  }
  // Next detect frame re-locks with the same id where prediction leads.
  const auto out = tracker.update({det(144, 100, 40, 80, 0.9f)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, id);
  EXPECT_EQ(out[0].state, TrackState::Confirmed);
}

TEST(Coast, IdsStableThroughCrossingAtIntervalTwo) {
  AssocConfig cfg;
  ByteTracker tracker(cfg);
  // Two targets crossing: A moves right, B moves left, meeting near x=150.
  // 2 px/frame: at interval 2 a track sees 4 px between detects, inside the
  // tentative-stage IoU gate. (Faster targets churn during track birth at
  // high N — the 0.7 tentative gate is the brittle point; see RESULTS.md.)
  int id_a = -1, id_b = -1;
  for (int f = 0; f < 44; ++f) {
    const float ax = 100.f + 2.f * static_cast<float>(f);
    const float bx = 200.f - 2.f * static_cast<float>(f);
    std::vector<Track> out;
    if (f % 2 == 0) {
      out = tracker.update({det(ax, 100, 30, 60, 0.9f), det(bx, 160, 30, 60, 0.9f)});
    } else {
      out = tracker.coast();  // detector skipped on odd frames
    }
    if (f == 8) {
      const auto a = find_confirmed_near(out, ax + 15, 130);
      const auto b = find_confirmed_near(out, bx + 15, 190);
      ASSERT_TRUE(a && b);
      id_a = a->id;
      id_b = b->id;
    }
  }
  // After the cross (f=44): A is right of B; same ids, no swap, no ghosts.
  const auto final_out = tracker.update({det(188, 100, 30, 60, 0.9f), det(112, 160, 30, 60, 0.9f)});
  const auto a = find_confirmed_near(final_out, 203, 130);
  const auto b = find_confirmed_near(final_out, 127, 190);
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->id, id_a);
  EXPECT_EQ(b->id, id_b);
  EXPECT_EQ(final_out.size(), 2u);
}

TEST(SortLifecycle, ConfirmsAfterNInitHits) {
  AssocConfig cfg;
  cfg.use_byte = false;
  ByteTracker tracker(cfg);

  auto out1 = tracker.update({det(100, 100, 40, 80, 0.9f)});
  ASSERT_EQ(out1.size(), 1u);
  EXPECT_EQ(out1[0].state, TrackState::Tentative);

  auto out2 = tracker.update({det(102, 100, 40, 80, 0.9f)});
  EXPECT_EQ(out2[0].state, TrackState::Tentative);

  auto out3 = tracker.update({det(104, 100, 40, 80, 0.9f)});
  ASSERT_EQ(out3.size(), 1u);
  EXPECT_EQ(out3[0].state, TrackState::Confirmed);
  EXPECT_EQ(out3[0].hits, 3);
}

TEST(SortLifecycle, TentativeMissIsDroppedImmediately) {
  AssocConfig cfg;
  cfg.use_byte = false;
  ByteTracker tracker(cfg);
  tracker.update({det(100, 100, 40, 80, 0.9f)});
  const auto out = tracker.update({});  // one miss while tentative
  EXPECT_TRUE(out.empty());
}

TEST(SortLifecycle, LostTrackCoastsThenIsRemoved) {
  AssocConfig cfg;
  cfg.use_byte = false;
  cfg.max_age = 5;
  ByteTracker tracker(cfg);
  for (int i = 0; i < 3; ++i)
    tracker.update({det(100.f + 2 * static_cast<float>(i), 100, 40, 80, 0.9f)});

  std::vector<Track> out;
  for (int miss = 0; miss < 5; ++miss) {
    out = tracker.update({});
    ASSERT_EQ(out.size(), 1u) << "miss " << miss;
    EXPECT_EQ(out[0].state, TrackState::Lost);
  }
  out = tracker.update({});  // exceeds max_age
  EXPECT_TRUE(out.empty());
}

TEST(SortLifecycle, ReacquireWithinMaxAgeKeepsId) {
  AssocConfig cfg;
  cfg.use_byte = false;
  ByteTracker tracker(cfg);
  int id = -1;
  for (int i = 0; i < 4; ++i) {
    const auto out = tracker.update({det(100.f + 3 * static_cast<float>(i), 100, 40, 80, 0.9f)});
    if (!out.empty()) id = out[0].id;
  }
  for (int i = 0; i < 8; ++i) tracker.update({});  // occlusion gap
  // Reappear where constant-velocity prediction leads (vx ~= 3 px/frame).
  const auto out = tracker.update({det(136, 100, 40, 80, 0.9f)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, id);
  EXPECT_EQ(out[0].state, TrackState::Confirmed);
}

TEST(SortLifecycle, IdsStableThroughCrossing) {
  AssocConfig cfg;
  cfg.use_byte = false;
  ByteTracker tracker(cfg);

  // A moves right along y=100, B moves left along y=112; they pass mid-way.
  auto box_a = [](int t) { return det(100.f + 6 * static_cast<float>(t), 100, 40, 80, 0.9f); };
  auto box_b = [](int t) { return det(400.f - 6 * static_cast<float>(t), 112, 40, 80, 0.9f); };

  int id_a = -1, id_b = -1;
  for (int t = 0; t < 50; ++t) {
    const auto out = tracker.update({box_a(t), box_b(t)});
    if (t == 5) {
      const auto a = find_confirmed_near(out, box_a(t).box.cx(), 140);
      const auto b = find_confirmed_near(out, box_b(t).box.cx(), 152);
      ASSERT_TRUE(a && b);
      id_a = a->id;
      id_b = b->id;
    }
  }
  const auto out = tracker.update({box_a(50), box_b(50)});
  const auto a = find_confirmed_near(out, box_a(50).box.cx(), 140);
  const auto b = find_confirmed_near(out, box_b(50).box.cx(), 152);
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->id, id_a) << "id swap on crossing";
  EXPECT_EQ(b->id, id_b) << "id swap on crossing";
}

TEST(SortLifecycle, LowScoreDetectionsNeverSpawnTracks) {
  AssocConfig cfg;
  cfg.use_byte = false;
  ByteTracker tracker(cfg);
  for (int i = 0; i < 10; ++i) EXPECT_TRUE(tracker.update({det(50, 50, 30, 30, 0.3f)}).empty());
}

// --- ByteTrack-specific behaviour (use_byte = true) ---

TEST(ByteTrack, LowScoreFlickerKeepsTrackUpdated) {
  AssocConfig byte_cfg;  // use_byte = true
  AssocConfig sort_cfg = byte_cfg;
  sort_cfg.use_byte = false;
  ByteTracker byte_tracker(byte_cfg);
  ByteTracker sort_tracker(sort_cfg);

  auto run = [](ByteTracker& tr, int frame) {
    // Score dips below track_thresh (0.5) during frames 10..14 — an
    // occlusion-dimmed detection — then recovers.
    const float score = (frame >= 10 && frame < 15) ? 0.3f : 0.9f;
    return tr.update({det(100.f + 2 * static_cast<float>(frame), 100, 40, 80, score)});
  };

  int byte_id = -1, sort_id = -1;
  bool sort_degraded = false;
  for (int f = 0; f < 25; ++f) {
    const auto byte_out = run(byte_tracker, f);
    const auto sort_out = run(sort_tracker, f);
    if (f >= 10 && f < 15) {
      // ByteTrack stage 2 keeps consuming the dimmed detection.
      ASSERT_EQ(byte_out.size(), 1u);
      EXPECT_EQ(byte_out[0].state, TrackState::Confirmed) << "frame " << f;
      // SORT ignores it and degrades to coasting.
      if (!sort_out.empty() && sort_out[0].state == TrackState::Lost) sort_degraded = true;
    }
    if (f == 24) {
      ASSERT_EQ(byte_out.size(), 1u);
      ASSERT_EQ(sort_out.size(), 1u);
      byte_id = byte_out[0].id;
      sort_id = sort_out[0].id;
    }
  }
  EXPECT_TRUE(sort_degraded) << "SORT mode should coast through the dip";
  EXPECT_EQ(byte_id, 1);
  EXPECT_EQ(sort_id, 1);  // both recover the id; byte never lost the lock
}

TEST(ByteTrack, LowScoreDetsOnlyFeedExistingTracks) {
  ByteTracker tracker{AssocConfig{}};
  // Low-score detections alone must not create tracks even in byte mode.
  for (int i = 0; i < 10; ++i) EXPECT_TRUE(tracker.update({det(50, 50, 30, 30, 0.4f)}).empty());
}

TEST(ByteTrack, StageTwoDoesNotFeedLostTracks) {
  AssocConfig cfg;
  ByteTracker tracker(cfg);
  for (int i = 0; i < 3; ++i)
    tracker.update({det(100.f + 2 * static_cast<float>(i), 100, 40, 80, 0.9f)});
  tracker.update({});  // miss -> Lost; no longer "matched last frame"
  const auto out = tracker.update({det(110, 100, 40, 80, 0.3f)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].state, TrackState::Lost) << "lost tracks must not eat low-score dets";
}

TEST(SortLifecycle, IdsAreNeverReused) {
  AssocConfig cfg;
  cfg.use_byte = false;
  cfg.n_init = 1;
  cfg.max_age = 0;
  ByteTracker tracker(cfg);
  const int first = tracker.update({det(100, 100, 40, 40, 0.9f)})[0].id;
  tracker.update({});  // drop it
  tracker.update({});
  const auto out = tracker.update({det(300, 300, 40, 40, 0.9f)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_GT(out[0].id, first);
}

}  // namespace
}  // namespace ctrk
