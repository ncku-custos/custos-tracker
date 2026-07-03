// Conversion layer: ctrk types <-> ROS messages, and the zero-copy
// Image -> FrameView bridge (stride and stamp edge cases included).
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "frame_view.hpp"
#include "msg_convert.hpp"

namespace {

ctrk::Track make_track(int id, ctrk::TrackState state) {
  ctrk::Track t;
  t.id = id;
  t.box = {10.f, 20.f, 30.f, 40.f};
  t.score = 0.9f;
  t.class_id = 2;
  t.state = state;
  return t;
}

}  // namespace

TEST(ToDetections, CenterConventionRoundTripsAndIdsStringify) {
  std_msgs::msg::Header header;
  header.frame_id = "camera";
  const auto msg = ctrk_ros::to_detections({make_track(7, ctrk::TrackState::Confirmed)}, header,
                                           /*include_lost=*/false);

  ASSERT_EQ(msg.detections.size(), 1u);
  const auto& d = msg.detections[0];
  EXPECT_EQ(msg.header.frame_id, "camera");
  EXPECT_EQ(d.header.frame_id, "camera");
  EXPECT_EQ(d.id, "7");
  // ctrk top-left (10,20,30,40) -> vision_msgs center (25,40) size (30,40).
  EXPECT_DOUBLE_EQ(d.bbox.center.position.x, 25.0);
  EXPECT_DOUBLE_EQ(d.bbox.center.position.y, 40.0);
  EXPECT_DOUBLE_EQ(d.bbox.size_x, 30.0);
  EXPECT_DOUBLE_EQ(d.bbox.size_y, 40.0);
  // ...and back: x = cx - size_x/2.
  EXPECT_DOUBLE_EQ(d.bbox.center.position.x - 0.5 * d.bbox.size_x, 10.0);
  ASSERT_EQ(d.results.size(), 1u);
  EXPECT_EQ(d.results[0].hypothesis.class_id, "2");
  EXPECT_NEAR(d.results[0].hypothesis.score, 0.9, 1e-6);
}

TEST(ToDetections, ConfirmedOnlyByDefaultLostOnRequest) {
  const std::vector<ctrk::Track> tracks = {
      make_track(1, ctrk::TrackState::Tentative),
      make_track(2, ctrk::TrackState::Confirmed),
      make_track(3, ctrk::TrackState::Lost),
      make_track(4, ctrk::TrackState::Removed),
  };
  std_msgs::msg::Header header;

  const auto confirmed_only = ctrk_ros::to_detections(tracks, header, false);
  ASSERT_EQ(confirmed_only.detections.size(), 1u);
  EXPECT_EQ(confirmed_only.detections[0].id, "2");

  const auto with_lost = ctrk_ros::to_detections(tracks, header, true);
  ASSERT_EQ(with_lost.detections.size(), 2u);
  EXPECT_EQ(with_lost.detections[0].id, "2");
  EXPECT_EQ(with_lost.detections[1].id, "3");
}

TEST(ToSotStatus, StatesMapAndBoxKeepsTopLeftConvention) {
  std_msgs::msg::Header header;
  ctrk::SotResult r;
  r.box = {5.f, 6.f, 7.f, 8.f};
  r.score = 0.42f;

  using Msg = ctrk_interfaces::msg::SotStatus;
  r.state = ctrk::SotState::Tracking;
  EXPECT_EQ(ctrk_ros::to_sot_status(r, header).state, Msg::STATE_TRACKING);
  r.state = ctrk::SotState::Unstable;
  EXPECT_EQ(ctrk_ros::to_sot_status(r, header).state, Msg::STATE_UNSTABLE);
  r.state = ctrk::SotState::Lost;
  const auto m = ctrk_ros::to_sot_status(r, header);
  EXPECT_EQ(m.state, Msg::STATE_LOST);
  EXPECT_FLOAT_EQ(m.x, 5.f);
  EXPECT_FLOAT_EQ(m.y, 6.f);
  EXPECT_FLOAT_EQ(m.w, 7.f);
  EXPECT_FLOAT_EQ(m.h, 8.f);
  EXPECT_FLOAT_EQ(m.score, 0.42f);

  const auto idle = ctrk_ros::idle_status(header);
  EXPECT_EQ(idle.state, Msg::STATE_IDLE);
  EXPECT_FLOAT_EQ(idle.w, 0.f);
  EXPECT_FLOAT_EQ(idle.score, 0.f);
}

TEST(ToFrameView, ZeroCopyBgr8WithPaddedStride) {
  sensor_msgs::msg::Image img;
  img.encoding = "bgr8";
  img.width = 4;
  img.height = 2;
  img.step = 16;  // 4 px * 3 bytes = 12, padded to 16
  img.data.resize(img.step * img.height);
  std::iota(img.data.begin(), img.data.end(), 0);
  img.header.stamp.sec = 3;
  img.header.stamp.nanosec = 500000000u;

  const auto v = ctrk_ros::to_frame_view(img);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->data, img.data.data());  // zero-copy: same buffer
  EXPECT_EQ(v->width, 4);
  EXPECT_EQ(v->height, 2);
  EXPECT_EQ(v->stride_bytes, 16);
  EXPECT_EQ(v->fmt, ctrk::PixelFormat::BGR8);
  EXPECT_EQ(v->t_ns, 3500000000LL);
}

TEST(ToFrameView, RejectsNonBgr8) {
  sensor_msgs::msg::Image img;
  img.encoding = "rgb8";
  img.width = img.height = 1;
  img.step = 3;
  img.data.resize(3);
  EXPECT_FALSE(ctrk_ros::to_frame_view(img).has_value());
}

TEST(StampToNs, TruncatedIndexStampsSplitLosslessly) {
  // The frames_player index mode computes t = int64(idx/fps*1e9) and splits
  // into sec/nanosec; the reassembly here must be exact for every frame index
  // or the KF dt sequence diverges from the CLI (parity gate, S4.1).
  for (int idx : {0, 1, 2, 29, 30, 31, 1000, 1049}) {
    const double fps = 30.0;
    const auto t_ns =
        static_cast<int64_t>(static_cast<double>(idx) / fps * 1e9);  // app_common.cpp math
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(t_ns / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(t_ns % 1000000000LL);
    EXPECT_EQ(ctrk_ros::stamp_to_ns(stamp), t_ns) << "frame " << idx;
  }
}
