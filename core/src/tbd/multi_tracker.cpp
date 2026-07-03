#include <algorithm>

#include "common/appearance.hpp"
#include "common/mat_view.hpp"
#include "ctrk/profile.hpp"
#include "ctrk/tbd.hpp"
#include "tbd/byte_tracker.hpp"
#include "tbd/gmc.hpp"

namespace ctrk {

struct MultiTracker::Impl {
  explicit Impl(const TbdConfig& config)
      : cfg(config), detector(make_yolov8_detector(config.detector)), tracker(config.assoc) {}

  TbdConfig cfg;
  std::unique_ptr<IDetector> detector;
  ByteTracker tracker;
  GmcEstimator gmc;               // stateful: previous downscaled gray (S3.4)
  std::vector<BBox> gmc_exclude;  // last frame's track boxes, masked from GMC corners
  int64_t prev_t_ns = -1;
  int until_detect = 0;  // frames left to coast before the next detect
};

MultiTracker::MultiTracker(const TbdConfig& config) : impl_(std::make_unique<Impl>(config)) {}
MultiTracker::~MultiTracker() = default;
MultiTracker::MultiTracker(MultiTracker&&) noexcept = default;
MultiTracker& MultiTracker::operator=(MultiTracker&&) noexcept = default;

std::vector<Track> MultiTracker::update(const FrameView& frame) {
  // dt in frame units from caller timestamps; clamped so stream glitches
  // (stalls, wrap-around) cannot catapult the motion model.
  float dt = 1.f;
  if (impl_->prev_t_ns >= 0 && frame.t_ns > impl_->prev_t_ns) {
    dt = std::clamp(static_cast<float>(static_cast<double>(frame.t_ns - impl_->prev_t_ns) * 1e-9 *
                                       impl_->cfg.nominal_fps),
                    0.1f, 5.f);
  }
  impl_->prev_t_ns = frame.t_ns;

  // The camera warp is per-frame state: it must be estimated on EVERY frame
  // (coast frames included) or the previous-gray chain skips frames and the
  // warp no longer matches the track states' pose. Last frame's track boxes
  // are masked out of the corner detector (moving objects hijack the
  // estimate in dense scenes — S3.4).
  Affine23 warp;
  if (impl_->cfg.gmc == GmcMethod::SparseFlow) {
    ProfileScope prof("gmc");
    warp = impl_->gmc.estimate(as_mat(frame), impl_->gmc_exclude);
  }

  const bool track_gmc = impl_->cfg.gmc != GmcMethod::Off;
  if (impl_->until_detect > 0) {
    impl_->until_detect--;
    ProfileScope prof("coast");
    auto out = impl_->tracker.coast(dt, warp);
    if (track_gmc) {
      impl_->gmc_exclude.clear();
      for (const auto& t : out) impl_->gmc_exclude.push_back(t.box);
    }
    return out;
  }
  impl_->until_detect = std::max(1, impl_->cfg.detect_interval) - 1;

  std::vector<Detection> dets = impl_->detector->detect(frame);
  if (impl_->cfg.assoc.appearance_weight > 0.f) {
    ProfileScope prof("embed");
    const cv::Mat img = as_mat(frame);
    for (auto& d : dets) d.embedding = hsv_embedding(img, d.box);
  }
  ProfileScope prof("assoc");
  auto out = impl_->tracker.update(dets, dt, warp);
  if (track_gmc) {
    impl_->gmc_exclude.clear();
    for (const auto& d : dets) impl_->gmc_exclude.push_back(d.box);
    for (const auto& t : out) impl_->gmc_exclude.push_back(t.box);
  }
  return out;
}

}  // namespace ctrk
