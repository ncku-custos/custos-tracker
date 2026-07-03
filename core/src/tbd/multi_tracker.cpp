#include <algorithm>

#include "ctrk/tbd.hpp"
#include "tbd/byte_tracker.hpp"

namespace ctrk {

struct MultiTracker::Impl {
  explicit Impl(const TbdConfig& config)
      : cfg(config), detector(make_yolov8_detector(config.detector)), tracker(config.assoc) {}

  TbdConfig cfg;
  std::unique_ptr<IDetector> detector;
  ByteTracker tracker;
  int64_t prev_t_ns = -1;
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
    dt = std::clamp(static_cast<float>(static_cast<double>(frame.t_ns - impl_->prev_t_ns) *
                                       1e-9 * impl_->cfg.nominal_fps),
                    0.1f, 5.f);
  }
  impl_->prev_t_ns = frame.t_ns;
  return impl_->tracker.update(impl_->detector->detect(frame), dt);
}

}  // namespace ctrk
