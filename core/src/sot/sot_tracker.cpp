#include <optional>

#include "common/mat_view.hpp"
#include "ctrk/sot.hpp"
#include "sot/mosse.hpp"
#include "sot/nanotrack.hpp"

namespace ctrk {

struct SotTracker::Impl {
  explicit Impl(const SotConfig& config) {
    if (config.backend == SotBackend::NanoTrack) {
      nano.emplace(config);
    } else {
      mosse.emplace(config);
    }
  }

  std::optional<NanoTracker> nano;
  std::optional<MosseTracker> mosse;
};

SotTracker::SotTracker(const SotConfig& config) : impl_(std::make_unique<Impl>(config)) {}
SotTracker::~SotTracker() = default;
SotTracker::SotTracker(SotTracker&&) noexcept = default;
SotTracker& SotTracker::operator=(SotTracker&&) noexcept = default;

void SotTracker::init(const FrameView& frame, const BBox& target) {
  const cv::Mat img = as_mat(frame);
  if (impl_->nano) {
    impl_->nano->init(img, target);
  } else {
    impl_->mosse->init(img, target);
  }
}

SotResult SotTracker::update(const FrameView& frame) {
  const cv::Mat img = as_mat(frame);
  return impl_->nano ? impl_->nano->update(img) : impl_->mosse->update(img);
}

}  // namespace ctrk
