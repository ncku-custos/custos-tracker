#include <optional>

#include "common/mat_view.hpp"
#include "ctrk/sot.hpp"
#include "sot/mosse.hpp"
#include "sot/nanotrack.hpp"

namespace ctrk {

struct SotTracker::Impl {
  explicit Impl(const SotConfig& config) : cfg(config) {
    if (cfg.lost_score_thr < 0.f)
      cfg.lost_score_thr = cfg.backend == SotBackend::NanoTrack ? 0.30f : 8.f;
    if (cfg.backend == SotBackend::NanoTrack) {
      nano.emplace(cfg);
    } else {
      mosse.emplace(cfg);
    }
  }

  // Tracking -> Unstable (1..patience-1 consecutive low scores) -> Lost.
  // A healthy score anywhere on the way restores Tracking.
  SotState classify(float score) {
    low_streak = score < cfg.lost_score_thr ? low_streak + 1 : 0;
    if (low_streak == 0) return SotState::Tracking;
    return low_streak >= cfg.lost_patience ? SotState::Lost : SotState::Unstable;
  }

  SotConfig cfg;
  std::optional<NanoTracker> nano;
  std::optional<MosseTracker> mosse;
  int low_streak = 0;
};

SotTracker::SotTracker(const SotConfig& config) : impl_(std::make_unique<Impl>(config)) {}
SotTracker::~SotTracker() = default;
SotTracker::SotTracker(SotTracker&&) noexcept = default;
SotTracker& SotTracker::operator=(SotTracker&&) noexcept = default;

void SotTracker::init(const FrameView& frame, const BBox& target) {
  const cv::Mat img = as_mat(frame);
  impl_->low_streak = 0;
  if (impl_->nano) {
    impl_->nano->init(img, target);
  } else {
    impl_->mosse->init(img, target);
  }
}

SotResult SotTracker::update(const FrameView& frame) {
  const cv::Mat img = as_mat(frame);
  SotResult r = impl_->nano ? impl_->nano->update(img) : impl_->mosse->update(img);
  r.state = impl_->classify(r.score);
  return r;
}

}  // namespace ctrk
