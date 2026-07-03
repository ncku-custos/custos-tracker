#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>

#include "common/mat_view.hpp"
#include "ctrk/log.hpp"
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

  void backend_init(const cv::Mat& img, const BBox& box) {
    if (nano) {
      nano->init(img, box);
    } else {
      mosse->init(img, box);
    }
  }

  // Tracking -> Unstable (1..patience-1 consecutive low scores) -> Lost.
  // A healthy score anywhere on the way restores Tracking.
  SotState classify(float score) {
    low_streak = score < cfg.lost_score_thr ? low_streak + 1 : 0;
    if (low_streak == 0) return SotState::Tracking;
    return low_streak >= cfg.lost_patience ? SotState::Lost : SotState::Unstable;
  }

  // Best re-acquisition candidate near the last confident box, or nullopt.
  // Nearest-passing-candidate wins (NOT highest detector score): when several
  // same-class objects pass the gates, proximity to the last confident
  // position is the least-wrong tie-break available without appearance
  // features (re-ID verification is step 3).
  std::optional<Detection> pick_candidate(const std::vector<Detection>& dets) const {
    const float diag = std::hypot(last_good.w, last_good.h);
    const float radius = diag * (reacquire_cfg.base_radius_frac +
                                 reacquire_cfg.growth_per_frame * static_cast<float>(frames_lost));
    std::optional<Detection> best;
    float best_dist = radius;
    for (const auto& d : dets) {
      if (d.score < reacquire_cfg.min_score) continue;
      if (reacquire_cfg.class_id >= 0 && d.class_id != reacquire_cfg.class_id) continue;
      const float size_ratio = std::sqrt((d.box.w * d.box.h) / (last_good.w * last_good.h));
      if (size_ratio < reacquire_cfg.size_low || size_ratio > reacquire_cfg.size_high) continue;
      const float dist = std::hypot(d.box.cx() - last_good.cx(), d.box.cy() - last_good.cy());
      if (dist <= best_dist) {
        best_dist = dist;
        best = d;
      }
    }
    return best;
  }

  SotConfig cfg;
  std::optional<NanoTracker> nano;
  std::optional<MosseTracker> mosse;
  int low_streak = 0;

  std::unique_ptr<IDetector> detector;
  ReacquireConfig reacquire_cfg;
  BBox last_good;
  int frames_lost = 0;
};

SotTracker::SotTracker(const SotConfig& config) : impl_(std::make_unique<Impl>(config)) {}
SotTracker::~SotTracker() = default;
SotTracker::SotTracker(SotTracker&&) noexcept = default;
SotTracker& SotTracker::operator=(SotTracker&&) noexcept = default;

void SotTracker::enable_reacquire(std::unique_ptr<IDetector> detector,
                                  const ReacquireConfig& config) {
  impl_->detector = std::move(detector);
  impl_->reacquire_cfg = config;
}

void SotTracker::init(const FrameView& frame, const BBox& target) {
  impl_->low_streak = 0;
  impl_->frames_lost = 0;
  impl_->last_good = target;
  impl_->backend_init(as_mat(frame), target);
}

SotResult SotTracker::update(const FrameView& frame) {
  const cv::Mat img = as_mat(frame);
  SotResult r = impl_->nano ? impl_->nano->update(img) : impl_->mosse->update(img);
  const SotState prev = impl_->low_streak >= impl_->cfg.lost_patience ? SotState::Lost
                        : impl_->low_streak > 0                       ? SotState::Unstable
                                                                      : SotState::Tracking;
  r.state = impl_->classify(r.score);
  if (r.state == SotState::Lost && prev != SotState::Lost) {
    char msg[96];
    std::snprintf(msg, sizeof(msg), "sot: target lost (score %.2f < %.2f)", r.score,
                  impl_->cfg.lost_score_thr);
    log(LogLevel::Info, msg);
  }

  if (r.state == SotState::Tracking) {
    impl_->last_good = r.box;
    impl_->frames_lost = 0;
    return r;
  }
  if (r.state != SotState::Lost || !impl_->detector) return r;

  // Lost with a detector available: periodically look for the target near
  // where it was last credibly seen and re-seed the template from the match.
  // The first attempt waits a full detect_every cycle — re-acquiring on the
  // very first lost frame tends to grab whatever is still visible next to a
  // just-started occlusion (e.g. the second jogger in OTB Jogging).
  impl_->frames_lost++;
  if (impl_->frames_lost % impl_->reacquire_cfg.detect_every != 0) return r;

  const auto candidate = impl_->pick_candidate(impl_->detector->detect(frame));
  if (!candidate) return r;

  impl_->backend_init(img, candidate->box);
  impl_->low_streak = impl_->cfg.lost_patience - 1;  // probation: prove the re-lock
  {
    char msg[96];
    std::snprintf(msg, sizeof(msg), "sot: re-acquired at %.0f,%.0f (det score %.2f, lost %d)",
                  candidate->box.cx(), candidate->box.cy(), candidate->score, impl_->frames_lost);
    log(LogLevel::Info, msg);
  }
  r.box = candidate->box;
  r.score = candidate->score;
  r.state = SotState::Unstable;
  return r;
}

}  // namespace ctrk
