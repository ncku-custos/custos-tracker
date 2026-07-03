#include <opencv2/imgproc.hpp>

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

namespace {

using ReidEmbedder = SotConfig::Reid::Embedder;

// H-S histogram of the box region (clipped to the image), L1-normalized.
// Empty when the clipped region is degenerate.
cv::Mat hsv_hist(const cv::Mat& img, const BBox& box) {
  const cv::Rect roi = cv::Rect(static_cast<int>(box.x), static_cast<int>(box.y),
                                static_cast<int>(box.w), static_cast<int>(box.h)) &
                       cv::Rect(0, 0, img.cols, img.rows);
  if (roi.width < 2 || roi.height < 2) return {};
  cv::Mat hsv;
  cv::cvtColor(img(roi), hsv, cv::COLOR_BGR2HSV);
  const int channels[] = {0, 1};
  const int bins[] = {30, 32};
  const float h_range[] = {0, 180}, s_range[] = {0, 256};
  const float* ranges[] = {h_range, s_range};
  cv::Mat hist;
  cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, bins, ranges);
  cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
  return hist;
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.empty() || a.size() != b.size()) return 0.f;
  float dot = 0.f, na = 0.f, nb = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const float denom = std::sqrt(na) * std::sqrt(nb);
  return denom > 0.f ? dot / denom : 0.f;
}

}  // namespace

struct SotTracker::Impl {
  explicit Impl(const SotConfig& config) : cfg(config) {
    if (cfg.lost_score_thr < 0.f)
      cfg.lost_score_thr = cfg.backend == SotBackend::NanoTrack ? 0.30f : 8.f;
    // Per-embedder threshold autos (RESULTS.md S3.3 sweeps).
    const bool nanoz = cfg.reid.embedder == ReidEmbedder::NanoZ;
    if (cfg.reid.accept < 0.f) cfg.reid.accept = nanoz ? 0.65f : 0.4f;
    if (cfg.reid.drift_thr < 0.f) cfg.reid.drift_thr = nanoz ? 0.55f : 0.4f;
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

  // Appearance similarity of `box` to the init-time reference, in the active
  // embedder's scale. 1 (pass) when no signal is available — degenerate
  // crops must not veto a rescue the geometric gates already accepted.
  float similarity(const cv::Mat& img, const BBox& box) {
    switch (cfg.reid.embedder) {
      case ReidEmbedder::HsvHist: {
        if (ref_hist.empty()) return 1.f;
        const cv::Mat hist = hsv_hist(img, box);
        if (hist.empty()) return 1.f;
        return 1.f - static_cast<float>(cv::compareHist(ref_hist, hist, cv::HISTCMP_BHATTACHARYYA));
      }
      case ReidEmbedder::NanoZ:
        if (!nano || ref_embed.empty()) return 1.f;
        return cosine(nano->embed(img, box), ref_embed);
      case ReidEmbedder::None:
        return 1.f;
    }
    return 1.f;
  }

  // Best re-acquisition candidate near the last confident box, or nullopt.
  // Without re-ID the nearest gate-passing candidate wins (proximity is the
  // least-wrong tie-break available without appearance features) — which
  // re-locks the WRONG jogger when two lookalikes are adjacent (M4). With
  // re-ID, candidates below `accept` similarity are vetoed (stay Lost) and
  // the most-similar survivor wins.
  std::optional<Detection> pick_candidate(const cv::Mat& img, const std::vector<Detection>& dets) {
    const float diag = std::hypot(last_good.w, last_good.h);
    const float radius = diag * (reacquire_cfg.base_radius_frac +
                                 reacquire_cfg.growth_per_frame * static_cast<float>(frames_lost));
    const bool use_reid = cfg.reid.embedder != ReidEmbedder::None;
    std::optional<Detection> best;
    float best_dist = radius;
    float best_sim = -1.f;
    for (const auto& d : dets) {
      if (d.score < reacquire_cfg.min_score) continue;
      if (reacquire_cfg.class_id >= 0 && d.class_id != reacquire_cfg.class_id) continue;
      const float size_ratio = std::sqrt((d.box.w * d.box.h) / (last_good.w * last_good.h));
      if (size_ratio < reacquire_cfg.size_low || size_ratio > reacquire_cfg.size_high) continue;
      const float dist = std::hypot(d.box.cx() - last_good.cx(), d.box.cy() - last_good.cy());
      if (dist > radius) continue;
      if (use_reid) {
        const float sim = similarity(img, d.box);
        if (sim < cfg.reid.accept) continue;
        if (sim > best_sim) {
          best_sim = sim;
          best = d;
        }
      } else if (dist <= best_dist) {
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

  cv::Mat ref_hist;              // HsvHist re-ID reference (init box)
  std::vector<float> ref_embed;  // NanoZ re-ID reference (init template)
  int since_drift_check = 0;
  bool drift_latched = false;
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
  impl_->drift_latched = false;
  impl_->since_drift_check = 0;
  const cv::Mat img = as_mat(frame);
  impl_->backend_init(img, target);
  // Capture the re-ID reference from the init-time target: it stays frozen
  // across re-locks so verification always compares against the original.
  if (impl_->cfg.reid.embedder == ReidEmbedder::HsvHist) {
    impl_->ref_hist = hsv_hist(img, target);
  } else if (impl_->cfg.reid.embedder == ReidEmbedder::NanoZ && impl_->nano) {
    impl_->ref_embed = impl_->nano->zf();
  }
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
    // Proactive drift verification (the Girl2 mode: confident drift to a
    // distractor never collapses the score, so the state machine alone is
    // blind to it). Every K frames — or every frame while latched — the
    // tracked box must still look like the init-time target; a failing check
    // latches Lost until appearance recovers or a verified re-lock lands.
    const bool check_due =
        impl_->cfg.reid.embedder != ReidEmbedder::None && impl_->cfg.reid.drift_check_every > 0 &&
        (impl_->drift_latched || ++impl_->since_drift_check >= impl_->cfg.reid.drift_check_every);
    if (!check_due) {
      impl_->last_good = r.box;
      impl_->frames_lost = 0;
      return r;
    }
    impl_->since_drift_check = 0;
    const float sim = impl_->similarity(img, r.box);
    if (sim >= impl_->cfg.reid.drift_thr) {
      impl_->drift_latched = false;
      impl_->last_good = r.box;
      impl_->frames_lost = 0;
      return r;
    }
    if (!impl_->drift_latched) {
      char msg[96];
      std::snprintf(msg, sizeof(msg), "sot: drift detected (similarity %.2f < %.2f, score %.2f)",
                    sim, impl_->cfg.reid.drift_thr, r.score);
      log(LogLevel::Info, msg);
    }
    impl_->drift_latched = true;
    impl_->low_streak = impl_->cfg.lost_patience;
    r.state = SotState::Lost;
  }
  if (r.state != SotState::Lost || !impl_->detector) return r;

  // Lost with a detector available: periodically look for the target near
  // where it was last credibly seen and re-seed the template from the match.
  // The first attempt waits a full detect_every cycle — re-acquiring on the
  // very first lost frame tends to grab whatever is still visible next to a
  // just-started occlusion (e.g. the second jogger in OTB Jogging).
  impl_->frames_lost++;
  if (impl_->frames_lost % impl_->reacquire_cfg.detect_every != 0) return r;

  const auto candidate = impl_->pick_candidate(img, impl_->detector->detect(frame));
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
