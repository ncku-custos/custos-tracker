#pragma once

#include <memory>
#include <string>

#include "ctrk/detector.hpp"
#include "ctrk/infer.hpp"
#include "ctrk/types.hpp"

namespace ctrk {

enum class SotBackend : uint8_t {
  NanoTrack,  // deep siamese, main line (needs the three model paths)
  Mosse,      // hand-written correlation filter, NN-free CPU fallback
};

struct SotConfig {
  SotBackend backend = SotBackend::NanoTrack;

  // NanoTrack v2 as three static graphs (tools/export/export_nanotrack.py).
  std::string backbone_z_path;  // template branch, run once at init
  std::string backbone_x_path;  // search branch, per frame
  std::string head_path;        // correlation head, per frame
  // Engine knobs, applied to all three graphs. Measured on the reference
  // host (docs/RESULTS.md S2.2): 2 threads with spinning DISABLED is both
  // the fastest and the cheapest-in-CPU configuration — three small
  // sessions' pools busy-waiting between the two sequential runs per frame
  // only fight each other. 1 thread trades ~0.6 ms for another ~60% CPU cut.
  EngineOptions engine{.intra_op_threads = 2, .allow_spinning = false};

  // Post-processing constants — cv::TrackerNano parity (do not tune before
  // the oracle differential test passes; tuning is step 3).
  float penalty_k = 0.055f;
  float window_influence = 0.455f;
  float size_lr = 0.37f;
  float context_amount = 0.5f;

  // Lost detection (score below threshold for `lost_patience` consecutive
  // frames). Score scales differ per backend, so -1 means auto: 0.30 for
  // NanoTrack (raw classifier peak), 8.0 for MOSSE (PSR). 1..patience-1 low
  // frames reports Unstable; recovery above threshold restores Tracking.
  float lost_score_thr = -1.f;
  int lost_patience = 5;

  // Appearance verification (RESULTS.md S3.3). Two embedders: an HSV H-S
  // colour histogram (backend-free, zero models) and the NanoTrack template
  // branch reused on candidate crops (nano backend only, zero new models).
  // Similarity is 1 - Bhattacharyya for HsvHist and cosine for NanoZ;
  // `accept`/`drift_thr` live in that per-embedder scale. Applied at two
  // places: re-lock candidates below `accept` are vetoed (stay Lost rather
  // than lock a lookalike — the Jogging failure), and with
  // drift_check_every > 0 the *tracked* box is verified while Tracking so
  // confident drift is caught even though the score never collapses (the
  // Girl2 failure); a drift verdict latches Lost until appearance recovers
  // or a verified re-acquisition lands.
  // Thresholds live in the embedder's own similarity scale, so -1 means
  // auto (measured in RESULTS.md S3.3): HsvHist accept 0.4 / drift 0.4,
  // NanoZ accept 0.65 / drift 0.55.
  struct Reid {
    enum class Embedder : uint8_t { None, HsvHist, NanoZ };
    Embedder embedder = Embedder::None;
    float accept = -1.f;        // min similarity for a re-lock candidate; -1 = auto
    int drift_check_every = 0;  // 0 = off; else verify the tracked box every K frames
    float drift_thr = -1.f;     // tracked-box similarity below this -> Lost; -1 = auto
  } reid;
};

// Detector-assisted re-acquisition, active while the state machine reports
// Lost: candidates must match the class, sit within a radius that grows with
// time-lost, and have a similar size to the last confidently-tracked box.
// The best candidate re-initializes the tracker template (probation:
// Unstable until the score proves the re-lock).
struct ReacquireConfig {
  int class_id = -1;       // required detector class; -1 accepts any
  int detect_every = 3;    // detector cadence (frames) while Lost
  float min_score = 0.4f;  // detector confidence floor for candidates
  float size_low = 0.5f;   // candidate/last size ratio gate
  float size_high = 2.f;
  float base_radius_frac = 1.5f;   // search radius, in last-box diagonals...
  float growth_per_frame = 0.05f;  // ...growing per lost frame
};

// Single-object tracker facade. init() with a target box, then update() per
// frame. Single-threaded, synchronous; one instance per thread.
class SotTracker {
 public:
  explicit SotTracker(const SotConfig& config);
  ~SotTracker();
  SotTracker(SotTracker&&) noexcept;
  SotTracker& operator=(SotTracker&&) noexcept;

  // Optional: enable re-acquisition while Lost. Takes ownership.
  void enable_reacquire(std::unique_ptr<IDetector> detector, const ReacquireConfig& config);

  void init(const FrameView& frame, const BBox& target);
  // score is the raw model confidence (pre-window classifier peak for the
  // siamese backend, PSR for MOSSE) — it drives the state machine.
  SotResult update(const FrameView& frame);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ctrk
