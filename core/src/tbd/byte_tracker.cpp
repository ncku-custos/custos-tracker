#include "tbd/byte_tracker.hpp"

#include <algorithm>
#include <cmath>

#include "common/appearance.hpp"
#include "common/geometry.hpp"
#include "common/munkres.hpp"

namespace ctrk {

namespace {

// IoU-gated Munkres. Returns (track_idx, det_idx) pairs whose IoU clears
// min_iou; forbidden pairs enter the cost matrix as a large finite value and
// are post-filtered (see munkres.hpp gating contract). When `track_embs` is
// given (stage 1, S3.5) the cost of gate-passing pairs fuses an appearance
// term: (1-w)*(1-IoU) + w*(1-cos) — appearance refines the ranking among
// geometrically plausible pairs, never overrides the gate.
std::vector<std::pair<int, int>> match_by_iou(
    const std::vector<BBox>& track_boxes, const std::vector<int>& track_ids,
    const std::vector<Detection>& dets, const std::vector<int>& det_ids, float min_iou,
    const std::vector<const std::vector<float>*>* track_embs = nullptr, float app_weight = 0.f) {
  std::vector<std::pair<int, int>> matches;
  if (track_ids.empty() || det_ids.empty()) return matches;

  constexpr float kForbidden = 1e6f;
  std::vector<std::vector<float>> cost(track_ids.size(), std::vector<float>(det_ids.size()));
  for (size_t i = 0; i < track_ids.size(); ++i)
    for (size_t j = 0; j < det_ids.size(); ++j) {
      const auto& det = dets[det_ids[j]];
      const float overlap = iou(track_boxes[track_ids[i]], det.box);
      if (overlap < min_iou) {
        cost[i][j] = kForbidden;
        continue;
      }
      float c = 1.f - overlap;
      if (track_embs && app_weight > 0.f && !det.embedding.empty()) {
        const std::vector<float>* emb = (*track_embs)[track_ids[i]];
        if (emb && !emb->empty())
          c = (1.f - app_weight) * c + app_weight * (1.f - cosine_similarity(*emb, det.embedding));
      }
      cost[i][j] = c;
    }

  const auto assignment = munkres_solve(cost);
  for (size_t i = 0; i < assignment.size(); ++i)
    if (assignment[i] >= 0 && cost[i][assignment[i]] < kForbidden)
      matches.emplace_back(track_ids[i], det_ids[assignment[i]]);
  return matches;
}

std::vector<int> remove_matched(const std::vector<int>& pool,
                                const std::vector<std::pair<int, int>>& matches, bool first) {
  std::vector<int> rest;
  for (int idx : pool) {
    bool taken = false;
    for (const auto& m : matches)
      if ((first ? m.first : m.second) == idx) taken = true;
    if (!taken) rest.push_back(idx);
  }
  return rest;
}

}  // namespace

void ByteTracker::mark_matched(STrack& track, const Detection& det) {
  // A newborn's first re-match carries its true displacement: seed the KF
  // velocity from it instead of blending from the zero-velocity prior.
  if (cfg_.velocity_seed && track.hits == 1) track.kf.seed_velocity(det.box, eff_dt_);
  track.kf.update(det.box, cfg_.nsa ? 1.f - det.score : 1.f);
  if (!det.embedding.empty()) {
    if (track.embedding.size() != det.embedding.size()) {
      track.embedding = det.embedding;
    } else {
      float norm = 0.f;
      for (size_t i = 0; i < track.embedding.size(); ++i) {
        track.embedding[i] =
            cfg_.embedding_ema * track.embedding[i] + (1.f - cfg_.embedding_ema) * det.embedding[i];
        norm += track.embedding[i] * track.embedding[i];
      }
      norm = std::sqrt(norm);
      if (norm > 0.f)
        for (auto& v : track.embedding) v /= norm;
    }
  }
  track.time_since_update = 0;
  track.hits++;
  track.score = det.score;
  track.class_id = det.class_id;
  if (track.state == TrackState::Lost) track.state = TrackState::Confirmed;
  if (track.state == TrackState::Tentative && track.hits >= cfg_.n_init)
    track.state = TrackState::Confirmed;
}

std::vector<Track> ByteTracker::coast(float dt, const Affine23& warp) {
  coasted_dt_ += dt;
  std::vector<Track> out;
  out.reserve(tracks_.size());
  for (auto& t : tracks_) {
    t.kf.predict(dt);
    if (!warp.identity()) t.kf.apply_affine(warp);
    t.age++;
    out.push_back({t.id, t.kf.box(), t.score, t.class_id, t.state, t.age, t.hits});
  }
  return out;
}

std::vector<Track> ByteTracker::update(const std::vector<Detection>& detections, float dt,
                                       const Affine23& warp) {
  const float coasted = coasted_dt_;
  coasted_dt_ = 0.f;
  eff_dt_ = coasted + dt;

  // Predict every live track forward, then warp into the current camera
  // pose (BoT-SORT order: predict, GMC-correct, associate).
  std::vector<BBox> predicted(tracks_.size());
  for (size_t i = 0; i < tracks_.size(); ++i) {
    tracks_[i].kf.predict(dt);
    if (!warp.identity()) tracks_[i].kf.apply_affine(warp);
    tracks_[i].age++;
    tracks_[i].time_since_update++;
    predicted[i] = tracks_[i].kf.box();
  }

  // Split detections by score.
  std::vector<int> high_dets, low_dets;
  for (size_t j = 0; j < detections.size(); ++j)
    (detections[j].score >= cfg_.track_thresh ? high_dets : low_dets)
        .push_back(static_cast<int>(j));

  std::vector<int> mature_tracks, tentative_tracks;
  for (size_t i = 0; i < tracks_.size(); ++i)
    (tracks_[i].state == TrackState::Tentative ? tentative_tracks : mature_tracks)
        .push_back(static_cast<int>(i));

  // Stage 1: high-score dets vs confirmed + lost tracks (appearance-fused
  // cost when enabled, S3.5).
  std::vector<const std::vector<float>*> track_embs;
  if (cfg_.appearance_weight > 0.f) {
    track_embs.reserve(tracks_.size());
    for (const auto& t : tracks_) track_embs.push_back(&t.embedding);
  }
  const float stage1_min_iou = 1.f - cfg_.match_thresh_high;
  auto matches =
      match_by_iou(predicted, mature_tracks, detections, high_dets, stage1_min_iou,
                   cfg_.appearance_weight > 0.f ? &track_embs : nullptr, cfg_.appearance_weight);
  auto unmatched_mature = remove_matched(mature_tracks, matches, /*first=*/true);
  auto unmatched_high = remove_matched(high_dets, matches, /*first=*/false);

  // Stage 2 (ByteTrack): low-score dets vs tracks matched as recently as the
  // previous frame. Recovers occlusion/blur-dimmed detections instead of
  // letting the track coast blind.
  if (cfg_.use_byte) {
    std::vector<int> recent;
    for (int idx : unmatched_mature)
      if (tracks_[idx].state == TrackState::Confirmed && tracks_[idx].time_since_update == 1)
        recent.push_back(idx);
    const auto low_matches =
        match_by_iou(predicted, recent, detections, low_dets, cfg_.match_thresh_low);
    matches.insert(matches.end(), low_matches.begin(), low_matches.end());
    unmatched_mature = remove_matched(unmatched_mature, low_matches, /*first=*/true);
  }

  // Stage 3: remaining high-score dets vs tentative tracks. The gate relaxes
  // with coasted frames (a newborn's prediction lags by the full interval);
  // zero coasting -> the configured gate, bit-identical N=1 behavior.
  const float tent_gate =
      std::max(cfg_.tentative_gate_floor,
               cfg_.tentative_match_thresh - cfg_.tentative_relax_per_coast * coasted);
  const auto tentative_matches =
      match_by_iou(predicted, tentative_tracks, detections, unmatched_high, tent_gate);
  matches.insert(matches.end(), tentative_matches.begin(), tentative_matches.end());
  const auto unmatched_tentative =
      remove_matched(tentative_tracks, tentative_matches, /*first=*/true);
  unmatched_high = remove_matched(unmatched_high, tentative_matches, /*first=*/false);

  for (const auto& [ti, dj] : matches) mark_matched(tracks_[ti], detections[dj]);

  // Lifecycle for unmatched tracks. Unconfirmed misses drop after
  // tentative_patience extra detect-frames (0 = immediately, the default).
  std::vector<bool> removed(tracks_.size(), false);
  for (int idx : unmatched_tentative)
    if (tracks_[idx].time_since_update > cfg_.tentative_patience) removed[idx] = true;
  for (int idx : unmatched_mature) {
    auto& t = tracks_[idx];
    if (t.state == TrackState::Confirmed) t.state = TrackState::Lost;
    if (t.time_since_update > cfg_.max_age) removed[idx] = true;
  }

  // Spawn tracks from confident unmatched high-score detections.
  for (int dj : unmatched_high) {
    const auto& det = detections[dj];
    if (det.score < cfg_.new_track_thresh) continue;
    STrack t;
    t.kf.initiate(det.box);
    t.id = next_id_++;
    t.state = cfg_.n_init <= 1 ? TrackState::Confirmed : TrackState::Tentative;
    t.score = det.score;
    t.class_id = det.class_id;
    tracks_.push_back(std::move(t));
    removed.push_back(false);
  }

  std::vector<STrack> alive;
  for (size_t i = 0; i < tracks_.size(); ++i)
    if (!removed[i]) alive.push_back(std::move(tracks_[i]));
  tracks_ = std::move(alive);

  std::vector<Track> out;
  out.reserve(tracks_.size());
  for (const auto& t : tracks_)
    out.push_back({t.id, t.kf.box(), t.score, t.class_id, t.state, t.age, t.hits});
  return out;
}

}  // namespace ctrk
