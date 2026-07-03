#include "common/mat_view.hpp"
#include "ctrk/sot.hpp"
#include "sot/nanotrack.hpp"

namespace ctrk {

struct SotTracker::Impl {
  explicit Impl(const SotConfig& config) : nano(config) {}
  NanoTracker nano;
};

SotTracker::SotTracker(const SotConfig& config) : impl_(std::make_unique<Impl>(config)) {}
SotTracker::~SotTracker() = default;
SotTracker::SotTracker(SotTracker&&) noexcept = default;
SotTracker& SotTracker::operator=(SotTracker&&) noexcept = default;

void SotTracker::init(const FrameView& frame, const BBox& target) {
  impl_->nano.init(as_mat(frame), target);
}

SotResult SotTracker::update(const FrameView& frame) { return impl_->nano.update(as_mat(frame)); }

}  // namespace ctrk
