#include <opencv2/core.hpp>

#include <cstdio>

#include "app_common.hpp"
#include "common/mat_view.hpp"
#include "ctrk/types.hpp"

namespace {

const char* kUsage =
    "{help h        |     | show this message}"
    "{input i       |     | video file, image-sequence pattern, or camera index}"
    "{bbox b        |     | initial target box as x,y,w,h (pixels, first frame)}"
    "{output o      | sot_out.mp4 | annotated output video ('' to disable)}"
    "{display       |     | show a live window (never default: headless CI/drone)}";

}  // namespace

int main(int argc, char** argv) {
  cv::CommandLineParser cli(argc, argv, kUsage);
  cli.about("ctrk single-object-tracking runner (M0 skeleton: I/O + timing only)");
  if (cli.has("help") || !cli.has("input") || !cli.has("bbox")) {
    cli.printMessage();
    return cli.has("help") ? 0 : 1;
  }

  ctrk::BBox init_box;
  if (!ctrk::app::parse_bbox(cli.get<std::string>("bbox"), init_box)) {
    std::fprintf(stderr, "bad --bbox, expected x,y,w,h\n");
    return 1;
  }

  auto src = ctrk::app::VideoSource::open(cli.get<std::string>("input"));
  if (!src) return 1;
  ctrk::app::VideoSink sink(cli.get<std::string>("output"), src->fps(), src->size(),
                            cli.has("display"));

  ctrk::StageTimer timer;
  cv::Mat frame;
  int64_t t_ns = 0;

  while (src->read(frame, t_ns) && sink.wants_more()) {
    {
      const auto scope = timer.scope("pipeline");
      const ctrk::FrameView view = ctrk::as_frame_view(frame, t_ns);
      (void)view;  // M2: SotTracker init/update consumes the view here
    }
    ctrk::app::draw_box(frame, init_box, {0, 200, 255}, "init box (tracker lands in M2)");
    ctrk::app::draw_hud(frame, timer, src->fps());
    sink.write(frame);
  }

  std::printf("frames: %d\n", src->frames_read());
  ctrk::app::print_stage_summary(timer);
  return 0;
}
