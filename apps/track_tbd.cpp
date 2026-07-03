#include <opencv2/core.hpp>

#include <cstdio>

#include "app_common.hpp"
#include "common/mat_view.hpp"
#include "ctrk/types.hpp"

namespace {

const char* kUsage =
    "{help h        |     | show this message}"
    "{input i       |     | video file, image-sequence pattern, or camera index}"
    "{output o      | tbd_out.mp4 | annotated output video ('' to disable)}"
    "{display       |     | show a live window (never default: headless CI/drone)}";

}  // namespace

int main(int argc, char** argv) {
  cv::CommandLineParser cli(argc, argv, kUsage);
  cli.about("ctrk tracking-by-detection runner (M0 skeleton: I/O + timing only)");
  if (cli.has("help") || !cli.has("input")) {
    cli.printMessage();
    return cli.has("help") ? 0 : 1;
  }

  auto src = ctrk::app::VideoSource::open(cli.get<std::string>("input"));
  if (!src) return 1;
  ctrk::app::VideoSink sink(cli.get<std::string>("output"), src->fps(), src->size(),
                            cli.has("display"));

  ctrk::StageTimer timer;
  cv::Mat frame;
  int64_t t_ns = 0;
  int64_t prev_t_ns = -1;

  while (src->read(frame, t_ns) && sink.wants_more()) {
    {
      const auto scope = timer.scope("pipeline");
      const ctrk::FrameView view = ctrk::as_frame_view(frame, t_ns);
      (void)view;  // M1: detector + association consume the view here
    }
    const double fps_now =
        prev_t_ns >= 0 ? 1e9 / static_cast<double>(t_ns - prev_t_ns) : src->fps();
    prev_t_ns = t_ns;
    ctrk::app::draw_hud(frame, timer, fps_now);
    sink.write(frame);
  }

  std::printf("frames: %d\n", src->frames_read());
  ctrk::app::print_stage_summary(timer);
  return 0;
}
