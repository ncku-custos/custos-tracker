// Param->config mapping: defaults land struct-identical (with the documented
// exceptions), overrides map through, bad enum strings throw — which
// on_configure turns into a FAILURE transition.
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "param_config.hpp"

namespace {

void ensure_rcl() {
  if (!rclcpp::ok()) rclcpp::init(0, nullptr);
}

std::shared_ptr<rclcpp_lifecycle::LifecycleNode> make_node(
    const std::vector<rclcpp::Parameter>& overrides = {}) {
  ensure_rcl();
  static int n = 0;
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(overrides);
  return std::make_shared<rclcpp_lifecycle::LifecycleNode>("param_test_" + std::to_string(n++),
                                                           opts);
}

}  // namespace

TEST(TbdParams, DefaultsMatchTheConfigStructs) {
  auto node = make_node();
  const auto p = ctrk_ros::declare_tbd_params(*node);

  const ctrk::TbdConfig def;
  EXPECT_EQ(p.cfg.detector.model_path, "models/cache/yolov8n_640.onnx");
  EXPECT_FLOAT_EQ(p.cfg.detector.conf_thr, def.detector.conf_thr);
  EXPECT_FLOAT_EQ(p.cfg.detector.nms_iou, def.detector.nms_iou);
  EXPECT_TRUE(p.cfg.detector.keep_classes.empty());
  EXPECT_EQ(p.cfg.detector.engine.intra_op_threads, def.detector.engine.intra_op_threads);
  EXPECT_EQ(p.cfg.detector.engine.allow_spinning, def.detector.engine.allow_spinning);
  EXPECT_EQ(p.cfg.detector.engine.warmup, def.detector.engine.warmup);
  EXPECT_EQ(p.cfg.detector.engine.use_dnnl, def.detector.engine.use_dnnl);

  const ctrk::AssocConfig ad;
  EXPECT_EQ(p.cfg.assoc.use_byte, ad.use_byte);
  EXPECT_FLOAT_EQ(p.cfg.assoc.track_thresh, ad.track_thresh);
  EXPECT_FLOAT_EQ(p.cfg.assoc.match_thresh_high, ad.match_thresh_high);
  EXPECT_FLOAT_EQ(p.cfg.assoc.match_thresh_low, ad.match_thresh_low);
  EXPECT_FLOAT_EQ(p.cfg.assoc.tentative_match_thresh, ad.tentative_match_thresh);
  EXPECT_FLOAT_EQ(p.cfg.assoc.new_track_thresh, ad.new_track_thresh);
  EXPECT_EQ(p.cfg.assoc.n_init, ad.n_init);
  EXPECT_EQ(p.cfg.assoc.max_age, ad.max_age);
  EXPECT_EQ(p.cfg.assoc.nsa, ad.nsa);
  EXPECT_FLOAT_EQ(p.cfg.assoc.tentative_relax_per_coast, ad.tentative_relax_per_coast);
  EXPECT_FLOAT_EQ(p.cfg.assoc.tentative_gate_floor, ad.tentative_gate_floor);
  EXPECT_EQ(p.cfg.assoc.tentative_patience, ad.tentative_patience);
  EXPECT_EQ(p.cfg.assoc.velocity_seed, ad.velocity_seed);
  EXPECT_FLOAT_EQ(p.cfg.assoc.appearance_weight, ad.appearance_weight);
  EXPECT_FLOAT_EQ(p.cfg.assoc.embedding_ema, ad.embedding_ema);

  EXPECT_EQ(p.cfg.gmc, ctrk::GmcMethod::Off);
  EXPECT_DOUBLE_EQ(p.cfg.nominal_fps, def.nominal_fps);
  EXPECT_EQ(p.cfg.detect_interval, def.detect_interval);

  EXPECT_FALSE(p.publish_lost);
  EXPECT_TRUE(p.common.dump_path.empty());
  EXPECT_EQ(p.common.qos_reliability, "best_effort");
  EXPECT_EQ(p.common.qos_depth, 1);
  EXPECT_FALSE(p.common.bench);
}

TEST(TbdParams, OverridesMapThrough) {
  auto node = make_node({
      {"detector.model_path", "/abs/yolo_int8.onnx"},
      {"detector.conf_thr", 0.25},
      {"detector.keep_classes", std::vector<int64_t>{0, 2}},
      {"detector.engine.intra_op_threads", 8},
      {"detector.engine.allow_spinning", false},
      {"assoc.n_init", 5},
      {"assoc.nsa", false},
      {"gmc", "sparse_flow"},
      {"detect_interval", 2},
      {"publish_lost", true},
      {"dump_path", "/tmp/tbd.txt"},
      {"image_qos_reliability", "reliable"},
      {"image_qos_depth", 10},
      {"bench", true},
  });
  const auto p = ctrk_ros::declare_tbd_params(*node);

  EXPECT_EQ(p.cfg.detector.model_path, "/abs/yolo_int8.onnx");
  EXPECT_FLOAT_EQ(p.cfg.detector.conf_thr, 0.25f);
  EXPECT_EQ(p.cfg.detector.keep_classes, (std::vector<int>{0, 2}));
  EXPECT_EQ(p.cfg.detector.engine.intra_op_threads, 8);
  EXPECT_FALSE(p.cfg.detector.engine.allow_spinning);
  EXPECT_EQ(p.cfg.assoc.n_init, 5);
  EXPECT_FALSE(p.cfg.assoc.nsa);
  EXPECT_EQ(p.cfg.gmc, ctrk::GmcMethod::SparseFlow);
  EXPECT_EQ(p.cfg.detect_interval, 2);
  EXPECT_TRUE(p.publish_lost);
  EXPECT_EQ(p.common.dump_path, "/tmp/tbd.txt");
  EXPECT_EQ(p.common.qos_reliability, "reliable");
  EXPECT_EQ(p.common.qos_depth, 10);
  EXPECT_TRUE(p.common.bench);
}

TEST(TbdParams, BadEnumsThrow) {
  EXPECT_THROW(ctrk_ros::declare_tbd_params(*make_node({{"gmc", "bogus"}})), std::invalid_argument);
  EXPECT_THROW(ctrk_ros::declare_tbd_params(*make_node({{"image_qos_reliability", "sometimes"}})),
               std::invalid_argument);
}

TEST(SotParams, DefaultsMatchTheConfigStructsWithDocumentedExceptions) {
  auto node = make_node();
  const auto p = ctrk_ros::declare_sot_params(*node);

  const ctrk::SotConfig def;
  EXPECT_EQ(p.cfg.backend, ctrk::SotBackend::NanoTrack);
  EXPECT_EQ(p.cfg.backbone_z_path, "models/cache/nanotrack_backbone_z.onnx");
  EXPECT_EQ(p.cfg.engine.intra_op_threads, def.engine.intra_op_threads);
  EXPECT_EQ(p.cfg.engine.allow_spinning, def.engine.allow_spinning);  // SOT: false
  EXPECT_FLOAT_EQ(p.cfg.penalty_k, def.penalty_k);
  EXPECT_FLOAT_EQ(p.cfg.window_influence, def.window_influence);
  EXPECT_FLOAT_EQ(p.cfg.size_lr, def.size_lr);
  EXPECT_FLOAT_EQ(p.cfg.context_amount, def.context_amount);
  EXPECT_FLOAT_EQ(p.cfg.lost_score_thr, def.lost_score_thr);
  EXPECT_EQ(p.cfg.lost_patience, def.lost_patience);
  EXPECT_EQ(p.cfg.template_update_interval, def.template_update_interval);

  // Documented exception: node default mirrors the CLI operating default
  // (hsv re-lock veto, S3.3), not the struct default None.
  EXPECT_EQ(p.cfg.reid.embedder, ctrk::SotConfig::Reid::Embedder::HsvHist);
  EXPECT_FLOAT_EQ(p.cfg.reid.accept, def.reid.accept);
  EXPECT_EQ(p.cfg.reid.drift_check_every, def.reid.drift_check_every);

  EXPECT_FALSE(p.reacquire.enable);
  // Documented exceptions: CLI defaults for the reacquire detector.
  EXPECT_FLOAT_EQ(p.reacquire.detector.conf_thr, 0.25f);
  EXPECT_EQ(p.reacquire.config.class_id, 0);
  const ctrk::ReacquireConfig rd;
  EXPECT_EQ(p.reacquire.config.detect_every, rd.detect_every);
  EXPECT_FLOAT_EQ(p.reacquire.config.min_score, rd.min_score);

  EXPECT_TRUE(p.init_bbox.empty());
}

TEST(SotParams, OverridesMapThrough) {
  auto node = make_node({
      {"backend", "mosse"},
      {"reid.embedder", "nanoz"},
      {"reid.drift_check_every", 5},
      {"reacquire.enable", true},
      {"reacquire.class_id", 2},
      {"reacquire.detector.model_path", "/abs/det.onnx"},
      {"init_bbox", std::vector<double>{70.0, 51.0, 107.0, 87.0}},
  });
  const auto p = ctrk_ros::declare_sot_params(*node);

  EXPECT_EQ(p.cfg.backend, ctrk::SotBackend::Mosse);
  EXPECT_EQ(p.cfg.reid.embedder, ctrk::SotConfig::Reid::Embedder::NanoZ);
  EXPECT_EQ(p.cfg.reid.drift_check_every, 5);
  EXPECT_TRUE(p.reacquire.enable);
  EXPECT_EQ(p.reacquire.config.class_id, 2);
  EXPECT_EQ(p.reacquire.detector.model_path, "/abs/det.onnx");
  ASSERT_EQ(p.init_bbox.size(), 4u);
  EXPECT_DOUBLE_EQ(p.init_bbox[2], 107.0);
}

TEST(SotParams, BadValuesThrow) {
  EXPECT_THROW(ctrk_ros::declare_sot_params(*make_node({{"backend", "goturn"}})),
               std::invalid_argument);
  EXPECT_THROW(ctrk_ros::declare_sot_params(*make_node({{"reid.embedder", "osnet"}})),
               std::invalid_argument);
  EXPECT_THROW(
      ctrk_ros::declare_sot_params(*make_node({{"init_bbox", std::vector<double>{1.0, 2.0}}})),
      std::invalid_argument);
  EXPECT_THROW(ctrk_ros::declare_sot_params(
                   *make_node({{"init_bbox", std::vector<double>{0.0, 0.0, -5.0, 10.0}}})),
               std::invalid_argument);
}
