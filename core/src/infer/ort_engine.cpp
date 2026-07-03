#include <dnnl_provider_options.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#include <map>
#include <stdexcept>

#include "ctrk/infer.hpp"
#include "ctrk/log.hpp"

namespace ctrk {

namespace {

// One process-wide environment: the env owns ORT's logging/telemetry state
// and (unlike sessions) is safe to share; per-engine envs would each carry
// their own bookkeeping for no benefit.
Ort::Env& shared_env() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ctrk");
  return env;
}

class OrtEngine final : public IEngine {
 public:
  OrtEngine(const std::string& model_path, const EngineOptions& options)
      : mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(options.intra_op_threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (!options.allow_spinning) {
      so.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    }
    if (options.use_dnnl) {
      OrtDnnlProviderOptions dnnl{};
      dnnl.use_arena = 1;
      so.AppendExecutionProvider_Dnnl(dnnl);
    }
    session_ = Ort::Session(shared_env(), model_path.c_str(), so);

    const Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < session_.GetInputCount(); ++i) {
      const auto name = session_.GetInputNameAllocated(i, alloc);
      input_descs_.push_back(describe(name.get(), session_.GetInputTypeInfo(i)));
    }
    for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
      const auto name = session_.GetOutputNameAllocated(i, alloc);
      output_descs_.push_back(describe(name.get(), session_.GetOutputTypeInfo(i)));
    }
    // Name arrays for Session::Run, built once the desc vectors are final
    // (string storage is stable from here on).
    for (const auto& d : input_descs_) in_names_.push_back(d.name.c_str());
    for (const auto& d : output_descs_) out_names_.push_back(d.name.c_str());

    if (options.warmup) {
      std::vector<std::vector<float>> zeros;
      std::vector<TensorView> views;
      for (const auto& d : input_descs_) {
        zeros.emplace_back(static_cast<size_t>(d.elements()), 0.f);
        views.push_back({d.name, d.shape, zeros.back().data()});
      }
      run(views);
    }
  }

  const std::vector<TensorDesc>& input_descs() const override { return input_descs_; }
  const std::vector<TensorDesc>& output_descs() const override { return output_descs_; }

  std::vector<TensorView> run(const std::vector<TensorView>& inputs) override {
    if (inputs.size() != input_descs_.size())
      throw std::runtime_error("engine expects " + std::to_string(input_descs_.size()) +
                               " inputs, got " + std::to_string(inputs.size()));

    ort_inputs_.clear();
    for (const auto& desc : input_descs_) {
      const TensorView* match = nullptr;
      for (const auto& in : inputs)
        if (in.name == desc.name) match = &in;
      if (!match) throw std::runtime_error("missing input tensor: " + desc.name);
      if (match->shape != desc.shape)
        throw std::runtime_error("shape mismatch for input: " + desc.name);
      ort_inputs_.push_back(Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(match->data),
                                                            static_cast<size_t>(desc.elements()),
                                                            desc.shape.data(), desc.shape.size()));
    }

    last_outputs_ = session_.Run(Ort::RunOptions{nullptr}, in_names_.data(), ort_inputs_.data(),
                                 ort_inputs_.size(), out_names_.data(), out_names_.size());

    std::vector<TensorView> views;
    for (size_t i = 0; i < last_outputs_.size(); ++i) {
      const auto info = last_outputs_[i].GetTensorTypeAndShapeInfo();
      views.push_back(
          TensorView{output_descs_[i].name, info.GetShape(), last_outputs_[i].GetTensorData<float>()});
    }
    return views;
  }

 private:
  static TensorDesc describe(const char* name, const Ort::TypeInfo& info) {
    TensorDesc d;
    d.name = name;
    d.shape = info.GetTensorTypeAndShapeInfo().GetShape();
    for (auto& dim : d.shape) {
      if (dim < 0)
        throw std::runtime_error("dynamic shape on tensor '" + d.name +
                                 "' — ctrk requires static graphs (re-export)");
    }
    return d;
  }

  Ort::MemoryInfo mem_;
  Ort::Session session_{nullptr};
  std::vector<TensorDesc> input_descs_, output_descs_;
  std::vector<const char*> in_names_, out_names_;  // point into the descs
  std::vector<Ort::Value> ort_inputs_;             // reused per run
  std::vector<Ort::Value> last_outputs_;           // keeps output buffers alive until next run
};

}  // namespace

std::unique_ptr<IEngine> make_ort_engine(const std::string& model_path,
                                         const EngineOptions& options) {
  return std::make_unique<OrtEngine>(model_path, options);
}

}  // namespace ctrk
