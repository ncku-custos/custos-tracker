#include <onnxruntime_cxx_api.h>

#include <map>
#include <stdexcept>

#include "ctrk/infer.hpp"
#include "ctrk/log.hpp"

namespace ctrk {

namespace {

class OrtEngine final : public IEngine {
 public:
  OrtEngine(const std::string& model_path, const EngineOptions& options)
      : env_(ORT_LOGGING_LEVEL_WARNING, "ctrk") {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(options.intra_op_threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, model_path.c_str(), so);

    const Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < session_.GetInputCount(); ++i) {
      const auto name = session_.GetInputNameAllocated(i, alloc);
      input_descs_.push_back(describe(name.get(), session_.GetInputTypeInfo(i)));
      input_names_.emplace_back(input_descs_.back().name);
    }
    for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
      const auto name = session_.GetOutputNameAllocated(i, alloc);
      output_descs_.push_back(describe(name.get(), session_.GetOutputTypeInfo(i)));
      output_names_.emplace_back(output_descs_.back().name);
    }
  }

  const std::vector<TensorDesc>& input_descs() const override { return input_descs_; }
  const std::vector<TensorDesc>& output_descs() const override { return output_descs_; }

  std::vector<TensorView> run(const std::vector<TensorView>& inputs) override {
    if (inputs.size() != input_descs_.size())
      throw std::runtime_error("engine expects " + std::to_string(input_descs_.size()) +
                               " inputs, got " + std::to_string(inputs.size()));

    const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> in_names;
    for (const auto& desc : input_descs_) {
      const TensorView* match = nullptr;
      for (const auto& in : inputs)
        if (in.name == desc.name) match = &in;
      if (!match) throw std::runtime_error("missing input tensor: " + desc.name);
      if (match->shape != desc.shape)
        throw std::runtime_error("shape mismatch for input: " + desc.name);
      ort_inputs.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(match->data),
                                                           static_cast<size_t>(desc.elements()),
                                                           desc.shape.data(), desc.shape.size()));
      in_names.push_back(desc.name.c_str());
    }

    std::vector<const char*> out_names;
    for (const auto& n : output_names_) out_names.push_back(n.c_str());

    last_outputs_ = session_.Run(Ort::RunOptions{nullptr}, in_names.data(), ort_inputs.data(),
                                 ort_inputs.size(), out_names.data(), out_names.size());

    std::vector<TensorView> views;
    for (size_t i = 0; i < last_outputs_.size(); ++i) {
      const auto info = last_outputs_[i].GetTensorTypeAndShapeInfo();
      views.push_back(
          TensorView{output_names_[i], info.GetShape(), last_outputs_[i].GetTensorData<float>()});
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

  Ort::Env env_;
  Ort::Session session_{nullptr};
  std::vector<TensorDesc> input_descs_, output_descs_;
  std::vector<std::string> input_names_, output_names_;
  std::vector<Ort::Value> last_outputs_;  // keeps output buffers alive until next run
};

}  // namespace

std::unique_ptr<IEngine> make_ort_engine(const std::string& model_path,
                                         const EngineOptions& options) {
  return std::make_unique<OrtEngine>(model_path, options);
}

}  // namespace ctrk
