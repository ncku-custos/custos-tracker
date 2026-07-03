#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ctrk {

// Static tensor description. Shapes are fixed at model load — dynamic shapes
// are deliberately unsupported (NPU portability contract, docs/PLAN.md).
// quant_scale/quant_zero are carried for future INT8 NPU backends; float
// backends leave them at identity.
struct TensorDesc {
  std::string name;
  std::vector<int64_t> shape;
  float quant_scale = 1.f;
  int32_t quant_zero = 0;

  int64_t elements() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

// Non-owning float tensor. For inputs the caller owns the buffer for the
// duration of run(); for outputs the engine owns it and it stays valid until
// the next run() on the same engine.
struct TensorView {
  std::string name;
  std::vector<int64_t> shape;
  const float* data = nullptr;
};

struct EngineOptions {
  int intra_op_threads = 4;
  // false trades a little tail latency for much lower idle CPU (the pool
  // stops busy-waiting between runs) — the SoC-friendly setting.
  bool allow_spinning = true;
  // One zero-input inference at load so the first real frame does not pay
  // one-time lazy initialization.
  bool warmup = true;
  // Experimental oneDNN execution provider (host benchmarking only).
  bool use_dnnl = false;
};

// One loaded model graph. Engines are single-threaded like all core objects;
// create one per thread. Different engines of one pipeline may use different
// backends (e.g. siamese backbones on the NPU, correlation head on CPU) —
// that per-engine freedom is a hard design requirement.
class IEngine {
 public:
  virtual ~IEngine() = default;
  virtual const std::vector<TensorDesc>& input_descs() const = 0;
  virtual const std::vector<TensorDesc>& output_descs() const = 0;
  // Inputs are matched to graph inputs by name. Returns one view per graph
  // output, in graph order.
  virtual std::vector<TensorView> run(const std::vector<TensorView>& inputs) = 0;
};

// ONNX Runtime (CPU) backend — the host reference implementation.
// Throws std::runtime_error on load failure.
std::unique_ptr<IEngine> make_ort_engine(const std::string& model_path,
                                         const EngineOptions& options = {});

}  // namespace ctrk
