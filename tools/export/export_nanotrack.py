#!/usr/bin/env python3
"""Re-emit NanoTrack v2 as three static single-shape graphs for ctrk.

The published backbone is one graph declared [1,3,255,255]; cv::TrackerNano
feeds it both 127 (template) and 255 (search) inputs relying on OpenCV-dnn
dynamic reshape. Static-shape NPUs cannot do that, so we emit:

  nanotrack_backbone_z.onnx  [1,3,127,127] -> [1,48,8,8]    (init, once)
  nanotrack_backbone_x.onnx  [1,3,255,255] -> [1,48,16,16]  (per frame)
  nanotrack_head.onnx        zf+xf -> cls [1,2,16,16], reg [1,4,16,16]

Upstream files are opset 14 (not the project's preferred 12); a
version_converter downgrade is attempted and verified numerically — the
outcome is printed and recorded in docs/DECISIONS.md.

M2 spike verdict (v3): nanotrackv3 head emits 15x15 score maps;
cv::TrackerNano hardcodes a 16x16 grid, so v3 cannot ride the oracle path.
v2 ships; v3 needs bespoke postproc and is deferred to step 3.

Usage: tools/.venv/bin/python tools/export/export_nanotrack.py
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import version_converter
from onnxsim import simplify

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models" / "cache"
BACKBONE = CACHE / "nanotrackv2_nanotrack_backbone_sim.onnx"
HEAD = CACHE / "nanotrackv2_nanotrack_head_sim.onnx"


def session(model: onnx.ModelProto) -> ort.InferenceSession:
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def random_feeds(sess: ort.InferenceSession) -> dict:
    rng = np.random.default_rng(0)
    return {i.name: rng.standard_normal(i.shape, dtype=np.float32) for i in sess.get_inputs()}


def reshape_static(model: onnx.ModelProto, hw: int) -> onnx.ModelProto:
    """Set the single input to [1,3,hw,hw] and drop all stale shape metadata
    so inference recomputes everything from scratch."""
    dims = model.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_value = hw
    dims[3].dim_value = hw
    del model.graph.value_info[:]
    for out in model.graph.output:
        for d in out.type.tensor_type.shape.dim:
            d.Clear()
    return onnx.shape_inference.infer_shapes(model)


def try_downgrade(model: onnx.ModelProto) -> onnx.ModelProto:
    ref = session(model)
    feeds = random_feeds(ref)
    expected = ref.run(None, feeds)
    try:
        down = version_converter.convert_version(model, 12)
        onnx.checker.check_model(down)
        got = session(down).run(None, feeds)
        if all(np.allclose(a, b, atol=1e-4) for a, b in zip(expected, got)):
            print("  opset 12 downgrade: OK (verified numerically)")
            return down
        print("  opset 12 downgrade: numeric mismatch — keeping opset 14")
    except Exception as e:  # noqa: BLE001 — any converter failure keeps 14
        print(f"  opset 12 downgrade failed ({type(e).__name__}) — keeping opset 14")
    return model


def emit(src: Path, dst: Path, hw: int | None, expect_outputs: dict) -> None:
    model = onnx.load(src)
    if hw is not None:
        model = reshape_static(model, hw)
    simplified, ok = simplify(model)
    assert ok, f"onnxsim failed on {src.name}"
    simplified = try_downgrade(simplified)
    onnx.checker.check_model(simplified)

    # Verify by RUNNING, not by declared metadata.
    sess = session(simplified)
    outs = sess.run(None, random_feeds(sess))
    got = {o.name: list(t.shape) for o, t in zip(sess.get_outputs(), outs)}
    assert got == expect_outputs, f"{dst.name}: outputs {got}, expected {expect_outputs}"
    onnx.save(simplified, dst)
    print(f"  {dst.name}: outputs {got}")


def main() -> None:
    assert BACKBONE.exists() and HEAD.exists(), "run models/get_models.sh first"
    print("backbone_z (template, 127):")
    emit(BACKBONE, CACHE / "nanotrack_backbone_z.onnx", 127, {"output": [1, 48, 8, 8]})
    print("backbone_x (search, 255):")
    emit(BACKBONE, CACHE / "nanotrack_backbone_x.onnx", 255, {"output": [1, 48, 16, 16]})
    print("head:")
    emit(HEAD, CACHE / "nanotrack_head.onnx", None,
         {"output1": [1, 2, 16, 16], "output2": [1, 4, 16, 16]})
    print("export: contract verified")


if __name__ == "__main__":
    main()
