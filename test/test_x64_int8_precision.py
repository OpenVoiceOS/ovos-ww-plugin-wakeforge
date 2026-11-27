"""int8 featurizers score the same on every x86 CPU.

A QuantizeLinear/DequantizeLinear model (as ``wakehubert-int8``) is fused into ``QLinearConv`` from
``ORT_ENABLE_EXTENDED`` on. On x86 CPUs without AVX-512 VNNI, onnxruntime's default u8s8 kernel sums pairs of
products in 16 bits and saturates; ``session_options()`` asks for the exact path (``session.x64quantprecision``).

The model here makes every pair sum 2 * 255 * 127, past int16, so a saturating kernel returns about half the true
value. The reference is the same graph at ``ORT_ENABLE_BASIC``, run unfused in float, which is the exact integer
result on every CPU. On VNNI and ARM CPUs both paths are exact and the test passes either way; on AVX2 it fails
without the option.
"""
import platform

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from ovos_ww_plugin_wakeforge.inference import session_options

C_IN, C_OUT, T = 16, 4, 8


def _qdq_conv(path):
    x_scale, w_scale, y_scale = 1.0 / 255, 1.0 / 127, 0.1
    inits = [
        numpy_helper.from_array(np.array(x_scale, np.float32), "x_scale"),
        numpy_helper.from_array(np.array(0, np.uint8), "x_zp"),
        numpy_helper.from_array(np.full((C_OUT, C_IN, 1), 127, np.int8), "w_q"),
        numpy_helper.from_array(np.array(w_scale, np.float32), "w_scale"),
        numpy_helper.from_array(np.array(0, np.int8), "w_zp"),
        numpy_helper.from_array(np.array(y_scale, np.float32), "y_scale"),
        numpy_helper.from_array(np.array(0, np.uint8), "y_zp"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zp"], ["x_q"]),
        helper.make_node("DequantizeLinear", ["x_q", "x_scale", "x_zp"], ["x_dq"]),
        helper.make_node("DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"]),
        helper.make_node("Conv", ["x_dq", "w_dq"], ["y"], kernel_shape=[1]),
        helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zp"], ["y_q"]),
        helper.make_node("DequantizeLinear", ["y_q", "y_scale", "y_zp"], ["out"]),
    ]
    graph = helper.make_graph(nodes, "qdq_conv", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C_IN, T])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, C_OUT, T])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _run(path, level, options):
    options.graph_optimization_level = level
    sess = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return sess.run(None, {"x": np.ones((1, C_IN, T), np.float32)})[0]


def test_session_options_ask_for_exact_int8():
    assert session_options().get_session_config_entry("session.x64quantprecision") == "1"


def test_fused_int8_conv_is_exact(tmp_path):
    path = tmp_path / "qdq_conv.onnx"
    _qdq_conv(path)
    exact = _run(path, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, ort.SessionOptions())
    assert np.allclose(exact, C_IN), exact            # 16 channels of 1.0 * 1.0
    for level in (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
        fused = _run(path, level, session_options())
        assert np.array_equal(fused, exact), (
            f"{platform.machine()} {platform.processor()}: fused int8 conv gave {fused.ravel()[:4]} instead of "
            f"{exact.ravel()[:4]} at {level}")
