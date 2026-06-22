"""Functional tests for ovos-ww-plugin-wakeforge.

A real trained wakeforge model is large and language/phrase specific, so these
tests synthesise a tiny two-file ONNX pipeline (featurizer + energy head) and
drive the plugin through the real onnxruntime sessions. This exercises the full
plumbing — PCM decode, feature cache, smoothing, trigger/reset — without torch
or a downloaded model.

The synthetic head fires on high-energy audio and stays quiet on silence, so the
detection assertions are deterministic.
"""
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from ovos_ww_plugin_wakeforge import WakeForgeHotwordPlugin


def _build_featurizer(path):
    """audio [1, T] float32 -> feats [1, T, 1] float32 (Unsqueeze on axis 2)."""
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")
    node = helper.make_node("Unsqueeze", ["audio", "axes"], ["feats"])
    graph = helper.make_graph(
        [node], "featurizer",
        inputs=[helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "T"])],
        outputs=[helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, "T", 1])],
        initializer=[axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)


def _build_head(path):
    """feats [1, T, 1] -> logit [1] = mean(|feats|) * 40 - 4 (energy detector)."""
    scale = numpy_helper.from_array(np.array(40.0, dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.array(4.0, dtype=np.float32), name="bias")
    nodes = [
        helper.make_node("Abs", ["feats"], ["a"]),
        helper.make_node("ReduceMean", ["a"], ["m"], axes=[1, 2], keepdims=0),
        helper.make_node("Mul", ["m", "scale"], ["s"]),
        helper.make_node("Sub", ["s", "bias"], ["logit"]),
    ]
    graph = helper.make_graph(
        nodes, "head",
        inputs=[helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, "T", 1])],
        outputs=[helper.make_tensor_value_info("logit", TensorProto.FLOAT, [1])],
        initializer=[scale, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("wakeforge_models")
    feat, head = str(d / "feat.onnx"), str(d / "head.onnx")
    _build_featurizer(feat)
    _build_head(head)
    return feat, head


def _engine(models, **cfg):
    feat, head = models
    config = {"featurizer": feat, "model": head}
    config.update(cfg)
    return WakeForgeHotwordPlugin(key_phrase="hey jarvis", config=config)


def _pcm(value, n=1600):
    """n int16 samples all set to `value`, as raw little-endian bytes."""
    return (np.full(n, value, dtype=np.int16)).tobytes()


def test_missing_config_raises(models):
    with pytest.raises(ValueError):
        WakeForgeHotwordPlugin(key_phrase="x", config={})


def test_loads(models):
    eng = _engine(models)
    assert eng.engine is not None
    assert not eng.found_wake_word()


def test_silence_does_not_trigger(models):
    eng = _engine(models)
    for _ in range(10):
        eng.update(_pcm(0))
        assert not eng.found_wake_word()


def test_loud_audio_triggers(models):
    eng = _engine(models, patience=3)
    detected = False
    for _ in range(8):
        eng.update(_pcm(16384))  # ~0.5 full-scale -> high energy
        if eng.found_wake_word():
            detected = True
            break
    assert detected, "high-energy audio should fire the synthetic detector"


def test_found_wake_word_resets(models):
    eng = _engine(models, patience=3)
    for _ in range(8):
        eng.update(_pcm(16384))
        if eng.found_wake_word():
            break
    # consumed once; immediately polling again must be False
    assert not eng.found_wake_word()


def test_opm_discovery():
    """Plugin is discoverable through the ovos-plugin-manager entry point."""
    from ovos_plugin_manager.wakewords import find_wake_word_plugins
    assert "ovos-ww-plugin-wakeforge" in find_wake_word_plugins()
