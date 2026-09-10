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
    model.ir_version = 10  # readable by every onnxruntime the plugin allows
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
    model.ir_version = 10  # readable by every onnxruntime the plugin allows
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


# ---------------------------------------------------------------------------
# Chunk-size independence: the listener's chunk size must not change decisions
# and must never reach the featurizer with fewer samples than its STFT needs.
# ---------------------------------------------------------------------------

def _build_reflect_featurizer(path, pad=200):
    """Like the real MFCC featurizer's STFT: reflect-pads `pad` samples on both
    ends, which onnxruntime refuses when the input is shorter than pad + 1."""
    pads = numpy_helper.from_array(np.array([0, 0, pad, 0, 0, pad], dtype=np.int64), name="pads")
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")
    nodes = [
        helper.make_node("Unsqueeze", ["audio", "axes"], ["a3"]),          # [1, T, 1]
        helper.make_node("Transpose", ["a3"], ["a3t"], perm=[0, 2, 1]),    # [1, 1, T]
        helper.make_node("Pad", ["a3t", "pads"], ["padded"], mode="reflect"),
        helper.make_node("Transpose", ["padded"], ["feats"], perm=[0, 2, 1]),  # [1, T+2pad, 1]
    ]
    graph = helper.make_graph(
        nodes, "featurizer",
        inputs=[helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "T"])],
        outputs=[helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, "T2", 1])],
        initializer=[pads, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, path)


@pytest.fixture(scope="module")
def reflect_models(tmp_path_factory):
    d = tmp_path_factory.mktemp("wakeforge_reflect")
    feat, head = str(d / "feat.onnx"), str(d / "head.onnx")
    _build_reflect_featurizer(feat)
    _build_head(head)
    return feat, head


def _clip(n=28 * 1280 + 192, sr=16000):
    """Loud burst in the middle of silence; 1280-sample chunks leave a 192-sample tail."""
    audio = np.zeros(n, dtype=np.int16)
    audio[sr // 2: sr // 2 + int(0.8 * sr)] = 16384
    return audio.tobytes()


def _first_detection(models, chunk_samples, **cfg):
    """Sample position of the chunk in which the plugin first fires, or None."""
    eng = _engine(models, patience=3, **cfg)
    pcm = _clip()
    for i in range(0, len(pcm), chunk_samples * 2):
        eng.update(pcm[i:i + chunk_samples * 2])
        if eng.found_wake_word():
            return i // 2 + chunk_samples
    return None


def test_short_trailing_chunk_does_not_crash_the_featurizer(reflect_models):
    # 1280-sample chunks leave a 192-sample tail, shorter than the featurizer's
    # 200-sample reflect pad; it must be buffered, not fed to the featurizer.
    eng = _engine(reflect_models, patience=3)
    pcm, fired = _clip(), 0
    for i in range(0, len(pcm), 1280 * 2):
        eng.update(pcm[i:i + 1280 * 2])
        fired += eng.found_wake_word()
    assert fired >= 1


def test_decisions_do_not_depend_on_chunk_size(reflect_models):
    # The burst starts at 0.5 s; the plugin must fire at the same point in the
    # audio whatever the chunk size, within one chunk of slack.
    first = {n: _first_detection(reflect_models, n) for n in (1280, 1600, 2048, 4096)}
    assert None not in first.values(), first
    assert max(first.values()) - min(first.values()) <= 4096, first
