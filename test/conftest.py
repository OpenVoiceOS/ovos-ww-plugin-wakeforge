"""Stand-in pretrained featurizer and heads for the pretrained-featurizer tests.

The stand-in featurizer has the WakeHuBERT I/O contract, ``[B, N] float32 ->
[B, N // 320, D]``, and a bounded causal receptive field of ``K`` frames, so a
stream reproduces offline frames only when enough past audio is carried with
each chunk. The Hub download is replaced by a function serving these files, so
no test touches the network.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

HOP = 320
FEAT_DIM = 16
K = 8                      # receptive field in frames
CONTEXT = K * HOP          # samples of past audio one frame depends on
HIDDEN = 8
WINDOW = 10


def _save(graph, path, metadata=None):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for key, value in (metadata or {}).items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _const(name, value, dtype=np.int64):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def make_featurizer(path, seed=0):
    """waveform [B, N] -> features [B, N // HOP, FEAT_DIM], causal over K frames."""
    rng = np.random.default_rng(seed)
    inits = [
        _const("zero", 0), _const("one", 1), _const("hop", HOP), _const("hop1", [HOP]),
        _const("ax0", [0]), _const("ax1", [1]),
        _const("pads", [0, 0, K - 1, 0, 0, 0]),
        numpy_helper.from_array(rng.normal(0, 2.0, (FEAT_DIM, 2, K)).astype(np.float32), "W"),
        numpy_helper.from_array(rng.normal(0, 0.1, FEAT_DIM).astype(np.float32), "bias"),
    ]
    nodes = [
        helper.make_node("Shape", ["waveform"], ["shape"]),
        helper.make_node("Gather", ["shape", "zero"], ["b"]),
        helper.make_node("Gather", ["shape", "one"], ["n_samples"]),
        helper.make_node("Div", ["n_samples", "hop"], ["n"]),
        helper.make_node("Mul", ["n", "hop"], ["n_used"]),
        helper.make_node("Unsqueeze", ["n_used", "ax0"], ["end"]),
        helper.make_node("Slice", ["waveform", "ax0", "end", "ax1"], ["x"]),
        helper.make_node("Unsqueeze", ["b", "ax0"], ["b1"]),
        helper.make_node("Unsqueeze", ["n", "ax0"], ["n1"]),
        helper.make_node("Concat", ["b1", "n1", "hop1"], ["frames_shape"], axis=0),
        helper.make_node("Reshape", ["x", "frames_shape"], ["frames"]),       # [B, n, HOP]
        helper.make_node("Mul", ["frames", "frames"], ["sq"]),
        helper.make_node("ReduceMean", ["sq"], ["energy"], axes=[2], keepdims=1),
        helper.make_node("ReduceMean", ["frames"], ["dc"], axes=[2], keepdims=1),
        helper.make_node("Concat", ["energy", "dc"], ["stats"], axis=2),     # [B, n, 2]
        helper.make_node("Transpose", ["stats"], ["stats_t"], perm=[0, 2, 1]),
        helper.make_node("Pad", ["stats_t", "pads"], ["padded"]),            # causal left pad
        helper.make_node("Conv", ["padded", "W", "bias"], ["conv"]),         # [B, D, n]
        helper.make_node("Transpose", ["conv"], ["conv_t"], perm=[0, 2, 1]),
        helper.make_node("Tanh", ["conv_t"], ["features"]),
    ]
    graph = helper.make_graph(
        nodes, "standin_wakehubert",
        inputs=[helper.make_tensor_value_info("waveform", TensorProto.FLOAT, ["batch", "samples"])],
        outputs=[helper.make_tensor_value_info("features", TensorProto.FLOAT,
                                               ["batch", "frames", FEAT_DIM])],
        initializer=inits,
    )
    _save(graph, path)


def make_batch_head(path, metadata=None, seed=1):
    """input_features [1, T, D] -> logits [1] = mean_t(features @ w) * 4 - 0.5."""
    rng = np.random.default_rng(seed)
    inits = [
        numpy_helper.from_array(rng.normal(0, 1, (FEAT_DIM, 1)).astype(np.float32), "w"),
        _const("scale", 4.0, np.float32), _const("offset", 0.5, np.float32),
    ]
    nodes = [
        helper.make_node("MatMul", ["input_features", "w"], ["proj"]),
        helper.make_node("ReduceMean", ["proj"], ["pooled"], axes=[1, 2], keepdims=0),
        helper.make_node("Mul", ["pooled", "scale"], ["scaled"]),
        helper.make_node("Sub", ["scaled", "offset"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, "standin_batch_head",
        inputs=[helper.make_tensor_value_info("input_features", TensorProto.FLOAT,
                                              [1, "T", FEAT_DIM])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1])],
        initializer=inits,
    )
    _save(graph, path, metadata)


def make_streaming_head(path, metadata=None, seed=2):
    """wakeforge streaming-head I/O:
    (feat_frame, h_in, out_window) -> (logit, h_out, out_window_next)."""
    rng = np.random.default_rng(seed)
    inits = [
        numpy_helper.from_array(rng.normal(0, 0.5, (FEAT_DIM, HIDDEN)).astype(np.float32), "A"),
        numpy_helper.from_array(rng.normal(0, 0.5, (HIDDEN, HIDDEN)).astype(np.float32), "U"),
        numpy_helper.from_array(rng.normal(0, 1, (HIDDEN, 1)).astype(np.float32), "v"),
        _const("s1", [1]), _const("e1", [WINDOW + 1]), _const("ax1", [1]),
        _const("scale", 1.0, np.float32),
    ]
    nodes = [
        helper.make_node("MatMul", ["feat_frame", "A"], ["xa"]),
        helper.make_node("MatMul", ["h_in", "U"], ["hu"]),
        helper.make_node("Add", ["xa", "hu"], ["pre"]),
        helper.make_node("Tanh", ["pre"], ["h_out"]),
        helper.make_node("Concat", ["out_window", "h_out"], ["cat"], axis=1),
        helper.make_node("Slice", ["cat", "s1", "e1", "ax1"], ["out_window_next"]),
        helper.make_node("MatMul", ["out_window_next", "v"], ["proj"]),
        helper.make_node("ReduceMean", ["proj"], ["pooled"], axes=[1, 2], keepdims=0),
        helper.make_node("Mul", ["pooled", "scale"], ["logit"]),
    ]
    graph = helper.make_graph(
        nodes, "standin_streaming_head",
        inputs=[
            helper.make_tensor_value_info("feat_frame", TensorProto.FLOAT, [1, 1, FEAT_DIM]),
            helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [1, 1, HIDDEN]),
            helper.make_tensor_value_info("out_window", TensorProto.FLOAT, [1, WINDOW, HIDDEN]),
        ],
        outputs=[
            helper.make_tensor_value_info("logit", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [1, 1, HIDDEN]),
            helper.make_tensor_value_info("out_window_next", TensorProto.FLOAT,
                                          [1, WINDOW, HIDDEN]),
        ],
        initializer=inits,
    )
    _save(graph, path, metadata)


@dataclass
class StandinHub:
    """Serves the stand-in featurizer for every pretrained name; records requests."""
    featurizer: str
    config: dict
    tmp: object
    calls: list = field(default_factory=list)
    hop: int = HOP
    context: int = CONTEXT
    window: int = WINDOW
    hidden: int = HIDDEN

    def batch_head(self, name="head.onnx", **metadata):
        path = str(self.tmp / name)
        make_batch_head(path, metadata)
        return path

    def streaming_head(self, name="head_streaming.onnx", **metadata):
        path = str(self.tmp / name)
        make_streaming_head(path, metadata)
        return path

    def download(self, repo_id, filename, revision=None, **kwargs):
        self.calls.append((repo_id, filename, revision))
        if filename == "config.json":
            path = self.tmp / f"config-{len(self.calls)}.json"
            path.write_text(json.dumps(self.config))
            return str(path)
        return self.featurizer


@pytest.fixture
def standin_hub(tmp_path, monkeypatch):
    """Replace the plugin's Hub download with the stand-in featurizer."""
    from ovos_ww_plugin_wakeforge import pretrained

    feat = tmp_path / "wakehubert.onnx"
    make_featurizer(str(feat))
    config = {
        "feature_dim": FEAT_DIM,
        "output": {"hop_samples": HOP, "frame_rate_hz": 16000 / HOP},
        "files": {"float32": "wakehubert.onnx", "int8": "wakehubert_int8.onnx"},
        "streaming": True,
        "receptive_field_samples": CONTEXT,
        "license": "apache-2.0",
    }
    hub = StandinHub(str(feat), config, tmp_path)
    monkeypatch.setattr(pretrained, "hf_hub_download", hub.download)
    monkeypatch.setattr(pretrained, "BUNDLED_REVISIONS", {})
    return hub


@pytest.fixture
def offline(monkeypatch):
    """Fail any Hub download or socket connection, as on a device with no network."""
    import socket

    from ovos_ww_plugin_wakeforge import pretrained

    def refuse(*args, **kwargs):
        raise OSError("network access in an offline test")

    monkeypatch.setattr(pretrained, "hf_hub_download", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def pcm():
    """4 s of int16 PCM: quiet noise, a loud burst, quiet noise."""
    return speech_like()


def speech_like(seconds=4.0, seed=3):
    """int16 PCM: quiet noise, a loud burst, quiet noise — deterministic."""
    rng = np.random.default_rng(seed)
    n = int(16000 * seconds)
    audio = rng.normal(0, 300, n)
    start, stop = int(0.35 * n), int(0.65 * n)
    audio[start:stop] += rng.normal(0, 9000, stop - start) + 3000
    return np.clip(audio, -32768, 32767).astype(np.int16)
