"""WakePhoneHuBERT featurizer: a head that names it loads, featurizes a real clip and scores.

The featurizer is downloaded from the Hugging Face repository TigreGotico/wakephonehubert at the
pinned revision. The head is a random GRU classifier of the right input width, since no
wakephonehubert wake-word model is published.
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import soundfile as sf
from onnx import TensorProto, helper, numpy_helper

from ovos_ww_plugin_wakeforge import WakeForgeHotwordPlugin
from ovos_ww_plugin_wakeforge.pretrained import (
    PRETRAINED_FEATURIZERS,
    OnnxWindowedWakeWord,
    PretrainedOnnxFeaturizer,
)

NAME = "wakephonehubert-int8"
WIDTH = 521
HIDDEN = 16
CLIP = Path(__file__).parent / "clips" / "hey_jarvis.wav"


def make_gru_head(path, metadata):
    """input_features [1, T, 521] -> logits [1]: a random GRU, last hidden state projected."""
    rng = np.random.default_rng(3)
    init = [
        numpy_helper.from_array(rng.normal(0, 0.1, (1, 3 * HIDDEN, WIDTH)).astype(np.float32), "W"),
        numpy_helper.from_array(rng.normal(0, 0.1, (1, 3 * HIDDEN, HIDDEN)).astype(np.float32), "R"),
        numpy_helper.from_array(rng.normal(0, 1, (HIDDEN, 1)).astype(np.float32), "v"),
        numpy_helper.from_array(np.asarray([-1], np.int64), "flat"),
        numpy_helper.from_array(np.asarray([1, -1], np.int64), "row"),
    ]
    nodes = [
        helper.make_node("Transpose", ["input_features"], ["seq"], perm=[1, 0, 2]),
        helper.make_node("GRU", ["seq", "W", "R"], ["all", "last"], hidden_size=HIDDEN),
        helper.make_node("Reshape", ["last", "row"], ["h"]),
        helper.make_node("MatMul", ["h", "v"], ["proj"]),
        helper.make_node("Reshape", ["proj", "flat"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, "random_gru_head",
        inputs=[helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [1, "T", WIDTH])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1])],
        initializer=init,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.fixture
def head(tmp_path):
    path = str(tmp_path / "wakephonehubert_random.onnx")
    make_gru_head(path, {"pretrained_featurizer": NAME})
    return path


@pytest.fixture
def pcm16():
    audio, rate = sf.read(CLIP, dtype="int16")
    assert rate == 16000
    return audio if audio.ndim == 1 else audio[:, 0]


def test_registry_entry():
    entry = PRETRAINED_FEATURIZERS[NAME]
    assert entry.repo_id == "TigreGotico/wakephonehubert"
    assert entry.variant == "features_int8"
    assert len(entry.revision) == 40


def test_featurizer_facts_follow_the_model_config():
    feat = PretrainedOnnxFeaturizer.from_pretrained(NAME)
    assert feat.feature_dim == WIDTH
    assert feat.hop_samples == 320
    assert feat.frame_rate_hz == 50.0
    assert feat.context_samples == 41920
    assert feat.streaming
    assert feat.model_path.endswith("wakephonehubert_features_int8.onnx")


def test_features_of_a_clip_are_521_wide(pcm16):
    feat = PretrainedOnnxFeaturizer.from_pretrained(NAME)
    sess = ort.InferenceSession(feat.model_path, providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["features"]
    wav = (pcm16.astype(np.float32) / 32768.0)[np.newaxis, :]
    out = sess.run(None, {sess.get_inputs()[0].name: wav})[0]
    assert out.shape[0] == 1 and out.shape[2] == WIDTH
    assert abs(out.shape[1] - len(pcm16) // 320) <= 1
    assert np.isfinite(out).all()
    assert out.std() > 0


def test_head_naming_the_featurizer_loads_and_scores(head, pcm16):
    eng = WakeForgeHotwordPlugin("hey jarvis", {"model": head, "threshold": 1.1})
    assert isinstance(eng.engine, OnnxWindowedWakeWord)
    assert eng.featurizer.name == NAME
    assert eng.engine.window == 75
    scores, push = [], eng.engine.push

    def recording_push(block):
        scores.append(push(block))
        return scores[-1]

    eng.engine.push = recording_push
    for i in range(0, len(pcm16), 1280):
        eng.update(pcm16[i:i + 1280].tobytes())
        assert not eng.found_wake_word()
    assert len(scores) == len(pcm16) // eng.block
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert np.ptp(scores) > 0
    assert eng.engine._feats.shape[2] == WIDTH
