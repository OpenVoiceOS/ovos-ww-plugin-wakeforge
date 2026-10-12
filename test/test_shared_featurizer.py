"""Hotwords on the same featurizer share one ONNX session and one featurizer run per window.

The featurizers and ready models are downloaded from the Hugging Face Hub; the audio is real clips from
``test/clips`` (synthetic speech, edge-tts voices) run through the engines as a listener feeds them.
"""
import gc
import weakref
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf

from ovos_ww_plugin_wakeforge import WakeForgeHotwordPlugin
from ovos_ww_plugin_wakeforge.inference import featurize
from ovos_ww_plugin_wakeforge.pretrained import resolve_pretrained
from ovos_ww_plugin_wakeforge.zeroshot import FEATURIZER, WakePhoneHuBERTZeroShotPlugin

CLIPS = Path(__file__).parent / "clips"
CHUNK = 1024
ZERO_SHOT = {
    "hey jarvis": "h eɪ dʒ ɑːɹ v ɪ s",
    "hey chatterbox": "h eɪ tʃ æ ɾ ɚ b ɑː k s",
    "computer": "k ə m p j uː ɾ ɚ",
}
DEFAULTS = {"hey mycroft": "wakehubert_hey_mycroft", "wake up": "wakehubert_wake_up"}


def _zero_shot(word):
    return WakePhoneHuBERTZeroShotPlugin(word, {"ipa": ZERO_SHOT[word]})


def _default(word):
    return WakeForgeHotwordPlugin(word, {"model": DEFAULTS[word]})


@pytest.fixture(scope="module")
def pcm():
    silence = np.zeros(8000, np.int16)
    parts = [silence]
    for name in ("hey_jarvis", "hey_mycroft", "wake_up", "computer", "alexa"):
        audio, sr = sf.read(CLIPS / f"{name}.wav", dtype="int16")
        assert sr == 16000
        parts += [audio, silence]
    return np.concatenate(parts)


def _feed(engines, pcm):
    """Feed every engine each chunk in turn, as the listener does; per engine, its score after every chunk."""
    scores = [[] for _ in engines]
    for i in range(0, len(pcm), CHUNK):
        chunk = pcm[i:i + CHUNK].tobytes()
        for eng, out in zip(engines, scores):
            eng.update(chunk)
            out.append(eng.last_score)
    return scores


@pytest.fixture
def featurizer_runs(monkeypatch):
    """Session runs per model file, counted at the session's run method."""
    runs = {}
    run = ort.InferenceSession.run

    def spy(self, output_names, input_feed, *args, **kwargs):
        name = Path(self._model_path).name
        runs[name] = runs.get(name, 0) + 1
        return run(self, output_names, input_feed, *args, **kwargs)

    monkeypatch.setattr(ort.InferenceSession, "run", spy)
    return runs


def test_instances_on_one_featurizer_share_its_session():
    a, b = _zero_shot("hey jarvis"), _zero_shot("computer")
    assert a.session is b.session
    m, w = _default("hey mycroft"), _default("wake up")
    assert m.engine.ext is w.engine.ext
    assert m.engine.head is not w.engine.head


def test_featurizer_runs_once_per_window_for_all_instances(pcm, featurizer_runs):
    zero_shot_file = Path(resolve_pretrained(FEATURIZER)[0]).name
    _feed([_zero_shot("hey jarvis"), _default("hey mycroft")], pcm)
    alone = dict(featurizer_runs)
    assert 0 < alone[zero_shot_file] <= len(pcm) // 1280
    assert 0 < alone["wakehubert_int8.onnx"] <= len(pcm) // 1280
    featurizer_runs.clear()
    _feed([_zero_shot(w) for w in ZERO_SHOT] + [_default(w) for w in DEFAULTS], pcm)
    assert featurizer_runs[zero_shot_file] == alone[zero_shot_file]
    assert featurizer_runs["wakehubert_int8.onnx"] == alone["wakehubert_int8.onnx"]


def test_scores_with_sharing_equal_each_instance_alone(pcm):
    words = list(ZERO_SHOT) + list(DEFAULTS)
    make = {**{w: _zero_shot for w in ZERO_SHOT}, **{w: _default for w in DEFAULTS}}
    alone = {}
    for word in words:
        alone[word] = _feed([make[word](word)], pcm)[0]
        gc.collect()
    together = _feed([make[w](w) for w in words], pcm)
    for word, scores in zip(words, together):
        assert scores == alone[word], word
    assert max(s for s in alone["hey jarvis"] if s is not None) > -0.3
    assert max(s for s in alone["hey mycroft"] if s is not None) > 0.5


def test_different_optimisation_levels_do_not_share():
    from ovos_ww_plugin_wakeforge.inference import shared_session
    eng = _zero_shot("hey jarvis")
    path = resolve_pretrained(FEATURIZER)[0]
    fused = shared_session(path, 1)
    assert fused is not eng.session
    assert eng.session is shared_session(path, 1, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    assert eng.session.get_session_options().graph_optimization_level == \
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    assert fused.get_session_options().graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert shared_session(path, 2, ort.GraphOptimizationLevel.ORT_DISABLE_ALL) is not eng.session


def test_releasing_every_instance_releases_the_session_and_a_new_one_works(pcm):
    from ovos_ww_plugin_wakeforge.inference import shared_session
    engines = [_zero_shot("hey jarvis"), _zero_shot("computer"), _default("hey mycroft"), _default("wake up")]
    before = _feed(engines, pcm)
    sessions = [weakref.ref(engines[0].session), weakref.ref(engines[2].engine.ext)]
    del engines
    gc.collect()
    assert [s() for s in sessions] == [None, None]
    again = [_zero_shot("hey jarvis"), _default("wake up")]
    assert _feed(again, pcm) == [before[0], before[3]]
    assert again[0].session is shared_session(resolve_pretrained(FEATURIZER)[0], 1,
                                              ort.GraphOptimizationLevel.ORT_DISABLE_ALL)


class _Echo:
    """A featurizer stand-in that returns its input and counts its runs."""

    def __init__(self):
        self.runs = 0

    def get_inputs(self):
        return [type("Input", (), {"name": "audio"})()]

    def get_outputs(self):
        return [type("Output", (), {"name": "features"})()]

    def run(self, names, feed):
        self.runs += 1
        return [feed["audio"].copy()]


def test_same_bytes_in_another_shape_are_not_reused():
    session, audio = _Echo(), np.arange(8, dtype=np.float32)
    assert featurize(session, audio.reshape(1, 8)).shape == (1, 8)
    assert featurize(session, audio.reshape(2, 4)).shape == (2, 4)
    assert session.runs == 2
