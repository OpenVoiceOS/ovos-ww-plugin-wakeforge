"""Streamed features (``streaming_features: true``): a stateful graph of the featurizer carries each layer's past
between blocks, and the head scores the last ``window`` streamed frames.

The stand-in featurizer and its streaming graph are built in conftest.py. The reference for every stream is the
whole stream, after ``window`` frames of silence, featurized offline in one call; a broken state carry disagrees
with it.
"""
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf

from test.conftest import K, make_featurizer
from ovos_ww_plugin_wakeforge import WakeForgeHotwordPlugin
from ovos_ww_plugin_wakeforge.inference import StreamingFeaturizer

WINDOW = 20


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _offline_streamed_scores(hub, head, pcm, block, window):
    """Score after each block: the head over the last ``window`` frames of the whole stream, featurized offline
    after ``window`` frames of silence."""
    feat = ort.InferenceSession(hub.featurizer, providers=["CPUExecutionProvider"])
    sess = ort.InferenceSession(head, providers=["CPUExecutionProvider"])
    lead = window * hub.hop
    wav = np.concatenate([np.zeros(lead, np.float32), pcm.astype(np.float32) / 32768.0])
    feats = feat.run(None, {"waveform": wav[np.newaxis, :]})[0]
    scores = []
    for end in range(block, len(pcm) + 1, block):
        last = (lead + end) // hub.hop
        scores.append(float(_sigmoid(sess.run(None, {"input_features": feats[:, last - window:last, :]})[0][0])))
    return scores


def _scores(eng, pcm, chunk):
    scores, push = [], type(eng.engine).push.__get__(eng.engine)

    def recording_push(block):
        scores.append(push(block))
        return scores[-1]

    eng.engine.push = recording_push
    for i in range(0, len(pcm), chunk):
        eng.update(pcm[i:i + chunk].tobytes())
        assert not eng.found_wake_word()
    return scores


def _engine(hub, **config):
    head = hub.batch_head(pretrained_featurizer="wakehubert", window_frames=str(WINDOW))
    return WakeForgeHotwordPlugin("hey computer", {"model": head, "threshold": 1.1,
                                                   "streaming_features": True, **config}), head


def _matches(hub, head, eng, pcm, got, atol=1e-5):
    return np.allclose(got, _offline_streamed_scores(hub, head, pcm, eng.block, WINDOW)[:len(got)], atol=atol)


@pytest.mark.parametrize("chunk", [480, 1000, 1280, 1337, 4096])
def test_streamed_scores_match_the_whole_stream_featurized_offline(standin_hub, pcm, chunk):
    eng, head = _engine(standin_hub)
    assert eng.engine.stream is not None and not eng.engine.isolated
    got = _scores(eng, pcm, chunk)
    assert len(got) > 40
    assert _matches(standin_hub, head, eng, pcm, got)


@pytest.mark.parametrize("block_ms", [70, 100])
def test_block_length_rounded_to_whole_hops(standin_hub, pcm, block_ms):
    eng, head = _engine(standin_hub, block_ms=block_ms)
    assert eng.block % standin_hub.hop == 0
    assert _matches(standin_hub, head, eng, pcm, _scores(eng, pcm, 1000))


def test_audio_not_in_whole_hops_refused(standin_hub):
    eng, _ = _engine(standin_hub)
    with pytest.raises(ValueError, match="whole number"):
        eng.engine.stream.push(np.zeros(standin_hub.hop + 1, np.float32))


def test_streamed_scores_differ_from_isolated_windows(standin_hub, pcm):
    eng, head = _engine(standin_hub)
    iso = WakeForgeHotwordPlugin("hey computer", {"model": head, "threshold": 1.1})
    assert iso.engine.stream is None and iso.engine.isolated
    assert not np.allclose(_scores(eng, pcm, 1280), _scores(iso, pcm, 1280), atol=1e-3)


def test_off_by_default_downloads_no_streaming_graph(standin_hub):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head})
    assert eng.engine.stream is None
    assert not any(f.endswith("_stream.onnx") for _, f, _ in standin_hub.calls)


def test_state_shapes_stay_fixed(standin_hub, pcm):
    eng, _ = _engine(standin_hub)
    stream = eng.engine.stream
    assert [s.shape for s in stream.state] == [(1, 2, K - 1)]
    _scores(eng, pcm, 1024)
    assert [s.shape for s in stream.state] == [(1, 2, K - 1)]


def test_listeners_share_the_session_and_keep_their_own_state(standin_hub, pcm):
    a, head = _engine(standin_hub)
    b, _ = _engine(standin_hub)
    assert a.engine.stream.session is b.engine.stream.session
    assert a.engine.stream is not b.engine.stream
    quiet = (pcm // 8).astype(np.int16)
    got_a, got_b = [], []
    for i in range(0, len(pcm), 1280):
        got_a += _scores(a, pcm[i:i + 1280], 1280)
        got_b += _scores(b, quiet[i:i + 1280], 1280)
    assert _matches(standin_hub, head, a, pcm, got_a)
    assert _matches(standin_hub, head, b, quiet, got_b)


def test_hotwords_on_one_stream_share_each_featurizer_run(standin_hub, pcm):
    a, head = _engine(standin_hub)
    b, _ = _engine(standin_hub)
    c, _ = _engine(standin_hub)
    session = a.engine.stream.session
    run, calls = session.run, []

    def counting_run(*args, **kwargs):
        calls.append(1)
        return run(*args, **kwargs)

    session.run = counting_run
    quiet = (pcm // 8).astype(np.int16)
    got = {k: [] for k in "abc"}
    blocks = 0
    for i in range(0, len(pcm) - 1279, 1280):
        got["a"] += _scores(a, pcm[i:i + 1280], 1280)
        got["b"] += _scores(b, pcm[i:i + 1280], 1280)
        got["c"] += _scores(c, quiet[i:i + 1280], 1280)
        blocks += 1
    assert len(calls) == 2 * blocks
    assert _matches(standin_hub, head, a, pcm, got["a"]) and _matches(standin_hub, head, b, pcm, got["b"])
    assert _matches(standin_hub, head, c, quiet, got["c"])


def test_a_hotword_reset_alone_leaves_the_shared_stream(standin_hub, pcm):
    a, head = _engine(standin_hub)
    b, _ = _engine(standin_hub)
    half = len(pcm) // 2 // 1280 * 1280
    _scores(a, pcm[:half], 1280)
    _scores(b, pcm[:half], 1280)
    a.reset()
    rest = pcm[half:].copy()
    got_a, got_b = [], []
    for i in range(0, len(rest) - 1279, 1280):
        got_a += _scores(a, rest[i:i + 1280], 1280)
        got_b += _scores(b, rest[i:i + 1280], 1280)
    assert _matches(standin_hub, head, a, rest, got_a)
    whole = _offline_streamed_scores(standin_hub, head, pcm, b.block, WINDOW)
    assert np.allclose(got_b, whole[half // 1280:half // 1280 + len(got_b)], atol=1e-5)
    assert not np.allclose(got_a, got_b, atol=1e-3)


def test_reset_restarts_the_stream(standin_hub, pcm):
    eng, head = _engine(standin_hub)
    _scores(eng, pcm[::-1].copy(), 1280)
    eng.reset()
    assert _matches(standin_hub, head, eng, pcm, _scores(eng, pcm, 1280))


def test_agc_refused_with_streamed_features(standin_hub):
    with pytest.raises(ValueError, match="agc"):
        _engine(standin_hub, agc=True)


def test_streaming_graph_from_another_featurizer_refused(standin_hub, tmp_path):
    other = tmp_path / "other_stream.onnx"
    make_featurizer(str(other), stateful="0" * 64)
    standin_hub.stream = str(other)
    with pytest.raises(ValueError, match="not made from the featurizer"):
        _engine(standin_hub)


def test_graph_without_state_refused(standin_hub):
    with pytest.raises(ValueError, match="not a stateful streaming featurizer"):
        StreamingFeaturizer(standin_hub.featurizer, standin_hub.hop)


def test_state_carried_one_frame_late_breaks_the_match(standin_hub, pcm, tmp_path):
    path = tmp_path / "late_stream.onnx"
    with open(standin_hub.featurizer, "rb") as f:
        make_featurizer(str(path), stateful=hashlib.sha256(f.read()).hexdigest(), state_offset=1)
    standin_hub.stream = str(path)
    eng, head = _engine(standin_hub)
    assert not _matches(standin_hub, head, eng, pcm, _scores(eng, pcm, 1280), atol=1e-3)


def test_state_not_carried_breaks_the_match(standin_hub, pcm, monkeypatch):
    def push(self, audio):
        feeds = {self._wav: audio.astype(np.float32)[np.newaxis, :]}
        feeds.update(zip(self._names, [np.zeros(s, np.float32) for s in self._shapes]))
        return self.session.run(self._outputs, feeds)[0]

    monkeypatch.setattr(StreamingFeaturizer, "push", push)
    eng, head = _engine(standin_hub)
    assert not _matches(standin_hub, head, eng, pcm, _scores(eng, pcm, 1280), atol=1e-3)


CLIPS = Path(__file__).parent / "clips"


def test_published_streaming_graph_follows_the_published_featurizer():
    """The published streaming file, at its pinned revision, against the published int8 featurizer run once over the
    same audio: equal but for int8 rounding flips."""
    from ovos_ww_plugin_wakeforge.pretrained import PretrainedOnnxFeaturizer

    feat = PretrainedOnnxFeaturizer.from_pretrained("wakehubert-int8")
    stream = StreamingFeaturizer(feat.streaming_path(), feat.hop_samples)
    audio, sr = sf.read(CLIPS / "alexa.wav", dtype="float32")
    wav = np.concatenate([audio, np.zeros(16000, np.float32)])
    wav = wav[:len(wav) // 1280 * 1280]
    got = np.concatenate([stream.push(wav[i:i + 1280]) for i in range(0, len(wav), 1280)], axis=1)[0]
    sess = ort.InferenceSession(feat.model_path, providers=["CPUExecutionProvider"])
    want = sess.run(None, {"waveform": wav[np.newaxis, :]})[0][0]
    close = np.abs(got - want).max(1) <= 1e-4
    cos = np.sum(got * want, 1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
    assert got.shape == want.shape
    assert close.mean() > 0.9 and cos.min() > 0.98


def test_ready_model_detects_its_word_on_streamed_features():
    audio, sr = sf.read(CLIPS / "alexa.wav", dtype="int16")
    pcm = np.concatenate([np.zeros(32000, np.int16), audio, np.zeros(32000, np.int16)])
    eng = WakeForgeHotwordPlugin("alexa", {"model": "wakehubert_alexa", "streaming_features": True})
    assert eng.engine.stream is not None
    fired = 0
    for i in range(0, len(pcm), 1024):
        eng.update(pcm[i:i + 1024].tobytes())
        fired += eng.found_wake_word()
    assert fired == 1
