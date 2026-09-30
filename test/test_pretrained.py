"""Models trained on a pretrained featurizer (``ww_trainer-train --tier wakehubert``).

The featurizer is a stand-in with the WakeHuBERT I/O contract and a bounded
causal receptive field (see conftest.py); the Hub download is monkeypatched.
The reference for every stream is the same audio featurized offline in one
call, so a scorer that drops the past audio a frame depends on disagrees.
"""
import numpy as np
import onnxruntime as ort
import pytest

from ovos_ww_plugin_wakeforge import WakeForgeHotwordPlugin
from ovos_ww_plugin_wakeforge.pretrained import (
    PRETRAINED_FEATURIZERS,
    OnnxWindowedWakeWord,
)
from ovos_ww_plugin_wakeforge.inference import OnnxStreamingWakeWord


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _offline_features(hub, pcm):
    sess = ort.InferenceSession(hub.featurizer, providers=["CPUExecutionProvider"])
    wav = (pcm.astype(np.float32) / 32768.0)[np.newaxis, :]
    return sess.run(None, {sess.get_inputs()[0].name: wav})[0]


def _offline_batch_scores(hub, head, pcm, block, window):
    """Score after each block: the batch head over the last `window` offline frames."""
    feats = _offline_features(hub, pcm)
    sess = ort.InferenceSession(head, providers=["CPUExecutionProvider"])
    scores = []
    for end in range(block, len(pcm) + 1, block):
        frames = feats[:, max(0, end // hub.hop - window):end // hub.hop, :]
        scores.append(float(_sigmoid(sess.run(None, {"input_features": frames})[0][0])))
    return scores


def _offline_streaming_scores(hub, head, pcm, block):
    """Score after each block: the streaming head stepped over offline frames."""
    feats = _offline_features(hub, pcm)
    sess = ort.InferenceSession(head, providers=["CPUExecutionProvider"])
    h = np.zeros((1, 1, hub.hidden), np.float32)
    ow = np.zeros((1, hub.window, hub.hidden), np.float32)
    per_frame = []
    for t in range(feats.shape[1]):
        logit, h, ow = sess.run(None, {"feat_frame": feats[:, t:t + 1, :],
                                       "h_in": h, "out_window": ow})
        per_frame.append(float(_sigmoid(logit[0])))
    return [per_frame[end // hub.hop - 1] for end in range(block, len(pcm) + 1, block)]


def _stream_scores(eng, pcm, chunk):
    """Feed int16 PCM through update() in `chunk`-sample pieces; record each block's score."""
    scores, push = [], eng.engine.push

    def recording_push(block):
        scores.append(push(block))
        return scores[-1]

    eng.engine.push = recording_push
    for i in range(0, len(pcm), chunk):
        eng.update(pcm[i:i + chunk].tobytes())
        assert not eng.found_wake_word()
    return scores


def test_featurizer_named_by_head_metadata(standin_hub, pcm):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head, "threshold": 1.1})
    assert isinstance(eng.engine, OnnxWindowedWakeWord)
    assert eng.featurizer.name == "wakehubert"
    assert standin_hub.calls[0] == ("TigreGotico/wakehubert-tiny", "config.json", None)
    assert standin_hub.calls[1][1] == "wakehubert.onnx"
    assert eng.engine.context == standin_hub.context


def test_featurizer_named_by_config(standin_hub):
    head = standin_hub.batch_head()
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head, "featurizer": "wakehubert-int8"})
    assert eng.featurizer.name == "wakehubert-int8"
    assert standin_hub.calls[1] == ("TigreGotico/wakehubert-tiny", "wakehubert_int8.onnx", None)


def test_head_metadata_wins_over_config(standin_hub):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert-int8")
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head, "featurizer": "wakehubert"})
    assert eng.featurizer.name == "wakehubert-int8"


def test_revision_pinned_by_config_then_metadata(standin_hub):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert", featurizer_revision="meta-rev")
    WakeForgeHotwordPlugin("x", {"model": head})
    assert {c[2] for c in standin_hub.calls} == {"meta-rev"}
    standin_hub.calls.clear()
    WakeForgeHotwordPlugin("x", {"model": head, "featurizer_revision": "cfg-rev"})
    assert {c[2] for c in standin_hub.calls} == {"cfg-rev"}


@pytest.mark.parametrize("chunk", [1000, 1280, 2048, 4096])
def test_batch_head_stream_with_context_matches_offline(standin_hub, pcm, chunk):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head, "threshold": 1.1,
                                                  "isolated_windows": False})
    assert not eng.engine.isolated
    window = eng.engine.window
    assert window == 75   # 1.5 s at the featurizer's 50 frames per second
    streamed = _stream_scores(eng, pcm, chunk)
    offline = _offline_batch_scores(standin_hub, head, pcm, eng.block, window)
    assert len(streamed) == len(pcm) // eng.block
    assert np.allclose(streamed, offline[:len(streamed)], atol=1e-5)
    assert np.ptp(offline) > 0.2   # the scores move, so agreement is not trivial


@pytest.mark.parametrize("chunk", [1000, 1280, 4096])
def test_streaming_head_stream_matches_offline(standin_hub, pcm, chunk):
    head = standin_hub.streaming_head(pretrained_featurizer="wakehubert")
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head, "threshold": 1.1})
    assert isinstance(eng.engine, OnnxStreamingWakeWord)
    assert (eng.engine.window, eng.engine.hidden_dim) == (standin_hub.window, standin_hub.hidden)
    assert eng.engine.context == standin_hub.context
    streamed = _stream_scores(eng, pcm, chunk)
    offline = _offline_streaming_scores(standin_hub, head, pcm, eng.block)
    assert np.allclose(streamed, offline[:len(streamed)], atol=1e-5)
    assert np.ptp(offline) > 0.2


def test_detection_fires_where_offline_crosses_threshold(standin_hub, pcm):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    probe = WakeForgeHotwordPlugin("x", {"model": head, "isolated_windows": False})
    offline = _offline_batch_scores(standin_hub, head, pcm, probe.block, probe.engine.window)
    threshold = (min(offline) + max(offline)) / 2
    expected = next(i for i, s in enumerate(offline) if s >= threshold)

    eng = WakeForgeHotwordPlugin("x", {"model": head, "threshold": threshold, "isolated_windows": False,
                                       "smoothing": "max", "window_size": 1, "patience": 1})
    fired_at = None
    for i in range(0, len(pcm), 1000):
        eng.update(pcm[i:i + 1000].tobytes())
        blocks_now = (i + 1000) // eng.block
        if eng.found_wake_word():
            fired_at = blocks_now - 1
            break
    assert fired_at == expected


def test_calibrated_head_by_path_uses_ready_model_defaults(standin_hub):
    # A head loaded by path whose own ONNX metadata carries default_threshold
    # (a Platt-calibrated head, as ww_trainer-train writes) is scored like the
    # bundled models, not with the EMA/patience-3 defaults meant for an
    # uncalibrated head.
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert", default_threshold="0.75")
    eng = WakeForgeHotwordPlugin("x", {"model": head})
    assert eng.threshold == 0.75
    assert (eng.smoother.method, eng.smoother.window_size, eng.smoother.patience) == ("max", 1, 1)
    assert eng.debounce_sec == 2.0


def test_calibrated_head_threshold_and_smoothing_still_overridable(standin_hub):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert", default_threshold="0.75")
    eng = WakeForgeHotwordPlugin("x", {"model": head, "threshold": 0.3, "smoothing": "ema",
                                       "patience": 3, "window_size": 5, "debounce_sec": 1.0})
    assert eng.threshold == 0.3
    assert (eng.smoother.method, eng.smoother.window_size, eng.smoother.patience) == ("ema", 5, 3)
    assert eng.debounce_sec == 1.0


def test_uncalibrated_head_by_path_keeps_ema_defaults(standin_hub):
    # No default_threshold metadata: the head is not known to be calibrated,
    # so the plugin keeps the EMA/patience-3 defaults meant for a fresh
    # wakeforge export.
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    eng = WakeForgeHotwordPlugin("x", {"model": head})
    assert eng.threshold == 0.5
    assert (eng.smoother.method, eng.smoother.window_size, eng.smoother.patience) == ("ema", 5, 3)
    assert eng.debounce_sec == 1.0


def test_block_rounded_to_featurizer_hop(standin_hub):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    eng = WakeForgeHotwordPlugin("x", {"model": head, "block_ms": 90})
    assert eng.block == 1600   # 1440 samples rounded up to whole 320-sample frames


def test_non_streamable_featurizer_refused_before_download(standin_hub):
    head = standin_hub.batch_head()
    assert PRETRAINED_FEATURIZERS["wakehubert-mel-bigru"].context_samples is None
    with pytest.raises(ValueError, match="not streamable"):
        WakeForgeHotwordPlugin("x", {"model": head, "featurizer": "wakehubert-mel-bigru"})
    assert standin_hub.calls == []


def test_featurizer_published_as_non_streaming_refused(standin_hub):
    standin_hub.config["streaming"] = False
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    with pytest.raises(ValueError, match="not streamable"):
        WakeForgeHotwordPlugin("x", {"model": head})


def test_vad_refused_with_pretrained_featurizer(standin_hub):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    with pytest.raises(ValueError, match="vad"):
        WakeForgeHotwordPlugin("x", {"model": head, "vad": head})


def test_unknown_metadata_featurizer_needs_config(standin_hub):
    # wakeforge writes the extractor class name for featurizers it exports itself.
    head = standin_hub.batch_head(featurizer="MfccExtractor")
    with pytest.raises(ValueError, match="featurizer"):
        WakeForgeHotwordPlugin("x", {"model": head})
    assert standin_hub.calls == []


def _offline_isolated_scores(hub, head, pcm, block, window):
    """Score after each block: the last `window` frames of audio featurized on their own."""
    feat = ort.InferenceSession(hub.featurizer, providers=["CPUExecutionProvider"])
    sess = ort.InferenceSession(head, providers=["CPUExecutionProvider"])
    x, n = pcm.astype(np.float32) / 32768.0, window * hub.hop
    scores = []
    for end in range(block, len(pcm) + 1, block):
        audio = np.zeros(n, np.float32)
        seg = x[max(0, end - n):end]
        audio[n - len(seg):] = seg
        frames = feat.run(None, {"waveform": audio[np.newaxis, :]})[0]
        scores.append(float(_sigmoid(sess.run(None, {"input_features": frames})[0][0])))
    return scores


@pytest.mark.parametrize("chunk", [1000, 1280, 4096])
def test_batch_head_isolated_windows_match_offline_windows(standin_hub, pcm, chunk):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert", window_frames="20")
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head, "threshold": 1.1})
    assert eng.engine.isolated and eng.engine.window == 20
    streamed = _stream_scores(eng, pcm, chunk)
    isolated = _offline_isolated_scores(standin_hub, head, pcm, eng.block, 20)
    assert np.allclose(streamed, isolated[:len(streamed)], atol=1e-5)
    # a 20-frame window is shorter than the featurizer's context, so the two modes differ
    context = _offline_batch_scores(standin_hub, head, pcm, eng.block, 20)
    assert not np.allclose(isolated, context, atol=1e-3)


def test_agc_levels_isolated_windows(standin_hub, pcm):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    loud = WakeForgeHotwordPlugin("x", {"model": head, "threshold": 1.1, "agc": True})
    quiet = WakeForgeHotwordPlugin("x", {"model": head, "threshold": 1.1, "agc": True})
    a = _stream_scores(loud, pcm, 1280)
    b = _stream_scores(quiet, (pcm // 2).astype(np.int16), 1280)
    tail = slice(len(a) // 2, None)  # once the window holds the burst, its peak sets the gain
    assert np.allclose(a[tail], b[tail], atol=1e-3)


def _long_burst(seconds=3.0, seed=4):
    rng = np.random.default_rng(seed)
    audio = rng.normal(0, 300, int(16000 * (seconds + 2)))
    audio[16000:16000 + int(16000 * seconds)] += rng.normal(0, 9000, int(16000 * seconds)) + 3000
    return np.clip(audio, -32768, 32767).astype(np.int16)


def _detection_times(eng, pcm, chunk=1024):
    times = []
    for i in range(0, len(pcm), chunk):
        eng.update(pcm[i:i + chunk].tobytes())
        if eng.found_wake_word():
            times.append((i + chunk) / 16000)
    return times


@pytest.mark.parametrize("debounce", [None, 2.0])
def test_no_second_detection_within_debounce(standin_hub, debounce):
    # found_wake_word() resets the smoother and the window; a sustained positive
    # then refills the window and fired again a few blocks later.
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert")
    pcm = _long_burst()
    probe = WakeForgeHotwordPlugin("x", {"model": head})
    offline = _offline_isolated_scores(standin_hub, head, pcm, probe.block, probe.engine.window)
    cfg = {"model": head, "threshold": (min(offline) + max(offline)) / 2,
           "smoothing": "max", "window_size": 1, "patience": 1}
    if debounce is not None:
        cfg["debounce_sec"] = debounce
    times = _detection_times(WakeForgeHotwordPlugin("x", cfg), pcm)
    gap = 1.0 if debounce is None else debounce   # the documented default is 1 s
    assert len(times) >= 2, times                  # the burst outlasts the debounce
    assert min(np.diff(times)) >= gap - 1024 / 16000, times
