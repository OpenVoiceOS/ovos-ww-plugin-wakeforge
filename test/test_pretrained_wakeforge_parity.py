"""The plugin's pretrained-featurizer runtime agrees with wakeforge's.

The plugin mirrors ``ww_trainer.pretrained`` so that it runs without torch.
These tests import wakeforge itself and fail when the registry, the resolved
featurizer attributes or the streamed scores drift apart.
"""
import numpy as np
from ww_trainer import pretrained as wf_pretrained
from ww_trainer.feats import OnnxFeatureExtractor
from ww_trainer.inference import OnnxStreamingWakeWord as WakeforgeStreamer

from ovos_ww_plugin_wakeforge import WakeForgeHotwordPlugin
from ovos_ww_plugin_wakeforge.pretrained import (
    PRETRAINED_FEATURIZERS,
    PretrainedOnnxFeaturizer,
)


def _fields(entry):
    return entry.repo_id, entry.variant, entry.license, entry.context_samples, entry.revision


def test_registry_matches_wakeforge():
    ours = {name: _fields(e) for name, e in PRETRAINED_FEATURIZERS.items()}
    theirs = {name: _fields(e) for name, e in wf_pretrained.PRETRAINED_FEATURIZERS.items()}
    assert ours == theirs


def test_resolved_featurizer_matches_wakeforge(standin_hub, monkeypatch):
    monkeypatch.setattr(wf_pretrained, "hf_hub_download", standin_hub.download)
    for name in ("wakehubert", "wakehubert-int8", "wakewav-mel-tcn-wide"):
        ours = PretrainedOnnxFeaturizer.from_pretrained(name, revision="r1")
        theirs = OnnxFeatureExtractor.from_pretrained(name, revision="r1", device="cpu")
        for attr in ("model_path", "feature_dim", "hop_samples", "frame_rate_hz",
                     "context_samples", "streaming", "license"):
            assert getattr(ours, attr) == getattr(theirs, attr), (name, attr)


def test_streamed_scores_match_wakeforge_streamer(standin_hub, monkeypatch, pcm):
    monkeypatch.setattr(wf_pretrained, "hf_hub_download", standin_hub.download)
    head = standin_hub.streaming_head(pretrained_featurizer="wakehubert")
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head, "threshold": 1.1})

    ext = OnnxFeatureExtractor.from_pretrained("wakehubert", device="cpu")
    reference = WakeforgeStreamer.from_extractor(ext, head, window=standin_hub.window,
                                                 hidden_dim=standin_hub.hidden)
    audio = pcm.astype(np.float32) / 32768.0
    expected = [reference.push(audio[i:i + eng.block])
                for i in range(0, len(audio) - eng.block + 1, eng.block)]

    scores, push = [], eng.engine.push
    eng.engine.push = lambda block: scores.append(push(block)) or scores[-1]
    for i in range(0, len(pcm), 1000):
        eng.update(pcm[i:i + 1000].tobytes())
    assert len(scores) == len(expected)
    assert np.allclose(scores, expected, atol=1e-6)
