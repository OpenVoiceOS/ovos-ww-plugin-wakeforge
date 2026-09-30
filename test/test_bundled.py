"""The ready models and the featurizer that ship inside the package, used with no network.

The test clips are synthetic speech (edge-tts, one voice) saying the wake word.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf

import ovos_ww_plugin_wakeforge
from ovos_ww_plugin_wakeforge import BUNDLED_MODELS, WakeForgeHotwordPlugin
from ovos_ww_plugin_wakeforge.pretrained import BUNDLED_REVISIONS, resolve_pretrained

PACKAGE = Path(ovos_ww_plugin_wakeforge.__file__).parent
CLIPS = Path(__file__).parent / "clips"

# Per-head metadata that is expected to differ (license wording, training data,
# the literal wake-word string); everything else is checked the same way for
# every bundled head.
_METADATA = {
    "alexa": {"wake_word": "alexa", "license_has": "CC-BY-4.0",
              "training_data_has": "synthetic-wakeword-alexa"},
    "hey_mycroft": {"wake_word": "hey mycroft", "license_has": "Apache-2.0",
                    "training_data_has": "not disclosed"},
    "wake_up": {"wake_word": "wake up", "license_has": "Apache-2.0",
                "training_data_has": "synthetic-wakeword-wake_up"},
}


def _pcm(path, lead=2.0, tail=1.0):
    audio, sr = sf.read(path, dtype="int16")
    assert sr == 16000
    return np.concatenate([np.zeros(int(lead * sr), np.int16), audio,
                           np.zeros(int(tail * sr), np.int16)])


def _run(eng, pcm, chunk=1024):
    scores, fired = [], []
    for i in range(0, len(pcm), chunk):
        eng.update(pcm[i:i + chunk].tobytes())
        scores.append(eng.last_score)
        if eng.found_wake_word():
            fired.append(i / 16000)
    return scores, fired


def test_every_head_file_has_a_default_trigger():
    assert {p.stem for p in (PACKAGE / "models").glob("*.onnx")} == set(BUNDLED_MODELS)
    assert set(BUNDLED_MODELS) == set(_METADATA)
    assert all(0 < t < 1 for t in BUNDLED_MODELS.values())


@pytest.mark.parametrize("word", sorted(BUNDLED_MODELS))
def test_bundled_heads_io_and_metadata(word):
    expected = _METADATA[word]
    sess = ort.InferenceSession(str(PACKAGE / "models" / f"{word}.onnx"))
    (inp,), (out,) = sess.get_inputs(), sess.get_outputs()
    assert (inp.name, inp.shape[2]) == ("features", 128)
    assert out.name == "logit_calibrated"
    meta = sess.get_modelmeta().custom_metadata_map
    assert meta["wake_word"] == expected["wake_word"]
    assert meta["pretrained_featurizer"] == "wakehubert"
    assert meta["window_frames"] == "75"
    assert expected["license_has"] in meta["license"]
    assert expected["training_data_has"] in meta["training_data"]
    if expected["training_data_has"] != "not disclosed":
        assert "CC" in meta["training_data"] and "4.0" in meta["training_data"]
    logit = sess.run(None, {"features": np.zeros((1, 75, 128), np.float32)})[0]
    assert logit.shape == (1,)


def test_bundled_featurizer_resolves_without_network(offline):
    for name, file in (("wakehubert", "wakehubert.onnx"), ("wakehubert-int8", "wakehubert_int8.onnx")):
        path, config = resolve_pretrained(name)
        folder = PACKAGE / "featurizers" / "wakehubert-tiny"
        assert Path(path) == folder / file
        assert config == json.loads((folder / "config.json").read_text())
        assert config["streaming_context_samples"] == 40000
        sess = ort.InferenceSession(path)
        assert sess.run(None, {"waveform": np.zeros((1, 24000), np.float32)})[0].shape == (1, 75, 128)
    assert BUNDLED_REVISIONS["TigreGotico/wakehubert-tiny"] in (folder / "NOTICE").read_text()


def test_other_revision_goes_to_the_hub(offline):
    with pytest.raises(OSError, match="network"):
        resolve_pretrained("wakehubert", revision="main")


@pytest.mark.parametrize("word", sorted(BUNDLED_MODELS))
def test_bundled_model_detects_its_wake_word_offline(offline, word):
    trigger = BUNDLED_MODELS[word]
    eng = WakeForgeHotwordPlugin(word, {"model": word})
    # the model fires on the block whose probability crosses the trigger
    assert eng.threshold == trigger
    assert (eng.smoother.method, eng.smoother.window_size, eng.smoother.patience) == ("max", 1, 1)
    assert eng.debounce_sec == 2.0
    assert eng.featurizer.name == "wakehubert"
    assert eng.engine.isolated and eng.engine.window == 75
    pcm = _pcm(CLIPS / f"{word}.wav")
    scores, fired = _run(eng, pcm)
    assert len(fired) == 1, (fired, max(s for s in scores if s is not None))
    assert 2.0 <= fired[0] <= len(pcm) / 16000   # after the word starts, not in the leading silence


@pytest.mark.parametrize("word", sorted(BUNDLED_MODELS))
def test_bundled_model_ignores_the_other_words(offline, word):
    eng = WakeForgeHotwordPlugin(word, {"model": word})
    for other in sorted(BUNDLED_MODELS):
        if other == word:
            continue
        scores, fired = _run(eng, _pcm(CLIPS / f"{other}.wav"))
        assert fired == [], (word, other, max(scores))
        eng.reset()


def test_bundled_model_quiet_on_silence_and_noise(offline):
    eng = WakeForgeHotwordPlugin("hey_mycroft", {"model": "hey_mycroft"})
    noise = (np.random.default_rng(0).normal(0, 300, 16000 * 5)).astype(np.int16)
    _, fired = _run(eng, np.concatenate([np.zeros(16000 * 3, np.int16), noise]))
    assert fired == []


def test_bundled_model_threshold_overridable(offline):
    eng = WakeForgeHotwordPlugin("alexa", {"model": "alexa", "threshold": 0.5, "patience": 2})
    assert eng.threshold == 0.5 and eng.smoother.patience == 2


@pytest.mark.parametrize("threads", [None, 2])
def test_sessions_use_few_threads_and_do_not_spin(offline, threads):
    config = {"model": "alexa"} if threads is None else {"model": "alexa", "onnx_threads": threads}
    eng = WakeForgeHotwordPlugin("alexa", config)
    for sess in (eng.engine.ext, eng.engine.head):
        options = sess.get_session_options()
        assert options.intra_op_num_threads == (threads or 1)
        assert options.inter_op_num_threads == 1
        assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
