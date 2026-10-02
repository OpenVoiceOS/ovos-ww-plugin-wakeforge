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
    "alexa": {"wake_word": "alexa", "license_has": "Apache-2.0",
              "training_data_has": "synthetic-wakeword-alexa"},
    "computer": {"wake_word": "computer", "license_has": "Apache-2.0",
                 "training_data_has": "synthetic-wakeword-computer"},
    "jarvis": {"wake_word": "jarvis", "license_has": "Apache-2.0",
               "training_data_has": "synthetic-wakeword-jarvis"},
    "ok_nabu": {"wake_word": "ok nabu", "license_has": "Apache-2.0",
                "training_data_has": "synthetic-wakeword-ok_nabu"},
    "hey_mycroft": {"wake_word": "hey mycroft", "license_has": "Apache-2.0",
                    "training_data_has": "human recordings"},
    "hey_mycroft_synthetic": {"wake_word": "hey mycroft", "license_has": "Apache-2.0",
                              "training_data_has": "synthetic-wakeword-hey_mycroft"},
    "wake_up": {"wake_word": "wake up", "license_has": "Apache-2.0",
                "training_data_has": "synthetic-wakeword-wake_up"},
    "acorda": {"wake_word": "acorda", "license_has": "Apache-2.0",
               "training_data_has": "synthetic-wakeword-acorda"},
    "android": {"wake_word": "android", "license_has": "Apache-2.0",
                "training_data_has": "synthetic-wakeword-android"},
    "athena": {"wake_word": "athena", "license_has": "Apache-2.0",
               "training_data_has": "synthetic-wakeword-athena"},
    "hello_nabu": {"wake_word": "hello nabu", "license_has": "Apache-2.0",
                   "training_data_has": "synthetic-wakeword-hello_nabu"},
    "hey_computer": {"wake_word": "hey computer", "license_has": "Apache-2.0",
                     "training_data_has": "synthetic-wakeword-hey_computer"},
    "hey_floyd": {"wake_word": "hey floyd", "license_has": "Apache-2.0",
                  "training_data_has": "synthetic-wakeword-hey_floyd"},
    "hey_jarvis": {"wake_word": "hey jarvis", "license_has": "Apache-2.0",
                   "training_data_has": "synthetic-wakeword-hey_jarvis"},
    "hey_robin": {"wake_word": "hey robin", "license_has": "Apache-2.0",
                  "training_data_has": "synthetic-wakeword-hey_robin"},
    "home_assistant": {"wake_word": "home assistant", "license_has": "Apache-2.0",
                       "training_data_has": "synthetic-wakeword-home_assistant"},
    "voice_assistant": {"wake_word": "voice assistant", "license_has": "Apache-2.0",
                        "training_data_has": "synthetic-wakeword-voice_assistant"},
    "marvin": {"wake_word": "marvin", "license_has": "Apache-2.0",
               "training_data_has": "Speech Commands"},
    "marvin_synthetic": {"wake_word": "marvin", "license_has": "Apache-2.0",
                         "training_data_has": "synthetic-wakeword-marvin"},
    "sheila_synthetic": {"wake_word": "sheila", "license_has": "Apache-2.0",
                         "training_data_has": "synthetic-wakeword-sheila"},
    "stop": {"wake_word": "stop", "license_has": "Apache-2.0",
             "training_data_has": "Speech Commands"},
    "stop_synthetic": {"wake_word": "stop", "license_has": "Apache-2.0",
                       "training_data_has": "synthetic-wakeword-stop"},
}

# Heads scored on the int8 featurizer; every other head runs on the float one.
_INT8 = {"alexa", "jarvis", "marvin_synthetic"}

# Model -> other words' clips it also fires on. Each pair is a phrase that contains the
# other ("jarvis", "hey jarvis"), or two phrases that share a syllable pattern; listed here so a
# new confusion shows up as a failure.
_CONFUSED_WITH = {
    "jarvis": {"hey_jarvis"},
    "computer": {"hey_computer"},
    "hey_computer": {"computer"},
    "home_assistant": {"voice_assistant"},
    "hello_nabu": {"ok_nabu"},
    "ok_nabu": {"hello_nabu"},
    "athena": {"ok_nabu"},
    "hey_robin": {"ok_nabu"},
}


def _featurizer(word):
    return "wakehubert-int8" if word in _INT8 else "wakehubert"


# Test clips are edge-tts voices held out of training, except hey_floyd's: its head trained on every
# edge-tts English voice, and Kokoro and OmniVoice speech did not reach its trigger.
def _clip(word):
    return word.removesuffix("_synthetic")


# Silence before the clip, in seconds. The plugin scores 80 ms blocks, so four leads 20 ms
# apart put the word at four different positions in a block; the longer leads repeat that
# at another absolute alignment.
_LEADS = (1.0, 1.02, 1.04, 1.06)
_OWN_LEADS = _LEADS + (2.0, 2.02, 2.04, 2.06)


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
    assert out.name in ("logit_calibrated", "logit")
    meta = sess.get_modelmeta().custom_metadata_map
    assert meta["wake_word"] == expected["wake_word"]
    assert meta["pretrained_featurizer"] == _featurizer(word)
    assert meta["window_frames"] == "75"
    assert expected["license_has"] in meta["license"]
    if "default_threshold" in meta:
        assert abs(float(meta["default_threshold"]) - BUNDLED_MODELS[word]) <= 5e-4
    if word == "hey_mycroft":
        assert meta["training_data"] == expected["training_data_has"]
    assert expected["training_data_has"] in meta["training_data"]
    if "synthetic" in expected["training_data_has"]:
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
    assert eng.featurizer.name == _featurizer(word)
    assert eng.engine.isolated and eng.engine.window == 75
    for lead in _OWN_LEADS:
        pcm = _pcm(CLIPS / f"{_clip(word)}.wav", lead=lead, tail=0.5)
        scores, fired = _run(eng, pcm)
        assert len(fired) == 1, (lead, fired, max(s for s in scores if s is not None))
        assert lead <= fired[0] <= len(pcm) / 16000   # after the word starts, not in the leading silence
        eng.reset()


@pytest.mark.parametrize("word", sorted(BUNDLED_MODELS))
def test_bundled_model_ignores_the_other_words(offline, word):
    eng = WakeForgeHotwordPlugin(word, {"model": word})
    for other in sorted({_clip(w) for w in BUNDLED_MODELS} - {_clip(word)}):
        if other in _CONFUSED_WITH.get(_clip(word), ()):
            continue
        for lead in _LEADS:
            scores, fired = _run(eng, _pcm(CLIPS / f"{other}.wav", lead=lead, tail=0.5))
            assert fired == [], (word, other, lead, max(s for s in scores if s is not None))
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
