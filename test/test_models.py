"""The ready models of the Hub repository and the featurizer they run on, downloaded into the Hugging Face cache.

The test clips are synthetic speech (edge-tts voices) saying the wake word; the clips added with the
calibrated models are in edge-tts voices held out of their training, and those of the localised wake-up words
(despierta, aufwachen, wakker worden, sveglia) are OmniVoice voices drawn from seeds no training clip uses.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf

import ovos_ww_plugin_wakeforge
from ovos_ww_plugin_wakeforge import (
    MODELS_REVISION,
    WakeForgeHotwordPlugin,
    download_listed_model,
    model_entries,
    model_names,
)
from ovos_ww_plugin_wakeforge.pretrained import PRETRAINED_FEATURIZERS, resolve_pretrained

ENTRIES = model_entries()
CLIPS = Path(__file__).parent / "clips"

# Per-head metadata that is expected to differ (license wording, training data,
# the literal wake-word string, the featurizer); everything else is checked the
# same way for every head.
_SYNTHETIC_ONLY = "synthetic only"
_METADATA = {
    "wakehubert_alexa": {"wake_word": "alexa", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_android": {"wake_word": "android", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_aufwachen": {"wake_word": "aufwachen", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_computer": {"wake_word": "computer", "training_data_has": "synthetic-wakeword-computer"},
    "wakehubert_despierta": {"wake_word": "despierta", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hello_nabu": {"wake_word": "hello nabu", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_chatterbox": {"wake_word": "hey chatterbox", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_computer": {"wake_word": "hey computer", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_floyd": {"wake_word": "hey floyd", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_jarvis": {"wake_word": "hey jarvis", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_k9": {"wake_word": "hey k9", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_marvin": {"wake_word": "hey marvin", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_mycroft": {"wake_word": "hey mycroft", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_potato": {"wake_word": "hey potato", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_rhasspy": {"wake_word": "hey rhasspy", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_robin": {"wake_word": "hey robin", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_scout": {"wake_word": "hey scout", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_stemcom": {"wake_word": "hey stemcom", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_hey_ziggy": {"wake_word": "hey ziggy", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_home_assistant": {"wake_word": "home assistant", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_jarvis": {"wake_word": "jarvis", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_marvin": {"wake_word": "marvin", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_okay_nabu": {"wake_word": "okay nabu", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_sheila": {"wake_word": "sheila", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_stop": {"wake_word": "stop", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_sveglia": {"wake_word": "sveglia", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_wake_up": {"wake_word": "wake up", "training_data_has": _SYNTHETIC_ONLY},
    "wakehubert_wakker_worden": {"wake_word": "wakker worden", "training_data_has": _SYNTHETIC_ONLY},
}
# Heads exported with their calibration folded into the graph, on the int8
# featurizer: 0.5 is the probability the calibration targets.
CALIBRATED = sorted(w for w, m in _METADATA.items() if m["training_data_has"] == _SYNTHETIC_ONLY)

# Clips a head fires on although they hold another word; real behaviour of
# similar-sounding words, documented in the README.
_CONFUSED_WITH = {
    "jarvis": {"hey_jarvis"},
    "android": {"hey_floyd"},
    "computer": {"hey_computer"},
    "hey_computer": {"computer"},
    "marvin": {"hey_marvin"},
    "hey_jarvis": {"hey_chatterbox"},
    "hello_nabu": {"hey_marvin", "okay_nabu"},
    "hey_marvin": {"hey_rhasspy", "hey_robin", "marvin", "okay_nabu"},
    "hey_robin": {"hey_marvin", "hey_rhasspy", "okay_nabu"},
    "okay_nabu": {"hey_k9", "hey_marvin", "hey_rhasspy"},
    "sheila": {"computer"},
}


def _featurizer(word):
    return "wakehubert-int8" if word in CALIBRATED else "wakehubert"


def _clip(word):
    return word.removeprefix("wakehubert_")


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


def test_listed_models_are_the_documented_words():
    assert model_names() == sorted(_METADATA)
    assert all(name.startswith("wakehubert_") for name in ENTRIES)
    assert all(0 < e["default_threshold"] < 1 for e in ENTRIES.values())
    assert {n for n, e in ENTRIES.items() if e["calibrated"]} == set(CALIBRATED)


@pytest.mark.parametrize("word", sorted(ENTRIES))
def test_listed_heads_io_and_metadata(word):
    expected = _METADATA[word]
    sess = ort.InferenceSession(download_listed_model(ENTRIES[word]))
    (inp,), (out,) = sess.get_inputs(), sess.get_outputs()
    assert (inp.name, inp.shape[2]) == ("features", 128)
    assert out.name == "logit_calibrated"
    meta = sess.get_modelmeta().custom_metadata_map
    assert meta["wake_word"] == expected["wake_word"]
    assert meta["pretrained_featurizer"] == _featurizer(word) == ENTRIES[word]["featurizer"]
    assert meta["window_frames"] == "75"
    assert "Apache-2.0" in meta["license"]
    if "default_threshold" in meta:
        assert abs(float(meta["default_threshold"]) - ENTRIES[word]["default_threshold"]) <= 5e-4
    assert expected["training_data_has"] in meta["training_data"]
    if expected["training_data_has"].startswith("synthetic-wakeword"):
        assert "CC" in meta["training_data"] and "4.0" in meta["training_data"]
    logit = sess.run(None, {"features": np.zeros((1, 75, 128), np.float32)})[0]
    assert logit.shape == (1,)


# Heads calibrated on a 20 h stream featurized as the plugin featurizes audio,
# where 1 FA/h is read without extrapolating the tail; their shipped defaults.
_RECALIBRATED = {"wakehubert_aufwachen": 0.57, "wakehubert_despierta": 0.51,
                 "wakehubert_sveglia": 0.47, "wakehubert_wakker_worden": 0.56}


@pytest.mark.parametrize("word", sorted(_RECALIBRATED))
def test_recalibrated_heads_carry_their_unextrapolated_calibration(word):
    assert ENTRIES[word]["default_threshold"] == _RECALIBRATED[word]
    meta = ort.InferenceSession(download_listed_model(ENTRIES[word])).get_modelmeta().custom_metadata_map
    assert meta["calib_extrapolated"] == "False"
    assert float(meta["calib_hours"]) >= 20.0
    assert abs(float(meta["default_threshold"]) - _RECALIBRATED[word]) <= 5e-4

def test_featurizer_resolves_from_the_hub_at_its_pinned_revision():
    for name, file in (("wakehubert", "wakehubert.onnx"), ("wakehubert-int8", "wakehubert_int8.onnx")):
        path, config = resolve_pretrained(name)
        assert Path(path).name == file
        assert "e30726f3a1c28bb5dffadf101fe26c97e0e3c3e1" in Path(path).parts
        assert config["streaming_context_samples"] == 40000
        sess = ort.InferenceSession(path)
        assert sess.run(None, {"waveform": np.zeros((1, 24000), np.float32)})[0].shape == (1, 75, 128)


@pytest.mark.parametrize("word", sorted(ENTRIES))
def test_listed_name_downloads_and_loads_with_the_entry_default_threshold(word):
    entry = ENTRIES[word]
    eng = WakeForgeHotwordPlugin(word, {"model": word})
    assert eng.threshold == entry["default_threshold"]
    assert eng.smoother.threshold == entry["default_threshold"]
    assert (eng.smoother.method, eng.smoother.window_size, eng.smoother.patience) == ("max", 1, 1)
    assert eng.debounce_sec == 2.0
    assert eng.featurizer.name == entry["featurizer"]
    assert eng.engine.isolated and eng.engine.window == 75
    want = ort.InferenceSession(download_listed_model(entry)).get_modelmeta().custom_metadata_map
    assert eng.engine.head.get_modelmeta().custom_metadata_map == want


def test_configured_threshold_overrides_the_entry_default():
    eng = WakeForgeHotwordPlugin("alexa", {"model": "wakehubert_alexa", "threshold": 0.8, "patience": 2})
    assert ENTRIES["wakehubert_alexa"]["default_threshold"] != 0.8
    assert eng.threshold == 0.8 and eng.smoother.threshold == 0.8 and eng.smoother.patience == 2


def test_sha256_mismatch_raises(monkeypatch):
    entries = {n: dict(e) for n, e in ENTRIES.items()}
    entries["wakehubert_alexa"]["sha256"] = "0" * 64
    monkeypatch.setattr(ovos_ww_plugin_wakeforge, "model_entries", lambda revision=None: entries)
    with pytest.raises(ValueError, match="SHA-256"):
        WakeForgeHotwordPlugin("alexa", {"model": "wakehubert_alexa"})
    WakeForgeHotwordPlugin("jarvis", {"model": "wakehubert_jarvis"})


def test_tampered_download_is_refused(monkeypatch, tmp_path):
    real = ovos_ww_plugin_wakeforge.hf_hub_download

    def tampered(repo, filename, revision=None):
        data = bytearray(Path(real(repo, filename, revision=revision)).read_bytes())
        if filename.endswith(".onnx"):
            data[len(data) // 2] ^= 1
        path = tmp_path / Path(filename).name
        path.write_bytes(bytes(data))
        return str(path)

    monkeypatch.setattr(ovos_ww_plugin_wakeforge, "hf_hub_download", tampered)
    with pytest.raises(ValueError, match="SHA-256"):
        WakeForgeHotwordPlugin("alexa", {"model": "wakehubert_alexa"})


def test_truncated_download_is_refused(monkeypatch, tmp_path):
    real = ovos_ww_plugin_wakeforge.hf_hub_download

    def truncated(repo, filename, revision=None):
        path = tmp_path / Path(filename).name
        data = Path(real(repo, filename, revision=revision)).read_bytes()
        path.write_bytes(data[:1000] if filename.endswith(".onnx") else data)
        return str(path)

    monkeypatch.setattr(ovos_ww_plugin_wakeforge, "hf_hub_download", truncated)
    with pytest.raises(ValueError, match="SHA-256"):
        WakeForgeHotwordPlugin("alexa", {"model": "wakehubert_alexa"})


def test_model_missing_from_the_repository_is_refused(monkeypatch):
    entries = {n: dict(e) for n, e in ENTRIES.items()}
    entries["wakehubert_alexa"]["file"] = "models/wakehubert_absent.onnx"
    monkeypatch.setattr(ovos_ww_plugin_wakeforge, "model_entries", lambda revision=None: entries)
    with pytest.raises(Exception, match="wakehubert_absent.onnx"):
        WakeForgeHotwordPlugin("alexa", {"model": "wakehubert_alexa"})


_FIRST_LOAD = ("from ovos_ww_plugin_wakeforge import WakeForgeHotwordPlugin; "
               "WakeForgeHotwordPlugin('alexa', {'model': 'wakehubert_alexa'})")


@pytest.mark.parametrize("env", [{"HF_HUB_OFFLINE": "1"}, {"HF_ENDPOINT": "http://127.0.0.1:9"}])
def test_first_load_without_network_and_with_an_empty_cache_fails_fast(env, tmp_path):
    run_env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "empty"), **env}
    run_env.pop("HF_HOME", None)
    done = subprocess.run([sys.executable, "-c", _FIRST_LOAD], env=run_env, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode != 0
    assert "wakehubert_alexa" in done.stderr
    assert "is not in the local Hugging Face cache" in done.stderr
    assert not list((tmp_path / "empty").glob("**/*.onnx"))


def test_pinned_revisions_are_commits():
    commit = re.compile(r"[0-9a-f]{40}")
    assert commit.fullmatch(MODELS_REVISION)
    for name in ("wakehubert", "wakehubert-int8"):
        assert commit.fullmatch(PRETRAINED_FEATURIZERS[name].revision or ""), name


def test_models_revision_selects_the_hub_revision(monkeypatch):
    real = ovos_ww_plugin_wakeforge.hf_hub_download
    seen = []

    def download(repo, filename, revision=None):
        seen.append((repo, filename, revision))
        return real(repo, filename, revision=MODELS_REVISION)

    monkeypatch.setattr(ovos_ww_plugin_wakeforge, "hf_hub_download", download)
    WakeForgeHotwordPlugin("alexa", {"model": "wakehubert_alexa", "models_revision": "other-revision"})
    assert seen == [("OpenVoiceOS/wakehubert-wakewords", "models.json", "other-revision"),
                    ("OpenVoiceOS/wakehubert-wakewords", "models/wakehubert_alexa.onnx", "other-revision")]


def test_unlisted_name_falls_through_to_the_path_behaviour(tmp_path):
    head = Path(download_listed_model(ENTRIES["wakehubert_jarvis"]))
    copy = tmp_path / "my_head.onnx"
    copy.write_bytes(head.read_bytes())
    eng = WakeForgeHotwordPlugin("jarvis", {"model": str(copy)})
    assert eng.threshold == ENTRIES["wakehubert_jarvis"]["default_threshold"]
    assert eng.featurizer.name == "wakehubert-int8"
    for name in ("alexa", "ok_nabu", "wakehubert_hey_mycroft_synthetic", str(tmp_path / "absent.onnx")):
        with pytest.raises(ValueError, match="Model not found"):
            WakeForgeHotwordPlugin("x", {"model": name})


@pytest.mark.parametrize("word", sorted(ENTRIES))
def test_listed_model_detects_its_wake_word(word):
    trigger = ENTRIES[word]["default_threshold"]
    eng = WakeForgeHotwordPlugin(word, {"model": word})
    # the model fires on the block whose probability crosses the trigger
    assert eng.threshold == trigger
    assert (eng.smoother.method, eng.smoother.window_size, eng.smoother.patience) == ("max", 1, 1)
    assert eng.debounce_sec == 2.0
    assert eng.featurizer.name == _featurizer(word)
    assert eng.engine.isolated and eng.engine.window == 75
    pcm = _pcm(CLIPS / f"{_clip(word)}.wav")
    scores, fired = _run(eng, pcm)
    assert len(fired) == 1, (fired, max(s for s in scores if s is not None))
    assert 2.0 <= fired[0] <= len(pcm) / 16000   # after the word starts, not in the leading silence


@pytest.mark.parametrize("word", sorted(ENTRIES))
def test_listed_model_ignores_the_other_words(word):
    eng = WakeForgeHotwordPlugin(word, {"model": word})
    others = {_clip(w) for w in ENTRIES} - {_clip(word)} - _CONFUSED_WITH.get(_clip(word), set())
    for other in sorted(others):
        scores, fired = _run(eng, _pcm(CLIPS / f"{other}.wav"))
        assert fired == [], (word, other, max(scores))
        eng.reset()


def test_listed_model_quiet_on_silence_and_noise():
    eng = WakeForgeHotwordPlugin("hey_mycroft", {"model": "wakehubert_hey_mycroft"})
    noise = (np.random.default_rng(0).normal(0, 300, 16000 * 5)).astype(np.int16)
    _, fired = _run(eng, np.concatenate([np.zeros(16000 * 3, np.int16), noise]))
    assert fired == []


@pytest.mark.parametrize("threads", [None, 2])
def test_sessions_use_few_threads_and_do_not_spin(threads):
    config = {"model": "wakehubert_alexa"}
    if threads is not None:
        config["onnx_threads"] = threads
    eng = WakeForgeHotwordPlugin("alexa", config)
    for sess in (eng.engine.ext, eng.engine.head):
        options = sess.get_session_options()
        assert options.intra_op_num_threads == (threads or 1)
        assert options.inter_op_num_threads == 1
        assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"


@pytest.mark.parametrize("word", CALIBRATED)
def test_calibrated_heads_run_on_the_int8_featurizer_at_their_metadata_trigger(word):
    sess = ort.InferenceSession(download_listed_model(ENTRIES[word]))
    meta = sess.get_modelmeta().custom_metadata_map
    assert meta["calibrated"] == "1" and "calib_a" in meta and "calib_b" in meta
    eng = WakeForgeHotwordPlugin(word, {"model": word})
    assert eng.threshold == ENTRIES[word]["default_threshold"]
    assert abs(eng.threshold - float(meta["default_threshold"])) <= 5e-4
    assert Path(eng.featurizer.model_path).name == "wakehubert_int8.onnx"
