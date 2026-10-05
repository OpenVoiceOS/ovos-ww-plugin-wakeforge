"""Zero-shot wake words from IPA: the WakePhoneHuBERT featurizer is downloaded from the Hugging Face repository
TigreGotico/wakephonehubert at the pinned revision, and real clips are scored for a word given only as phones.

The clips are synthetic speech (edge-tts voices) from ``test/clips``. The scorer parity test reads stored log
posteriors from ``test/data``, so it does not depend on the CPU the featurizer runs on.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf
from ovos_plugin_manager.wakewords import find_wake_word_plugins, load_wake_word_plugin

from ovos_ww_plugin_wakeforge import zeroshot
from ovos_ww_plugin_wakeforge.pretrained import resolve_pretrained
from ovos_ww_plugin_wakeforge.zeroshot import (
    FEATURIZER,
    WakePhoneHuBERTZeroShotPlugin,
    default_threshold,
    ipa_symbol_table,
    keyword_score,
    tokenize_ipa,
)

CLIPS = Path(__file__).parent / "clips"
LOG_POSTERIORS = Path(__file__).parent / "data" / "zeroshot_jarvis_log_posteriors.npz"
JARVIS = "dʒ ɑːɹ v ɪ s"
HEY_JARVIS = "h eɪ dʒ ɑːɹ v ɪ s"
HEY_CHATTERBOX = "h eɪ tʃ æ ɾ ɚ b ɑː k s"
VOICE_ASSISTANT = "v ɔɪ s ɐ s ɪ s t ə n t"

# Score of every window of test/clips/jarvis.wav (1 s of silence on both sides, an isolated 1.5 s window after
# every 80 ms block), from the evaluation code the default thresholds were calibrated with (its "best path per
# frame" scorer), over the log posteriors stored in LOG_POSTERIORS.
REF_JARVIS = [
    -1.072458, -1.072458, -1.072458, -1.072458, -1.072458, -1.072458, -1.072458, -1.072458, -1.072458, -1.072458,
    -1.072458, -1.072458, -1.072458, -1.072458, -1.072458, -1.049662, -0.890005, -0.910298, -0.970477, -1.030979,
    -0.929349, -0.655032, -0.412829, -0.141756, -0.13487, -0.140434, -0.303899, -0.254905, -0.171784, -0.165942,
    -0.1585, -0.170081, -0.16051, -0.33579, -0.451036, -0.418532, -0.612593, -0.813195, -0.910757, -0.934162,
    -1.028459, -1.034786, -1.061752, -1.072551, -1.074417, -1.070804, -1.064093, -1.059111]
REF_HEY_CHATTERBOX = [
    -1.97174, -1.97174, -1.97174, -1.97174, -1.97174, -1.97174, -1.97174, -1.97174, -1.97174, -1.97174,
    -1.97174, -1.97174, -1.97174, -1.97174, -1.97174, -1.957727, -1.753167, -1.593109, -1.353567, -1.227951,
    -1.057738, -0.98726, -0.942924, -0.941559, -0.925455, -0.922889, -0.919033, -0.954782, -0.936375, -0.913194,
    -0.891015, -0.877488, -0.822877, -1.070554, -1.241547, -1.233413, -1.572702, -1.826751, -1.861525, -1.895064,
    -1.971188, -1.985572, -2.044078, -2.085918, -2.088971, -2.087897, -2.079235, -2.076036]

# Vocab ids of the evaluation's targets: eSpeak NG en-us through the teacher's tokenizer.
TEACHER_IDS = {
    JARVIS: [60, 68, 25, 17, 5],
    HEY_CHATTERBOX: [39, 44, 66, 36, 15, 43, 26, 41, 11, 5],
    "k ə m p j uː ɾ ɚ": [11, 7, 13, 18, 24, 46, 15, 43],
    VOICE_ASSISTANT: [25, 100, 5, 20, 5, 17, 5, 6, 7, 4, 6],
}


@pytest.fixture(scope="module")
def table():
    _, config = resolve_pretrained(FEATURIZER)
    return ipa_symbol_table(config["vocab"]["symbols"])


def _pcm(name, pad=1.0):
    audio, sr = sf.read(CLIPS / f"{name}.wav", dtype="int16")
    assert sr == 16000
    silence = np.zeros(int(pad * sr), np.int16)
    return np.concatenate([silence, audio, silence])


def _run(eng, pcm, chunk=1024):
    fired = []
    for i in range(0, len(pcm), chunk):
        eng.update(pcm[i:i + chunk].tobytes())
        if eng.found_wake_word():
            fired.append(i / 16000)
    return fired


def _stored_windows():
    """[windows, 75, 392] log posteriors: the stored columns, each frame's maximum in column 1 (never a phone),
    everything else far below it."""
    z = np.load(LOG_POSTERIORS)
    lp = np.full(z["log_post"].shape[:2] + (392,), -40.0)
    lp[:, :, z["columns"]] = z["log_post"]
    lp[:, :, 1] = z["row_max"]
    return lp


def test_scores_match_the_evaluation_code(table):
    windows = _stored_windows()
    for ipa, ref in ((JARVIS, REF_JARVIS), (HEY_CHATTERBOX, REF_HEY_CHATTERBOX)):
        ids = tokenize_ipa(ipa, table)
        np.testing.assert_allclose([keyword_score(w, ids) for w in windows], ref, rtol=0, atol=1e-6)


def _peak(eng, pcm):
    """Highest score of the first pronunciation over the clip. Detections are not collected, since a detection
    resets the window and would cut the peak short."""
    scores, score_window = [], eng.score_window

    def recording(audio):
        scores.append(score_window(audio))
        return scores[-1]

    eng.score_window = recording
    for i in range(0, len(pcm), 1024):
        eng.update(pcm[i:i + 1024].tobytes())
    return max(s[0] for s in scores)


# Smallest distance kept between a clip's peak score and the default threshold, so the decisions below hold on
# CPUs whose featurizer output differs slightly from the reference machine's.
MARGIN = 0.05


def test_detection_holds_with_a_margin():
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS})
    assert _peak(eng, _pcm("hey_jarvis")) >= default_threshold(7) + MARGIN
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS})
    assert len(_run(eng, _pcm("hey_jarvis"))) == 1
    for ipa, n in ((HEY_CHATTERBOX, 10), (VOICE_ASSISTANT, 11)):
        eng = WakePhoneHuBERTZeroShotPlugin("other", {"ipa": ipa})
        assert _peak(eng, _pcm("hey_jarvis")) <= default_threshold(n) - MARGIN
        eng = WakePhoneHuBERTZeroShotPlugin("other", {"ipa": ipa})
        assert _run(eng, _pcm("hey_jarvis")) == []


def test_alternative_pronunciations_fire_on_any():
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": [HEY_CHATTERBOX, HEY_JARVIS]})
    assert eng.thresholds == [default_threshold(10), default_threshold(7)]
    assert len(_run(eng, _pcm("hey_jarvis"))) == 1


def test_detection_is_debounced_and_reported_once():
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS, "debounce_sec": 30})
    pcm = _pcm("hey_jarvis")
    assert len(_run(eng, np.concatenate([pcm, pcm, pcm]))) == 1
    assert not eng.found_wake_word()


def _scripted(eng, scores):
    """Replace the featurizer and scorer with a fixed sequence of window scores, one per 80 ms block."""
    seq = iter(scores)
    eng.score_window = lambda audio: [next(seq)]
    return np.zeros(len(scores) * zeroshot.BLOCK, np.int16)


def test_one_window_above_threshold_does_not_fire():
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS, "threshold": -0.2})
    assert eng.confirm_blocks == 2
    pcm = _scripted(eng, [-0.9, -0.1, -0.9, -0.9, -0.1, -0.5, -0.1, -0.9])
    assert _run(eng, pcm, chunk=zeroshot.BLOCK) == []


def test_consecutive_windows_above_threshold_fire_once():
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS, "threshold": -0.2})
    pcm = _scripted(eng, [-0.9, -0.1, -0.15, -0.1, -0.05, -0.9])
    assert len(_run(eng, pcm, chunk=zeroshot.BLOCK)) == 1


def test_confirm_blocks_one_fires_on_a_single_window():
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS, "threshold": -0.2, "confirm_blocks": 1})
    pcm = _scripted(eng, [-0.9, -0.1, -0.9, -0.9])
    assert len(_run(eng, pcm, chunk=zeroshot.BLOCK)) == 1
    with pytest.raises(ValueError, match="confirm_blocks"):
        WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS, "confirm_blocks": 0})


def test_a_hit_in_the_first_window_only_does_not_fire():
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS, "threshold": -0.2})
    pcm = _scripted(eng, [-0.1, -0.9, -0.9, -0.9])
    assert _run(eng, pcm, chunk=zeroshot.BLOCK) == []


def test_confirmation_needs_the_same_pronunciation_twice():
    config = {"ipa": [HEY_JARVIS, HEY_CHATTERBOX], "threshold": -0.2}
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", config)
    seq = iter([[-0.1, -0.9], [-0.9, -0.1]] * 3)
    eng.score_window = lambda audio: next(seq)
    pcm = np.zeros(6 * zeroshot.BLOCK, np.int16)
    assert _run(eng, pcm, chunk=zeroshot.BLOCK) == []

    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", config)
    seq = iter([[-0.9, -0.1], [-0.9, -0.1], [-0.9, -0.9]])
    eng.score_window = lambda audio: next(seq)
    pcm = np.zeros(3 * zeroshot.BLOCK, np.int16)
    assert len(_run(eng, pcm, chunk=zeroshot.BLOCK)) == 1


def test_a_detection_clears_the_confirmation_history():
    eng = WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS, "threshold": -0.2, "debounce_sec": 0})
    pcm = _scripted(eng, [-0.1, -0.1, -0.1, -0.9])
    assert len(_run(eng, pcm, chunk=zeroshot.BLOCK)) == 1


def test_confirm_blocks_must_be_an_integer_of_at_least_one():
    for value in (0, -1, 2.7, "abc", "2", True, None):
        with pytest.raises(ValueError, match=r"confirm_blocks.*got"):
            WakePhoneHuBERTZeroShotPlugin("hey jarvis", {"ipa": HEY_JARVIS, "confirm_blocks": value})


def test_ipa_tokens_are_the_teacher_ids(table):
    for ipa, ids in TEACHER_IDS.items():
        assert tokenize_ipa(ipa, table) == ids
        assert tokenize_ipa(ipa.replace(" ", ""), table) == ids


def test_multi_character_symbols_are_one_phone(table):
    assert len(tokenize_ipa("tʃ", table)) == 1
    assert len(tokenize_ipa("aɪ", table)) == 1
    assert len(tokenize_ipa("ɑːɹ", table)) == 1
    assert tokenize_ipa("tʃaɪ", table) == tokenize_ipa("tʃ aɪ", table)
    assert tokenize_ipa("ˈdʒɑːɹ.vɪs", table) == TEACHER_IDS[JARVIS]
    assert tokenize_ipa("d͡ʒɑːɹvɪs", table) == TEACHER_IDS[JARVIS]
    assert tokenize_ipa("hˈeɪ dʒˈɑːɹvɪs", table) == tokenize_ipa(HEY_JARVIS, table)


def test_ascii_g_is_ipa_g(table):
    assert tokenize_ipa("g", table) == tokenize_ipa("ɡ", table) == [table["ɡ"]]


def test_nfc_and_nfd_spellings_match(table):
    assert tokenize_ipa("\u00e3", table) == tokenize_ipa("a\u0303", table) == [330]
    assert tokenize_ipa("u\u0303", table) == [191]
    assert tokenize_ipa("\u0169", table) == [374]


def test_unknown_characters_are_named(table):
    with pytest.raises(ValueError, match=r"'ʘ' \(U\+0298\)"):
        tokenize_ipa("dʒɑːɹʘɪs", table)
    with pytest.raises(ValueError, match=r"'ʘ' \(U\+0298\), '😀'"):
        WakePhoneHuBERTZeroShotPlugin("jarvis", {"ipa": "dʒ ɑːɹ ʘ ɪ s 😀"})
    with pytest.raises(ValueError, match=r"'ɝ' \(U\+025D\).*eSpeak NG"):
        tokenize_ipa("ˈkɑmpətɝ", table)
    with pytest.raises(ValueError, match="ipa"):
        WakePhoneHuBERTZeroShotPlugin("jarvis", {})


def test_default_threshold_follows_phone_count():
    assert default_threshold(5) == pytest.approx(-0.10)
    assert default_threshold(7) == pytest.approx(-0.20)
    assert default_threshold(9) == default_threshold(14) == -0.29
    eng = WakePhoneHuBERTZeroShotPlugin("jarvis", {"ipa": JARVIS, "threshold": -0.5})
    assert eng.thresholds == [-0.5]


def test_session_runs_the_int8_graph_unfused(monkeypatch):
    seen = {}
    session = ort.InferenceSession

    def spy(path, options, **kwargs):
        seen["level"] = options.graph_optimization_level
        return session(path, options, **kwargs)

    monkeypatch.setattr(zeroshot.ort, "InferenceSession", spy)
    WakePhoneHuBERTZeroShotPlugin("jarvis", {"ipa": JARVIS})
    assert seen["level"] == ort.GraphOptimizationLevel.ORT_DISABLE_ALL


def test_loads_through_opm_entry_point():
    assert "ovos-ww-plugin-wakeforge-zeroshot" in find_wake_word_plugins()
    cls = load_wake_word_plugin("ovos-ww-plugin-wakeforge-zeroshot")
    assert cls is WakePhoneHuBERTZeroShotPlugin
    eng = cls("hey jarvis", {"ipa": HEY_JARVIS})
    assert len(_run(eng, _pcm("hey_jarvis"))) == 1
