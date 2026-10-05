"""Zero-shot wake words from IPA with WakePhoneHuBERT.

WakePhoneHuBERT's ``ipa`` output is a CTC posteriorgram over the 392 symbols of
wav2vec2-xlsr-53-espeak-cv-ft, 50 frames per second. A wake word written as IPA
phones is spotted in it with a CTC keyword search, so a word needs no recordings
and no trained head: only its phones and a threshold. Pure runtime: numpy +
onnxruntime + huggingface_hub, no torch and no phonemiser.
"""
import hashlib
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnxruntime as ort
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG

from ovos_ww_plugin_wakeforge.inference import session_options
from ovos_ww_plugin_wakeforge.pretrained import PRETRAINED_FEATURIZERS, resolve_pretrained

FEATURIZER = "wakephonehubert-int8"
# SHA-256 of wakephonehubert_features_int8.onnx at the featurizer's pinned revision.
FEATURIZER_SHA256 = "98ec3c40ed89e5e37d9a8f6fa09492681fd4b4a92f7524d3001bad242c8194e1"

SAMPLE_RATE = 16000
BLOCK = 1280               # 80 ms: one score per block
WINDOW = 24000             # 1.5 s isolated window, 75 frames of 320 samples

# Written in IPA but not phones: stress marks and tie bars (the table writes dʒ, not d͡ʒ).
IGNORED = "ˈˌ͜͡"
# Split phones like whitespace does.
BOUNDARIES = ".|‿"

_NEG = -1e30


def ipa_symbol_table(symbols: Sequence[str]) -> Dict[str, int]:
    """Symbol -> vocab id for the phones a keyword may use.

    Ids 0-3 (blank, ``<s>``, ``</s>``, ``<unk>``) are never phones, and the
    tone-numbered, dotted, bracketed and ``??`` entries of the teacher's table
    are left out, as in the evaluation the thresholds come from. Each symbol is
    also reachable in its NFC and NFD spellings, so a precomposed ``ã`` and a
    decomposed one both match.
    """
    import unicodedata
    table: Dict[str, int] = {}
    kept = [(i, s) for i, s in enumerate(symbols) if i >= 4 and s != "??"
            and not any(c.isdigit() or c in ".[" for c in s)]
    for i, s in kept:
        table.setdefault(s, i)
    for i, s in kept:
        for form in ("NFC", "NFD"):
            table.setdefault(unicodedata.normalize(form, s), i)
    return table


def tokenize_ipa(ipa: str, table: Dict[str, int]) -> List[int]:
    """Vocab ids of an IPA string, by longest match against ``table``.

    Whitespace, ``.``, ``|`` and ``‿`` separate phones but are optional, so
    ``"dʒ ɑːɹ v ɪ s"`` and ``"dʒɑːɹvɪs"`` give the same ids; multi-character
    symbols such as ``tʃ``, ``aɪ`` or ``ɑːɹ`` count as one phone. Stress marks
    and tie bars are dropped, and an ASCII ``g`` is read as IPA ``ɡ``. Where the
    longest match leaves a remainder no symbol covers (``pjuː`` would leave a
    lone ``ː`` after ``ju``), the next shorter match is tried.

    Raises:
        ValueError: naming the characters no symbol contains, or the stretch
            that cannot be split into symbols, or when no phone is left.
    """
    chars = set("".join(table))
    longest = max(map(len, table))
    text = ipa
    for b in BOUNDARIES:
        text = text.replace(b, " ")
    text = "".join(c for c in text if c not in IGNORED).replace("g", "ɡ")
    unknown = sorted({c for c in text if not c.isspace() and c not in chars})
    if unknown:
        raise ValueError(
            f"IPA {ipa!r} has characters that are not in WakePhoneHuBERT's phone table: "
            + ", ".join(f"{c!r} (U+{ord(c):04X})" for c in unknown)
            + ". Write the phones as eSpeak NG en-us prints them (espeak-ng -q --ipa -v en-us).")
    ids: List[int] = []
    for chunk in text.split():
        split = _segment(chunk, table, longest)
        if split is None:
            raise ValueError(f"IPA {ipa!r}: {chunk!r} cannot be split into WakePhoneHuBERT phones")
        ids += split
    if not ids:
        raise ValueError(f"IPA {ipa!r} holds no phones")
    return ids


def _segment(chunk: str, table: Dict[str, int], longest: int) -> Optional[List[int]]:
    @lru_cache(maxsize=None)
    def rest(i):
        if i == len(chunk):
            return ()
        for n in range(min(longest, len(chunk) - i), 0, -1):
            sym = chunk[i:i + n]
            if sym in table:
                tail = rest(i + n)
                if tail is not None:
                    return (table[sym],) + tail
        return None
    out = rest(0)
    return None if out is None else list(out)


def default_threshold(n_phones: int) -> float:
    """Starting threshold for a keyword of ``n_phones`` phones.

    Per-word thresholds set at 1 false activation per hour on negatives only
    (LibriSpeech dev-clean and dev-other, and held-out non-speech, 13.1 h) fall
    with the keyword's length, from -0.01 to -0.04 for four phones to -0.29 to
    -0.39 for ten or eleven, about 0.048 per phone over 23 words. The default
    is that line, rounded to 0.045 per phone so it errs to fewer false
    activations, and held at -0.30 from ten phones on, where the calibrated
    thresholds stop falling. A single threshold for all lengths does not work:
    -0.20, the median of the 23, gives "jarvis" 39 and "alexa" 74 false
    activations per hour. Checked on the public 31.1 h of test negatives
    (LibriSpeech test-clean, test-other, a train-other-500 sample, non-speech):
    a median of 0.67 false activations per hour over the 23 words (0.42 over
    those of six phones or more), with "alexa" the worst at 7.6, and mean
    recall 74.8% against 79.7% at each word's own calibrated threshold.
    """
    return max(-0.045 * (n_phones - 3), -0.30)


def keyword_score(log_post: np.ndarray, ids: Sequence[int]) -> float:
    """Length-normalised best-path CTC score of a keyword in a window.

    ``log_post`` is ``[frames, 392]`` log posteriors. Each frame is measured
    against its own most likely symbol, the keyword (blank-interleaved CTC
    labels) may start and end on any frame, and the score is the best path's
    log likelihood divided by the frames it spans, maximised over end frames.
    0 is a perfect match. This is the evaluation's ``fl_maxpf`` scorer, with
    the same tie-breaking.
    """
    lp = np.asarray(log_post, dtype=np.float64)
    labels = np.zeros(2 * len(ids) + 1, dtype=np.int64)
    labels[1::2] = ids
    n_states = len(labels)
    skip = np.zeros(n_states, dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    emit = lp[:, labels] - lp.max(axis=1, keepdims=True)
    start = np.full(n_states, _NEG)
    start[:2] = 0.0
    a = np.full(n_states, _NEG)
    origin = np.zeros(n_states, dtype=np.int64)
    best = _NEG
    for t in range(len(lp)):
        prev, src = a.copy(), origin.copy()
        for shift, allowed in ((1, None), (2, skip)):
            cand = np.full(n_states, _NEG)
            cand[shift:] = a[:-shift]
            if allowed is not None:
                cand = np.where(allowed, cand, _NEG)
            csrc = np.zeros(n_states, dtype=np.int64)
            csrc[shift:] = origin[:-shift]
            better = cand > prev
            prev, src = np.where(better, cand, prev), np.where(better, csrc, src)
        better = start > prev
        prev, src = np.where(better, start, prev), np.where(better, t, src)
        a = prev + emit[t]
        origin = src
        for s in (n_states - 1, n_states - 2):
            best = max(best, a[s] / (t + 1 - origin[s]))
    return float(best)


def log_posteriors(ipa: np.ndarray) -> np.ndarray:
    """``ipa`` posteriors -> the log posteriors the scorer and the calibration use."""
    return np.log(np.clip(ipa.astype(np.float32), 1e-10, None)).astype(np.float64)


class WakePhoneHuBERTZeroShotPlugin(HotWordEngine):
    """Detect a wake word from its IPA phones alone, with WakePhoneHuBERT.

    No recordings and no trained model: the word is given as IPA in the
    ``hotwords.<phrase>`` section of ``mycroft.conf``::

        "hotwords": {
            "hey_jarvis": {
                "module": "ovos-ww-plugin-wakeforge-zeroshot",
                "ipa": "h eɪ dʒ ɑːɹ v ɪ s"
            }
        }

    Every 80 ms the last 1.5 s of audio is featurized on its own and the
    keyword is scored over the window's ``ipa`` frames. Those frames describe
    the audio 100 ms (5 frames) earlier than their position, so a word is found
    once its last phone is 100 ms into the past, by the next windows if not by
    this one; all 75 frames are scored, as they were when the thresholds were
    calibrated.

    Recognised config keys:
        ipa (str | list[str]): the wake word's phones in IPA (required). A
            list gives alternative pronunciations; each is scored and the word
            fires when any of them reaches its threshold.
        threshold (float): detection threshold on the length-normalised score
            (0 is a perfect match), for every pronunciation. Default per
            pronunciation from its phone count, see :func:`default_threshold`.
        debounce_sec (float): minimum seconds between detections, default 2.0.
        featurizer_revision (str): Hub revision of TigreGotico/wakephonehubert,
            default the pinned revision, whose ONNX is checked against its
            SHA-256.
        onnx_threads (int): intra-op threads of the ONNX session, default 1.
    """

    def __init__(self, key_phrase="hey jarvis", config=None):
        super().__init__(key_phrase, config)
        ipa = self.config.get("ipa")
        if not ipa:
            raise ValueError(
                "wakeforge zero-shot plugin needs the wake word's phones as 'ipa' in the hotword "
                "config, e.g. \"ipa\": \"dʒ ɑːɹ v ɪ s\"")
        self.ipa = [ipa] if isinstance(ipa, str) else list(ipa)

        entry = PRETRAINED_FEATURIZERS[FEATURIZER]
        revision = self.config.get("featurizer_revision") or entry.revision
        model_path, config = resolve_pretrained(FEATURIZER, revision)
        if revision == entry.revision:
            with open(model_path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if digest != FEATURIZER_SHA256:
                raise ValueError(f"WakePhoneHuBERT featurizer at {model_path} has SHA-256 {digest}, "
                                 f"expected {FEATURIZER_SHA256}")
        self.ipa_slice = slice(*config["output"]["layout"]["ipa"])
        table = ipa_symbol_table(config["vocab"]["symbols"])
        self.keywords = [tokenize_ipa(p, table) for p in self.ipa]

        if "threshold" in self.config:
            self.thresholds = [float(self.config["threshold"])] * len(self.keywords)
        else:
            self.thresholds = [default_threshold(len(k)) for k in self.keywords]
        self.debounce_sec = float(self.config.get("debounce_sec", 2.0))

        options = session_options(int(self.config.get("onnx_threads", 1)))
        # Graph optimisation fuses the int8 graph into kernels that differ between CPUs (AVX2-only x86 against
        # VNNI), and block scores move by up to 0.8. Unfused, x86 CPUs agree to 1e-7 and aarch64 to 1e-3 where
        # the thresholds act, at about 1.4 times the featurizer cost.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self._in = self.session.get_inputs()[0].name
        self._out = self.session.get_outputs()[0].name

        self.trigger_flag = False
        self.last_score = None
        self._quiet_samples = 0
        self.reset()
        LOG.info(f"wakeforge zero-shot wake word '{self.key_phrase}' loaded: "
                 f"{len(self.keywords)} pronunciation(s), phones {[len(k) for k in self.keywords]}, "
                 f"thresholds {[round(t, 3) for t in self.thresholds]}")

    def score_window(self, audio: np.ndarray) -> List[float]:
        """Score of every pronunciation over one window of float audio."""
        feats = self.session.run([self._out], {self._in: audio[np.newaxis, :].astype(np.float32)})[0]
        lp = log_posteriors(feats[0, :, self.ipa_slice])
        return [keyword_score(lp, ids) for ids in self.keywords]

    def update(self, chunk):
        """Buffer a raw 16-bit PCM chunk and score the window after every 80 ms block."""
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return
        self._buffer = np.concatenate([self._buffer, audio])
        while self._buffer.size >= BLOCK:
            block, self._buffer = self._buffer[:BLOCK], self._buffer[BLOCK:]
            self._window = np.concatenate([self._window[BLOCK:], block])
            scores = self.score_window(self._window)
            self.last_score = max(scores)
            if self._quiet_samples > 0:
                self._quiet_samples -= BLOCK
                continue
            if any(s >= t for s, t in zip(scores, self.thresholds)):
                self.trigger_flag = True
                self._quiet_samples = int(self.debounce_sec * SAMPLE_RATE)

    def found_wake_word(self):
        """Return True (once) if the wake word fired since the last call."""
        if self.trigger_flag:
            self.trigger_flag = False
            self.reset()
            return True
        return False

    def reset(self):
        """Clear the audio window between detections; the debounce carries on."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._window = np.zeros(WINDOW, dtype=np.float32)
