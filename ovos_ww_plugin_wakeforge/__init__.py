"""OVOS wake-word plugin for models trained with wakeforge.

Loads a wakeforge two-file ONNX pipeline (featurizer + classifier head) and runs
it as an always-on hotword detector. Pure runtime: numpy + onnxruntime, no torch.
"""
import hashlib
import json
from os import makedirs
from os.path import isfile, join, expanduser

import numpy as np
import onnxruntime as ort
import requests
from huggingface_hub import hf_hub_download
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home

from ovos_ww_plugin_wakeforge.inference import (
    OnnxStreamingWakeWord,
    OnnxWakeWordInferencer,
    PredictionSmoother,
    session_options,
)
from ovos_ww_plugin_wakeforge.pretrained import (
    PRETRAINED_FEATURIZERS,
    OnnxWindowedWakeWord,
    PretrainedOnnxFeaturizer,
)

# Head metadata keys that may name the pretrained featurizer, in order.
_FEATURIZER_META_KEYS = ("pretrained_featurizer", "featurizer")

# Ready models are Hub files named in the repository's models.json, pinned to a
# revision. Each head is a recurrent classifier on WakeHuBERT features with its
# calibration folded in, so it outputs the probability that the window holds
# the wake word; its ``pretrained_featurizer`` metadata names the featurizer.
MODELS_REPO = "OpenVoiceOS/wakehubert-wakewords"
MODELS_REVISION = "d7c4c3d46f7c95d9c21989d43b99fecffc6d946c"


def _download_failed(name, revision, error):
    return RuntimeError(
        f"wakeforge model '{name}' ({MODELS_REPO} at revision {revision}) could not be "
        f"downloaded and is not in the local Hugging Face cache: {error}")


def model_entries(revision=MODELS_REVISION):
    """The ready models of the Hub repository at ``revision``, by name."""
    with open(hf_hub_download(MODELS_REPO, "models.json", revision=revision), encoding="utf-8") as f:
        return {entry["name"]: entry for entry in json.load(f)["models"]}


def model_names(revision=MODELS_REVISION):
    """Names of the ready models at ``revision``."""
    return sorted(model_entries(revision))


def download_listed_model(entry, revision=MODELS_REVISION):
    """Download a ready model into the shared Hugging Face cache and check its SHA-256."""
    try:
        path = hf_hub_download(MODELS_REPO, entry["file"], revision=revision)
    except OSError as e:
        raise _download_failed(entry["name"], revision, e) from e
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"wakeforge model '{entry['name']}' at {path} has SHA-256 {digest}, "
                         f"models.json lists {entry['sha256']}")
    return path


class WakeForgeHotwordPlugin(HotWordEngine):
    """Detect a wake word using a model trained with wakeforge.

    wakeforge exports a two-file ONNX pipeline — a feature extractor and a
    classifier head. Point this plugin at both (local paths or HTTP URLs) via the
    ``hotwords.<phrase>`` section of ``mycroft.conf``::

        "hotwords": {
            "hey_jarvis": {
                "module": "ovos-ww-plugin-wakeforge",
                "featurizer": "/path/to/best_f1_featurizer.onnx",
                "model": "/path/to/best_f1.onnx",
                "threshold": 0.5
            }
        }

    A model trained on a pretrained featurizer (``ww_trainer-train --tier
    wakehubert``) names it instead of a path: ``"featurizer": "wakehubert"``.
    Pretrained featurizers are downloaded from the Hugging Face Hub into the
    shared cache. Ready models are downloaded from the Hub repository
    ``OpenVoiceOS/wakehubert-wakewords`` the same way, named
    ``wakehubert_<word>``: ``"model": "wakehubert_jarvis"`` and the other names
    ``model_names()`` returns.

    A head loaded by path is calibrated the same way when its own ONNX
    metadata carries a ``default_threshold`` (a Platt-calibrated head, as
    ``ww_trainer-train`` writes): its threshold and smoothing default to the
    same values as the ready models, not to the EMA/patience-3 defaults
    meant for an uncalibrated head. Config keys still override.

    Recognised config keys:
        featurizer (str): path/URL to the feature-extractor ONNX, or the name
            of a pretrained featurizer (``wakehubert``, ``wakehubert-int8``,
            ...). Optional when the head's ONNX metadata names one.
        featurizer_revision (str): Hub revision (branch, tag or commit) of a
            pretrained featurizer.
        model (str): path/URL to the classifier-head ONNX, or a ready model
            name from ``model_names()`` (``wakehubert_alexa``,
            ``wakehubert_jarvis``, ...) (required).
        models_revision (str): Hub revision of the ready-model repository,
            default ``MODELS_REVISION``.
        vad (str): optional path/URL to a VAD ONNX (extra channel).
        threshold (float): detection threshold, default 0.5, or the head's
            own ``default_threshold`` metadata when it has one (a ready
            model's value is its ``models.json`` entry).
        smoothing (str): ``"ema"`` | ``"mean"`` | ``"max"``, default ``"ema"``
            (``"max"`` over one block for a calibrated head).
        patience (int): consecutive above-threshold frames to fire, default 3
            (1 for a calibrated head).
        debounce_sec (float): minimum seconds between detections, default 1.0
            (2.0 for a calibrated head).
        window_size (int): smoother rolling-window size (mean/max), default 5
            (1 for a calibrated head).
        ema_alpha (float): EMA responsiveness, default 0.3.
        block_ms (float): scoring block length, default 80.
        streaming (bool): use the stateful streaming head (GRU), default False.
            With a pretrained featurizer the head type is read from the ONNX.
        gru_window (int): frames the head averages over. Default 100 for the
            streaming head; with a pretrained featurizer, read from a streaming
            head, or the head's ``window_frames`` metadata, or 1.5 s of frames
            for a batch head.
        isolated_windows (bool): with a pretrained featurizer and a batch
            head, featurize each window on its own, as training clips were,
            default True; False featurizes with the past audio each frame
            depends on instead.
        agc (bool): with isolated windows, level each window (peak to 0.5,
            gain 0.25-4x), default False.
        hidden_dim (int): GRU hidden size for the streaming head, default 128.
        onnx_threads (int): intra-op threads per ONNX session, default 1.
    """

    def __init__(self, key_phrase="hey jarvis", config=None):
        super().__init__(key_phrase, config)
        self.trigger_flag = False

        self.last_score = None
        self.streaming = bool(self.config.get("streaming", False))

        model_name = self.config.get("model")
        if not model_name:
            raise ValueError(
                "wakeforge plugin needs a 'model' ONNX path in the hotword config "
                "(train one with `wakeforge-quickstart`)."
            )
        revision = self.config.get("models_revision", MODELS_REVISION)
        entry = None
        if not model_name.startswith("http") and not isfile(expanduser(model_name)):
            try:
                entry = model_entries(revision).get(model_name)
            except OSError as e:
                raise _download_failed(model_name, revision, e) from e
        # Fires when the probability crosses the trigger; "ready-model defaults".
        ready_defaults = {"smoothing": "max", "window_size": 1, "patience": 1, "debounce_sec": 2.0}
        if entry:
            model = download_listed_model(entry, revision)
            cfg = {"threshold": entry["default_threshold"], **ready_defaults}
        else:
            model = self._resolve(model_name)
            cfg = {"threshold": 0.5, "smoothing": "ema", "window_size": 5, "patience": 3,
                   "debounce_sec": 1.0}
        self.threads = int(self.config.get("onnx_threads", 1))
        head = ort.InferenceSession(model, session_options(self.threads),
                                    providers=["CPUExecutionProvider"])
        meta = head.get_modelmeta().custom_metadata_map
        if not entry and "default_threshold" in meta:
            # The head's own metadata says it is Platt-calibrated (as
            # ww_trainer-train writes): score it like the ready models,
            # not with the EMA/patience-3 defaults meant for an uncalibrated
            # head, which never cross a calibrated head's near-1.0 scores.
            cfg = {"threshold": float(meta["default_threshold"]), **ready_defaults}
        cfg.update(self.config)
        self.threshold = float(cfg["threshold"])
        featurizer = self.config.get("featurizer")
        pretrained = self._pretrained_name(meta, featurizer)
        if pretrained is None and not featurizer:
            raise ValueError(
                "wakeforge plugin needs both 'featurizer' and 'model' ONNX paths "
                "in the hotword config (train one with `wakeforge-quickstart`)."
            )

        # No detection within debounce_sec of the last one. Kept by the plugin
        # because found_wake_word() resets the smoother, which forgets it.
        self.debounce_sec = float(cfg["debounce_sec"])
        self._quiet_samples = 0
        self.smoother = PredictionSmoother(
            method=cfg["smoothing"],
            window_size=int(cfg["window_size"]),
            threshold=self.threshold,
            patience=int(cfg["patience"]),
            debounce_sec=self.debounce_sec,
            ema_alpha=float(self.config.get("ema_alpha", 0.3)),
        )
        # Audio is scored in fixed blocks whatever chunk size the listener
        # sends, so the smoother's patience counts time, not chunks, and the
        # featurizer never sees fewer samples than its STFT window.
        self.block = int(16000 * float(self.config.get("block_ms", 80)) / 1000)

        self.featurizer = None
        if pretrained is not None:
            self.engine = self._load_pretrained(pretrained, head, model, meta)
            hop = self.featurizer.hop_samples
            self.block = -(-self.block // hop) * hop
        else:
            featurizer = self._resolve(featurizer)
            vad = self.config.get("vad")
            vad = self._resolve(vad) if vad else None
            if self.streaming:
                self.engine = OnnxStreamingWakeWord(
                    featurizer, model,
                    window=int(self.config.get("gru_window", 100)),
                    hidden_dim=int(self.config.get("hidden_dim", 128)),
                    threads=self.threads,
                )
            else:
                self.engine = OnnxWakeWordInferencer(featurizer, model, vad_path=vad,
                                                     device="cpu", threads=self.threads)
        self._cache = None
        self._buffer = np.zeros(0, dtype=np.float32)
        LOG.info(f"wakeforge wake word '{self.key_phrase}' loaded "
                 f"(featurizer={pretrained or 'onnx'}, streaming={self.streaming}, "
                 f"threshold={self.threshold})")

    @staticmethod
    def _pretrained_name(meta, featurizer):
        """Name of the pretrained featurizer the head runs on, or None.

        The head's ONNX metadata is the authority, since it records what the
        head was trained on; the ``featurizer`` config key is the fallback.
        """
        from_meta = next((meta[k] for k in _FEATURIZER_META_KEYS
                          if meta.get(k) in PRETRAINED_FEATURIZERS), None)
        from_config = (featurizer if featurizer in PRETRAINED_FEATURIZERS
                       and not isfile(expanduser(featurizer)) else None)
        if from_meta and featurizer and from_config != from_meta:
            LOG.warning(f"wakeforge model metadata names featurizer '{from_meta}'; "
                        f"ignoring configured featurizer '{featurizer}'")
        return from_meta or from_config

    def _load_pretrained(self, name, head, model, meta):
        """Download a pretrained featurizer and build the matching stream scorer."""
        if self.config.get("vad"):
            raise ValueError("wakeforge plugin: 'vad' is not supported with a "
                             f"pretrained featurizer ('{name}').")
        if PRETRAINED_FEATURIZERS[name].context_samples is None:
            raise ValueError(
                f"wakeforge plugin: pretrained featurizer '{name}' is not streamable "
                "(its frames depend on later audio or on unbounded history), so it "
                "cannot run as a live wake word. Train on a streaming featurizer, "
                "such as 'wakehubert'.")
        revision = (self.config.get("featurizer_revision")
                    or meta.get("featurizer_revision") or None)
        self.featurizer = PretrainedOnnxFeaturizer.from_pretrained(name, revision)
        inputs = {i.name: i.shape for i in head.get_inputs()}
        if "out_window" in inputs:
            self.streaming = True
            _, window, hidden = inputs["out_window"]
            if not isinstance(window, int):
                window = int(self.config.get("gru_window", 100))
            if not isinstance(hidden, int):
                hidden = int(self.config.get("hidden_dim", 128))
            return OnnxStreamingWakeWord.from_extractor(
                self.featurizer, model, window=window, hidden_dim=hidden, threads=self.threads)
        self.streaming = False
        window = int(self.config.get("gru_window") or meta.get("window_frames")
                     or round(1.5 * self.featurizer.frame_rate_hz))
        return OnnxWindowedWakeWord.from_extractor(
            self.featurizer, model, window,
            isolated=bool(self.config.get("isolated_windows", True)),
            agc=bool(self.config.get("agc", False)), threads=self.threads)

    def _resolve(self, model):
        """Resolve a model reference to a local file path, downloading URLs."""
        if model.startswith("http"):
            return self.download_model(model)
        path = expanduser(model)
        if not isfile(path):
            raise ValueError(f"Model not found: {path}")
        return path

    @staticmethod
    def download_model(url):
        """Download an ONNX model to the XDG data dir (cached) and return its path."""
        name = url.split("/")[-1]
        folder = join(xdg_data_home(), "wakeforge")
        model_path = join(folder, name)
        if not isfile(model_path):
            LOG.info(f"Downloading wakeforge ONNX model: {url}")
            response = requests.get(url)
            response.raise_for_status()
            makedirs(folder, exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(response.content)
            LOG.info(f"Model downloaded to {model_path}")
        return model_path

    def update(self, chunk):
        """Buffer a raw 16-bit PCM chunk and score every complete block."""
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return
        self._buffer = np.concatenate([self._buffer, audio])
        while self._buffer.size >= self.block:
            block, self._buffer = self._buffer[:self.block], self._buffer[self.block:]
            if self.streaming or self.featurizer is not None:
                self.last_score = self.engine.push(block)
            else:
                self.last_score, self._cache = self.engine.infer_streaming(block, self._cache)
            if self._quiet_samples > 0:
                self._quiet_samples -= block.size
                continue
            self.smoother.update(self.last_score)
            if self.smoother.is_triggered():
                self.trigger_flag = True
                self._quiet_samples = int(self.debounce_sec * 16000)

    def found_wake_word(self):
        """Return True (once) if the wake word fired since the last call."""
        if self.trigger_flag:
            self.trigger_flag = False
            self.reset()
            return True
        return False

    def reset(self):
        """Clear streaming/smoothing state between detections."""
        self.smoother.reset()
        self._cache = None
        self._buffer = np.zeros(0, dtype=np.float32)
        if self.streaming or self.featurizer is not None:
            self.engine.reset()
