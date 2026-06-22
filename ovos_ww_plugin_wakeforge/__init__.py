"""OVOS wake-word plugin for models trained with wakeforge.

Loads a wakeforge two-file ONNX pipeline (featurizer + classifier head) and runs
it as an always-on hotword detector. Pure runtime: numpy + onnxruntime, no torch.
"""
from os import makedirs
from os.path import isfile, join, expanduser

import numpy as np
import requests
from ovos_plugin_manager.templates.hotwords import HotWordEngine
from ovos_utils.log import LOG
from ovos_utils.xdg_utils import xdg_data_home

from ovos_ww_plugin_wakeforge.inference import (
    OnnxStreamingWakeWord,
    OnnxWakeWordInferencer,
    PredictionSmoother,
)


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

    Recognised config keys:
        featurizer (str): path/URL to the feature-extractor ONNX (required).
        model (str): path/URL to the classifier-head ONNX (required).
        vad (str): optional path/URL to a VAD ONNX (extra channel).
        threshold (float): detection threshold, default 0.5.
        smoothing (str): ``"ema"`` | ``"mean"`` | ``"max"``, default ``"ema"``.
        patience (int): consecutive above-threshold frames to fire, default 3.
        debounce_sec (float): minimum seconds between triggers, default 1.0.
        window_size (int): smoother rolling-window size (mean/max), default 5.
        ema_alpha (float): EMA responsiveness, default 0.3.
        streaming (bool): use the stateful streaming head (GRU), default False.
        gru_window (int): window the streaming head was exported with, default 100.
        hidden_dim (int): GRU hidden size for the streaming head, default 128.
    """

    def __init__(self, key_phrase="hey jarvis", config=None):
        super().__init__(key_phrase, config)
        self.trigger_flag = False

        self.threshold = float(self.config.get("threshold", 0.5))
        self.streaming = bool(self.config.get("streaming", False))

        featurizer = self.config.get("featurizer")
        model = self.config.get("model")
        if not featurizer or not model:
            raise ValueError(
                "wakeforge plugin needs both 'featurizer' and 'model' ONNX paths "
                "in the hotword config (train one with `wakeforge-quickstart`)."
            )
        featurizer = self._resolve(featurizer)
        model = self._resolve(model)
        vad = self.config.get("vad")
        vad = self._resolve(vad) if vad else None

        self.smoother = PredictionSmoother(
            method=self.config.get("smoothing", "ema"),
            window_size=int(self.config.get("window_size", 5)),
            threshold=self.threshold,
            patience=int(self.config.get("patience", 3)),
            debounce_sec=float(self.config.get("debounce_sec", 1.0)),
            ema_alpha=float(self.config.get("ema_alpha", 0.3)),
        )

        if self.streaming:
            self.engine = OnnxStreamingWakeWord(
                featurizer, model,
                window=int(self.config.get("gru_window", 100)),
                hidden_dim=int(self.config.get("hidden_dim", 128)),
            )
        else:
            self.engine = OnnxWakeWordInferencer(featurizer, model, vad_path=vad,
                                                 device="cpu")
        self._cache = None
        LOG.info(f"wakeforge wake word '{self.key_phrase}' loaded "
                 f"(streaming={self.streaming}, threshold={self.threshold})")

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
        """Process a raw 16-bit PCM chunk and update the trigger flag."""
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return
        if self.streaming:
            prob = self.engine.push(audio)
            self.smoother.update(prob)
        else:
            _, self._cache = self.engine.infer_streaming(
                audio, self._cache, smoother=self.smoother)
        if self.smoother.is_triggered():
            self.trigger_flag = True

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
        if self.streaming:
            self.engine.reset()
