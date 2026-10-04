# The registry and resolution below mirror wakeforge's ww_trainer/pretrained.py
# (https://github.com/TigreGotico/wakeforge). wakeforge is the source of truth;
# test/test_pretrained_wakeforge_parity.py fails when the two drift apart.
# Pure runtime: numpy + onnxruntime + huggingface_hub, no torch.
"""Pretrained ONNX featurizers resolved by name from the Hugging Face Hub.

A model trained with ``ww_trainer-train --tier wakehubert`` (or
``--featurizer-type <name>``) runs on a featurizer that wakeforge downloads by
name, such as ``wakehubert``. This module resolves the same names to the same
Hub files, in the shared Hugging Face cache, and scores a stream with them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ovos_ww_plugin_wakeforge.inference import session_options

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PretrainedFeaturizer:
    """A named ONNX featurizer published in a Hub model repository.

    Args:
        repo_id: Hub model repository holding the ONNX files and ``config.json``.
        variant: Key into the repository config's ``files`` mapping
            (``"float32"`` or ``"int8"``).
        license: Licence as published; the downloaded ``config.json`` is the
            authority.
        description: One line on what the extractor is.
        context_samples: Past audio that determines one output frame, or
            ``None`` when frames cannot be recomputed exactly from a bounded
            window. A ``receptive_field_samples`` entry in the config takes
            precedence.
        revision: Default revision (branch, tag or commit) to download.
    """
    repo_id: str
    variant: str = "float32"
    license: str = ""
    description: str = ""
    context_samples: Optional[int] = None
    revision: Optional[str] = None


_APACHE, _CC_BY_SA, _CC_BY_NC_SA = "apache-2.0", "cc-by-sa-3.0", "cc-by-nc-sa-4.0"

# Streaming context in 320-sample frames, (float32, int8), as measured by
# wakeforge on each published ONNX. None: no bounded context reproduces the
# offline features.
_REPOS = {
    "wakehubert-tiny": (_APACHE, "HuBERT-base student, causal TCN (default)", 125, 125),
    "wakehubert-mel-tcn-deep": (_APACHE, "HuBERT-base student, 16-block causal TCN", 250, 250),
    "wakehubert-mel-tcn-wide": (_APACHE, "HuBERT-base student, wide causal TCN, 256-d", 200, 200),
    "wakehubert-mel-gru": (_APACHE, "HuBERT-base student, causal GRU", None, None),
    "wakehubert-mel-bigru": (_APACHE, "HuBERT-base student, bidirectional GRU (offline)", None, None),
    "wakehubert-mel-attn": (_APACHE, "HuBERT-base student, windowed causal attention", 200, None),
    "wakehubert-mixconv": (_APACHE, "HuBERT-base student, causal mixed-kernel TCN", 250, 250),
    "wakewav-mel-tcn": (_CC_BY_SA, "WavLM-base+ student, causal TCN", 125, 125),
    "wakewav-mel-tcn-wide": (_CC_BY_SA, "WavLM-base+ student, wide causal TCN, 256-d", 200, 200),
    "wakewav-mel-gru": (_CC_BY_SA, "WavLM-base+ student, causal GRU", None, None),
    "wakewav-mel-bigru": (_CC_BY_SA, "WavLM-base+ student, bidirectional GRU (offline)", None, None),
    "wakexeus-mel-tcn": (_CC_BY_NC_SA, "XEUS student, causal TCN", 125, 125),
    "wakexeus-mel-tcn-wide": (_CC_BY_NC_SA, "XEUS student, wide causal TCN, 256-d", 200, 200),
    "wakexeus-mel-gru": (_CC_BY_NC_SA, "XEUS student, causal GRU", None, None),
    "wakexeus-mel-bigru": (_CC_BY_NC_SA, "XEUS student, bidirectional GRU (offline)", None, None),
}


_REVISIONS = {"wakehubert-tiny": "e30726f3a1c28bb5dffadf101fe26c97e0e3c3e1"}


def _entry(repo: str, variant: str, frames: Optional[int]) -> PretrainedFeaturizer:
    licence, description, _, _ = _REPOS[repo]
    return PretrainedFeaturizer(f"TigreGotico/{repo}", variant, licence, description,
                                None if frames is None else frames * 320, _REVISIONS.get(repo))


PRETRAINED_FEATURIZERS: Dict[str, PretrainedFeaturizer] = {}
for _repo, (_, _, _fp32, _int8) in _REPOS.items():
    PRETRAINED_FEATURIZERS[_repo] = _entry(_repo, "float32", _fp32)
    PRETRAINED_FEATURIZERS[f"{_repo}-int8"] = _entry(_repo, "int8", _int8)
PRETRAINED_FEATURIZERS["wakehubert"] = PRETRAINED_FEATURIZERS["wakehubert-tiny"]
PRETRAINED_FEATURIZERS["wakehubert-int8"] = PRETRAINED_FEATURIZERS["wakehubert-tiny-int8"]


def is_non_commercial(licence: str) -> bool:
    """True for Creative Commons NonCommercial licences (``*-nc-*``)."""
    return "-nc" in licence.lower()


def resolve_pretrained(name: str, revision: Optional[str] = None) -> tuple[str, dict]:
    """Return ``(onnx_path, config)`` for a pretrained featurizer.

    The files are downloaded into the shared Hugging Face cache at the
    featurizer's pinned revision, or at ``revision`` when given.
    """
    if name not in PRETRAINED_FEATURIZERS:
        raise ValueError(
            f"Unknown pretrained featurizer {name!r}. "
            f"Available: {', '.join(sorted(PRETRAINED_FEATURIZERS))}"
        )
    entry = PRETRAINED_FEATURIZERS[name]
    rev = revision or entry.revision
    config_path = hf_hub_download(entry.repo_id, "config.json", revision=rev)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    onnx_path = hf_hub_download(entry.repo_id, config["files"][entry.variant], revision=rev)
    return onnx_path, config


@dataclass(frozen=True)
class PretrainedOnnxFeaturizer:
    """A downloaded pretrained featurizer and the facts needed to stream it.

    Carries the attributes that
    :meth:`~ovos_ww_plugin_wakeforge.inference.OnnxStreamingWakeWord.from_extractor`
    reads from wakeforge's ``OnnxFeatureExtractor``, without torch.
    """
    name: str
    model_path: str
    feature_dim: int
    hop_samples: int
    frame_rate_hz: float
    context_samples: int
    streaming: bool
    license: str = ""

    @classmethod
    def from_pretrained(cls, name: str, revision: Optional[str] = None,
                        sample_rate: int = 16000) -> "PretrainedOnnxFeaturizer":
        """Download a named featurizer (e.g. ``"wakehubert"``) from the Hub."""
        onnx_path, config = resolve_pretrained(name, revision)
        entry = PRETRAINED_FEATURIZERS[name]
        licence = config.get("license", entry.license)
        if is_non_commercial(licence):
            logger.warning("Pretrained featurizer %r is licensed %s: non-commercial use only.",
                           name, licence)
        output = config.get("output", {})
        hop = int(output.get("hop_samples", 160))
        context = entry.context_samples
        if context is not None and "receptive_field_samples" in config:
            context = int(config["receptive_field_samples"])
        return cls(
            name=name,
            model_path=onnx_path,
            feature_dim=int(config["feature_dim"]),
            hop_samples=hop,
            frame_rate_hz=float(output.get("frame_rate_hz", sample_rate / hop)),
            context_samples=context or 0,
            streaming=bool(config.get("streaming", False)) and context is not None,
            license=licence,
        )


class OnnxWindowedWakeWord:
    """Score a stream with a batch head over a rolling window of feature frames.

    The batch head ``best_f1.onnx`` that ``ww_trainer-train`` writes scores a
    whole clip of features. Two ways to produce the window it scores:

    - isolated (the default in the plugin): the last ``window`` frames' worth
      of audio is featurized on its own, as the training clips were, so the
      head sees windows like the ones it was trained and calibrated on.
    - with context: each chunk is featurized together with ``context_samples``
      of past audio, so its frames equal the frames of the same audio
      featurized offline, and the head scores the last ``window`` of them.

    Args:
        featurizer_path: Feature extractor ONNX, ``[1, samples] -> [1, frames, F]``.
        head_path: Batch head ONNX, ``[1, frames, F] -> logit``.
        window: Number of feature frames the head scores.
        hop_samples: Featurizer hop; chunks must be a multiple of it.
        context_samples: Past audio that determines one output frame.
        isolated: Featurize each window on its own instead of with context.
        agc: With isolated windows, bring each window's peak to 0.5 with a gain
            limited to 0.25-4x, so a quiet or hot microphone scores like a
            normal one and silence stays quiet.
        threads: Intra-op threads per ONNX session.
    """

    def __init__(self, featurizer_path: str, head_path: str, window: int,
                 hop_samples: int, context_samples: int, isolated: bool = False,
                 agc: bool = False, threads: int = 1) -> None:
        providers, so = ["CPUExecutionProvider"], session_options(threads)
        self.ext = ort.InferenceSession(featurizer_path, so, providers=providers)
        self.head = ort.InferenceSession(head_path, so, providers=providers)
        self._ext_in = self.ext.get_inputs()[0].name
        self._ext_out = self.ext.get_outputs()[0].name
        self._head_in = self.head.get_inputs()[0].name
        self._head_out = self.head.get_outputs()[0].name
        self.window = window
        self.hop = hop_samples
        self.context = context_samples
        self.isolated = isolated
        self.agc = agc
        self.reset()

    @classmethod
    def from_extractor(cls, extractor, head_path: str, window: int, isolated: bool = False,
                       agc: bool = False, threads: int = 1) -> "OnnxWindowedWakeWord":
        """Build a scorer for a :class:`PretrainedOnnxFeaturizer`.

        Raises:
            ValueError: If the featurizer cannot be streamed from a bounded
                window of past audio.
        """
        if not extractor.streaming:
            raise ValueError(
                f"Featurizer {extractor.model_path} is not streamable: its frames depend "
                "on later audio or on unbounded history, so a live stream cannot "
                "reproduce the features the head was trained on.")
        return cls(extractor.model_path, head_path, window=window,
                   hop_samples=extractor.hop_samples,
                   context_samples=extractor.context_samples,
                   isolated=isolated, agc=agc, threads=threads)

    def reset(self) -> None:
        """Clear the carried audio and feature window."""
        self._ctx = np.zeros(0, dtype=np.float32)
        self._audio = np.zeros(self.window * self.hop, dtype=np.float32)
        self._feats: Optional[np.ndarray] = None

    def _featurize(self, audio: np.ndarray) -> np.ndarray:
        return self.ext.run([self._ext_out], {self._ext_in: audio[np.newaxis, :]})[0]

    def push(self, audio_chunk: np.ndarray) -> float:
        """Feed one chunk (a multiple of the hop) and return the head's probability."""
        chunk = audio_chunk.astype(np.float32)
        if self.isolated:
            self._audio = np.concatenate([self._audio, chunk])[-self.window * self.hop:]
            audio = self._audio
            if self.agc:
                audio = audio * np.clip(0.5 / (np.abs(audio).max() + 1e-6), 0.25, 4.0)
            self._feats = self._featurize(audio.astype(np.float32))
        else:
            buf = np.concatenate([self._ctx, chunk])
            feats = self._featurize(buf)
            n_new = min(max(1, len(chunk) // self.hop), feats.shape[1])
            new = feats[:, feats.shape[1] - n_new:, :]
            self._feats = new if self._feats is None else np.concatenate([self._feats, new], axis=1)
            self._feats = self._feats[:, -self.window:, :]
            self._ctx = buf[-self.context:] if self.context else np.zeros(0, dtype=np.float32)
        logit = self.head.run([self._head_out], {self._head_in: self._feats.astype(np.float32)})[0]
        return float(1.0 / (1.0 + np.exp(-float(np.asarray(logit).ravel()[0]))))
