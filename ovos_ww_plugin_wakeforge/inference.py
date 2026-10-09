# Vendored from wakeforge/inference.py (https://github.com/TigreGotico/wakeforge).
# wakeforge is the source of truth — keep this copy in sync on runtime changes.
# Local addition: session_options(), shared_session(), featurize(), StreamingFeaturizer and the `threads` arguments.
# Pure runtime: numpy + onnxruntime only, no torch.
"""ONNX-only wake word inference — no PyTorch dependency at runtime."""
from __future__ import annotations

import hashlib
import math
import os
import threading
import weakref
from collections import OrderedDict, deque
from typing import Optional

import numpy as np
import onnxruntime as ort


def session_options(threads: int = 1) -> ort.SessionOptions:
    """Options for an always-on detector: `threads` intra-op threads, one inter-op
    thread, and no spin-waiting between the small runs a stream makes."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


_SESSIONS: "weakref.WeakValueDictionary[tuple, ort.InferenceSession]" = weakref.WeakValueDictionary()
_OUTPUTS: "weakref.WeakKeyDictionary[ort.InferenceSession, OrderedDict]" = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()
# Windows whose featurizer output is kept per session: one listener chunk makes one new window per
# distinct input, and every hotword on the same featurizer asks for it within that chunk.
RECENT_WINDOWS = 4
# Chunks of a streaming featurizer kept per session, keyed by the stream they continue: every hotword on the same
# stream pushes the same chunk within one listener chunk, and a reset stream primes itself with one more.
_STREAMS: "weakref.WeakKeyDictionary[ort.InferenceSession, OrderedDict]" = weakref.WeakKeyDictionary()
RECENT_STREAM_CHUNKS = 8


def shared_session(path: str, threads: int = 1,
                   level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                   providers: "tuple[str, ...]" = ("CPUExecutionProvider",)) -> ort.InferenceSession:
    """The process's session for a featurizer file, thread count, graph optimisation level and providers.

    Every detector that loads the same featurizer the same way holds the same session, so its weights are
    in memory once; the session is released with the last detector that holds it.
    """
    key = (os.path.realpath(path), threads, level, tuple(providers))
    with _LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            options = session_options(threads)
            options.graph_optimization_level = level
            session = ort.InferenceSession(path, options, providers=list(providers))
            _SESSIONS[key] = session
            _OUTPUTS[session] = OrderedDict()
    return session


def featurize(session: ort.InferenceSession, audio: np.ndarray) -> np.ndarray:
    """Run a featurizer session on ``audio``, reusing its output for input it ran on moments ago.

    Hotwords fed the same audio featurize the same window; only an input with exactly the same shape
    and bytes reuses an output, so a score never changes. The output is read-only.
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    key = (audio.shape, hashlib.blake2b(audio.tobytes(), digest_size=16).digest())
    with _LOCK:
        recent = _OUTPUTS.setdefault(session, OrderedDict())
        feats = recent.get(key)
    if feats is None:
        feats = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: audio})[0]
        feats.flags.writeable = False
        with _LOCK:
            recent[key] = feats
            while len(recent) > RECENT_WINDOWS:
                recent.popitem(last=False)
    return feats


class StreamingFeaturizer:
    """One listener's stream through a stateful featurizer graph.

    The graph (``stream_trunk.py streamify`` in wakeforge) takes ``waveform`` [1, samples], a whole number of hops,
    and one state input per causal layer, and returns ``features`` and the next state of each layer. The session is
    the process's shared one for the file. Fed block by block from the zero state, it returns the frames the source
    featurizer computes over the whole stream at once.

    A stream is named by everything it has been fed since its last reset, so hotwords fed the same audio share one
    featurizer run per chunk: the first to push a chunk computes its frames and next state, and the others read
    them. A stream fed different audio, or reset at another moment, has another name and runs on its own.
    """

    def __init__(self, path: str, hop_samples: int, threads: int = 1) -> None:
        self.session = shared_session(path, threads)
        self.hop = hop_samples
        inputs = self.session.get_inputs()
        self._wav = inputs[0].name
        self._names = [i.name for i in inputs[1:]]
        self._shapes = [i.shape for i in inputs[1:]]
        if not self._names or not all(isinstance(d, int) for s in self._shapes for d in s):
            raise ValueError(f"{path} is not a stateful streaming featurizer: it needs state inputs of fixed shape "
                             "after the waveform")
        self._outputs = [o.name for o in self.session.get_outputs()]
        if len(self._outputs) != len(self._names) + 1:
            raise ValueError(f"{path}: {len(self._names)} state inputs but {len(self._outputs) - 1} state outputs")
        self.reset()

    def reset(self) -> None:
        """Start a new stream: every state at zero, as before the first sample."""
        self.state = [np.zeros(s, dtype=np.float32) for s in self._shapes]
        self._name = b""

    def push(self, audio: np.ndarray) -> np.ndarray:
        """Feature frames ``[1, frames, D]`` of ``audio``, a whole number of hops, continuing the stream. The
        frames and the state are read-only.

        Raises:
            ValueError: If ``audio`` is not a whole number of hops: the graph would drop the remainder and the
                stream would lose its place without any error.
        """
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size % self.hop:
            raise ValueError(f"{audio.size} samples is not a whole number of {self.hop}-sample hops")
        name = hashlib.blake2b(self._name + audio.tobytes(), digest_size=16).digest()
        with _LOCK:
            recent = _STREAMS.setdefault(self.session, OrderedDict())
            hit = recent.get(name)
        if hit is None:
            feeds = {self._wav: audio[np.newaxis, :]}
            feeds.update(zip(self._names, self.state))
            hit = self.session.run(self._outputs, feeds)
            for out in hit:
                out.flags.writeable = False
            with _LOCK:
                recent[name] = hit
                while len(recent) > RECENT_STREAM_CHUNKS:
                    recent.popitem(last=False)
        self._name = name
        feats, *self.state = hit
        return feats


class PredictionSmoother:
    """Smooths raw per-frame predictions for robust wake word detection.

    Inspired by openWakeWord's prediction buffer + patience mechanism
    and EfficientWord-Net's relaxation time.

    Supports three smoothing methods:
    - ``"ema"``: Exponential moving average (fast response, smooth decay).
    - ``"mean"``: Rolling window mean (stable, uniform weighting).
    - ``"max"``: Rolling window max (aggressive, catches peaks).

    Args:
        method: Smoothing method — ``"ema"``, ``"mean"``, or ``"max"``.
        window_size: Number of frames in the rolling buffer (for mean/max).
        threshold: Detection threshold for ``is_triggered()``.
        patience: Number of consecutive above-threshold frames required.
        debounce_sec: Minimum seconds between consecutive triggers.
        frame_rate_hz: Approximate frame rate for debounce timing.
        ema_alpha: Smoothing factor for EMA (default 0.3). Higher = more responsive.
    """

    def __init__(
        self,
        method: str = "ema",
        window_size: int = 5,
        threshold: float = 0.5,
        patience: int = 3,
        debounce_sec: float = 1.0,
        frame_rate_hz: float = 10.0,
        ema_alpha: float = 0.3,
    ) -> None:
        if method not in ("ema", "mean", "max"):
            raise ValueError(f"Unknown smoothing method: {method!r}. Use 'ema', 'mean', or 'max'.")
        self.method = method
        self.window_size = window_size
        self.threshold = threshold
        self.patience = patience
        self.debounce_frames = int(debounce_sec * frame_rate_hz)
        self.ema_alpha = ema_alpha

        self._buffer: deque = deque(maxlen=window_size)
        self._ema_value: float = 0.0
        self._consecutive_above: int = 0
        self._frames_since_trigger: int = self.debounce_frames  # allow first trigger
        self._smoothed: float = 0.0

    def update(self, raw_prob: float) -> float:
        """Feed a new raw prediction and return the smoothed value.

        Args:
            raw_prob: Raw sigmoid probability from model.

        Returns:
            Smoothed probability.
        """
        self._frames_since_trigger += 1

        if self.method == "ema":
            self._ema_value = self.ema_alpha * raw_prob + (1.0 - self.ema_alpha) * self._ema_value
            self._smoothed = self._ema_value
        elif self.method == "mean":
            self._buffer.append(raw_prob)
            self._smoothed = sum(self._buffer) / len(self._buffer)
        elif self.method == "max":
            self._buffer.append(raw_prob)
            self._smoothed = max(self._buffer)

        if self._smoothed >= self.threshold:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0

        return self._smoothed

    def is_triggered(self) -> bool:
        """Check if a detection should fire.

        Returns True when smoothed probability exceeds threshold for
        ``patience`` consecutive frames AND debounce window has elapsed.

        Returns:
            True if wake word detected.
        """
        if (self._consecutive_above >= self.patience
                and self._frames_since_trigger >= self.debounce_frames):
            self._frames_since_trigger = 0
            self._consecutive_above = 0
            return True
        return False

    def reset(self) -> None:
        """Reset smoother state."""
        self._buffer.clear()
        self._ema_value = 0.0
        self._consecutive_above = 0
        self._frames_since_trigger = self.debounce_frames
        self._smoothed = 0.0


class OnnxWakeWordInferencer:
    """Run wake word detection using multiple ONNX sessions.

    This class has no dependency on PyTorch. Only numpy and onnxruntime are required.

    A text featurizer ONNX can optionally be supplied alongside the audio featurizer.
    When present, :meth:`infer` accepts ``text_token_ids`` and appends the resulting
    text embedding as extra feature channels before running the classifier head —
    exactly the same pattern used for the optional VAD channel.

    Args:
        extractor_path: Path to the audio feature extractor ONNX file.
        head_path: Path to the classifier head ONNX file.
        vad_path: Optional path to a VAD ONNX file (e.g. Silero VAD).
        text_extractor_path: Optional path to a text-encoder ONNX file.
            The ONNX must accept ``[1, seq_len]`` int64 token IDs and return
            ``[1, 1, D]`` or ``[1, D]`` float32 embeddings.
        text_emb_dim: Output embedding dimension of the text featurizer (``D``).
        sample_rate: Expected audio sample rate in Hz (default 16000).
        device: ``"cpu"``, ``"cuda"``, or ``"auto"`` (selects CUDA if available).
        threads: Intra-op threads per ONNX session.
    """

    def __init__(self, extractor_path: str, head_path: str,
                 vad_path: Optional[str] = None,
                 text_extractor_path: Optional[str] = None,
                 text_emb_dim: int = 128,
                 sample_rate: int = 16000, device: str = "auto", threads: int = 1) -> None:
        if device == "auto":
            available = ort.get_available_providers()
            device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        so = session_options(threads)
        self.extractor = shared_session(extractor_path, threads, providers=tuple(providers))
        self.head = ort.InferenceSession(head_path, so, providers=providers)
        self.vad = ort.InferenceSession(vad_path, so, providers=providers) if vad_path else None
        self.text_ext = (
            ort.InferenceSession(text_extractor_path, so, providers=providers)
            if text_extractor_path else None
        )
        self._text_emb_dim = text_emb_dim
        self.sample_rate = sample_rate

        self._head_input = self.head.get_inputs()[0].name
        self._head_output = self.head.get_outputs()[0].name

        if self.vad:
            self._vad_input = self.vad.get_inputs()[0].name
            self._vad_output = self.vad.get_outputs()[0].name
        if self.text_ext:
            self._text_input = self.text_ext.get_inputs()[0].name
            self._text_output = self.text_ext.get_outputs()[0].name

    def _run_text(self, token_ids: "list[int]", target_t: int) -> np.ndarray:
        """Encode token IDs and expand to ``[1, target_t, D]``."""
        ids = np.array([token_ids], dtype=np.int64)  # [1, seq_len]
        out = self.text_ext.run([self._text_output], {self._text_input: ids})[0]
        # out may be [1, 1, D] or [1, D]
        if out.ndim == 2:
            out = out[:, np.newaxis, :]   # [1, 1, D]
        return np.repeat(out, target_t, axis=1)  # [1, target_t, D]

    def _run_vad(self, wav: np.ndarray, target_t: int) -> np.ndarray:
        """Run VAD and interpolate to match target time dimension."""
        # Silero VAD (v4+) specific: expects 512 sample chunks
        chunk_size = 512
        B, L = wav.shape
        # Pad to multiple of chunk_size
        pad_len = (chunk_size - (L % chunk_size)) % chunk_size
        if pad_len > 0:
            wav = np.pad(wav, ((0, 0), (0, pad_len)))
        
        L_padded = wav.shape[1]
        T_chunks = L_padded // chunk_size
        
        # Reshape to batch of chunks
        chunks = wav.reshape(B * T_chunks, chunk_size)
        
        # Run VAD
        # Note: some Silero versions expect [B, L] or [B, T, 512]. 
        # We assume [B_total, 512] for consistency with the feats.py wrapper.
        vad_out = self.vad.run([self._vad_output], {self._vad_input: chunks})[0]
        # [B * T_chunks, 1]
        vad_probs = vad_out.reshape(B, T_chunks, 1)
        
        if T_chunks == target_t:
            return vad_probs
            
        # Linear interpolation to match base feature length
        # [B, T_chunks, 1] -> [B, target_t, 1]
        x = np.linspace(0, 1, T_chunks)
        x_new = np.linspace(0, 1, target_t)
        
        aligned = np.zeros((B, target_t, 1), dtype=np.float32)
        for b in range(B):
            # numpy.interp expects 1D arrays
            aligned[b, :, 0] = np.interp(x_new, x, vad_probs[b, :, 0])
            
        return aligned

    def infer(self, audio: np.ndarray,
              text_token_ids: "Optional[list[int]]" = None) -> float:
        """Infer wake word probability for a single audio waveform.

        Args:
            audio: 1-D float32 numpy array at ``self.sample_rate``.
            text_token_ids: Optional list of integer token IDs for the keyword.
                Required when ``text_extractor_path`` was supplied at init.
                The token scheme must match whatever the text-encoder ONNX expects.

        Returns:
            Sigmoid probability in [0, 1].
        """
        if audio.ndim != 1:
            raise ValueError("audio must be a 1-D numpy array")
        wav = audio[np.newaxis, :].astype(np.float32)  # [1, T]
        feats = featurize(self.extractor, wav)  # [1, T, F]

        if self.vad:
            vad_probs = self._run_vad(wav, feats.shape[1])
            feats = np.concatenate([feats, vad_probs], axis=-1)

        if self.text_ext is not None and text_token_ids is not None:
            text_feats = self._run_text(text_token_ids, feats.shape[1])
            feats = np.concatenate([feats, text_feats], axis=-1)  # [1, T, F+D]

        logit = self.head.run(
            [self._head_output], {self._head_input: feats}
        )[0]  # [1] or scalar
        logit_val = float(np.asarray(logit).ravel()[0])
        return float(1.0 / (1.0 + np.exp(-logit_val)))

    def infer_batch(self, audio_batch: np.ndarray,
                    text_token_ids: "Optional[list[int]]" = None) -> np.ndarray:
        """Infer for a batch of equal-length waveforms.

        Args:
            audio_batch: float32 array of shape ``[B, T]``.
            text_token_ids: Optional token IDs shared across the batch.

        Returns:
            Sigmoid probabilities array of shape ``[B]``.
        """
        if audio_batch.ndim != 2:
            raise ValueError("audio_batch must be a 2-D numpy array [B, T]")
        wav = audio_batch.astype(np.float32)
        feats = featurize(self.extractor, wav)

        if self.vad:
            vad_probs = self._run_vad(wav, feats.shape[1])
            feats = np.concatenate([feats, vad_probs], axis=-1)

        if self.text_ext is not None and text_token_ids is not None:
            B = feats.shape[0]
            text_feats = self._run_text(text_token_ids, feats.shape[1])  # [1, T, D]
            text_feats = np.repeat(text_feats, B, axis=0)               # [B, T, D]
            feats = np.concatenate([feats, text_feats], axis=-1)

        logits = self.head.run([self._head_output], {self._head_input: feats})[0]
        logits = np.asarray(logits).ravel()
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def infer_streaming(self, audio_chunk: np.ndarray,
                        cache: np.ndarray | None,
                        smoother: Optional[PredictionSmoother] = None,
                        ) -> tuple[float, np.ndarray]:
        """Process one audio chunk with a rolling feature cache.

        Args:
            audio_chunk: 1-D float32 array (one chunk of audio).
            cache: Previous feature cache [T_cached, F] or None for first call.
            smoother: Optional PredictionSmoother for temporal smoothing.

        Returns:
            (probability, updated_cache) — probability is smoothed if
            smoother is provided, raw otherwise.
        """
        wav = audio_chunk[np.newaxis, :].astype(np.float32)  # [1, T_chunk]
        new_feats = featurize(self.extractor, wav)  # [1, T_new, F]
        
        if self.vad:
            vad_probs = self._run_vad(wav, new_feats.shape[1])
            new_feats = np.concatenate([new_feats, vad_probs], axis=-1)

        new_feats = new_feats.squeeze(0) # [T_new, F_total]

        if cache is None:
            cache = new_feats
        else:
            cache = np.concatenate([cache, new_feats], axis=0)

        # Keep last 50 frames (matching SlidingFeatureCacheTensor default)
        window = 50
        if cache.shape[0] > window:
            cache = cache[-window:]

        feats_input = cache[np.newaxis, :, :]  # [1, T_cached, F]
        logit = self.head.run(
            [self._head_output], {self._head_input: feats_input}
        )[0]
        logit_val = float(np.asarray(logit).ravel()[0])
        prob = float(1.0 / (1.0 + np.exp(-logit_val)))
        if smoother is not None:
            prob = smoother.update(prob)
        return prob, cache


class OnnxStreamingWakeWord:
    """Stateful streaming wake-word inference — ONNX only, O(chunk) per step.

    Pairs a featurizer ONNX with a *streaming* head ONNX exported via
    :meth:`wakeforge.model.GruClassifierHead.export_streaming_onnx`. Unlike
    :meth:`OnnxWakeWordInferencer.infer_streaming` (which re-runs the head over a
    cached feature window each call), this carries the GRU hidden state and a
    sliding window of GRU outputs as model state, so the head never recomputes
    history — the always-on / MCU pattern. No PyTorch at runtime.

    Usage::

        sw = OnnxStreamingWakeWord("feat.onnx", "head_streaming.onnx")
        for chunk in audio_chunks:          # e.g. 0.1 s of 16 kHz audio
            prob = sw.push(chunk)

    Args:
        featurizer_path: Feature extractor ONNX.
        streaming_head_path: Streaming head ONNX (state I/O).
        window: GRU-output window length the head was exported with.
        hidden_dim: GRU hidden size of the head.
        threads: Intra-op threads per ONNX session.
    """

    def __init__(self, featurizer_path: str, streaming_head_path: str,
                 window: int = 100, hidden_dim: int = 128,
                 hop_samples: int = 160, context_samples: int = 640,
                 threads: int = 1) -> None:
        providers, so = ["CPUExecutionProvider"], session_options(threads)
        self.ext = shared_session(featurizer_path, threads)
        self.head = ort.InferenceSession(streaming_head_path, so, providers=providers)
        self.window = window
        self.hidden_dim = hidden_dim
        self.hop = hop_samples            # featurizer hop (for frame accounting)
        self.context = context_samples    # left-context carried for clean MFCC
        self.reset()

    @classmethod
    def from_extractor(cls, extractor, streaming_head_path: str, window: int = 100,
                       hidden_dim: int = 128, threads: int = 1) -> "OnnxStreamingWakeWord":
        """Build a streamer for an :class:`~ww_trainer.feats.OnnxFeatureExtractor`.

        Takes the featurizer path, hop and context from the extractor, e.g. one
        from :meth:`~ww_trainer.feats.OnnxFeatureExtractor.from_pretrained`.

        Raises:
            ValueError: If the extractor cannot be streamed: its frames depend
                on later audio (bidirectional) or on unbounded history
                (recurrent), so re-featurizing a window of past audio would not
                reproduce the features the head was trained on.
        """
        if not extractor.streaming:
            raise ValueError(
                f"Featurizer {extractor.model_path} is not streamable: its frames depend "
                "on later audio or on unbounded history. Score whole windows with "
                "OnnxWakeWordInferencer instead, or pick a streaming featurizer.")
        kwargs = {"context_samples": extractor.context_samples} if extractor.context_samples else {}
        return cls(extractor.model_path, streaming_head_path, window=window,
                   hidden_dim=hidden_dim, hop_samples=extractor.hop_samples, threads=threads,
                   **kwargs)

    def reset(self) -> None:
        """Clear the carried GRU state and audio context (call between utterances)."""
        self._h = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)
        self._ow = np.zeros((1, self.window, self.hidden_dim), dtype=np.float32)
        self._ctx = np.zeros(0, dtype=np.float32)

    def push(self, audio_chunk: np.ndarray) -> float:
        """Feed one audio chunk, advance state frame-by-frame, return the score.

        The chunk is featurized *with* a left-context tail from the previous
        call so MFCC frames are computed with proper context (avoiding chunk-edge
        artifacts); only the frames belonging to the new audio are streamed
        through the stateful head (O(1) per frame). Returns the sigmoid
        probability after the last new frame.
        """
        chunk = audio_chunk.astype(np.float32)
        buf = np.concatenate([self._ctx, chunk])
        wav = buf[np.newaxis, :]
        feats = featurize(self.ext, wav)  # [1, T, F]
        # Frames attributable to the new chunk (the rest came from context).
        n_new = max(1, int(round(len(chunk) / self.hop)))
        n_new = min(n_new, feats.shape[1])
        self._ctx = buf[-self.context:] if self.context else np.zeros(0, dtype=np.float32)
        logit_val = 0.0
        for t in range(feats.shape[1] - n_new, feats.shape[1]):
            frame = feats[:, t:t + 1, :].astype(np.float32)
            logit, self._h, self._ow = self.head.run(
                None, {"feat_frame": frame, "h_in": self._h, "out_window": self._ow})
            logit_val = float(np.asarray(logit).ravel()[0])
        return float(1.0 / (1.0 + np.exp(-logit_val)))


def cli_main() -> None:
    """CLI entry point for ``wakeforge-infer``.

    Run a trained ONNX wake-word model on an audio file and print the
    confidence score.
    """
    import argparse
    import logging
    import sys

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="wakeforge-infer",
        description="Score an audio file with a trained ONNX wake-word model.",
    )
    parser.add_argument("--featurizer", required=True, help="Path to featurizer ONNX file.")
    parser.add_argument("--model", required=True, help="Path to head ONNX file.")
    parser.add_argument("--audio", required=True, help="Path to WAV file (16 kHz mono float32).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (default 0.5).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        try:
            import torchaudio
            wav, sr = torchaudio.load(args.audio)
            if sr != 16000:
                import torchaudio.functional as F_ta
                wav = F_ta.resample(wav, sr, 16000)
            audio = wav.squeeze().numpy().astype(np.float32)
        except ImportError:
            print("Install soundfile or torchaudio to load audio files.", file=sys.stderr)
            sys.exit(1)
    else:
        audio, sr = sf.read(args.audio, dtype="float32", always_2d=False)
        if sr != 16000:
            print(f"Warning: sample rate is {sr} Hz, expected 16000.", file=sys.stderr)

    model = OnnxWakeWordInferencer(args.featurizer, args.model, device=args.device)
    score = model.infer(audio)
    detected = score >= args.threshold
    print(f"score={score:.4f}  threshold={args.threshold}  detected={detected}")
    sys.exit(0 if detected else 1)


if __name__ == "__main__":
    cli_main()
