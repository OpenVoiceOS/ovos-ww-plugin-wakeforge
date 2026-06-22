# ovos-ww-plugin-wakeforge

OVOS wake-word plugin for custom models trained with
[wakeforge](https://github.com/TigreGotico/wakeforge).

wakeforge trains a wake-word detector from a single phrase and exports a two-file
ONNX pipeline — a feature extractor and a classifier head. This plugin loads that
pipeline and runs it as an always-on hotword engine. Runtime is pure
`onnxruntime` + `numpy` (no PyTorch).

## Install

```bash
pip install ovos-ww-plugin-wakeforge
```

## Train a model

```bash
pip install wakeforge
wakeforge-quickstart "hey jarvis" ./hey_jarvis
# → ./hey_jarvis/best_f1_featurizer.onnx  +  ./hey_jarvis/best_f1.onnx
```

## Configure

In `mycroft.conf`, point a hotword at the two ONNX files (local paths or URLs):

```json
{
  "listener": {
    "wake_word": "hey_jarvis"
  },
  "hotwords": {
    "hey_jarvis": {
      "module": "ovos-ww-plugin-wakeforge",
      "listen": true,
      "featurizer": "~/hey_jarvis/best_f1_featurizer.onnx",
      "model": "~/hey_jarvis/best_f1.onnx",
      "threshold": 0.5
    }
  }
}
```

### Config keys

| key | default | description |
|-----|---------|-------------|
| `featurizer` | — (required) | feature-extractor ONNX (path or URL) |
| `model` | — (required) | classifier-head ONNX (path or URL) |
| `vad` | none | optional VAD ONNX for an extra channel |
| `threshold` | `0.5` | detection threshold |
| `smoothing` | `ema` | `ema` \| `mean` \| `max` |
| `patience` | `3` | consecutive above-threshold frames to fire |
| `debounce_sec` | `1.0` | minimum seconds between triggers |
| `window_size` | `5` | smoother rolling window (mean/max) |
| `ema_alpha` | `0.3` | EMA responsiveness |
| `streaming` | `false` | use the stateful streaming (GRU) head |
| `gru_window` | `100` | window the streaming head was exported with |
| `hidden_dim` | `128` | GRU hidden size of the streaming head |

URL models are cached under `${XDG_DATA_HOME}/wakeforge/`.

## Credits

Developed by [TigreGótico](https://github.com/TigreGotico) for
[OpenVoiceOS](https://openvoiceos.org). Funded by the NGI0 Commons Fund / NLnet.
