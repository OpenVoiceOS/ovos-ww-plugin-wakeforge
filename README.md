# ovos-ww-plugin-wakeforge

An OVOS wake-word plugin for custom models trained with
[wakeforge](https://github.com/TigreGotico/wakeforge).

Wakeforge trains a wake-word detector from a single phrase and exports a two-file
ONNX pipeline: a feature extractor and a classifier head. This plugin loads that
pipeline and runs it as an always-on hotword engine. The runtime uses
`onnxruntime` and `numpy` only. It does not need PyTorch.

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

In `mycroft.conf`, point a hotword at the two ONNX files. Use local paths or URLs.

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

## Related projects

- [TigreGotico/wakeforge](https://github.com/TigreGotico/wakeforge) trains the ONNX models this plugin loads.
- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager) loads and manages this plugin alongside other STT, TTS, and wake-word plugins.
- [OpenVoiceOS/ovos-ww-plugin-precise-onnx](https://github.com/OpenVoiceOS/ovos-ww-plugin-precise-onnx) is a sibling wake-word plugin that runs Precise models through ONNX.

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).

---

## License

Apache-2.0
