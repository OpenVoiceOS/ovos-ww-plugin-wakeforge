# ovos-ww-plugin-wakeforge

An OpenVoiceOS wake-word plugin that runs models made with
[wakeforge](https://github.com/TigreGotico/wakeforge). It ships ready models for
"hey mycroft", "alexa" and "wake up", together with the feature extractor they run on, so it works
offline with nothing else to download. The runtime needs only `onnxruntime` and `numpy`.

This plugin only runs models. To train a model for your own wake word, use
[wakeforge](https://github.com/TigreGotico/wakeforge), a wake-word research framework.

## Install

```bash
pip install --pre ovos-ww-plugin-wakeforge
```

## Use a ready model

Put this in `mycroft.conf`:

```json
{
  "listener": {
    "wake_word": "hey_mycroft"
  },
  "hotwords": {
    "hey_mycroft": {
      "module": "ovos-ww-plugin-wakeforge",
      "model": "hey_mycroft",
      "listen": true
    }
  }
}
```

For the other ready models, set `model` to `alexa` or `wake_up`, and name the hotword to match.

Each ready model is a small classifier on features from
[WakeHuBERT tiny](https://huggingface.co/TigreGotico/wakehubert-tiny), a 0.64M-parameter
speech feature extractor distilled from HuBERT-base. The score is a calibrated probability
that the last 1.5 s holds the wake word. Each model fires at its own default trigger (0.965 for `hey_mycroft`,
0.99 for `alexa`, 0.990 for `wake_up`), and after a detection it stays quiet for 2 s. Set
`"threshold"` to change the trade-off: a lower value fires more readily and falsely more often. With a
microphone whose gain is far too high or too low, `"agc": true` levels the audio before scoring.
Model licences and training data are listed in `ovos_ww_plugin_wakeforge/models/NOTICE`.

## Use your own model

Point `model` at the classifier ONNX that wakeforge exported, and `featurizer` at its feature
extractor. Both can be local paths or URLs. A model trained on a pretrained featurizer names
that featurizer in its own metadata, so `featurizer` can be left out. A model whose metadata
carries a calibrated `default_threshold` uses it, together with the ready-model smoothing.

```json
{
  "listener": {
    "wake_word": "hey_jarvis"
  },
  "hotwords": {
    "hey_jarvis": {
      "module": "ovos-ww-plugin-wakeforge",
      "listen": true,
      "featurizer": "~/hey_jarvis/featurizer.onnx",
      "model": "~/hey_jarvis/model.onnx"
    }
  }
}
```

## Config keys

| key | default | description |
|-----|---------|-------------|
| `featurizer` | — (required) | feature-extractor ONNX (path or URL), or a pretrained featurizer name such as `wakehubert`; optional when the model names its featurizer |
| `featurizer_revision` | bundled or latest | Hugging Face revision (branch, tag or commit) of a pretrained featurizer; a revision other than the bundled one is downloaded |
| `model` | — (required) | classifier-head ONNX (path or URL), or a ready model: `alexa`, `hey_mycroft`, `wake_up` |
| `vad` | none | optional VAD ONNX for an extra channel |
| `threshold` | `0.5` | detection threshold; a ready model or any head whose ONNX metadata carries `default_threshold` uses that instead (`0.99` alexa, `0.965` hey_mycroft, `0.990` wake_up) |
| `smoothing` | `ema` | `ema` \| `mean` \| `max` (a calibrated head: `max`) |
| `patience` | `3` | consecutive above-threshold frames to fire (a calibrated head: `1`) |
| `debounce_sec` | `1.0` | minimum seconds between detections (a calibrated head: `2.0`) |
| `window_size` | `5` | smoother rolling window (mean/max) (a calibrated head: `1`) |
| `ema_alpha` | `0.3` | EMA responsiveness |
| `block_ms` | `80` | audio is scored in blocks of this length whatever chunk size the listener sends, so patience counts time and the featurizer always gets a full block |
| `onnx_threads` | `1` | intra-op threads per ONNX session; sessions never spin-wait |
| `streaming` | `false` | use the stateful streaming (GRU) head; with a pretrained featurizer the head type is read from the model |
| `gru_window` | `100` | frames the head averages over; with a pretrained featurizer it is read from a streaming head, or from a batch head's `window_frames` metadata, else 1.5 s of frames |
| `isolated_windows` | `true` | pretrained featurizer with a batch head: featurize each window on its own, as training clips were; `false` featurizes with the past audio each frame depends on |
| `agc` | `false` | with isolated windows, level each window (peak to 0.5, gain 0.25–4x) before scoring |
| `hidden_dim` | `128` | GRU hidden size of the streaming head |

URL models are cached under `${XDG_DATA_HOME}/wakeforge/`.

Pretrained featurizers other than the bundled `wakehubert` and `wakehubert-int8` are downloaded from the Hugging Face Hub on first use. A featurizer that depends on future audio, or on unbounded history, cannot run on a live stream and is refused at load time.

## Compare against other engines

`ovos-wakeforge-demo` listens to the microphone and runs the ready model next to three
other wake-word engines, each fed the same 80 ms chunks:

```bash
pip install --pre "ovos-ww-plugin-wakeforge[demo]"
ovos-wakeforge-demo --wakeword hey_mycroft
```

| engine | what runs |
|---|---|
| `wakehubert` | this plugin with its ready model for the word |
| `openwakeword` | the `openwakeword` library with its pretrained model for the word |
| `microwakeword` | the `pymicro-wakeword` library with its v2 model for the word |
| `precise-onnx` | the OVOS precise-onnx plugin with its hey mycroft model (no alexa or wake_up model exists) |

`openwakeword` and `microwakeword` have no `wake_up` model either; running `--wakeword
wake_up` shows those two (and `precise-onnx`) in the red "not loaded" row, with `wakehubert`
the only bar on the panel.

Each engine has a row with a bar showing its wake-word probability and a `|` where it
triggers. When a bar crosses its marker the row turns green with "WAKE WORD!", the
detection is counted and logged under "recent detections", and that engine's detections
are ignored for 2 s. A meter shows the microphone level and warns when the input clips.
An engine that fails to load is listed in a red "not loaded" row with the reason, and an
engine that fails while running shows the error in its own row.

Options: `--wakeword alexa|hey_mycroft|wake_up`, `--threshold` (WakeHuBERT trigger probability),
`--oww-threshold` (default 0.5), `--mww-cutoff` (default: the model's shipped cutoff),
`--precise-sensitivity` (default 0.5), `--only wakehubert,openwakeword,...`, `--device`
(see `python -m sounddevice`), `--agc` (level each WakeHuBERT window) and `--file clip.wav`,
which scores a 16 kHz recording instead of the microphone and prints each engine's
detection times.

## Related projects

- [TigreGotico/wakeforge](https://github.com/TigreGotico/wakeforge) is the wake-word research framework that trains the models this plugin runs.
- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager) loads this plugin alongside other speech-to-text, text-to-speech and wake-word plugins.
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
