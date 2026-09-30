# ovos-ww-plugin-wakeforge

An OVOS wake-word plugin for custom models trained with
[wakeforge](https://github.com/TigreGotico/wakeforge).

Wakeforge trains a wake-word detector from a single phrase and exports a two-file
ONNX pipeline: a feature extractor and a classifier head. This plugin loads that
pipeline and runs it as an always-on hotword engine. The runtime uses
`onnxruntime` and `numpy` only. It does not need PyTorch.

## Install

```bash
pip install --pre ovos-ww-plugin-wakeforge
```

## Ready models: alexa and hey mycroft

The package ships two ready models and the featurizer they run on, so they work offline
with nothing else to download. Put one of these in `mycroft.conf`:

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

```json
{
  "listener": {
    "wake_word": "alexa"
  },
  "hotwords": {
    "alexa": {
      "module": "ovos-ww-plugin-wakeforge",
      "model": "alexa",
      "listen": true
    }
  }
}
```

Each model is a small GRU head on features from
[WakeHuBERT tiny](https://huggingface.co/TigreGotico/wakehubert-tiny), a 0.64M-parameter
speech feature extractor distilled from HuBERT-base (Apache-2.0, bundled in
`ovos_ww_plugin_wakeforge/featurizers/`). The heads were trained on synthetic speech only,
from CC-BY-4.0 synthetic wake-word datasets (see `ovos_ww_plugin_wakeforge/models/NOTICE`),
and calibrated, so the score is the probability that the last 1.5 s holds the wake word.
A detection fires when that probability reaches the trigger, 0.99 for `alexa` and 0.98
for `hey_mycroft` by default, and no second detection follows within 2 s.

The triggers were chosen on LibriSpeech dev-clean. On audio they were not chosen on, two
separate runs streamed the defaults in 2048-byte chunks and counted the same false
activations:

| held-out audio | `alexa` | `hey_mycroft` |
|---|---|---|
| LibriSpeech test-clean, 1.0 h | 1 | 2 |
| LibriSpeech test-other, 0.5 h | 1 | 2 |
| AudioSet noise, 0.5 h | 0 | 0 |

That is about 1.3 false activations per hour for `alexa` and 2.7 for `hey_mycroft` on read
speech. Set `"threshold"` to change the trade-off: lower fires more readily and falsely
more often. With a microphone whose gain is far too high or too low, `"agc": true` levels
each window before scoring it.

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
| `featurizer` | — (required) | feature-extractor ONNX (path or URL), or a pretrained featurizer name such as `wakehubert`; optional when the model names its featurizer |
| `featurizer_revision` | bundled or latest | Hugging Face revision (branch, tag or commit) of a pretrained featurizer; a revision other than the bundled one is downloaded |
| `model` | — (required) | classifier-head ONNX (path or URL), or a ready model: `alexa`, `hey_mycroft` |
| `vad` | none | optional VAD ONNX for an extra channel |
| `threshold` | `0.5` | detection threshold (ready models: `0.99` alexa, `0.98` hey_mycroft) |
| `smoothing` | `ema` | `ema` \| `mean` \| `max` (ready models: `max`) |
| `patience` | `3` | consecutive above-threshold frames to fire (ready models: `1`) |
| `debounce_sec` | `1.0` | minimum seconds between detections (ready models: `2.0`) |
| `window_size` | `5` | smoother rolling window (mean/max) (ready models: `1`) |
| `ema_alpha` | `0.3` | EMA responsiveness |
| `block_ms` | `80` | audio is scored in blocks of this length whatever chunk size the listener sends, so patience counts time and the featurizer always gets a full block |
| `onnx_threads` | `1` | intra-op threads per ONNX session; sessions never spin-wait |
| `streaming` | `false` | use the stateful streaming (GRU) head; with a pretrained featurizer the head type is read from the model |
| `gru_window` | `100` | frames the head averages over; with a pretrained featurizer it is read from a streaming head, or from a batch head's `window_frames` metadata, else 1.5 s of frames |
| `isolated_windows` | `true` | pretrained featurizer with a batch head: featurize each window on its own, as training clips were; `false` featurizes with the past audio each frame depends on |
| `agc` | `false` | with isolated windows, level each window (peak to 0.5, gain 0.25–4x) before scoring |
| `hidden_dim` | `128` | GRU hidden size of the streaming head |

URL models are cached under `${XDG_DATA_HOME}/wakeforge/`.

## Models on a pretrained featurizer

wakeforge can train a head on a pretrained featurizer downloaded from the Hugging Face Hub,
such as WakeHuBERT (`ww_trainer-train --tier wakehubert`, or `--featurizer-type wakehubert-int8`
or any other published name). Such a model has no featurizer file of its own. Name the
featurizer instead of a path:

```json
"hey_computer": {
  "module": "ovos-ww-plugin-wakeforge",
  "featurizer": "wakehubert",
  "model": "~/hey_computer/head_streaming.onnx",
  "threshold": 0.5
}
```

`wakehubert` and `wakehubert-int8` are bundled with the package and need no network. Other
names are downloaded into the shared Hugging Face cache on first use, so a device that runs
offline needs them cached once. When the model's ONNX metadata carries a
`pretrained_featurizer` entry (or a `featurizer` entry holding a pretrained name), that name
wins over the config key, and a `featurizer_revision` entry pins the revision unless the
config sets one.

`model` can be the batch head that training writes (`best_f1.onnx`) or the streaming head
exported from it with `export_streaming_onnx`; the plugin tells them apart by their inputs.
A batch head scores the last `gru_window` frames. By default each such window is featurized
on its own, the way wakeforge featurizes its training clips, so the head is scored on the kind
of window it was trained and calibrated on. With `"isolated_windows": false` each block is
instead featurized together with the past audio its frames depend on (2.5 s for
`wakehubert`), so the frames equal those of the whole stream featurized at once. For the
bundled heads the two agree on which wake words they detect, but on speech without the wake
word the context path gives more high scores, so isolated windows are the default. The
streaming head always carries context, as it was exported to. Blocks are rounded up to whole
featurizer frames (320 samples).

Featurizers whose frames depend on later audio or on unbounded history (the `-gru` and
`-bigru` extractors, `wakehubert-mel-attn-int8`) cannot run on a live stream and are refused
when the plugin loads. The `vad` channel is not available with a pretrained featurizer.

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
| `precise-onnx` | the OVOS precise-onnx plugin with its hey mycroft model (no alexa model exists) |

Each engine has a row with a bar showing its wake-word probability and a `|` where it
triggers. When a bar crosses its marker the row turns green with "WAKE WORD!", the
detection is counted and logged under "recent detections", and that engine's detections
are ignored for 2 s. A meter shows the microphone level and warns when the input clips.
An engine that fails to load is listed in a red "not loaded" row with the reason, and an
engine that fails while running shows the error in its own row.

Options: `--wakeword alexa|hey_mycroft`, `--threshold` (WakeHuBERT trigger probability),
`--oww-threshold` (default 0.5), `--mww-cutoff` (default: the model's shipped cutoff),
`--precise-sensitivity` (default 0.5), `--only wakehubert,openwakeword,...`, `--device`
(see `python -m sounddevice`), `--agc` (level each WakeHuBERT window) and `--file clip.wav`,
which scores a 16 kHz recording instead of the microphone and prints each engine's
detection times.

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
