# ovos-ww-plugin-wakeforge

An OpenVoiceOS wake-word plugin. It runs wake-word models made with
[wakeforge](https://github.com/TigreGotico/wakeforge). The models download from Hugging Face on first use.

## Install

```bash
pip install --pre ovos-ww-plugin-wakeforge
```

## Try it in your browser

Everything runs locally on your machine.

- [WakeHuBERT wake words](https://huggingface.co/spaces/OpenVoiceOS/wakehubert-wakewords-space): every ready model, on your microphone or an uploaded file.
- [WakePhoneHuBERT](https://huggingface.co/spaces/TigreGotico/wakephonehubert-space): spot a keyword you type, with no trained model.

## Try it on your laptop

The demo listens to your microphone. It runs a ready model next to openWakeWord, microWakeWord and Precise, so you can compare them. Use `--help` for the options.

```bash
pip install --pre "ovos-ww-plugin-wakeforge[demo]"
ovos-wakeforge-demo --wakeword hey_mycroft
```

![The live demo: one probability bar per engine](https://raw.githubusercontent.com/OpenVoiceOS/ovos-ww-plugin-wakeforge/dev/docs/demo.svg)

## Use a ready model

Add this to `mycroft.conf`:

```json
{
  "listener": {
    "wake_word": "hey_mycroft"
  },
  "hotwords": {
    "hey_mycroft": {
      "module": "ovos-ww-plugin-wakeforge",
      "model": "wakehubert_hey_mycroft",
      "listen": true
    }
  }
}
```

Each ready model is named `wakehubert_<word>`. The model card lists all words, thresholds, test results and training data: https://huggingface.co/OpenVoiceOS/wakehubert-wakewords.

## Threshold

Calibrated models make `threshold` a sensitivity setting. A higher value gives fewer false activations and fewer detections. A lower value gives more of both. Each model has its own default threshold. Set `"threshold"` in the hotword config to change it.

## WakePhoneHuBERT

[WakePhoneHuBERT](https://huggingface.co/TigreGotico/wakephonehubert) is a second featurizer, named `wakephonehubert-int8`. It adds a voice-activity head and an IPA phoneme head to the WakeHuBERT trunk. Models trained on it are named `wakephonehubert_<word>`. They name their featurizer in their own metadata, so `model` is the only key you need. The plugin downloads the featurizer on first use.

## Zero-shot wake words

The `ovos-ww-plugin-wakeforge-zeroshot` engine detects a wake word from its IPA phones alone, with no recordings and no trained model. It scores WakePhoneHuBERT's phoneme output for the phones in `ipa`, every 80 ms over the last 1.5 s of audio.

```json
"hotwords": {
  "hey_jarvis": {"module": "ovos-ww-plugin-wakeforge-zeroshot", "ipa": "h eɪ dʒ ɑːɹ v ɪ s"}
}
```

`ipa` can be a list of pronunciations. Get the phones once with eSpeak NG, which the default thresholds were calibrated with: `espeak-ng -q --ipa -v en-us "hey jarvis"`. The plugin needs no phonemiser. The default threshold depends on the number of phones and is a starting point: set `"threshold"` (0 is a perfect match) to tune it. Use a phrase of at least six phones, ideally two words. Short single words are unreliable: train a model for them.

## Use your own model

Set `model` to the classifier ONNX that wakeforge exported. Set `featurizer` to its feature-extractor ONNX. Both can be local paths or URLs. A model trained on a pretrained featurizer names it in its metadata, so you can leave `featurizer` out. Train models with [wakeforge](https://github.com/TigreGotico/wakeforge).

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

| key | default | meaning |
|-----|---------|---------|
| `model` | required | classifier ONNX (path or URL), or a ready model name |
| `featurizer` | none | feature-extractor ONNX (path or URL), or a pretrained name such as `wakehubert`; optional when the model names one |
| `threshold` | `0.5` | detection threshold; a ready model or calibrated head uses its own default |
| `debounce_sec` | `1.0` | minimum seconds between detections (`2.0` for a calibrated head) |
| `agc` | `false` | level each audio window before scoring; helps when the microphone gain is far too high or low |
| `models_revision` | pinned | Hugging Face revision of the ready-model repository |
| `featurizer_revision` | pinned | Hugging Face revision of a pretrained featurizer |
| `onnx_threads` | `1` | threads per ONNX session |

The source documents the other keys (`smoothing`, `patience`, `window_size`, `ema_alpha`, `block_ms`, `streaming`, `gru_window`, `isolated_windows`, `hidden_dim`, `vad`) in `ovos_ww_plugin_wakeforge/__init__.py`. URL models are cached under `${XDG_DATA_HOME}/wakeforge/`.

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
