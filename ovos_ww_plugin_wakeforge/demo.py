"""Live wake-word comparison: WakeHuBERT next to openWakeWord, microWakeWord and precise-onnx.

    ovos-wakeforge-demo [--wakeword alexa|hey_mycroft] [--threshold PROB] [--only NAMES] [--file clip.wav]

Every engine hears the same microphone audio in 80 ms chunks, as a live assistant would feed it:

  wakehubert       this plugin with its ready model for the word, on its WakeHuBERT featurizer;
                   the score is the probability that the last 1.5 s holds the wake word
  openwakeword     the openwakeword library with its pretrained model for the word
  microwakeword    the pymicro-wakeword library with its v2 model for the word, at its shipped cutoff
  precise-onnx     the OVOS precise-onnx plugin with its hey mycroft model (no alexa model exists)

Each engine is shown on the same 0-100% scale with a marker where it triggers; crossing the marker is a
detection, after which that engine's detections are ignored for 2 s. --file scores a wav instead of the
microphone and prints the detection times. Needs the demo extra:
pip install "ovos-ww-plugin-wakeforge[demo]".
"""
import argparse
import queue
import time

import numpy as np

from ovos_ww_plugin_wakeforge import model_names

SAMPLE_RATE = 16000
CHUNK = 1280  # 80 ms, the block openWakeWord predicts on
REFRACTORY_SEC = 2.0
ENGINES = ("wakehubert", "openwakeword", "microwakeword", "precise-onnx")
PRECISE_MODEL = ("https://github.com/OpenVoiceOS/precise-lite-models/raw/master/"
                 "wakewords/en/{}.onnx")


class Engine:
    """One wake-word engine: ``feed`` a chunk, get a detection; ``score`` is 0-1 or None."""
    name = ""
    line = 0.5
    score = None
    error = None

    def feed(self, pcm: bytes) -> bool:
        raise NotImplementedError


class WakeHuBERT(Engine):
    """This plugin with its ready model for the word."""
    name = "wakehubert"

    def __init__(self, word, threshold=None, agc=False):
        from ovos_ww_plugin_wakeforge import WakeForgeHotwordPlugin
        config = {"model": f"wakehubert_{word}", "agc": agc}
        if threshold is not None:
            config["threshold"] = threshold
        self.plugin = WakeForgeHotwordPlugin(word, config)
        self.line = self.plugin.threshold

    def feed(self, pcm):
        self.plugin.update(pcm)
        self.score = self.plugin.last_score
        return self.plugin.found_wake_word()


class OpenWakeWord(Engine):
    """The openwakeword library with its pretrained model for the word."""
    name = "openwakeword"

    def __init__(self, word, threshold=0.5):
        from pathlib import Path

        import openwakeword
        from openwakeword.model import Model
        model = Path(openwakeword.__file__).parent / "resources" / "models" / f"{word}_v0.1.onnx"
        if not model.exists():
            from openwakeword.utils import download_models
            download_models([model.stem])
        self.model = Model(wakeword_models=[str(model)], inference_framework="onnx")
        self.line = threshold

    def feed(self, pcm):
        self.score = float(next(iter(self.model.predict(np.frombuffer(pcm, "<i2")).values())))
        if self.score < self.line:
            return False
        self.model.reset()
        return True


class MicroWakeWord(Engine):
    """The pymicro-wakeword library with its v2 model for the word."""
    name = "microwakeword"

    def __init__(self, word, cutoff=None):
        from pymicro_wakeword import MicroWakeWord as Mww, MicroWakeWordFeatures, Model
        self.mww = Mww.from_builtin(Model(word))
        if cutoff is not None:
            self.mww.probability_cutoff = cutoff
        self.features = MicroWakeWordFeatures()
        self.line = self.mww.probability_cutoff

    def feed(self, pcm):
        hit = False
        for frame in self.features.process_streaming(pcm):
            hit |= bool(self.mww.process_streaming(frame))
            if self.mww._probabilities:
                self.score = float(np.mean(self.mww._probabilities))
        return hit


class PreciseOnnx(Engine):
    """The OVOS precise-onnx plugin."""
    name = "precise-onnx"

    def __init__(self, word, sensitivity=0.5):
        if word != "hey_mycroft":
            raise ValueError(f"no precise-onnx model for '{word}'")
        from ovos_plugin_manager.wakewords import load_wake_word_plugin
        self.plugin = load_wake_word_plugin("ovos-ww-plugin-precise-onnx")(
            word, {"model": PRECISE_MODEL.format(word), "sensitivity": sensitivity,
                   "trigger_level": 3})
        self.line = 1.0 - sensitivity
        engine_update = self.plugin.engine.update

        def update(stream):
            self.score = float(engine_update(stream))
            return self.score

        self.plugin.engine.update = update

    def feed(self, pcm):
        self.plugin.update(pcm)
        return self.plugin.found_wake_word()


def build_engines(args):
    """Load the selected engines; return them and a list of 'name: reason' for those that failed."""
    makers = {
        "wakehubert": lambda: WakeHuBERT(args.wakeword, args.threshold, args.agc),
        "openwakeword": lambda: OpenWakeWord(args.wakeword, args.oww_threshold),
        "microwakeword": lambda: MicroWakeWord(args.wakeword, args.mww_cutoff),
        "precise-onnx": lambda: PreciseOnnx(args.wakeword, args.precise_sensitivity),
    }
    only = set(filter(None, (args.only or "").split(",")))
    engines, skipped = [], []
    for name, make in makers.items():
        if only and name not in only:
            continue
        try:
            engines.append(make())
        except Exception as e:  # one missing engine must not stop the comparison
            skipped.append(f"{name}: {type(e).__name__}: {str(e)[:100]}")
    return engines, skipped


class Comparison:
    """Feeds every engine the same chunk and keeps detections, counts and the log."""

    def __init__(self, engines):
        self.engines = engines
        self.counts = {e.name: 0 for e in engines}
        self.quiet_until = {e.name: 0.0 for e in engines}
        self.last_hit = {}
        self.log = []

    def step(self, pcm, now):
        hits = []
        for e in self.engines:
            try:
                fired = e.feed(pcm)
            except Exception as ex:  # the row shows the error; the other engines go on
                fired, e.score, e.error = False, None, f"{type(ex).__name__}: {ex}"
            if fired and now >= self.quiet_until[e.name]:
                self.quiet_until[e.name] = now + REFRACTORY_SEC
                self.counts[e.name] += 1
                self.last_hit[e.name] = now
                hits.append(e)
        if hits:
            self.log.append(f"{time.strftime('%H:%M:%S')}  {now:6.1f}s  " + ", ".join(
                e.name if e.score is None else f"{e.name} ({100 * min(e.score, 1.0):.0f}%)"
                for e in hits))
        return hits


def score_file(path, engines):
    """Feed a wav file (plus 1 s of silence) through the engines; return detection times per engine."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr} Hz; the engines need {SAMPLE_RATE} Hz")
    audio = np.concatenate([audio.mean(axis=1), np.zeros(SAMPLE_RATE, np.float32)])
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    comparison, found = Comparison(engines), {e.name: [] for e in engines}
    for i in range(len(pcm) // (CHUNK * 2)):
        now = i * CHUNK / SAMPLE_RATE
        for e in comparison.step(pcm[i * CHUNK * 2:(i + 1) * CHUNK * 2], now):
            found[e.name].append(round(now, 2))
    return found


def render(comparison, skipped, word, level_db, clipping, now):
    """The live panel: help line, mic meter, one bar per engine, recent detections."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    width = 40
    table = Table(expand=True, show_edge=False, box=None, padding=(0, 1))
    table.add_column("engine", width=21)
    table.add_column("probability", ratio=1)
    table.add_column("", width=8, justify="right")
    table.add_column("detections", width=10, justify="right")
    table.add_column("", width=14)
    for e in comparison.engines:
        score = None if e.score is None else float(np.clip(e.score, 0, 1))
        mark = min(int(e.line * width), width - 1)
        cells = ["|" if i == mark else ("█" if score is not None and i < int(score * width) else "·")
                 for i in range(width)]
        hot = now - comparison.last_hit.get(e.name, -99.0) < 1.5
        style = "bold green" if hot else ("yellow" if score is not None and score >= 0.8 * e.line else "white")
        bar = Text(f"error: {e.error}"[:60], style="red") if e.error else Text("".join(cells), style=style)
        shown = "—" if score is None else (f"{100 * score:5.1f}%" if score < 0.995 else f"{100 * score:6.2f}%")
        status = (Text("WAKE WORD!", style="bold black on green") if hot
                  else Text(f"trigger {100 * e.line:.4g}%", style="dim"))
        table.add_row(Text(e.name, style="bold"), bar, shown, str(comparison.counts[e.name]), status)
    if skipped:
        table.add_row(Text("not loaded", style="bold red"), Text("; ".join(skipped)[:200], style="red"),
                      "", "", "")
    level = int(np.clip((level_db + 60) / 60, 0, 1) * width)
    mic = Text("mic   " + "█" * level + "·" * (width - level) + f"  {level_db:5.0f} dBFS", style="cyan")
    if clipping > 0.001:
        mic.append(f"   CLIPPING ({100 * clipping:.1f}% of samples at full scale) - lower the mic gain",
                   style="bold red")
    spoken = word.replace("_", " ")
    help_line = Text(f'Say "{spoken}". Each bar is an engine\'s wake-word probability; | is where it '
                     "triggers. Crossing it is a detection (the row turns green), then that engine's "
                     "detections are ignored for 2 s. Ctrl+C to quit.", style="dim")
    events = Text("\n".join(comparison.log[-6:]) or "no detections yet", style="" if comparison.log else "dim")
    return Panel(Group(help_line, Text(""), mic, Text(""), table, Text(""),
                       Text("recent detections", style="bold"), events),
                 title=f"wake-word live test — {spoken}", border_style="blue")


def listen(engines, skipped, word, device=None):
    """Run the live comparison on the microphone until Ctrl+C."""
    import sounddevice as sd
    from rich.live import Live
    comparison, chunks = Comparison(engines), queue.Queue()
    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK,
                           device=device, callback=lambda data, *_: chunks.put(bytes(data))):
        start, clipping = time.monotonic(), 0.0
        with Live(render(comparison, skipped, word, -90.0, 0.0, 0.0), refresh_per_second=12) as live:
            try:
                while True:
                    pcm = chunks.get()
                    now = time.monotonic() - start
                    x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
                    level_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
                    clipping = 0.9 * clipping + 0.1 * float(np.mean(np.abs(x) > 0.99))
                    comparison.step(pcm, now)
                    live.update(render(comparison, skipped, word, level_db, clipping, now))
            except KeyboardInterrupt:
                pass
    return comparison.counts


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="ovos-wakeforge-demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("--wakeword", choices=[w.removeprefix("wakehubert_") for w in model_names()],
                    default="alexa")
    ap.add_argument("--threshold", type=float,
                    help="WakeHuBERT trigger probability, 0-1 (default: the model's)")
    ap.add_argument("--oww-threshold", type=float, default=0.5, help="openWakeWord threshold (default 0.5)")
    ap.add_argument("--mww-cutoff", type=float,
                    help="microWakeWord probability cutoff (default: the model's shipped value)")
    ap.add_argument("--precise-sensitivity", type=float, default=0.5,
                    help="precise-onnx sensitivity (default 0.5)")
    ap.add_argument("--agc", action="store_true",
                    help="level each WakeHuBERT window (peak to 0.5, gain 0.25-4x); helps when the mic "
                         "gain is too high or too low")
    ap.add_argument("--only", help=f"comma-separated engines to run: {','.join(ENGINES)}")
    ap.add_argument("--device", help="input device name or index (see: python -m sounddevice)")
    ap.add_argument("--file", help="score this 16 kHz wav instead of the microphone")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("loading engines...")
    engines, skipped = build_engines(args)
    for reason in skipped:
        print(f"  not loaded: {reason}")
    if args.file:
        print(score_file(args.file, engines))
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    print("detections:", listen(engines, skipped, args.wakeword, device))


if __name__ == "__main__":
    main()
