"""The comparison demo: lazy imports, --file mode end to end, the refractory and the panel."""
import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from rich.console import Console

from ovos_ww_plugin_wakeforge import demo

CLIPS = Path(__file__).parent / "clips"
DEMO_ONLY = ("rich", "sounddevice", "soundfile", "openwakeword", "pymicro_wakeword",
             "ovos_ww_plugin_precise_onnx")


class Stub(demo.Engine):
    """Fires whenever the chunk is loud; its score is the chunk's peak."""

    def __init__(self, name, line=0.5):
        self.name, self.line = name, line

    def feed(self, pcm):
        self.score = float(np.abs(np.frombuffer(pcm, "<i2")).max()) / 32768
        return self.score >= self.line


class Broken(demo.Engine):
    name = "broken"

    def feed(self, pcm):
        raise RuntimeError("model went away")


def _wav(tmp_path, bursts=(1.0, 4.0), seconds=6.0):
    audio = np.zeros(int(16000 * seconds), np.float32)
    for start in bursts:
        audio[int(16000 * start):int(16000 * (start + 0.5))] = 0.8
    path = tmp_path / "clip.wav"
    sf.write(path, audio, 16000)
    return str(path)


def test_importing_the_demo_loads_none_of_its_extra_dependencies():
    code = ("import sys, ovos_ww_plugin_wakeforge.demo; "
            f"print([m for m in {DEMO_ONLY!r} if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_extra_dependencies_are_imported_inside_functions_only():
    tree = ast.parse(Path(demo.__file__).read_text())
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
           for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module)])}
    assert not top & set(DEMO_ONLY), top


def test_file_mode_end_to_end_with_stub_engines(tmp_path, monkeypatch, capsys):
    engines = [Stub("loud"), Stub("never", line=2.0), Broken()]
    monkeypatch.setattr(demo, "build_engines", lambda args: (engines, ["missing: ImportError: nope"]))
    demo.main(["--file", _wav(tmp_path)])
    out = capsys.readouterr().out
    assert "not loaded: missing: ImportError: nope" in out
    found = eval(out.strip().splitlines()[-1])
    assert found == {"loud": [0.96, 4.0], "never": [], "broken": []}
    assert engines[2].error == "RuntimeError: model went away" and engines[2].score is None


def test_refractory_ignores_detections_for_two_seconds():
    comparison = demo.Comparison([Stub("loud")])
    loud = (np.full(demo.CHUNK, 30000, np.int16)).tobytes()
    times = [i * demo.CHUNK / 16000 for i in range(100)]   # 8 s of continuous detections
    hits = [t for t in times if comparison.step(loud, t)]
    assert hits == [0.0, 2.0, 4.0, 6.0]
    assert comparison.counts == {"loud": 4} and len(comparison.log) == 4



def test_wake_up_has_no_other_engine_and_skips_cleanly():
    # wake_up is only a model of this plugin; openWakeWord, microWakeWord and
    # precise-onnx have no model for it and must be skipped cleanly, not crash
    # the comparison.
    args = demo.parse_args(["--wakeword", "wake_up"])
    engines, skipped = demo.build_engines(args)
    assert [e.name for e in engines] == ["wakehubert"]
    assert {s.split(":")[0] for s in skipped} == {"openwakeword", "microwakeword", "precise-onnx"}


def test_only_selects_engines_and_failures_are_reported(monkeypatch):
    args = demo.parse_args(["--only", "precise-onnx", "--wakeword", "alexa"])
    engines, skipped = demo.build_engines(args)
    assert engines == [] and skipped == ["precise-onnx: ValueError: no precise-onnx model for 'alexa'"]


def test_panel_shows_bars_triggers_detections_and_failures():
    engines = [Stub("loud"), Broken()]
    comparison = demo.Comparison(engines)
    comparison.step((np.full(demo.CHUNK, 30000, np.int16)).tobytes(), 1.0)
    console = Console(width=140, record=True)
    console.print(demo.render(comparison, ["precise-onnx: no model"], "hey_mycroft", -20.0, 0.05, 1.2))
    text = console.export_text()
    for expected in ('Say "hey mycroft"', "WAKE WORD!", "error: RuntimeError: model went away",
                     "not loaded", "precise-onnx: no model", "CLIPPING", "recent detections", "loud (92%)"):
        assert expected in text, expected


@pytest.mark.parametrize("word, trigger", [("hey_mycroft", 0.18), ("wake_up", 0.36)])
def test_wakehubert_engine_scores_a_clip_through_the_plugin(word, trigger):
    engine = demo.WakeHuBERT(word)
    assert engine.line == trigger
    found = demo.score_file(str(CLIPS / f"{word}.wav"), [engine])
    assert len(found["wakehubert"]) == 1
    assert 0.0 < engine.score <= 1.0
