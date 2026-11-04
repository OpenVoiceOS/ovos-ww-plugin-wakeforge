"""Models under a non-ASCII path load (a Windows user name with an accent puts the Hugging Face cache under one)."""
import shutil
import sys

import numpy as np
import onnxruntime as ort

from ovos_ww_plugin_wakeforge import inference
from ovos_ww_plugin_wakeforge.inference import model_source, session_options
from test.conftest import make_featurizer


def test_model_source_reads_non_ascii_paths_on_windows(tmp_path, monkeypatch):
    model = tmp_path / "zoé.onnx"
    model.write_bytes(b"onnx bytes")
    monkeypatch.setattr(inference.sys, "platform", "win32")
    assert model_source(model) == b"onnx bytes"
    assert model_source(tmp_path / "plain.onnx") == str(tmp_path / "plain.onnx")
    (tmp_path / "zoé.onnx.data").write_bytes(b"")       # external weights: keep the path
    assert model_source(model) == str(model)
    monkeypatch.setattr(inference.sys, "platform", "linux")
    (tmp_path / "zoé.onnx.data").unlink()
    assert model_source(model) == str(model)


def test_session_from_non_ascii_directory(tmp_path):
    src = tmp_path / "featurizer.onnx"
    make_featurizer(str(src))
    home = tmp_path / "Zoé Gaëtan" / "modèles"
    home.mkdir(parents=True)
    path = home / "réveil.onnx"
    shutil.copy(src, path)
    sess = ort.InferenceSession(model_source(path), session_options(), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3200), np.float32)})[0]
    assert out.shape[0] == 1, sys.platform
