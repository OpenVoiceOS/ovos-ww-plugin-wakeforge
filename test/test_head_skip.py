"""``head_skip_below``: while the last score is low, the head scores every other block and repeats its last score in
between; every block it does score is scored exactly as without the option."""
import numpy as np

from ovos_ww_plugin_wakeforge import WakeForgeHotwordPlugin

WINDOW = 20


def _scores(standin_hub, pcm, **config):
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert", window_frames=str(WINDOW))
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head, "threshold": 1.1, "streaming_features": True,
                                                  **config})
    runs, run = [0], eng.engine.head.run

    class Counting:
        def run(self, *a, **k):
            runs[0] += 1
            return run(*a, **k)

        def __getattr__(self, name):
            return getattr(run.__self__, name)

    eng.engine.head = Counting()
    scores = []
    for i in range(0, len(pcm) - 1279, 1280):
        eng.update(pcm[i:i + 1280].tobytes())
        scores.append(eng.last_score)
    return np.array(scores), runs[0]


def test_off_by_default_scores_every_block(standin_hub, pcm):
    scores, runs = _scores(standin_hub, pcm)
    assert runs == len(scores)


def test_low_scores_are_computed_every_other_block(standin_hub, pcm):
    full, full_runs = _scores(standin_hub, pcm)
    skip, runs = _scores(standin_hub, pcm, head_skip_below=1.0)      # every score is below 1: always skip
    assert runs == (len(full) + 1) // 2
    assert np.array_equal(skip[0::2], full[0::2])                     # the scored blocks are unchanged
    assert np.array_equal(skip[1::2], full[0::2][:len(skip[1::2])])   # the others repeat the last score


def test_a_score_at_the_floor_scores_every_block(standin_hub, pcm):
    full, _ = _scores(standin_hub, pcm)
    skip, runs = _scores(standin_hub, pcm, head_skip_below=0.0)      # no score is below 0: never skip
    assert runs == len(full)
    assert np.array_equal(skip, full)


def test_a_floor_above_the_threshold_repeats_no_score_that_counts(standin_hub):
    """A repeated score is never above the threshold: one real frame of 0.55 and its repeat would otherwise fire at
    threshold 0.5 and patience 2, though the next real score is 0.2."""
    head = standin_hub.batch_head(pretrained_featurizer="wakehubert", window_frames=str(WINDOW))
    eng = WakeForgeHotwordPlugin("hey computer", {"model": head, "threshold": 0.5, "smoothing": "max",
                                                  "window_size": 1, "patience": 2, "streaming_features": True,
                                                  "head_skip_below": 0.9})
    assert eng.engine.skip_below == 0.5
    scores = iter([0.55, 0.2] * 20)

    class Alternating:
        def run(self, *a, **k):
            p = next(scores)
            return [np.array([[np.log(p / (1 - p))]], np.float32)]

        def __getattr__(self, name):
            return getattr(eng.engine.__dict__["_real_head"], name)

    eng.engine._real_head, eng.engine.head = eng.engine.head, Alternating()
    for _ in range(10):
        eng.update(np.zeros(1280, np.int16).tobytes())
    assert not eng.found_wake_word()
