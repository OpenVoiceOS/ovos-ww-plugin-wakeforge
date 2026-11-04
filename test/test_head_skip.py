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
