from pathlib import Path

import numpy as np

from semcache.embed import HashingEmbedder
from semcache.evaluate import Policy, best_cascade, outcome, run, threshold_for_precision

DATA = Path(__file__).resolve().parents[1] / "data"


def test_threshold_for_precision():
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 1, 0, 1, 0])
    assert threshold_for_precision(s, y, 1.0) == 0.8
    assert threshold_for_precision(s, y, 0.75) == 0.6
    assert threshold_for_precision(s, np.zeros(5, int), 0.5) == float("inf")


def test_outcome_counts():
    served = np.array([True, True, False, False])
    y = np.array([1, 0, 1, 0])
    o = outcome(served, np.zeros(4, bool), y)
    assert o["served_rate"] == 0.5 and o["precision"] == 0.5 and o["recall"] == 0.5
    assert o["wrong_answers_per_1000"] == 250


def test_cascade_policy_routes_gray_zone():
    pol = Policy(accept=0.9, verify_low=0.5, llm_threshold=0.5)
    cos = np.array([0.95, 0.7, 0.7, 0.3])
    p = np.array([0.0, 0.9, 0.1, 1.0])
    served, called = pol.decide(cos, p)
    assert served.tolist() == [True, True, False, False]
    assert called.tolist() == [False, True, True, False]


def test_best_cascade_beats_threshold_with_perfect_verifier():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    cos = 0.5 + 0.2 * y + 0.2 * rng.standard_normal(400)
    p = y.astype(float)  # oracle verifier
    pol = best_cascade(cos, p, y, target=0.95)
    th = Policy(threshold_for_precision(cos, y, 0.95))
    rec_c = outcome(*pol.decide(cos, p), y)["recall"]
    rec_t = outcome(*th.decide(cos, None), y)["recall"]
    assert rec_c > rec_t


def test_run_on_committed_data(tmp_path):
    res = run(DATA, tmp_path, [HashingEmbedder()], log=lambda *_: None)
    e = next(iter(res["embedders"].values()))
    assert 0.5 < e["pawsx_auc_test"] < 1.0
    assert e["stsb_spearman_test"] > 0.3
    assert (tmp_path / "results.md").exists()


def test_replay_counts_identical_question_as_correct():
    import pandas as pd

    from semcache.evaluate import replay

    df = pd.DataFrame(
        {
            "s1": ["Le chat dort sur le canapé.", "Paris est la capitale."],
            "s2": ["le chat dort sur le canapé", "Rome est la capitale."],
            "label": [0, 0],
        }
    )
    r = replay(HashingEmbedder(n_features=2**12), df, Policy(accept=0.999))
    assert r["routes"].get("exact") == 1
    assert r["precision"] == 1.0 and r["wrong_answers_per_1000"] == 0
