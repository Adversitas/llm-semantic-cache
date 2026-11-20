"""Measure how much a semantic cache saves, and how often it serves a wrong answer.

Data (French, committed under data/):

* PAWS-X fr dev/test: 2,000 + 2,000 sentence pairs, ~45% paraphrases, the rest
  adversarial near-duplicates (swapped entities, word order), exactly the
  pairs that make a cache serve a wrong answer;
* STS-B fr dev/test: graded similarity, used for a Spearman sanity check.

Thresholds are always chosen on dev and reported on test.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .cache import SemanticCache
from .embed import CachedEmbedder, Embedder

TARGETS = (0.90, 0.95, 0.99)


def load_pairs(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{name}.csv", keep_default_na=False)


def pair_cosine(emb: Embedder, df: pd.DataFrame) -> np.ndarray:
    a = emb.encode(df["s1"].tolist())
    b = emb.encode(df["s2"].tolist())
    return (a * b).sum(1)


# ------------------------------------------------------------------ decision rules


@dataclass
class Policy:
    accept: float
    verify_low: float | None = None
    llm_threshold: float | None = None

    def decide(self, cos: np.ndarray, p_llm: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (served, verifier_called) boolean arrays."""
        served = cos >= self.accept
        called = np.zeros_like(served)
        if self.verify_low is not None and p_llm is not None:
            called = (cos >= self.verify_low) & ~served
            served = served | (called & (p_llm >= self.llm_threshold))
        return served, called


def outcome(served: np.ndarray, called: np.ndarray, y: np.ndarray) -> dict:
    n_served = int(served.sum())
    correct = int((served & (y == 1)).sum())
    return {
        "served_rate": n_served / len(y),
        "precision": correct / n_served if n_served else 1.0,
        "recall": correct / max(int((y == 1).sum()), 1),
        "wrong_answers_per_1000": 1000 * (n_served - correct) / len(y),
        "verifier_call_rate": float(called.mean()),
    }


def threshold_for_precision(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """Lowest threshold whose precision on (scores, y) is >= target."""
    order = np.argsort(-scores)
    s, yy = scores[order], y[order]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return float("inf")
    return float(s[ok.max()])


def best_cascade(
    cos: np.ndarray, p: np.ndarray, y: np.ndarray, target: float, max_call_rate: float = 1.0
) -> Policy:
    """Grid search on dev: max recall s.t. precision >= target and verifier calls bounded."""
    qs = np.unique(np.quantile(cos, np.linspace(0, 1, 41)))
    best, best_rec = Policy(threshold_for_precision(cos, y, target)), -1.0
    for acc in qs:
        for low in qs[qs <= acc]:
            for lt in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
                pol = Policy(float(acc), float(low), lt)
                o = outcome(*pol.decide(cos, p), y)
                if (
                    o["precision"] >= target
                    and o["verifier_call_rate"] <= max_call_rate
                    and o["recall"] > best_rec
                ):
                    best, best_rec = pol, o["recall"]
    return best


# -------------------------------------------------------------------- simulation


def replay(emb: Embedder, df: pd.DataFrame, pol: Policy, verifier=None) -> dict:
    """Fill a real SemanticCache with every s1, then query every s2.

    A hit is counted as correct if the cached question is textually identical
    to the query after normalisation (same question, same answer), or if the
    matched entry is the query's own labelled paraphrase partner. Any other
    semantic hit is counted wrong, even if it happens to be an unlabelled
    paraphrase (conservative).
    """
    cache = SemanticCache(emb, accept=pol.accept, verify_low=pol.verify_low, verifier=verifier)
    emb.encode(df["s1"].tolist() + df["s2"].tolist())  # warm the memo
    for s1 in df["s1"].unique():
        cache.put(s1, s1)
    t0 = time.perf_counter()
    correct = wrong = 0
    for s1, s2, y in df[["s1", "s2", "label"]].itertuples(index=False):
        r = cache.get(s2)
        if r.hit:
            ok = r.route == "exact" or (r.answer == s1 and y == 1)
            correct += int(ok)
            wrong += int(not ok)
    dt = time.perf_counter() - t0
    n = len(df)
    return {
        "entries": len(cache),
        "served_rate": (correct + wrong) / n,
        "precision": correct / max(correct + wrong, 1),
        "wrong_answers_per_1000": 1000 * wrong / n,
        "verifier_call_rate": cache.stats.verifier_calls / n,
        "routes": cache.stats.by_route,
        "lookup_ms_per_query_excl_embedding": 1000 * dt / n,
    }


# ------------------------------------------------------------------------- runner


def run(
    data_dir: Path,
    reports: Path,
    embedders: list[Embedder],
    llm=None,
    gen_llm=None,
    log=print,
) -> dict:
    paws = {s: load_pairs(data_dir, f"pawsx_fr_{s}") for s in ("dev", "test")}
    sts = {s: load_pairs(data_dir, f"stsb_fr_{s}") for s in ("dev", "test")}
    res: dict = {"n_pairs": {k: len(v) for k, v in paws.items()}, "embedders": {}}

    p_llm = {}
    if llm is not None:
        log(f"LLM verifier {llm.name}: scoring {sum(map(len, paws.values()))} pairs ...")
        for s, df in paws.items():
            t0 = time.perf_counter()
            p_llm[s] = llm.p_same(list(zip(df["s1"], df["s2"], strict=True)))
            res.setdefault("llm_seconds_per_pair", {})[s] = (time.perf_counter() - t0) / len(df)
        res["llm_alone"] = {
            "name": llm.name,
            "auc_test": float(roc_auc_score(paws["test"]["label"], p_llm["test"])),
        }

    for raw in embedders:
        emb = CachedEmbedder(raw)
        log(f"embedder {emb.name}")
        cos = {s: pair_cosine(emb, df) for s, df in paws.items()}
        y = {s: paws[s]["label"].to_numpy() for s in paws}
        r: dict = {
            "pawsx_auc_test": float(roc_auc_score(y["test"], cos["test"])),
            "stsb_spearman_test": float(
                spearmanr(pair_cosine(emb, sts["test"]), sts["test"]["score"]).statistic
            ),
            "threshold_only": {},
            "cascade": {},
        }
        for t in TARGETS:
            thr = threshold_for_precision(cos["dev"], y["dev"], t)
            pol = Policy(thr)
            r["threshold_only"][str(t)] = {
                "accept": thr,
                **outcome(*pol.decide(cos["test"], None), y["test"]),
            }
            if llm is not None:
                pol = best_cascade(cos["dev"], p_llm["dev"], y["dev"], t)
                r["cascade"][str(t)] = {
                    "policy": pol.__dict__,
                    **outcome(*pol.decide(cos["test"], p_llm["test"]), y["test"]),
                }
        t95 = r["threshold_only"]["0.95"]["accept"]
        r["replay_threshold_0.95"] = replay(emb, paws["test"], Policy(t95))
        # single-query embedding latency (cold: not memoised)
        qs = paws["test"]["s2"].tolist()[:50]
        raw.encode(qs[:5])
        t0 = time.perf_counter()
        for q in qs:
            raw.encode([q])
        r["embed_ms_per_query"] = 1000 * (time.perf_counter() - t0) / len(qs)
        res["embedders"][emb.name] = r

    if gen_llm is not None:
        res["generation"] = generation_cost(gen_llm)
        if llm is not None:
            pairs = list(zip(paws["test"]["s1"][:50], paws["test"]["s2"][:50], strict=True))
            llm.p_same(pairs[:4], batch_size=1)
            t0 = time.perf_counter()
            for p in pairs:
                llm.p_same([p], batch_size=1)
            res["verifier_ms_per_call"] = 1000 * (time.perf_counter() - t0) / len(pairs)
    reports.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(res, indent=2, default=float)
    (reports / "results.json").write_text(payload, encoding="utf-8")
    (reports / "results.md").write_text(render(res), encoding="utf-8")
    return res


QUESTIONS = [
    "Comment déclarer un dégât des eaux à mon assureur ?",
    "Quel est le délai pour déclarer un sinistre automobile ?",
    "Mon assurance habitation couvre-t-elle le vol de mon vélo ?",
    "Comment résilier mon contrat d'assurance emprunteur ?",
    "Que faire après un bris de glace sur ma voiture ?",
]


def generation_cost(gen_llm) -> dict:
    gen_llm.generate("Bonjour", max_new_tokens=8)
    secs, toks = [], []
    for q in QUESTIONS:
        _, s, n = gen_llm.generate(q)
        secs.append(s)
        toks.append(n)
    return {
        "model": gen_llm.name,
        "seconds_per_answer": float(np.mean(secs)),
        "tokens_per_answer": float(np.mean(toks)),
    }


def render(r: dict) -> str:
    lines = ["# Results", ""]
    if "llm_alone" in r:
        lines.append(
            f"LLM verifier alone ({r['llm_alone']['name']}) PAWS-X fr test ROC-AUC: "
            f"{r['llm_alone']['auc_test']:.3f}"
        )
        lines.append("")
    lines += [
        "## Pair discrimination",
        "",
        "| Embedder | PAWS-X fr test ROC-AUC | STS-B fr test Spearman | Embed ms/query |",
        "|---|---|---|---|",
    ]
    for name, e in r["embedders"].items():
        lines.append(
            f"| {name} | {e['pawsx_auc_test']:.3f} | {e['stsb_spearman_test']:.3f} | "
            f"{e['embed_ms_per_query']:.1f} |"
        )
    lines += [
        "",
        "## Cache decisions on PAWS-X fr test (thresholds chosen on dev)",
        "",
        "| Embedder | Policy | Target precision | Served % | Precision % | Recall % "
        "| Wrong answers / 1000 req. | Verifier calls % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in r["embedders"].items():
        for kind in ("threshold_only", "cascade"):
            for t, o in e[kind].items():
                lines.append(
                    f"| {name} | {kind.replace('_', ' ')} | {float(t):.2f} | "
                    f"{100 * o['served_rate']:.1f} | {100 * o['precision']:.1f} | "
                    f"{100 * o['recall']:.1f} | {o['wrong_answers_per_1000']:.1f} | "
                    f"{100 * o['verifier_call_rate']:.1f} |"
                )
    lines += [
        "",
        "## Replay through the real cache (all test s1 cached, every s2 queried, 0.95 threshold)",
        "",
        "| Embedder | Entries | Served % | Precision % | Wrong / 1000 | Lookup ms (excl. embed) |",
        "|---|---|---|---|---|---|",
    ]
    for name, e in r["embedders"].items():
        rp = e["replay_threshold_0.95"]
        lines.append(
            f"| {name} | {rp['entries']} | {100 * rp['served_rate']:.1f} | "
            f"{100 * rp['precision']:.1f} | {rp['wrong_answers_per_1000']:.1f} | "
            f"{rp['lookup_ms_per_query_excl_embedding']:.2f} |"
        )
    if "generation" in r:
        g = r["generation"]
        lines += [
            "",
            "## Cost of a miss",
            "",
            f"Generating an answer with {g['model']}: {g['seconds_per_answer']:.2f} s, "
            f"{g['tokens_per_answer']:.0f} tokens on average (5 insurance questions).",
        ]
        if "verifier_ms_per_call" in r:
            lines.append(f"One verifier call: {r['verifier_ms_per_call']:.1f} ms.")
    return "\n".join(lines) + "\n"
