import numpy as np
import pytest

from semcache.cache import SemanticCache, normalize
from semcache.embed import CachedEmbedder, HashingEmbedder


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def emb():
    return HashingEmbedder(n_features=2**12)


def test_normalize():
    assert normalize("  Comment  déclarer un SINISTRE ?? ") == "comment déclarer un sinistre"


def test_exact_hit_ignores_case_and_punctuation(emb):
    c = SemanticCache(emb, accept=0.99)
    c.put("Comment déclarer un sinistre ?", "A")
    r = c.get("comment déclarer un sinistre")
    assert r.hit and r.route == "exact" and r.answer == "A"


def test_semantic_hit_and_miss(emb):
    c = SemanticCache(emb, accept=0.7)
    c.put("Comment déclarer un dégât des eaux ?", "A")
    assert c.get("Comment je déclare un dégât des eaux").hit
    assert not c.get("Quel est le prix d'une assurance auto ?").hit
    assert c.stats.lookups == 2 and c.stats.hits == 1


def test_namespaces_are_isolated(emb):
    c = SemanticCache(emb, accept=0.5)
    c.put("Quel est mon plafond de garantie ?", "client-A answer", namespace="tenant-a")
    r = c.get("Quel est mon plafond de garantie ?", namespace="tenant-b")
    assert not r.hit and r.route == "miss"


def test_invalidate_namespace(emb):
    c = SemanticCache(emb)
    c.put("q1", "a", namespace="model-v1")
    c.put("q2", "b", namespace="model-v1")
    c.put("q3", "c", namespace="model-v2")
    assert c.invalidate("model-v1") == 2
    assert len(c) == 1 and not c.get("q1", namespace="model-v1").hit


def test_ttl_expiry(emb):
    clock = Clock()
    c = SemanticCache(emb, ttl_s=10, clock=clock)
    c.put("Horaires du service client", "9h-18h")
    clock.t = 5
    assert c.get("Horaires du service client").hit
    clock.t = 11
    assert not c.get("Horaires du service client").hit
    assert len(c) == 0


def test_lru_capacity(emb):
    c = SemanticCache(emb, accept=0.99, capacity=2)
    c.put("alpha question", "1")
    c.put("beta question", "2")
    c.get("alpha question")  # alpha becomes most recent
    c.put("gamma question", "3")
    assert c.get("alpha question").hit
    assert not c.get("beta question").hit


def test_verifier_gray_zone(emb):
    calls = []

    def verifier(a, b):
        calls.append((a, b))
        return "eaux" in a

    c = SemanticCache(emb, accept=0.999, verify_low=0.3, verifier=verifier)
    c.put("Comment déclarer un dégât des eaux ?", "A")
    assert c.get("Déclarer un dégât des eaux, comment faire").route == "verified"
    assert c.get("Comment déclarer un vol de voiture ?").route == "rejected"
    assert c.stats.verifier_calls == 2 and len(calls) == 2


def test_get_or_call_only_calls_llm_on_miss(emb):
    c = SemanticCache(emb, accept=0.9)
    n = []

    def llm(q):
        n.append(q)
        return q.upper()

    a1, _ = c.get_or_call("Quels documents fournir ?", llm)
    a2, r2 = c.get_or_call("quels documents fournir", llm)
    assert a1 == a2 and r2.hit and len(n) == 1


def test_bad_thresholds(emb):
    with pytest.raises(ValueError):
        SemanticCache(emb, accept=0.5, verify_low=0.8)


def test_embeddings_normalised_and_memoised(emb):
    ce = CachedEmbedder(emb)
    v = ce.encode(["a b c", "a b c", "d e f"])
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert len(ce._memo) == 2
