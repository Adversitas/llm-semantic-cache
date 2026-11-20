"""A semantic response cache for LLM applications.

A request whose meaning matches an earlier one is answered from the cache
instead of calling the LLM again. Lookups are exact-match first (normalised
text hash), then nearest-neighbour on embeddings. Entries are isolated by
namespace (tenant, user, model version or prompt version) so that an answer
is never served across a security or configuration boundary.

Decision rule for a candidate with cosine similarity ``s``:

* ``s >= accept``                -> hit;
* ``verify_low <= s < accept``   -> ask the verifier (if any), hit if it agrees;
* otherwise                      -> miss.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .embed import Embedder

Verifier = Callable[[str, str], bool]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[\s?!.]+$", "", text)


def text_key(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


@dataclass
class Entry:
    query: str
    answer: str
    namespace: str
    created: float
    expires: float | None
    hits: int = 0


@dataclass
class Lookup:
    hit: bool
    answer: str | None
    score: float
    matched_query: str | None
    route: str  # "exact" | "semantic" | "verified" | "rejected" | "miss"


@dataclass
class CacheStats:
    lookups: int = 0
    exact: int = 0
    semantic: int = 0
    verified: int = 0
    rejected: int = 0
    verifier_calls: int = 0
    by_route: dict = field(default_factory=dict)

    @property
    def hits(self) -> int:
        return self.exact + self.semantic + self.verified

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        accept: float = 0.95,
        verify_low: float | None = None,
        verifier: Verifier | None = None,
        capacity: int = 10_000,
        ttl_s: float | None = None,
        clock: Callable[[], float] = time.time,
    ):
        if verify_low is not None and verify_low > accept:
            raise ValueError("verify_low must be <= accept")
        self.embedder = embedder
        self.accept = accept
        self.verify_low = verify_low
        self.verifier = verifier
        self.capacity = capacity
        self.ttl_s = ttl_s
        self.clock = clock
        self.stats = CacheStats()
        self._entries: OrderedDict[int, Entry] = OrderedDict()  # LRU order
        self._exact: dict[tuple[str, str], int] = {}
        self._vecs: dict[str, dict[int, np.ndarray]] = {}
        self._index: dict[str, tuple[list[int], np.ndarray]] = {}
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ writes
    def put(self, query: str, answer: str, namespace: str = "default") -> int:
        now = self.clock()
        key = (namespace, text_key(query))
        if key in self._exact:
            self._evict(self._exact[key])
        eid = self._next_id
        self._next_id += 1
        expires = now + self.ttl_s if self.ttl_s else None
        self._entries[eid] = Entry(query, answer, namespace, now, expires)
        self._exact[key] = eid
        self._vecs.setdefault(namespace, {})[eid] = self.embedder.encode([query])[0]
        while len(self._entries) > self.capacity:
            self._evict(next(iter(self._entries)))
        return eid

    def invalidate(self, namespace: str) -> int:
        """Drop every entry of a namespace (e.g. after a model or prompt upgrade)."""
        ids = list(self._vecs.get(namespace, {}))
        for eid in ids:
            self._evict(eid)
        return len(ids)

    def _evict(self, eid: int) -> None:
        e = self._entries.pop(eid, None)
        if e is None:
            return
        self._exact.pop((e.namespace, text_key(e.query)), None)
        self._vecs.get(e.namespace, {}).pop(eid, None)

    # ------------------------------------------------------------------- reads
    def _live(self, eid: int) -> bool:
        e = self._entries[eid]
        if e.expires is not None and e.expires < self.clock():
            self._evict(eid)
            return False
        return True

    def _record(self, res: Lookup, eid: int | None) -> Lookup:
        self.stats.lookups += 1
        self.stats.by_route[res.route] = self.stats.by_route.get(res.route, 0) + 1
        if res.route in ("exact", "semantic", "verified"):
            setattr(self.stats, res.route, getattr(self.stats, res.route) + 1)
            assert eid is not None
            self._entries[eid].hits += 1
            self._entries.move_to_end(eid)
        elif res.route == "rejected":
            self.stats.rejected += 1
        return res

    def nearest(self, query: str, namespace: str = "default") -> tuple[int | None, float]:
        vecs = self._vecs.get(namespace, {})
        if not vecs:
            return None, -1.0
        ids, mat = self._matrix(namespace)
        sims = mat @ self.embedder.encode([query])[0]
        for j in np.argsort(-sims):  # skip expired entries, best live one wins
            eid = ids[int(j)]
            if eid in self._entries and self._live(eid):
                return eid, float(sims[j])
        return None, -1.0

    def _matrix(self, namespace: str) -> tuple[list[int], np.ndarray]:
        """Dense index per namespace, rebuilt only when entries were added or removed."""
        vecs = self._vecs[namespace]
        cached = self._index.get(namespace)
        if cached is None or cached[0] != list(vecs):
            ids = list(vecs)
            cached = (ids, np.stack([vecs[i] for i in ids]))
            self._index[namespace] = cached
        return cached

    def get(self, query: str, namespace: str = "default") -> Lookup:
        eid = self._exact.get((namespace, text_key(query)))
        if eid is not None and self._live(eid):
            e = self._entries[eid]
            return self._record(Lookup(True, e.answer, 1.0, e.query, "exact"), eid)
        eid, score = self.nearest(query, namespace)
        if eid is None:
            return self._record(Lookup(False, None, score, None, "miss"), None)
        e = self._entries[eid]
        if score >= self.accept:
            return self._record(Lookup(True, e.answer, score, e.query, "semantic"), eid)
        if self.verifier is not None and self.verify_low is not None and score >= self.verify_low:
            self.stats.verifier_calls += 1
            if self.verifier(query, e.query):
                return self._record(Lookup(True, e.answer, score, e.query, "verified"), eid)
            return self._record(Lookup(False, None, score, e.query, "rejected"), None)
        return self._record(Lookup(False, None, score, e.query, "miss"), None)

    def get_or_call(self, query: str, llm: Callable[[str], str], namespace: str = "default"):
        res = self.get(query, namespace)
        if res.hit:
            return res.answer, res
        answer = llm(query)
        self.put(query, answer, namespace)
        return answer, res
