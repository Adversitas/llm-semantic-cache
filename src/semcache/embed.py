"""Sentence embedders. All return L2-normalised float32 vectors."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


class HashingEmbedder:
    """Character n-gram hashing vectoriser. No fitting, no download: the lexical baseline."""

    def __init__(self, n_features: int = 2**13, ngram=(2, 4)):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"char{ngram[0]}-{ngram[1]}-hashing"
        self._vec = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=ngram,
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        x = self._vec.transform(texts)
        x.data = np.log1p(x.data)
        return _l2(x.toarray())


class TransformerEmbedder:
    """Mean-pooled transformer encoder (e.g. intfloat/multilingual-e5-small)."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        prefix: str = "query: ",
        device: str | None = None,
        batch_size: int = 64,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model_name.split("/")[-1]
        self.prefix = prefix
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self._torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [self.prefix + t for t in texts[i : i + self.batch_size]]
                enc = self.tok(
                    batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                h = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                out.append(((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
        return _l2(np.concatenate(out))


class CachedEmbedder:
    """Memoises another embedder (experiments embed the same sentences many times)."""

    def __init__(self, inner: Embedder):
        self.inner = inner
        self.name = inner.name
        self._memo: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        todo = list(dict.fromkeys(t for t in texts if t not in self._memo))
        if todo:
            for t, v in zip(todo, self.inner.encode(todo), strict=True):
                self._memo[t] = v
        return np.stack([self._memo[t] for t in texts])
