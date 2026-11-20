"""semcache: a semantic response cache for LLM apps, measured for hit rate and wrong answers."""

from .cache import Lookup, SemanticCache

__all__ = ["Lookup", "SemanticCache"]
__version__ = "0.1.0"
