"""Compatibility exports for the paper-aligned LaDiR reasoner."""

from types import SimpleNamespace

from ladir_reasoner import (
    LaDiRReasoner,
    ReasonerSpecialTokens,
    ReasonerTokenIds,
    configure_reasoner_tokenizer,
)

# Names used by the initial release.
LMFusionModel = LaDiRReasoner
LMFusionConfig = SimpleNamespace

__all__ = [
    "LaDiRReasoner",
    "ReasonerSpecialTokens",
    "ReasonerTokenIds",
    "configure_reasoner_tokenizer",
    "LMFusionModel",
    "LMFusionConfig",
]
