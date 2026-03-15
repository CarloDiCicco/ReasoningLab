from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from reasoninglab.eval.metrics import AttemptRecord


# GenerationLike and SupportsGenerate describe required behavior/type shape;
# they are not concrete runtime data containers.
class GenerationLike(Protocol):
    """Minimal generation payload needed by H1 policies."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float


class SupportsGenerate(Protocol):
    """Model interface required by policy modules."""

    # Declares only the method signature so policies can type-check against
    # any adapter that exposes generate(...), not just QwenModel.
    def generate(
        self,
        prompt: str,
        return_hidden_states: bool = False,
    ) -> GenerationLike:
        ...


# PolicyResult is concrete data you actually instantiate and pass around.
@dataclass(frozen=True)
class PolicyResult:
    """Policy output for one task evaluation."""

    attempts: tuple[AttemptRecord, ...]
    selected_candidate: str | None
    # H2: per-attempt hidden states from the prompt forward pass.
    # Tuple of dicts (one per attempt), each mapping layer index to np.ndarray.
    # None when return_hidden_states=False (H1 default).
    hidden_states: tuple[dict, ...] | None = None