from __future__ import annotations

from reasoninglab.policies.contracts import (
    GenerationLike,
    PolicyResult,
    SupportsGenerate,
)

'''
__all__ declares the intended public API of the module.
It controls from reasoninglab.policies import *.
It also acts as a clear contract for humans/tools about what is "officially exported".
Not strictly required, but useful and standard.
'''
__all__ = [
    "GenerationLike",
    "SupportsGenerate",
    "PolicyResult",
]
