"""Type-checked replacements for pinned upstream functions.

Wrap a replacement in ``same_signature(original)`` so ty fails when the pinned
upstream signature drifts from it. CI installs the pinned upstream packages so
ty sees the real signatures. ty can't check an ``original`` it sees as ``Any``,
such as one read from a module passed in as an untyped argument.
"""

from collections.abc import Callable


def same_signature[**P, R](
    original: Callable[P, R],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return the replacement unchanged, typed to require ``original``'s signature."""
    return lambda new: new
