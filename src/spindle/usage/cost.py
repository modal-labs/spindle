"""Splitting each Modal app's cost across runs by the tokens its GPUs processed."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from .rows import UsageRow


def attribute_cost(
    rows: Iterable[UsageRow],
    app_costs: Mapping[str, float],
    *,
    sessions: Mapping[str, Mapping[str, str]],
    group_by: tuple[str, ...] = ("run_id",),
) -> dict[str | tuple[str, ...], float]:
    """Split each app's cost across groups by the tokens its GPUs processed.

    A group is the tuple of a session's values for the `group_by` label keys, with
    `unknown` for a missing label. Idle and cold-start time are priced at the app's
    cost per container-second and reported under the string keys `idle` and
    `cold_start`. Cost with no tokens to split it by is `unattributed`, so the
    groups add up to `app_costs`.
    """
    by_app: dict[str, list[UsageRow]] = defaultdict(list)
    for row in rows:
        by_app[row.app_name].append(row)

    def group(session_id: str | None) -> tuple[str, ...]:
        labels = sessions.get(session_id or "") or {}
        return tuple(labels.get(key, "unknown") for key in group_by)

    costs: dict[str | tuple[str, ...], float] = defaultdict(float)
    for app_name, cost in app_costs.items():
        app_rows = by_app.get(app_name, [])
        container_s = sum(
            row.busy_seconds + row.idle_seconds + row.cold_start_seconds
            for row in app_rows
        )
        per_second = cost / container_s if container_s else 0.0
        remaining = cost
        for overhead in ("idle", "cold_start"):
            if spent := per_second * sum(
                getattr(row, f"{overhead}_seconds") for row in app_rows
            ):
                costs[overhead] += spent
                remaining -= spent

        tokens: dict[str | None, int] = defaultdict(int)
        for row in app_rows:
            uncached = row.prompt_tokens - row.cached_prompt_tokens
            if count := row.train_tokens + uncached + row.completion_tokens:
                tokens[row.session_id] += count
        total = sum(tokens.values())
        if not total:
            costs["unattributed"] += remaining
            continue
        for session_id, count in tokens.items():
            costs[group(session_id)] += remaining * count / total
    return dict(costs)
