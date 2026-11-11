"""Splitting each Modal app's cost across runs by the tokens its GPUs processed."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from .rows import UsageRow

SUMMARY_LINES = ("idle", "cold_start", "unattributed", "total")


def attribute_cost(
    rows: Iterable[UsageRow],
    app_costs: Mapping[str, float],
    *,
    sessions: Mapping[str, Mapping[str, str]],
    group_by: str = "run_id",
) -> dict[str, float]:
    """Split each app's cost across groups by the tokens its GPUs processed.

    Idle and cold-start time are priced at the app's cost per container-second and
    reported as their own groups. Cost with no tokens to split it by is `unattributed`,
    so the groups add up to `app_costs`. A label value that is also a summary line's
    name is shown with the key, such as `idle (run_id)`.
    """
    by_app: dict[str, list[UsageRow]] = defaultdict(list)
    for row in rows:
        by_app[row.app_name].append(row)

    def group(session_id: str | None) -> str:
        if session_id is None:
            return "unknown"
        name = (sessions.get(session_id) or {}).get(group_by, "unknown")
        return f"{name} ({group_by})" if name in SUMMARY_LINES else name

    costs: dict[str, float] = defaultdict(float)
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
