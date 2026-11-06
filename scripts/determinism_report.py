"""Exact comparison of fixed-input training replay traces."""


def compare(reference, candidate):
    expected = {row["step"]: row for row in reference["steps"]}
    differences = []
    expected_steps = list(expected)
    if candidate["mode"] == "restore":
        expected_steps = expected_steps[len(expected_steps) // 2 :]
    if (
        not expected_steps
        or [row["step"] for row in candidate["steps"]] != expected_steps
    ):
        differences.append({"step": None, "field": "missing_or_reordered_steps"})
    state = "restored_midpoint" if candidate["mode"] == "restore" else "restored"
    baseline_state = "midpoint" if candidate["mode"] == "restore" else "restored"
    if reference[baseline_state] != candidate[state]:
        differences.append({"step": 0, "field": state})
    if candidate["deterministic"] and reference["initial"] != candidate["initial"]:
        differences.append({"step": 0, "field": "fresh_initialization"})
    for row in candidate["steps"]:
        if row["step"] not in expected:
            differences.append({"step": row["step"], "field": "unexpected_step"})
            continue
        base = expected[row["step"]]
        for field in ("forward", "before_optimizer", "optimizer", "after_optimizer"):
            if base[field] != row[field]:
                differences.append({"step": row["step"], "field": field})
    if reference["final_evaluation"] != candidate["final_evaluation"]:
        differences.append(
            {
                "step": expected_steps[-1] if expected_steps else None,
                "field": "final_evaluation",
            }
        )
    return {
        "equal": not differences,
        "first_divergence": differences[0] if differences else None,
        "differences": differences,
        "replay_nll_difference": candidate["final_nll"] - reference["final_nll"],
    }
