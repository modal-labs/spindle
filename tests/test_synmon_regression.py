import pytest

from synthetic_monitoring.config_validation import select_runs

SERVED = {
    "Qwen/Qwen3-8B": "qwen3_disable_thinking",
    "Qwen/Qwen3-4B": None,
    "deepseek-ai/DeepSeek-V3.1": "deepseekv3",
}


@pytest.mark.parametrize(
    ("models", "runnable", "skipped"),
    [
        (
            None,
            [
                ("Qwen/Qwen3-8B", "qwen3_disable_thinking"),
                ("deepseek-ai/DeepSeek-V3.1", "deepseekv3"),
            ],
            [("Qwen/Qwen3-4B", "no cookbook renderer")],
        ),
        (
            ["Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "openai/gpt-oss-20b"],
            [("Qwen/Qwen3-8B", "qwen3_disable_thinking")],
            [
                ("Qwen/Qwen3-4B", "no cookbook renderer"),
                ("openai/gpt-oss-20b", "not served"),
            ],
        ),
    ],
    ids=["all served models", "chosen models"],
)
def test_select_runs(models, runnable, skipped):
    assert select_runs(SERVED, models) == (runnable, skipped)
