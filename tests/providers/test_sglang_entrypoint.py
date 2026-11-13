"""The SGLang worker receives its resolved constructor options without argparse."""

import json
import runpy
import sys
from types import ModuleType

import pytest

from spindle.inference.kv_events import KV_EVENTS_CONFIG
from spindle.deployments import DeploymentConfig, config_path, load


@pytest.mark.parametrize("source", [None, "hook", "kv_events", "compare"])
def test_sglang_constructor_receives_recorded_settings(monkeypatch, source):
    row = DeploymentConfig.create(load(config_path("qwen35-9b-lora-16k")))
    row.inference_settings["kv_events_config"] = '{"publisher":"null"}'
    calls = []

    class ServerArgs:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.log_level = "info"

    modules = {
        "sglang": {"__path__": []},
        "sglang.srt": {"__path__": []},
        "sglang.launch_server": {"run_server": lambda args: calls.append("run")},
        "sglang.srt.plugins": {"load_plugins": lambda: calls.append("plugins")},
        "sglang.srt.server_args": {"ServerArgs": ServerArgs},
        "sglang.srt.utils": {"kill_process_tree": lambda *a, **k: calls.append("stop")},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        sys, "argv", ["sglang", row.asset_path, json.dumps(row.inference_settings)]
    )
    if source is not None:
        monkeypatch.setenv("SGLANG_SPINDLE_ROUTING_SOURCE", source)
    runpy.run_module("spindle.inference.sglang", run_name="__main__")
    assert calls == [
        "plugins",
        {
            "model_path": row.asset_path,
            "host": "127.0.0.1",
            "port": 8001,
            **row.inference_settings,
            **(
                {"kv_events_config": json.dumps(KV_EVENTS_CONFIG)}
                if source in {"kv_events", "compare"}
                else {}
            ),
        },
        "run",
        "stop",
    ]
