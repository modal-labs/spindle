import asyncio
import importlib
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.providers.local import InMemoryKeyValueStore
from synthetic_monitoring import checkout, config_validation
from synthetic_monitoring.checkout import isolated_configs, regression_recipes

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


def test_checkout_selection_prefers_standard_32k_specs():
    recipes = regression_recipes()
    assert len({recipe.model for recipe in recipes}) == len(recipes)
    selected = {recipe.model: recipe for recipe in recipes}
    for model in (
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.8-27B",
        "zai-org/GLM-5.3-Flash",
    ):
        assert selected[model].max_context_length == 32768
        assert selected[model].sglang_cfg["speculative_algorithm"] == "DFLASH"
    assert selected["Qwen/Qwen3.5-4B"].sglang_cfg["speculative_algorithm"] == "NEXTN"
    for config in isolated_configs(recipes, "sr-0123456789ab"):
        assert len(config.trainer_app_name) <= 64
        assert len(config.inference_app_name) <= 64


def test_explicit_config_and_unknown_model_selection():
    assert (
        regression_recipes(configs=["qwen38-27b-lora-128k"])[0].max_context_length
        == 131072
    )
    with pytest.raises(ValueError, match="No checkout LoRA recipe"):
        regression_recipes(["missing/model"])
    with pytest.raises(ValueError, match="one recipe per base model"):
        regression_recipes(configs=["qwen38-27b-lora-32k", "qwen38-27b-lora-128k"])


def test_checkout_configs_do_not_mutate_source_or_share_writable_run_storage():
    recipes = regression_recipes(["Qwen/Qwen3-8B"])
    before = deepcopy(vars(recipes[0]))
    first = isolated_configs(recipes, "sr-first")[0]
    second = isolated_configs(recipes, "sr-second")[0]
    assert vars(recipes[0]) == before
    assert first.trainer_app_name != second.trainer_app_name
    assert first.inference_app_name != second.inference_app_name
    assert (
        first.recipe.platform["storage"]["assets"]
        == second.recipe.platform["storage"]["assets"]
    )
    for storage in ("checkpoints", "bulletin"):
        assert (
            first.recipe.platform["storage"][storage]
            != second.recipe.platform["storage"][storage]
        )
    assert first.recipe.platform["secrets"]["api"] == "sr-first"
    assert first.recipe.platform["secrets"]["sampler_proxy"] == "sr-first"
    assert first.recipe.platform["requires_proxy_auth"] is True
    assert first.recipe.platform["modal"]["exclude_reserved_capacity"] is True
    assert first.recipe.platform["modal"]["environment"] == checkout.REGRESSION_ENV


@pytest.mark.parametrize(
    "failure",
    [None, "allow", "deploy", "auth", "commit", "recipe", "body", "interrupt"],
)
def test_checkout_lifecycle_cleans_up_and_rejects_wrong_server(monkeypatch, failure):
    recipes = regression_recipes(["Qwen/Qwen3-8B"])
    events = []
    captured = {}
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "test-proxy-id")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "test-proxy-secret")
    monkeypatch.setattr(checkout, "source_commit", lambda: "checkout-sha")

    def create(name, credentials, **kwargs):
        captured.update(frontend=name, key=credentials["TINKER_API_KEY"])
        events.append("secret")

    def allow(token_id, env):
        assert (token_id, env) == ("test-proxy-id", checkout.REGRESSION_ENV)
        if failure == "allow":
            raise RuntimeError("allow failed")

    tokens = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            token_id="test-proxy-id", token_secret="test-proxy-secret"
        ),
        allow=allow,
        delete=lambda token_id: events.append(("delete-proxy", token_id)),
    )
    monkeypatch.setattr(
        checkout.modal.Workspace,
        "from_context",
        lambda: SimpleNamespace(proxy_tokens=tokens),
    )

    def deploy(configs):
        assert configs[0].recipe.platform["frontend"] == captured["frontend"]
        events.append("deploy")
        if failure == "deploy":
            raise RuntimeError("deploy failed")

    def transport(request):
        assert str(request.url) in {
            "https://checkout.example/api/v1/get_server_capabilities",
            "https://checkout.example/api/v1/spindle/deployments",
        }
        assert request.headers["X-API-Key"] == captured["key"]
        assert request.headers["Modal-Key"] == "test-proxy-id"
        assert request.headers["Modal-Secret"] == "test-proxy-secret"
        if failure == "auth":
            return httpx.Response(401, text="invalid proxy authorization")
        if request.url.path.endswith("spindle/deployments"):
            return httpx.Response(
                200,
                json={
                    "deployments": []
                    if failure == "recipe"
                    else [{"name": recipes[0].name}]
                },
            )
        return httpx.Response(
            200,
            json={
                "spindle_commit": "wrong-sha"
                if failure == "commit"
                else "checkout-sha",
                "supported_models": [{"model_name": recipes[0].model}],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        monkeypatch.setattr(checkout.httpx, "get", client.get)
        monkeypatch.setattr(checkout, "deploy", deploy)
        monkeypatch.setattr(
            checkout, "stop", lambda name, env: events.append(("stop", name, env))
        )
        monkeypatch.setattr(
            checkout.modal.Secret,
            "objects",
            SimpleNamespace(
                create=create,
                delete=lambda name, **kwargs: events.append(("delete-secret", name)),
            ),
        )
        monkeypatch.setattr(
            checkout.modal.Volume,
            "objects",
            SimpleNamespace(
                delete=lambda name, **kwargs: events.append(("delete-volume", name))
            ),
        )
        monkeypatch.setattr(
            checkout.modal.Function,
            "from_name",
            lambda *a, **k: SimpleNamespace(
                get_web_url=lambda: "https://checkout.example"
            ),
        )

        def run():
            with checkout.checkout_server(recipes) as (url, key, commit, secret_name):
                checkout.verify_checkout(
                    url, key, commit, [recipe.name for recipe in recipes]
                )
                events.append("body")
                assert (url, key, commit) == (
                    "https://checkout.example",
                    captured["key"],
                    "checkout-sha",
                )
                if failure == "body":
                    raise RuntimeError("workload failed")
                if failure == "interrupt":
                    raise KeyboardInterrupt()

        if failure is None:
            run()
        else:
            expected = (
                KeyboardInterrupt
                if failure == "interrupt"
                else ValueError
                if failure in ("auth", "commit", "recipe")
                else RuntimeError
            )
            with pytest.raises(expected):
                run()
    frontend = captured.get("frontend") or next(
        e[1] for e in events if isinstance(e, tuple) and e[0] == "stop"
    )
    assert events[-6:] == [
        ("stop", frontend, checkout.REGRESSION_ENV),
        *[
            ("delete-volume", f"{frontend}-{suffix}")
            for suffix in ("checkpoints", "bulletin", "usage")
        ],
        ("delete-secret", frontend),
        ("delete-proxy", "test-proxy-id"),
    ]
    assert ("body" in events) == (failure in (None, "body", "interrupt"))


def test_regression_uses_checkout_credentials_and_cancels_active_work_on_interrupt(
    monkeypatch,
):
    recipes = regression_recipes(["Qwen/Qwen3-8B"])
    events = []
    monkeypatch.setenv("TINKER_BASE_URL", "https://production.example")
    monkeypatch.setenv("TINKER_API_KEY", "production-key")

    @contextmanager
    def server(selected):
        assert selected == recipes
        try:
            yield (
                "https://checkout.example",
                "checkout-key",
                "checkout-sha",
                "checkout-secret",
            )
        finally:
            events.append("stop-server")

    class Call:
        def get(self):
            raise KeyboardInterrupt()

        def cancel(self, terminate_containers=False):
            assert terminate_containers
            events.append("cancel-client")

    def spawn(model, recipe_name, renderer, url, key, commit):
        assert model == recipes[0].model
        assert recipe_name == recipes[0].name
        env = config_validation.client_env(url, key)
        assert env["TINKER_BASE_URL"] == "https://checkout.example"
        assert env["TINKER_API_KEY"] == "checkout-key"
        assert "/opt/proxy-auth" in env.get("PYTHONPATH", "")
        assert commit == "checkout-sha"
        return Call()

    monkeypatch.setattr(config_validation, "regression_recipes", lambda *a: recipes)
    monkeypatch.setattr(config_validation, "checkout_server", server)
    monkeypatch.setattr(
        config_validation,
        "verify_regression_server",
        SimpleNamespace(
            with_options=lambda **kwargs: SimpleNamespace(remote=lambda *args: None)
        ),
    )
    monkeypatch.setattr(
        config_validation,
        "regression_renderers",
        SimpleNamespace(remote=lambda models: {models[0]: "qwen3_disable_thinking"}),
    )
    monkeypatch.setattr(
        config_validation,
        "regression",
        SimpleNamespace(with_options=lambda **kw: SimpleNamespace(spawn=spawn)),
    )
    with pytest.raises(KeyboardInterrupt):
        config_validation.regress(model="Qwen/Qwen3-8B")
    assert events == ["cancel-client", "stop-server"]


def test_verification_uses_actual_capability_and_deployment_response_contracts(
    monkeypatch,
):
    recipes = regression_recipes(["Qwen/Qwen3-8B"])
    configs = isolated_configs(recipes, "sr-contract")
    server = create_control_plane_app(
        ControlPlane(InMemoryKeyValueStore(), SimpleNamespace()),
        configs,
        api_key="test-key",
    )
    http_module = importlib.import_module("spindle.control_plane.http")
    monkeypatch.setattr(http_module, "source_commit", lambda: "checkout-sha")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "test-proxy-id")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "test-proxy-secret")
    requests = []

    def get(url, **kwargs):
        async def request():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server)
            ) as client:
                response = await client.get(url, **kwargs)
                requests.append((response.request.url.path, response.json()))
                return response

        return asyncio.run(request())

    monkeypatch.setattr(checkout.httpx, "get", get)
    checkout.verify_checkout(
        "https://checkout.example", "test-key", "checkout-sha", [recipes[0].name]
    )
    assert requests[0][1]["supported_models"][0]["model_name"] == recipes[0].model
    assert requests[1][1]["deployments"][0]["name"] == recipes[0].name
