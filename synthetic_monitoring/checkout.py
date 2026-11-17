"""Deploy checkout recipes behind a temporary regression frontend."""

from collections.abc import Iterator
from contextlib import contextmanager
import secrets
import uuid

import httpx
import modal

from spindle.configuration import BaseConfig
from spindle.deployment_cli import config_file, deploy, stop
from spindle.deploy_wizard import bundled_config_paths
from spindle.deployments import DeploymentConfig, load
from spindle.version import source_commit
from spindle.providers.modal.fft_pool import proxy_auth_headers

REGRESSION_ENV = "spindle-ci"


def regression_recipes(
    models: list[str] | None = None, configs: list[str] | None = None
) -> list[BaseConfig]:
    if configs is not None:
        recipes = [load(config_file(name)) for name in dict.fromkeys(configs)]
        if any(recipe.parameterization != "lora" for recipe in recipes):
            raise ValueError("The guess-number regression requires LoRA recipes")
        if len({recipe.model for recipe in recipes}) != len(recipes):
            raise ValueError(
                "Select only one recipe per base model in a regression run"
            )
        return recipes
    recipes = [load(path) for path in bundled_config_paths()]
    recipes = [recipe for recipe in recipes if recipe.parameterization == "lora"]
    recipes.sort(
        key=lambda recipe: (
            abs(recipe.max_context_length - 32768),
            len(recipe.name),
            recipe.name,
        )
    )
    by_model = {}
    for recipe in recipes:
        by_model.setdefault(recipe.model, recipe)
    if models is not None:
        missing = set(models) - by_model.keys()
        if missing:
            raise ValueError(
                f"No checkout LoRA recipe for: {', '.join(sorted(missing))}"
            )
        return [by_model[model] for model in dict.fromkeys(models)]
    return [by_model[model] for model in sorted(by_model)]


def isolated_configs(
    recipes: list[BaseConfig], frontend: str
) -> list[DeploymentConfig]:
    configs = []
    for original in recipes:
        recipe = BaseConfig(**vars(original))
        recipe.platform["frontend"] = frontend
        recipe.platform["modal"]["environment"] = REGRESSION_ENV
        recipe.platform["modal"]["exclude_reserved_capacity"] = True
        recipe.platform["url_label"] = frontend
        recipe.platform["requires_proxy_auth"] = True
        recipe.platform["secrets"]["api"] = frontend
        recipe.platform["secrets"]["sampler_proxy"] = frontend
        recipe.platform["storage"]["checkpoints"] = f"{frontend}-checkpoints"
        recipe.platform["storage"]["bulletin"] = f"{frontend}-bulletin"
        configs.append(DeploymentConfig.create(recipe))
    return configs


def verify_checkout(
    base_url: str, api_key: str, commit: str, recipes: list[str]
) -> None:
    def get(path: str) -> dict:
        response = httpx.get(
            base_url.rstrip("/") + "/api/v1/" + path,
            headers={"X-API-Key": api_key, **proxy_auth_headers()},
            timeout=120,
        )
        if response.is_error:
            detail = response.text.replace(api_key, "[redacted]")[:200]
            raise ValueError(
                f"Regression verification failed (HTTP {response.status_code}): {detail}"
            )
        return response.json()

    capabilities = get("get_server_capabilities")
    if capabilities.get("spindle_commit") != commit:
        raise ValueError(
            f"Regression server reports {capabilities.get('spindle_commit')!r}; expected {commit!r}"
        )
    served = {row["name"] for row in get("spindle/deployments")["deployments"]}
    missing = set(recipes) - served
    if missing:
        raise ValueError(f"Regression server is missing recipes: {sorted(missing)}")


@contextmanager
def checkout_server(recipes: list[BaseConfig]) -> Iterator[tuple[str, str, str, str]]:
    commit = source_commit()
    if commit is None:
        raise ValueError("Run regression from a Spindle checkout")
    frontend = f"sr-{uuid.uuid4().hex[:12]}"
    api_key = "tml-spindle-" + secrets.token_urlsafe(32)
    configs = isolated_configs(recipes, frontend)
    print(f"Deploying checkout {commit} as {frontend} in {REGRESSION_ENV}", flush=True)
    tokens = modal.Workspace.from_context().proxy_tokens
    token = tokens.create(name=frontend)
    try:
        tokens.allow(token.token_id, REGRESSION_ENV)
        modal.Secret.objects.create(
            frontend,
            {
                "TINKER_API_KEY": api_key,
                "MODAL_PROXY_TOKEN_ID": token.token_id,
                "MODAL_PROXY_TOKEN_SECRET": token.token_secret,
            },
            environment_name=REGRESSION_ENV,
        )
        deploy(configs)
        url = modal.Function.from_name(
            frontend, "server", environment_name=REGRESSION_ENV
        ).get_web_url()
        if not url:
            raise RuntimeError("Regression frontend has no URL")
        print(f"Regression server: {url} ({commit})", flush=True)
        yield url, api_key, commit, frontend
    finally:
        try:
            stop(frontend, REGRESSION_ENV)
            for suffix in ("checkpoints", "bulletin", "usage"):
                modal.Volume.objects.delete(
                    f"{frontend}-{suffix}",
                    allow_missing=True,
                    environment_name=REGRESSION_ENV,
                )
            modal.Secret.objects.delete(
                frontend, allow_missing=True, environment_name=REGRESSION_ENV
            )
        finally:
            tokens.delete(token.token_id)
