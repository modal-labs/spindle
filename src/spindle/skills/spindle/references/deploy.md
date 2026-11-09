# Deploy

## Connect

`spindle.ServiceClient(base_url=url, api_key=key)` or env `TINKER_BASE_URL`/`TINKER_API_KEY`. Proxy-protected frontends also need `MODAL_PROXY_TOKEN_ID`/`MODAL_PROXY_TOKEN_SECRET`, which this client reads automatically.

## Install

`uv add 'modal-spindle @ git+https://github.com/modal-projects/spindle.git'`. Python 3.12 for scoped runs and deploys. Select the Modal workspace and `MODAL_ENVIRONMENT` before creating secrets.

## Credentials

| Credential | Purpose |
| --- | --- |
| Modal token / profile | Deploy and manage resources |
| `TINKER_API_KEY` in `spindle-api` secret | Authenticate Tinker clients |
| `MODAL_PROXY_TOKEN_ID`/`MODAL_PROXY_TOKEN_SECRET` in `spindle-proxy` secret | Reach protected sampler pools |
| `HF_TOKEN` in `huggingface-secret` secret | Download base weights |

## Commands

- `uv run spindle deploy`: interactive picker; without a terminal and without args it exits, deploying nothing. Checks secrets first and prints creation commands for missing ones.
- `uv run spindle deploy <name...>`: bundled configs by name. `--all`: every bundled config (in `src/spindle/configs`).
- `uv run spindle deploy config.py`: custom config file.
- Each deploy adds or updates the listed configs in the per-frontend registry; configs not listed remain served.
- `POST /api/v1/spindle/deployments` (JSON `name` + `base` and `overrides`, or `recipe`) registers a config on a running frontend and deploys its trainer and inference apps in the background; `DELETE /api/v1/spindle/deployments/{name}` removes one and stops its apps (HTTP 409 listing the models still training on it unless `?force=true`). Invalid configs return HTTP 400, a different config under an existing name 409, an unknown name 404. `GET /api/v1/spindle/deployments` reports each config's `apps` as `"deploying"`, `"ready"`, `"failed"` or `"missing"`; re-sending the same config retries a failed deploy.
- `uv run spindle add NAME --base BASE [--FIELD=VALUE | --FIELD VALUE ...]`: register a config through that API and wait until its apps are ready. Flags after `--base` are recipe fields (dotted fields work; values parse as JSON, else as strings); unknown fields are rejected before any request. Needs `TINKER_BASE_URL` and `TINKER_API_KEY`.
- `uv run spindle remove NAME [--force]`: remove a config from a running frontend. If models still train on it, lists them and prompts the user first; `--force` skips the interactive prompt.

## Custom configs

```bash
spindle config init --preset qwen35-9b-lora-16k > my_model.py
spindle config validate my_model.py
spindle deploy my_model.py
```

Subclass `BaseConfig` or a bundled `Config`. Dotted `overrides` set single backend options (`{"miles_cfg.max_tokens_per_gpu": 8192}`). `trainer_gpu`/`inference_gpu` accept a preference list like `["H200", "B300"]`; Modal schedules the first with capacity. Platform settings (`platform.frontend`, `.modal.environment`, `.modal.region`, `.secrets.*`, `.storage.*`) must match across all configs in one deploy; `--app`/`--env`/`--region` flags override.

Fixed a bundled config? See the `spindle-dev` skill's upstream-config.md.

## Update isolation

- Apps: `{frontend}-trainer-{name}`, `{frontend}-inference-{name}`; pools `{frontend}-lora-{name}`, `{frontend}-fft-{digest}-*`, where `{frontend}` is `spindle` or the `--app` value. Recipe names must be unique per frontend.
- Inference-only change reuses the trainer app and vice versa. `--refresh-trainer`/`--refresh-inference` for source-only upgrades.
- Inference changes redeploy existing pools before the frontend publishes; an interrupted update retries with the same command. Region/storage/secret changes refresh both apps.
- Drain jobs before removing a recipe or changing model/context/adapter incompatibly.
- A startup failure pauses new starts for five minutes; `spindle deployment retry --frontend <app> <definition_id>` clears it.

## Teardown

Stopped heartbeats trigger a periodic cleaner that reclaims idle models and latest pools (`modal app list` to check). Tear down: `spindle stop --app <frontend> --env <env>` stops the frontend, workers, and samplers; `modal app stop <frontend>` alone leaves samplers running.
