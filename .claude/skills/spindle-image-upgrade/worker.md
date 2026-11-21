# Image upgrade worker

You move one bundled Spindle config to newer images and prove it still works. The JSON entry at the end of this message names the config, its current pins, the target pins and the validation definitions that already deploy it. The message also names the base branch, the Modal environment and your app. Follow AGENTS.md. Work only in your own worktree and your own Modal app.

## 1. Change the pins

1. Create branch `image-upgrade/<config name>` from the base branch.
2. In the config file, set `trainer_image` and `inference_image` to the targets. Never create or edit an image: the targets are the newest registered images, and the only goal is to move the config onto them. That includes configs pinned to a model-specific image, such as one ending in `-glm53`.
3. Run `uv run --no-sync spindle config validate <config file>` and `uv run --no-sync pytest -q`.

If the config doesn't work on the targets, you may change config settings or backend code to make it work. Any change outside the config's pins and `reference_runs/` makes the result `changed`. Keep each change as small as the fix allows, since its PR is reviewed on its own. If it would need a new image, report `failed`.

## 2. Find the validation and its baseline

A validation is a definition in `reference_runs/configs/<name>.toml` that names the config, a workload script and fixed hardware. Its runs are logged in `reference_runs/index.toml`. Read reference_runs/README.md first.

- **Definition.** Use the one in `validations` in your entry. If there is none, write one: `config` set to your config, `script = "../scripts/guess_number.py"`, and a `[workload]` with `tokenizer_model` and the full 40-character `tokenizer_revision`. In `[config_overrides]`, fix the hardware with the fields `ConfigOverrides` in `run.py` declares: one trainer instance, one client, fixed sampler replicas, GPU types with `!` where Modal accepts them. Use `dapo_math.py` only if guess-number doesn't suit the model.
- **Baseline.** The newest run of the definition in `index.toml`. If none is logged, run the definition once on the current pins, as below with `--trainer-image` and `--inference-image` set to the current pins.

## 3. Run, compare, iterate

`spindle deploy` takes a Python config file, so `run.py --write-config` writes one from the definition: the bundled config plus the definition's hardware. Put it next to the run's output, in `reference_runs/outputs/<run id>/`, which Git ignores. Deploy it to your app, run the workload with the stock Tinker SDK, then stop the app:

```bash
DEF=reference_runs/configs/<name>.toml
RUN=reference_runs/outputs/<run id>
mkdir -p $RUN
uv run python reference_runs/run.py $DEF --write-config $RUN/deployment.py
uv run spindle deploy $RUN/deployment.py --app <app> --env <env>
export TINKER_BASE_URL=<server URL that spindle deploy printed>
export TINKER_API_KEY=<the key in the environment's spindle-api secret>
uv run python reference_runs/run.py $DEF --output $RUN.jsonl
uv run spindle stop --app <app> --env <env>
```

For the baseline, pass `--trainer-image <current> --inference-image <current>` to both `run.py` commands. If the server uses Modal proxy auth, run `run.py --output` through `examples/shared-server/with_proxy_auth.py`.

Compare the run with the baseline, and keep the chart for your report:

```bash
mkdir -p /tmp/image-upgrade-plots
uv run python reference_runs/plot.py <baseline id> <run id or JSONL> --labels Baseline Upgrade \
  --output /tmp/image-upgrade-plots/<config name>.png
```

Only compare runs with the same definition and actual hardware. The upgrade is good when:

- every step completes and every optimizer update is accepted;
- the final evaluation reward is within the run-to-run noise of the baseline; when unsure, run the baseline again to see that noise;
- the median time per update is no more than 10% slower.

If it isn't good, find the cause in the trainer and sampler logs, fix it within the rules above and rerun. `run.py` adds each completed run to `index.toml`. Commit the entries for the runs you report, with the actual hardware and notes on the results against the baseline and what the run doesn't show. Failed runs aren't indexed; keep their partial JSONL out of Git but use it for the chart.

Confirm the app is stopped with zero containers before you finish, also after a failure.

## 4. Report

Commit and push your branch, even if the upgrade failed. Finish with exactly one of:

- `clean`: the upgrade passed, and the branch changes only the config's pins and files under `reference_runs/`.
- `changed`: the upgrade passed after other changes. List each change and why it was needed.
- `failed`: the step that failed, the error, and what change would unblock it.

Every report includes the branch, the commit the runs used, the run IDs, whether the validation definition is new, the comparison table, and the absolute path of the chart, if there is one.
