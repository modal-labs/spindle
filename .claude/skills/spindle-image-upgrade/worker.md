# Image upgrade worker

You move one bundled Spindle config to newer images and prove it still works. The JSON entry at the end of this message names the config, its current pins, the target pins and the validation definitions that already deploy it. The message also names the base branch. Follow AGENTS.md. Work only in your own worktree and your own Modal app.

## 1. Change the pins

1. Create branch `image-upgrade/<config name>` from the base branch.
2. In the config file, set `trainer_image` and `inference_image` to the targets. Change only the pins.
3. If a current pin has a suffix, such as `-glm53`, the config needs model-specific build steps that the base target lacks. Add a new image that applies the same steps to the target base, named `<target>-<suffix>`, in a new module, right after the target in its registry. Never edit an existing image. A new image makes the result `changed`.
4. Run `uv run --no-sync spindle config validate <config file>` and `uv run --no-sync pytest -q`.

If the target image rejects the config, for example `miles-v0.2.0` with `megatron_to_hf_mode` set to `raw`, you may change what it takes to make the config work: a config setting, a new image variant, or a backend fix. Any change outside the config's pins and `reference_runs/` makes the result `changed`. Keep each change as small as the fix allows, since its PR is reviewed on its own. Never edit an existing named image; add a new one.

## 2. Find or propose a validation run

A validation run is a definition in `reference_runs/configs/<name>.toml` that names the config, a workload script in `reference_runs/scripts/` and fixed hardware, plus executions recorded in `reference_runs/index.toml` with their JSONL in `reference_runs/outputs/`. Read reference_runs/README.md first. Definitions don't select images; the deployed config's pins do.

- **One exists** (`validations` in your entry is not empty): the baseline is the newest execution of that definition in `index.toml`.
- **None exists:** propose one, then record a baseline on the current pins before you change anything:
  1. Write `reference_runs/configs/<name>.toml` with `config` set to your config, `script = "../scripts/dapo_math.py"` (or `guess_number.py` for a model that suits it better) and a `[tokenizer]` with the model's Hugging Face ID and full commit revision. In `[config_overrides]`, fix the hardware with the fields `ConfigOverrides` in `run.py` declares: one trainer instance, one client, fixed sampler replicas, GPU types with `!` where Modal accepts them so it doesn't upgrade them.
  2. If no script in `reference_runs/scripts/` suits the model, for example a base model that can't follow the chat prompt, stop and report `failed`: the config needs another workload script.
  3. Run it as below before step 1.2, while the config still has its current pins. This is the baseline, and the next upgrade of this config reuses it.

## 3. Run, compare, iterate

Prepare and deploy the definition to your app, run the workload with the stock Tinker SDK, then stop the app:

```bash
uv run python reference_runs/run.py reference_runs/configs/<name>.toml --write-config /tmp/<app>.py
uv run spindle deploy /tmp/<app>.py --app <app> --env <env>
# Set TINKER_BASE_URL and TINKER_API_KEY for <app>, then:
uv run python reference_runs/run.py reference_runs/configs/<name>.toml \
  --output reference_runs/outputs/<run id>.jsonl
uv run spindle stop --app <app> --env <env>
```

Compare the new run with the baseline, and keep the chart for your report:

```bash
mkdir -p /tmp/image-upgrade-plots
uv run python reference_runs/plot.py <baseline id> <run id or JSONL> --labels Baseline Upgrade \
  --output /tmp/image-upgrade-plots/<config name>.png
```

Only compare runs with the same definition and actual hardware. The upgrade is good when:

- every step completes and every optimizer update is accepted;
- the final evaluation accuracy is within the run-to-run noise of the baseline; when unsure, run the baseline again to see that noise;
- the median time between updates is no more than 10% slower.

If it isn't good, find the cause in the trainer and sampler logs, fix it within the rules above and rerun. `run.py` adds each completed run to `index.toml`. Commit the entries for the runs you report, and fill in `trainer_image` and `inference_image` with the image names, the actual hardware, and notes on the results against the baseline and what the run doesn't show. Failed runs aren't indexed; keep their partial JSONL out of Git but use it for the chart.

Confirm the app is stopped with zero containers before you finish, also after a failure.

## 4. Report

Commit and push your branch, even if the upgrade failed. Finish with exactly one of:

- `clean`: the upgrade passed, and the branch changes only the config's pins and files under `reference_runs/`.
- `changed`: the upgrade passed after other changes. List each change and why it was needed.
- `failed`: the step that failed, the error, and what change would unblock it.

Every report includes the branch, the commit the runs used, the run IDs, whether the validation definition is new, the comparison table, and the absolute path of the chart, if there is one.
