---
name: spindle-image-upgrade
description: Move a few bundled Spindle configs to the newest registered trainer and sampler images, validate each one with a reference run in parallel workers, and open one PR. Use for the rolling image upgrade, by hand or on a schedule.
---

# Rolling image upgrade

Every bundled config pins its own `trainer_image` and `inference_image`, and named images are frozen (docs/deployment-configs.md, "Changing an image"). New images are added by whoever needs them. This skill moves the configs that are furthest behind onto the newest images, a few at a time, so that no config stays on an old image for long.

You are the controller. You choose the configs, start one worker per config, wait for them, merge what they validated and open the PR. Workers do the GPU work; don't deploy anything yourself.

Inputs, from the invocation or the user's agent instructions:

- How many configs to attempt: `count`, a number of configs, or `fraction`, a share of all bundled configs. Default: `fraction` 0.2.
- Modal environment for validation deployments. Workers deploy one app each, named `<app prefix>-<config name>`; the default prefix is `upgrade`.
- `reviewer`: GitHub login to request review from. Default: assign the PR to the account `gh` is logged in as.

## 1. Choose configs

```bash
git fetch origin main
git switch -c image-upgrades/$(date +%Y-%m-%d) origin/main
uv sync --locked --python 3.12
uv run --no-sync python .claude/skills/spindle-image-upgrade/select_configs.py --fraction 0.2   # or --count 5
```

The script prints a JSON list. It orders configs by how many newer base images their pins are behind, breaks ties at random, and takes `count` configs, or `ceil(fraction * number of bundled configs)`. Each entry has the current and target pins and `validations`, the definitions in `reference_runs/configs/` that deploy that config. An empty list means every config is on the newest images: report that and stop.

## 2. Start one worker per config

Each worker gets its own worktree, branched from `origin/main`, and the instructions in [worker.md](worker.md) followed by its JSON entry, the Modal environment and its app name. With bb:

```bash
{ cat .claude/skills/spindle-image-upgrade/worker.md; echo; echo '```json'; echo "$ENTRY"; echo '```'; \
  echo "Modal environment: $ENV. App: $PREFIX-$CONFIG."; } \
| bb thread spawn --project "$(bb status --json | jq -r .project.id)" --parent-self \
    --new-environment worktree --base-branch origin/main \
    --title "Image upgrade: $CONFIG" --prompt-file -
```

Without bb, use your agent tool's subagents, one isolated worktree each. Start all workers before you wait on any of them.

## 3. Wait and answer

Wait for each worker (`bb thread wait <id>`, then `bb thread output <id>`). Answer worker questions with what you know; ask the user only when the answer changes shared code or another config. Each worker finishes with one of:

- `validated`: a pushed branch, the validation run and its comparison.
- `blocked`: the step that failed and why. A config that needs a change to a shared image or backend code is blocked, not fixed in place.

## 4. Merge and open the PR

Merge each validated branch into your branch with `git merge --no-ff origin/<branch>`. Two workers that each added an image will conflict in `TRAINER_IMAGES` or `SAMPLER_IMAGES`; keep both entries in the order their releases came out. Then run the checks from AGENTS.md:

```bash
uv run pre-commit run --all-files
uv run --no-sync pytest -q
```

Open the PR when at least one config validated. If none did, report the blocked list to the user and stop.

```bash
gh pr create --base main --title "Move <n> configs to newer images" --body-file <body> \
  --assignee @me   # or --reviewer <reviewer>
```

The body follows AGENTS.md. It has one row per attempted config:

| Config | Trainer image | Sampler image | Validation run | Result |
| --- | --- | --- | --- | --- |
| `qwen3-8b-lora-32k` | `miles-v0.1.1-52a7a99` → `miles-v0.2.0-8c5de26` | unchanged | `reference_runs/configs/<name>.toml` (new) | validated |

Mark validation runs that a worker proposed in this PR as new, since the reviewer has to approve the workload as well as the result. List blocked configs with their reasons below the table. Finish by telling the user the PR link and the blocked configs.
