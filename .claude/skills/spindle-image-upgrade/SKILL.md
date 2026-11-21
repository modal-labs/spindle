---
name: spindle-image-upgrade
description: Move a few bundled Spindle configs to the newest registered trainer and sampler images, validate each one with a reference run in parallel workers, and open PRs for the results. Use for the rolling image upgrade, by hand or on a schedule.
---

# Rolling image upgrade

Every bundled config pins its own `trainer_image` and `inference_image`, and named images are frozen (docs/deployment-configs.md, "Changing an image"). New images are added by whoever needs them. This skill moves the configs that are furthest behind onto the newest images, a few at a time, so that no config stays on an old image for long.

You are the controller. You choose the configs, start one worker per config, wait for them, then open the PRs. Workers do the GPU work; don't deploy anything yourself.

Inputs, from the invocation or the user's agent instructions:

- How many configs to attempt: `count`, a number of configs, or `fraction`, a share of all bundled configs. Default: `count` 5.
- Modal environment for validation deployments. Each worker deploys its own app, named by the `app` field of its entry.
- `TINKER_API_KEY`: the key in that environment's `spindle-api` secret. Modal can't print a secret, so it must already be set in the environment workers start with. Check that it is set, and never put it in a prompt.
- `reviewer`: GitHub login to request review from. Default: assign the PRs to the account `gh` is logged in as.

## 1. Choose configs

Work from the branch you are on; call it the base branch. Workers branch from it and every PR targets it, so it must exist on `origin`.

```bash
BASE=$(git branch --show-current)
git push -u origin "$BASE"   # skip if origin already has it
uv sync --locked --python 3.12
SKIP=$(gh pr list --state open --limit 200 --json headRefName \
  -q '.[].headRefName | select(startswith("image-upgrade/")) | sub("^image-upgrade/"; "") | sub("-[0-9a-f]{6}$"; "") | "--skip " + .')
uv run --no-sync python .claude/skills/spindle-image-upgrade/select_configs.py --count 5 $SKIP   # or --fraction 0.2
```

Configs that still have an open upgrade PR are skipped until that PR is merged or closed. The script prints a JSON list. It orders configs by how many newer base images their pins are behind, breaks ties at random, and takes `count` configs, or `ceil(fraction * number of bundled configs)`. Each entry has the current and target pins, `validations` (the definitions in `reference_runs/configs/` that deploy that config), a random `suffix` for branch and file names, and the worker's Modal `app`. An empty list means every config is on the newest images: report that and stop.

## 2. Start one worker per config

Each worker gets its own worktree on a new branch from `$BASE`, and the instructions in [worker.md](worker.md) followed by its JSON entry, the base branch and the Modal environment. With bb:

```bash
{ cat .claude/skills/spindle-image-upgrade/worker.md; echo; echo '```json'; echo "$ENTRY"; echo '```'; \
  echo "Base branch: $BASE. Modal environment: $ENV."; } \
| bb thread spawn --project "$(bb status --json | jq -r .project.id)" --parent-self \
    --new-environment worktree --base-branch "$BASE" \
    --title "Image upgrade: $CONFIG" --prompt-file -
```

Without bb, use your agent tool's subagents, one isolated worktree each, each on a new branch from `$BASE`. Start all workers before you wait on any of them.

## 3. Wait and answer

Wait for each worker with `bb thread wait <id> --timeout 6h`, then read `bb thread output <id>`. Reference runs can take hours: a wait that times out means the worker is still running, so wait again. Answer worker questions with what you know; ask the user only when the answer changes shared code or another config. Each worker pushes its branch and finishes with one of:

- `clean`: the upgrade passed and the branch changes only the config's pins and files under `reference_runs/`.
- `changed`: the upgrade passed, but only after other changes, such as a config setting or a backend fix. Workers never create images.
- `failed`: the upgrade didn't pass. The report says which step failed, why, and what would unblock it.

Each report includes the path of a `plot.py` chart comparing the baseline with the upgrade run, when an upgrade run produced metrics.

## 4. Open the PRs

Open one PR for all `clean` configs together, and one PR per `changed` or `failed` config. Every PR targets `$BASE` and is assigned the same way (`--assignee @me`, or `--reviewer <reviewer>`). Open `failed` PRs as drafts.

**Clean PR.** Merge the clean branches into one new branch and run the checks from AGENTS.md:

```bash
git fetch origin
git switch -c image-upgrades/$(date +%Y-%m-%d)-$(openssl rand -hex 3) "origin/$BASE"
INDEX=reference_runs/index.toml
for B in <clean worker branches>; do
  if ! git merge --no-ff --no-edit "origin/$B"; then
    # run.py only appends to index.toml, so a conflict there is two sets of new
    # entries. Keep ours and append the text the worker's branch added.
    [ "$(git diff --name-only --diff-filter=U)" = "$INDEX" ] || { echo "resolve $B by hand"; break; }
    START=$(git merge-base "origin/$BASE" "origin/$B")
    git checkout --ours $INDEX
    git show "origin/$B:$INDEX" | tail -c +$(( $(git show "$START:$INDEX" | wc -c) + 1 )) >> $INDEX
    git add $INDEX && git commit --no-edit
  fi
done
uv run pre-commit run --all-files
uv run --no-sync pytest -q
```

If a merge conflicts anywhere other than `index.toml`, stop and resolve it by hand. If the checks fail, move the config that causes it to its own PR as `changed`.

If a worker reports that its upgrade breaks something existing users rely on, such as restoring checkpoints saved on the old trainer image, add one CHANGELOG.md entry on the clean branch that says what breaks and how to keep the old behavior. Otherwise don't touch the changelog.

**Changed and failed PRs.** Open each from its worker's branch as it is, so the reviewer sees exactly what that config needed.

**Charts.** Attach each worker's chart with `gh pr create --attach` (gh 2.103 or later). Write the body next to the charts and reference them by relative path; gh uploads each attached file and rewrites the reference to the uploaded image:

```bash
# Charts are /tmp/image-upgrade-plots/<config>-<suffix>.png; write body-<suffix>.md beside them.
# It contains ![qwen3-8b-lora-32k](./qwen3-8b-lora-32k-1a2b3c.png)
gh pr create --base "$BASE" --title "..." --body-file /tmp/image-upgrade-plots/body-1a2b3c.md \
  --attach /tmp/image-upgrade-plots/qwen3-8b-lora-32k-1a2b3c.png --assignee @me
```

If a worker's chart can't be read, check out its branch and rerun `reference_runs/plot.py` with the run IDs from its report.

**Descriptions** follow AGENTS.md. The clean PR has one row per config, then every config's chart under a heading with its name:

| Config | Trainer image | Sampler image | Validation run | Result |
| --- | --- | --- | --- | --- |
| `qwen3-8b-lora-32k` | `miles-v0.1.1-52a7a99` → `miles-v0.2.0-e814245` | `sglang-v0.5.23-c4182b2` → `sglang-v0.5.23-3db151a` | `reference_runs/configs/<name>.toml` (new) | clean |

A changed PR says what had to change and why, with its chart. A failed PR says which step failed, the error and what would unblock it, with a chart if the run produced metrics. Mark validation definitions that a worker proposed as new in every PR, since the reviewer has to approve the workload as well as the result. When a baseline ran on other images, code or hardware than the current pins, say so next to its chart instead of hiding the difference.

Finish by telling the user each PR link and which configs it covers.
