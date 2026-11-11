# Cost attribution

When several runs share one trainer and sampler, Modal bills the shared apps, not the runs. Spindle records how many tokens each session uses and serves them through Tinker's `get_billing_usage`, labeled with the runs, teams or other labels you set.

## Label your runs

Pass labels as `user_metadata` when you create the Tinker client. Every training model and sample request in that session is counted under these labels.

```python
import tinker

service = tinker.ServiceClient(
    base_url=SPINDLE_URL,
    api_key=SPINDLE_API_KEY,
    user_metadata={"run_id": "run-A", "team_id": "t-ranking"},
)
training = service.create_lora_training_client(base_model="Qwen/Qwen3.5-9B-Base")
```

Spindle keeps `run_id` and `team_id` and drops other keys. A session that doesn't set one of them shows under `unknown` when you group by it. To keep other keys, such as `project`, list them all in `SPINDLE_USAGE_LABEL_KEYS` in the `spindle-api` Modal secret and redeploy:

```dotenv
SPINDLE_USAGE_LABEL_KEYS=run_id,team_id,project
```

Values must be strings of 1 to 256 characters. Spindle drops a longer or empty value and logs a warning; the session is still created, and its usage shows under `unknown`. Labels are fixed when the session is created.

## How it works

- Trainers count the input tokens of every completed `forward` and `forward_backward`, including each image's `expected_tokens`, and the time they spend busy, idle and starting up.
- Samplers count prompt tokens, how many of them came from the prefix cache, and sampled tokens. A request for 4 samples counts its prompt 4 times, since each sample is a separate request to the inference server. Each sampler pool container also records its start-up time and, once a second, whether SGLang had a request running, queued or finished. Spindle turns SGLang metrics on for this. A recipe that turns them off with `sglang_cfg.enable_metrics=False` gets a warning from `spindle deploy`, and its pools' whole cost is split by tokens.
- Each container adds its counts up per minute and session, and writes them to a Modal Volume named `<frontend>-usage`, such as `spindle-usage`. Training and sampling never wait on these writes.
- Each time the frontend's cleaner runs (every `sweep_interval_s`, 5 minutes by default), it rolls each finished hour and day into one file, so queries stay fast. It rolls up each hour once an hour has passed since it ended. Rows written more than 10 minutes after their hour, for example because a container couldn't reach the Volume, go to a separate folder that queries always read, so they're still counted.
- The cleaner deletes raw per-minute files 14 days after their hour, once that day is rolled up. The rollups keep the history, so older windows can still be queried.

## Get token counts

`get_billing_usage` returns hourly token counts per session, in the same format as Tinker. The window must start and end on the hour, be at most 14 days long, and not start in the future.

- Recent hours can be incomplete: containers write usage a minute or two after it happens, and rows that arrive late are added when they arrive.
- A window with more than 100,000 events returns an error asking for a shorter window.
- Reads hold one lock on the frontend, so they run one at a time. If a request times out, ask again: a read for the same window is kept for 10 minutes after it finishes.
- If the frontend can't read the Volume, the request fails with status 424, so the SDK doesn't retry it for minutes. Try again later.
- Anyone with the deployment's API key can read every session's usage and labels.

```python
from datetime import UTC, datetime

usage = service.create_rest_client().get_billing_usage(
    datetime(2026, 11, 10, 22, tzinfo=UTC),
    datetime(2026, 11, 10, 23, tzinfo=UTC),
).result()
```

For two runs that each trained, then sampled 4 sequences from a 64-token prompt, the endpoint returns:

```json
{
  "data": [
    {"bucket_start": "2026-11-10T22:00:00Z", "bucket_end": "2026-11-10T23:00:00Z", "base_model": "Qwen/Qwen3.5-9B-Base", "session_id": "8d41b07c", "event_info": {"type": "training", "token_count": 800}},
    {"bucket_start": "2026-11-10T22:00:00Z", "bucket_end": "2026-11-10T23:00:00Z", "base_model": "Qwen/Qwen3.5-9B-Base", "session_id": "3f2c9e1a", "event_info": {"type": "training", "token_count": 288}},
    {"bucket_start": "2026-11-10T22:00:00Z", "bucket_end": "2026-11-10T23:00:00Z", "base_model": "Qwen/Qwen3.5-9B-Base", "session_id": "8d41b07c", "event_info": {"type": "sampling_prefill", "cached": false, "token_count": 256}},
    {"bucket_start": "2026-11-10T22:00:00Z", "bucket_end": "2026-11-10T23:00:00Z", "base_model": "Qwen/Qwen3.5-9B-Base", "session_id": "3f2c9e1a", "event_info": {"type": "sampling_prefill", "cached": false, "token_count": 128}},
    {"bucket_start": "2026-11-10T22:00:00Z", "bucket_end": "2026-11-10T23:00:00Z", "base_model": "Qwen/Qwen3.5-9B-Base", "session_id": "3f2c9e1a", "event_info": {"type": "sampling_prefill", "cached": true, "token_count": 128}},
    {"bucket_start": "2026-11-10T22:00:00Z", "bucket_end": "2026-11-10T23:00:00Z", "base_model": "Qwen/Qwen3.5-9B-Base", "session_id": "3f2c9e1a", "event_info": {"type": "sampling_sample", "token_count": 64}},
    {"bucket_start": "2026-11-10T22:00:00Z", "bucket_end": "2026-11-10T23:00:00Z", "base_model": "Qwen/Qwen3.5-9B-Base", "session_id": "8d41b07c", "event_info": {"type": "sampling_sample", "token_count": 64}}
  ],
  "sessions": {
    "8d41b07c": {"user_metadata": {"run_id": "run-B", "team_id": "t-ads"}},
    "3f2c9e1a": {"user_metadata": {"run_id": "run-A", "team_id": "t-ranking"}}
  }
}
```

`training` includes `forward` tokens. `sessions` maps each session to its labels.

The Tinker CLI prints the same data as a table, or as CSV with `--csv -`. Point it at your Spindle server with `TINKER_BASE_URL` and `TINKER_API_KEY`:

```bash
tinker billing usage 2026-11-10T22:00:00Z 2026-11-10T23:00:00Z --csv -
```

```text
bucket_start,bucket_end,base_model,user_id,user_name,session_id,project_id,type,token_count,cached
2026-11-10T22:00:00Z,2026-11-10T23:00:00Z,Qwen/Qwen3.5-9B-Base,,,8d41b07c,,training,800,
2026-11-10T22:00:00Z,2026-11-10T23:00:00Z,Qwen/Qwen3.5-9B-Base,,,3f2c9e1a,,training,288,
2026-11-10T22:00:00Z,2026-11-10T23:00:00Z,Qwen/Qwen3.5-9B-Base,,,8d41b07c,,sampling_prefill,256,False
2026-11-10T22:00:00Z,2026-11-10T23:00:00Z,Qwen/Qwen3.5-9B-Base,,,3f2c9e1a,,sampling_prefill,128,False
2026-11-10T22:00:00Z,2026-11-10T23:00:00Z,Qwen/Qwen3.5-9B-Base,,,3f2c9e1a,,sampling_prefill,128,True
2026-11-10T22:00:00Z,2026-11-10T23:00:00Z,Qwen/Qwen3.5-9B-Base,,,3f2c9e1a,,sampling_sample,64,
2026-11-10T22:00:00Z,2026-11-10T23:00:00Z,Qwen/Qwen3.5-9B-Base,,,8d41b07c,,sampling_sample,64,
```

This is Tinker SDK 0.26; newer versions add cost columns, which Spindle leaves empty. Add `--sessions-csv -` for each session's labels.

## Split cost by run or team

`spindle usage` splits each app's Modal cost by the tokens each session made the GPU process: train tokens on trainers, and uncached prompt tokens and sampled tokens on samplers. Every token counts the same, so this is an estimate; a sampled token takes more GPU time than a prompt token.

Run it for a window that has already ended, naming your frontend and its Modal environment:

```bash
spindle usage --frontend spindle --env main \
  --since 2026-11-10T22:00:00Z --until 2026-11-10T23:00:00Z --group-by team_id
```

```text
team_id              USD
t-ads               0.41
t-ranking           0.22
------------------------
idle                0.53
cold_start          0.33
unattributed        0.05
total               1.54
```

It reads usage through the frontend, so it needs a frontend deployed with this version. It reads the window's cost of every app the deployment created from Modal's billing report. The report can lag a few hours behind; if it doesn't cover the whole window yet, `spindle usage` splits only the usage up to the last hour billed and says so. To use other costs, pass `--costs` with a JSON file of app name to USD, such as `{"spindle-trainer-qwen35-9b-lora-2k": 1.00}`.

The lines add up to those costs:

- `idle` and `cold_start`: time trainers and sampler pools spent waiting for work or starting up. No run is charged for it, but you still pay Modal for it. Idle and cold start are priced by the container time that was recorded, so they're overstated for a window where some containers recorded none, such as old pool containers still serving right after an upgrade.
- `unattributed`: cost with no tokens to split it by, such as the frontend, a pool that served nothing, or a trainer that only saved weights.
- `unknown`: usage from sessions without that label.

Add `--hide-overhead` to show only the groups; `total` is then their sum.

Add `--format csv` for CSV.
