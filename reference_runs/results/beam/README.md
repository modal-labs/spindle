# Beam LoRA reference run results

This validation stopped after **42 accepted updates**. The latest complete AIME25
evaluation is update 40; no update-42 accuracy is reported. The originally planned
100-update learning test was not completed.

| AIME25 evaluation | Accuracy | Mean response tokens |
| --- | ---: | ---: |
| Initial | 97/240 (40.42%) | 3,180.54 |
| Update 40 | 154/240 (64.17%) | 2,492.28 |

Each evaluation uses 30 problems, eight samples per problem and a 4,096-token
response cap. Independent strict regrading agreed with all 2,160 local responses
across nine evaluations. The update-40 accuracy change is +23.75 percentage points,
with a descriptive paired-problem 95% bootstrap interval of +12.92 to +35.83 points.
This is an intermediate comparison, not the predeclared update-100 test.

Median time between accepted updates was **621.90 seconds**, including the planned
cold restart. Median native forward/backward time was 449.40 seconds; median trainer
API wall time was 513.19 seconds. Wall time through the requested stop was 10.90 hours.

The green curves are approximate digitizations of the
[published Beam LoRA run](https://huggingface.co/blog/reflection-beam#train-with-lora).
Its reported step axis has not been verified as optimizer updates. Hardware and
orchestration differ; the comparison does not establish parity.

The run used 16 H200 trainer GPUs and eight B300 sampler GPUs. Runtime pins were
Miles `6f5a4a84`, trainer SGLang `9ac1c65b` and sampler SGLang `991fb37566`.
Client code was `6bcf5aa` through update 10 and `35f365f` afterward. These results
predate the Miles v0.2.0 recipe and validate only the runtime pins above.

Only [configuration](config.json), [metrics](metrics.jsonl) and this README are
checked in. Metrics contain update/evaluation aggregates and accuracy per problem
in dataset order, averaged over eight samples. They contain no prompts, answers,
labels or routing payloads. Detailed responses,
checkpoint journals and audit files stay local and are ignored by Git.

Charts are generated locally and shown in the PR description. Regenerate them
from these metrics without the detailed responses:

```bash
uv run --with matplotlib==3.11.2 python reference_runs/beam.py \
  --output reference_runs/results/beam --plot-only
```
