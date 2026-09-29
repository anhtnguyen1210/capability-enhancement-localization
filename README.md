# Capability Enhancement through Localization

Preliminary anonymous code artifact for attention-head ablation experiments.
The implementation supports all twelve model/task combinations below. It includes
intact evaluation, singleton screening, exhaustive one-to-three-head search,
fresh discovery confirmation, and evaluation of ten frozen finalists.

## Models

| CLI name | Public checkpoint | Query heads |
| --- | --- | --- |
| `qwen` | `Qwen/Qwen2.5-3B-Instruct` | 36 layers × 16 = 576 |
| `llama` | `meta-llama/Llama-3.2-3B-Instruct` | 28 layers × 24 = 672 |
| `smollm3` | `HuggingFaceTB/SmolLM3-3B` | 36 layers × 16 = 576 |

Checkpoint revisions are pinned in `configs/models.json`. Llama requires access
under its upstream model terms; authenticate locally using Hugging Face tooling.
Do not put access tokens in the repository. Native SmolLM3 layers, including
layers without rotary position embeddings, are preserved. SmolLM3 generation uses
`enable_thinking=False`, `/no_think`, and an empty input thinking block. Qwen2.5
and Llama3.2 have no equivalent thinking-mode switch. BoolQ and HellaSwag score
continuations directly and do not generate a reasoning response.

## Data and scoring

| Dataset | Discovery | Held-out | Metric |
| --- | ---: | ---: | --- |
| Arithmetic | 2,000 | 1,000 | Strict signed-integer accuracy |
| HellaSwag | 2,000 | 10,042 | Character-normalized choice likelihood (`acc_norm`) |
| BoolQ | 2,000 | 3,270 | Raw choice likelihood (`acc`) |
| MBPP | 374 | 500 | Greedy pass@1 on all three supplied tests |

`configs/panels.json` freezes ordered source indices and per-row hashes; datasets
are downloaded separately from pinned public revisions. Arithmetic uses 200
examples per subtask for discovery and 100 per subtask for held-out evaluation,
from ten arithmetic subtasks. Both panels are drawn from the source validation
split and have disjoint problem text. HellaSwag and BoolQ use training examples
for discovery and their full labeled validation split for held-out evaluation.
BoolQ discovery keeps one question per passage and excludes held-out passages.
MBPP uses only its 374 training tasks for discovery and all 500 test tasks for
held-out evaluation; its 90 validation tasks are excluded. Prompt task IDs 2–4
supply the three fixed few-shot examples and are excluded from both panels.

Held-out means excluded from selection in this workflow; it is not a claim that
these public benchmarks have never previously been evaluated. The release freezes
one common panel per task for all three models. It does not claim these memberships
match every historical experiment.

Inference uses batches of eight examples, BF16, SDPA and greedy generation.
Arithmetic has a 32-token cap. MBPP has a 256-token cap, a three-shot prompt, an
assistant `[BEGIN]` prefill, and `[DONE]` stopping. MBPP programs run only through
the isolated Apptainer/seccomp scorer with a three-second program timeout.
Formatting and token-cap diagnostics are saved. Generation applies the native
chat template once and then tokenizes without adding a second set of special
tokens. Templates use a fixed date. These choices define this preliminary release;
they are not a claim of bit-for-bit parity with all historical adapters/results.

## Installation

Use Linux x86-64, Python 3.12, a CUDA GPU supporting BF16, and sufficient GPU
memory for the model and batch. No cluster scheduler or site-specific paths are
required. The full search is expensive; baseline is the initial smoke/consistency
stage. This repository does not automatically submit compute jobs.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python prepare_data.py --output data
```

`prepare_data.py` verifies every selected source hash, panel count, and split
invariant. It refuses to overwrite existing prepared files. Dataset downloads
require network access. No model weights or dataset text are distributed here.
See `THIRD_PARTY.md` for upstream sources and terms.

MBPP additionally needs Apptainer and a C compiler. Build a Python-only sandbox
image on a machine where Apptainer builds are supported, then validate it:

```bash
apptainer build sandbox.sif sandbox.def
python sandbox_check.py --image sandbox.sif --output runs/sandbox-check
```

The image recipe is a starting point: each experiment hashes the exact built
image, which must remain identical across stages. A failed isolation check is a
hard error; there is no fallback to executing generated code on the host.

## Run one experiment

Run the stages in this order, using the same run directory. For example:

```bash
python experiment.py --model smollm3 --task boolq --stage prepare --run runs/smollm3-boolq
python experiment.py --model smollm3 --task boolq --stage baseline --run runs/smollm3-boolq
python experiment.py --model smollm3 --task boolq --stage singleton --run runs/smollm3-boolq
python experiment.py --model smollm3 --task boolq --stage search --run runs/smollm3-boolq
python experiment.py --model smollm3 --task boolq --stage confirmation --run runs/smollm3-boolq
python experiment.py --model smollm3 --task boolq --stage heldout --run runs/smollm3-boolq
```

Replace `smollm3` with `qwen` or `llama`, and `boolq` with `arithmetic`,
`hellaswag`, or `mbpp`. For MBPP add `--sandbox-image sandbox.sif` to **every**
stage, including preparation. Use a separate run directory for each model/task.
The twelve combinations are supported independently; no GPU allocation is
started by installation, preparation, or the unit tests.

The baseline stage evaluates intact twice and requires exact record agreement.
The singleton stage removes each query head at the input to the attention output
projection; KV groups are not removed. Discovery ranks heads by paired gain,
fewer damaged examples, higher correct count, then head coordinates. Ten nominees
produce 45 pairs and 120 triples; the ten singleton results are reused, giving
175 total candidates. Combinations use the same ranking with fewer removed heads
before the coordinate tie-break. The ten finalists are frozen before fresh
discovery confirmation and held-out evaluation. Confirmation does not reselect.
All ten are reported in discovery order; the primary result is discovery rank 1.

Each stage gets a local intact comparator. Conditions retain per-example outputs,
paired rescue/damage counts, model/software/GPU-type provenance, intervention
activity and hashes. Existing complete conditions are validated on resume;
incomplete attempts remain in a separate directory. A run lock prevents concurrent
writers. This preliminary launcher is sequential within each model/task and does
not include distributed scheduling. Stop and investigate repeat disagreement;
changing the code, sources, model or sandbox requires a fresh run directory.

Final results are in `runs/<experiment>/report.json`. No numerical paper results
are bundled or asserted by this preliminary implementation.

## Release checks and anonymity

See `VALIDATION.md` for checks performed and remaining validation limits. Only
reviewed source, tests, public checkpoint identifiers and dataset membership
metadata belong in a release. Generated `data/` and `runs/` are ignored.
`audit_release.py` checks tracked files and anonymous Git author/committer metadata:

```bash
python audit_release.py
```

Additional private names can be supplied with repeatable `--forbid` arguments;
keep those arguments out of published artifacts. Source hygiene cannot hide a
GitHub owner's username or repository activity. Distribute a separately verified
anonymous review link or a source archive that excludes `.git` and private files.
