# Preliminary validation status

Completed checks:

- All 21,186 selected dataset rows matched the per-row hashes in the frozen
  membership manifest, using locally cached public source data.
- Panel sizes and source-ID disjointness passed for all four tasks. Arithmetic
  balance/problem separation, BoolQ passage separation, and MBPP prompt/train/test
  separation passed. MBPP discovery contains 374 training rows only. Arithmetic
  discovery preserves 2,000 source rows with 1,999 distinct problem texts; its
  one existing repeated problem is disclosed in `configs/data_notes.json`.
- Actual checkpoint tokenizers rendered Arithmetic and three-shot MBPP prompts
  for all three models. SmolLM3's `/no_think` mode and empty thinking prefix passed.
- **25 CPU tests passed.** The tests cover task preparation, paired metrics, immutable result writes,
  native Qwen2/Llama/SmolLM3 head-hook activity, slice isolation and restoration,
  selection prerequisites, 175-candidate search, finalist freezing, confirmation,
  held-out rank preservation, resumption and tamper rejection. Further checks cover
  numerical likelihood offsets/normalization, full and partial generation batches
  on all three tiny native architectures, stop-token trimming, complete baseline
  evidence, stage/condition/input binding, and sandbox setup-error classification.
- Preparation and idempotent preparation resumption passed for all twelve
  model/task combinations, without loading model weights.
- All Python modules compile. The seccomp C helper compiles with
  `cc -O2 -Wall -Werror`.
- Working, staged and historical committed contents, commit messages and
  anonymous author/committer metadata are checked before publication.

Limits:

- No full 3B GPU evaluation or twelve-cell sweep was run for this release. Tiny
  native model tests establish hook behavior, not benchmark accuracy or complete
  GPU/runtime compatibility.
- MBPP sandbox execution could not complete in the preparation environment:
  Apptainer exited with a denied socket operation before executing the fixtures.
  The C helper compiles, but container runtime isolation and program scoring need
  `sandbox_check.py` to pass on the execution host. The experiment runner enforces
  that check before loading the model. No host execution fallback is provided.
- Source content was checked against cached Arrow datasets; a clean network
  download and fresh virtual-environment installation were not tested here.
- This release specifies its own documented adapters and fixed panels. It is
  preliminary reproducibility code, not a claim of reproducing all historical
  scores, prompt variants, or experiment settings exactly.
