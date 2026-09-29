"""Batched teacher-forced BoolQ raw-choice scoring."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch

from boolq_tasks import canonical_hash


def _validate_scope_record(record: dict) -> None:
    observed = record.get("record_sha256")
    unhashed = dict(record)
    unhashed.pop("record_sha256", None)
    if not isinstance(observed, str) or canonical_hash(unhashed) != observed:
        raise ValueError("BoolQ scope record content hash differs")
    if record.get("choices") != ["no", "yes"]:
        raise ValueError("BoolQ choices differ")
    tokenizations = record.get("choice_tokenizations")
    if not isinstance(tokenizations, list) or len(tokenizations) != 2:
        raise ValueError("BoolQ tokenization inventory differs")
    for tokenization in tokenizations:
        context_ids = tokenization.get("context_ids")
        continuation_ids = tokenization.get("continuation_ids")
        if not all(
            isinstance(values, list)
            and values
            and all(isinstance(value, int) for value in values)
            for values in (context_ids, continuation_ids)
        ):
            raise ValueError("BoolQ token IDs are invalid")


def _chunks(values: Sequence[dict], size: int) -> Iterable[Sequence[dict]]:
    if size <= 0:
        raise ValueError("batch_size_examples must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _score_batch(
    model,
    records: Sequence[dict],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, list[dict]]:
    sequences = []
    metadata = []
    for record in records:
        for choice_index, tokenization in enumerate(record["choice_tokenizations"]):
            context_ids = tokenization["context_ids"]
            continuation_ids = tokenization["continuation_ids"]
            sequences.append([*context_ids, *continuation_ids])
            metadata.append(
                {
                    "sample_id": record["sample_id"],
                    "choice_index": choice_index,
                    "context_token_count": len(context_ids),
                    "continuation_ids": continuation_ids,
                    "choice_character_length": tokenization[
                        "choice_character_length"
                    ],
                }
            )
    maximum_length = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), maximum_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, sequence in enumerate(sequences):
        input_ids[row_index, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        attention_mask[row_index, : len(sequence)] = 1
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    output: dict[str, list[dict]] = {}
    for row_index, item in enumerate(metadata):
        context_length = item["context_token_count"]
        continuation_ids = item["continuation_ids"]
        token_count = len(continuation_ids)
        window = logits[
            row_index,
            context_length - 1 : context_length - 1 + token_count,
            :,
        ].float()
        targets = torch.tensor(
            continuation_ids, dtype=torch.long, device=window.device
        )
        selected = torch.log_softmax(window, dim=-1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        total = float(selected.sum().item())
        choice = {
            "choice_index": int(item["choice_index"]),
            "sum_loglikelihood": total,
            "normalized_loglikelihood": total
            / int(item["choice_character_length"]),
            "token_loglikelihoods": [
                float(value) for value in selected.detach().cpu().tolist()
            ],
            "token_ids": list(continuation_ids),
            "token_count": token_count,
            "character_length": int(item["choice_character_length"]),
        }
        output.setdefault(item["sample_id"], []).append(choice)
    return output


def score_examples(
    model,
    records: Sequence[dict],
    *,
    condition: dict,
    pad_token_id: int,
    batch_size_examples: int,
    device: torch.device | str,
) -> list[dict]:
    if not records:
        raise ValueError("at least one BoolQ scope record is required")
    device = torch.device(device)
    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate BoolQ sample ID")
    for record in records:
        _validate_scope_record(record)
    outputs = []
    for batch in _chunks(records, batch_size_examples):
        scores = _score_batch(
            model, batch, pad_token_id=pad_token_id, device=device
        )
        for record in batch:
            choices = sorted(
                scores.get(record["sample_id"], []),
                key=lambda row: row["choice_index"],
            )
            if [row["choice_index"] for row in choices] != [0, 1]:
                raise ValueError("incomplete BoolQ scores")
            raw = [row["sum_loglikelihood"] for row in choices]
            if not all(math.isfinite(value) for value in raw):
                raise ValueError("non-finite BoolQ score")
            predicted = max(range(2), key=raw.__getitem__)
            gold = int(record["gold"])
            maximum = raw[predicted]
            output = {
                "schema_version": 1,
                "sample_id": record["sample_id"],
                "dataset_split": record["dataset_split"],
                "dataset_row_index": record["dataset_row_index"],
                "partition": record["partition"],
                "scope_record_sha256": record["record_sha256"],
                "condition": condition,
                "choices": choices,
                "gold": gold,
                "predicted": predicted,
                "margin": raw[gold] - raw[1 - gold],
                "correct": predicted == gold,
                "tie": sum(value == maximum for value in raw) > 1,
            }
            output["record_sha256"] = canonical_hash(output)
            outputs.append(output)
    return outputs


def summarize_condition(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("cannot summarize an empty BoolQ condition")
    margins = [float(record["margin"]) for record in records]
    correct = sum(bool(record["correct"]) for record in records)
    ties = sum(bool(record["tie"]) for record in records)
    return {
        "example_count": len(records),
        "accuracy": correct / len(records),
        "correct_count": correct,
        "incorrect_count": len(records) - correct,
        "tie_count": ties,
        "mean_margin": sum(margins) / len(margins),
        "minimum_margin": min(margins),
        "maximum_margin": max(margins),
    }


__all__ = ["score_examples", "summarize_condition"]
