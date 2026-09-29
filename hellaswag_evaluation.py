"""Batched teacher-forced HellaSwag normalized-choice scoring."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch

from hellaswag_tasks import canonical_hash


def _validate_scope_record(record: dict) -> None:
    observed = record.get("record_sha256")
    if not isinstance(observed, str):
        raise ValueError("HellaSwag scope record has no record_sha256")
    unhashed = dict(record)
    del unhashed["record_sha256"]
    if canonical_hash(unhashed) != observed:
        raise ValueError("HellaSwag scope record content hash differs")
    if len(record.get("choices", [])) != 4:
        raise ValueError("HellaSwag scope record does not have four choices")
    tokenizations = record.get("choice_tokenizations")
    if not isinstance(tokenizations, list) or len(tokenizations) != 4:
        raise ValueError("HellaSwag choice tokenization inventory differs")
    for tokenization, choice in zip(tokenizations, record["choices"], strict=True):
        context_ids = tokenization.get("context_ids")
        continuation_ids = tokenization.get("continuation_ids")
        if not all(
            isinstance(values, list)
            and values
            and all(isinstance(value, int) for value in values)
            for values in (context_ids, continuation_ids)
        ):
            raise ValueError("HellaSwag scope token IDs are invalid")
        if tokenization.get("context_token_count") != len(context_ids):
            raise ValueError("HellaSwag context token count differs")
        if tokenization.get("continuation_token_count") != len(continuation_ids):
            raise ValueError("HellaSwag continuation token count differs")
        if tokenization.get("choice_character_length") != len(choice):
            raise ValueError("HellaSwag choice character length differs")


def _chunks(values: Sequence[dict], size: int) -> Iterable[Sequence[dict]]:
    if size <= 0:
        raise ValueError("batch_size_examples must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _build_batch(
    records: Sequence[dict],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
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
    return input_ids, attention_mask, metadata


def _score_batch(
    model,
    records: Sequence[dict],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, list[dict]]:
    input_ids, attention_mask, metadata = _build_batch(
        records, pad_token_id=pad_token_id, device=device
    )
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("causal model returned invalid HellaSwag logits")
    output: dict[str, list[dict]] = {}
    for row_index, item in enumerate(metadata):
        context_length = item["context_token_count"]
        continuation_ids = item["continuation_ids"]
        token_count = len(continuation_ids)
        prediction_logits = logits[
            row_index,
            context_length - 1 : context_length - 1 + token_count,
            :,
        ].float()
        if prediction_logits.shape[0] != token_count:
            raise ValueError("HellaSwag prediction window differs")
        targets = torch.tensor(
            continuation_ids,
            dtype=torch.long,
            device=prediction_logits.device,
        )
        selected = torch.log_softmax(prediction_logits, dim=-1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        character_length = int(item["choice_character_length"])
        sum_loglikelihood = float(selected.sum().item())
        choice = {
            "choice_index": int(item["choice_index"]),
            "sum_loglikelihood": sum_loglikelihood,
            "normalized_loglikelihood": sum_loglikelihood / character_length,
            "token_loglikelihoods": [
                float(value) for value in selected.detach().cpu().tolist()
            ],
            "token_ids": list(continuation_ids),
            "token_count": token_count,
            "character_length": character_length,
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
        raise ValueError("at least one HellaSwag scope record is required")
    device = torch.device(device)
    seen_sample_ids = set()
    for record in records:
        _validate_scope_record(record)
        sample_id = record["sample_id"]
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate HellaSwag sample ID: {sample_id}")
        seen_sample_ids.add(sample_id)

    outputs = []
    for batch in _chunks(records, batch_size_examples):
        scores = _score_batch(
            model,
            batch,
            pad_token_id=pad_token_id,
            device=device,
        )
        for record in batch:
            choices = sorted(
                scores.get(record["sample_id"], []),
                key=lambda row: row["choice_index"],
            )
            if [row["choice_index"] for row in choices] != list(range(4)):
                raise ValueError(
                    f"incomplete HellaSwag scores for {record['sample_id']}"
                )
            normalized = [row["normalized_loglikelihood"] for row in choices]
            if not all(math.isfinite(value) for value in normalized):
                raise ValueError("non-finite HellaSwag normalized score")
            predicted = max(range(4), key=normalized.__getitem__)
            gold = int(record["gold"])
            best_nontarget = max(
                normalized[index] for index in range(4) if index != gold
            )
            maximum = normalized[predicted]
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
                "margin": normalized[gold] - best_nontarget,
                "correct": predicted == gold,
                "tie": sum(value == maximum for value in normalized) > 1,
            }
            output["record_sha256"] = canonical_hash(output)
            outputs.append(output)
    if len(outputs) != len(records):
        raise ValueError("HellaSwag output count differs")
    return outputs


def summarize_condition(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("cannot summarize an empty HellaSwag condition")
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
