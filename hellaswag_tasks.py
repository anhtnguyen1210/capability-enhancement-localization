"""Deterministic HellaSwag preparation for masked-head experiments."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping

from resilience_io import canonical_json_bytes


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def preprocess(text: str) -> str:
    """Match the installed lm-eval HellaSwag preprocessing function."""

    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def source_row_payload(
    row: Mapping[str, object], *, dataset_split: str, dataset_row_index: int
) -> dict:
    keys = (
        "ind",
        "activity_label",
        "ctx_a",
        "ctx_b",
        "ctx",
        "endings",
        "source_id",
        "split",
        "split_type",
        "label",
    )
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"HellaSwag row is missing fields: {missing}")
    return {
        "dataset_split": dataset_split,
        "dataset_row_index": int(dataset_row_index),
        **{key: row[key] for key in keys},
    }


def process_source(source: Mapping[str, object]) -> dict:
    endings = source["endings"]
    if not isinstance(endings, list) or len(endings) != 4:
        raise ValueError("HellaSwag row must have four endings")
    query = preprocess(
        f"{source['activity_label']}: {source['ctx_a']} "
        f"{str(source['ctx_b']).capitalize()}"
    )
    choices = [preprocess(str(ending)) for ending in endings]
    gold = int(source["label"])
    if not query or any(not choice for choice in choices):
        raise ValueError("HellaSwag query or choice became empty")
    if gold not in range(4):
        raise ValueError(f"HellaSwag gold label is invalid: {gold}")
    return {"query": query, "choices": choices, "gold": gold}


def encode_context_continuation(tokenizer, context: str, continuation: str) -> dict:
    """Apply lm-eval's causal boundary convention exactly."""

    if not context or not continuation:
        raise ValueError("HellaSwag context and continuation must be non-empty")
    trailing_space_count = len(context) - len(context.rstrip())
    if trailing_space_count:
        moved_spaces = context[-trailing_space_count:]
        context = context[:-trailing_space_count]
    else:
        moved_spaces = ""
    scored_continuation = moved_spaces + continuation
    context_ids = list(tokenizer.encode(context, add_special_tokens=True))
    whole_ids = list(
        tokenizer.encode(context + scored_continuation, add_special_tokens=True)
    )
    if not context_ids:
        raise ValueError("HellaSwag context tokenized to an empty sequence")
    continuation_ids = whole_ids[len(context_ids) :]
    if not continuation_ids:
        raise ValueError("HellaSwag choice tokenized to an empty continuation")
    return {
        "context_ids": context_ids,
        "continuation_ids": continuation_ids,
        "whole_ids": whole_ids,
        "context_token_count": len(context_ids),
        "continuation_token_count": len(continuation_ids),
        "scored_continuation": scored_continuation,
        "boundary_rule": "lm_eval_causal_move_trailing_spaces_v1",
        "add_special_tokens": True,
    }


def prepare_record(
    tokenizer,
    row: Mapping[str, object],
    *,
    dataset_split: str,
    dataset_row_index: int,
    partition: str,
    selection_key_sha256: str,
    experiment_id: str,
) -> dict:
    source = source_row_payload(
        row,
        dataset_split=dataset_split,
        dataset_row_index=dataset_row_index,
    )
    processed = process_source(source)
    choice_tokenizations = []
    for choice in processed["choices"]:
        tokenization = encode_context_continuation(
            tokenizer,
            processed["query"],
            f" {choice}",
        )
        tokenization["choice_character_length"] = len(choice)
        choice_tokenizations.append(tokenization)
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "dataset_path": "Rowan/hellaswag",
        "dataset_split": dataset_split,
        "dataset_row_index": int(dataset_row_index),
        "sample_id": f"{dataset_split}:{int(dataset_row_index)}",
        "partition": partition,
        "selection_key_sha256": selection_key_sha256,
        "source": source,
        "source_sha256": canonical_hash(source),
        "query": processed["query"],
        "choices": processed["choices"],
        "gold": processed["gold"],
        "choice_tokenizations": choice_tokenizations,
        "scoring_contract": {
            "target_delimiter": " ",
            "normalization": "choice_character_length",
            "prediction": "first_argmax_normalized_sum_loglikelihood",
            "primary_metric": "acc_norm",
        },
    }
    payload["record_sha256"] = canonical_hash(payload)
    return payload


def select_source_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    dataset_split: str,
    seed: int,
    count: int,
) -> list[dict]:
    ranked = []
    seen_source_hashes = set()
    for dataset_row_index, row in enumerate(rows):
        source = source_row_payload(
            row,
            dataset_split=dataset_split,
            dataset_row_index=dataset_row_index,
        )
        source_sha256 = canonical_hash(source)
        if source_sha256 in seen_source_hashes:
            raise ValueError("duplicate HellaSwag source identity")
        seen_source_hashes.add(source_sha256)
        selection_key = hashlib.sha256(
            f"{seed}\0{dataset_split}\0{source_sha256}".encode("utf-8")
        ).hexdigest()
        ranked.append(
            {
                "dataset_row_index": dataset_row_index,
                "source_sha256": source_sha256,
                "selection_key_sha256": selection_key,
            }
        )
    if count <= 0 or count > len(ranked):
        raise ValueError("HellaSwag selection count is invalid")
    ranked.sort(key=lambda item: (item["selection_key_sha256"], item["source_sha256"]))
    return ranked[:count]


__all__ = [
    "canonical_hash",
    "encode_context_continuation",
    "prepare_record",
    "preprocess",
    "process_source",
    "select_source_rows",
    "source_row_payload",
]
