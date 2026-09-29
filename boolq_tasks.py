"""Deterministic BoolQ preparation for masked-head experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

from hellaswag_tasks import canonical_hash, encode_context_continuation


def source_row_payload(
    row: Mapping[str, object], *, dataset_split: str, dataset_row_index: int
) -> dict:
    keys = ("question", "passage", "idx", "label")
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"BoolQ row is missing fields: {missing}")
    return {
        "dataset_split": dataset_split,
        "dataset_row_index": int(dataset_row_index),
        **{key: row[key] for key in keys},
    }


def process_source(source: Mapping[str, object]) -> dict:
    passage = str(source["passage"])
    question = str(source["question"])
    query = f"{passage}\nQuestion: {question}?\nAnswer:"
    choices = ["no", "yes"]
    gold = int(source["label"])
    if not passage or not question or gold not in (0, 1):
        raise ValueError("BoolQ source is invalid")
    return {"query": query, "choices": choices, "gold": gold}


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
    tokenizations = []
    for choice in processed["choices"]:
        tokenization = encode_context_continuation(
            tokenizer, processed["query"], f" {choice}"
        )
        tokenization["choice_character_length"] = len(choice)
        tokenizations.append(tokenization)
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "dataset_path": "aps/super_glue",
        "dataset_name": "boolq",
        "dataset_split": dataset_split,
        "dataset_row_index": int(dataset_row_index),
        "sample_id": f"{dataset_split}:{int(dataset_row_index)}",
        "partition": partition,
        "selection_key_sha256": selection_key_sha256,
        "source": source,
        "source_sha256": canonical_hash(source),
        "passage_sha256": canonical_hash(str(source["passage"])),
        "query": processed["query"],
        "choices": processed["choices"],
        "gold": processed["gold"],
        "choice_tokenizations": tokenizations,
        "scoring_contract": {
            "target_delimiter": " ",
            "normalization": None,
            "prediction": "first_argmax_sum_loglikelihood",
            "primary_metric": "acc",
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
    excluded_passage_sha256: set[str] | frozenset[str] | None = None,
) -> list[dict]:
    excluded = excluded_passage_sha256 or set()
    ranked = []
    seen = set()
    for dataset_row_index, row in enumerate(rows):
        source = source_row_payload(
            row,
            dataset_split=dataset_split,
            dataset_row_index=dataset_row_index,
        )
        source_sha256 = canonical_hash(source)
        if source_sha256 in seen:
            raise ValueError("duplicate BoolQ source identity")
        seen.add(source_sha256)
        passage_sha256 = canonical_hash(str(source["passage"]))
        if passage_sha256 in excluded:
            continue
        selection_key = hashlib.sha256(
            f"{seed}\0{dataset_split}\0{source_sha256}".encode("utf-8")
        ).hexdigest()
        ranked.append(
            {
                "dataset_row_index": dataset_row_index,
                "source_sha256": source_sha256,
                "passage_sha256": passage_sha256,
                "selection_key_sha256": selection_key,
            }
        )
    if count <= 0 or count > len(ranked):
        raise ValueError("BoolQ selection count is invalid")
    ranked.sort(key=lambda item: (item["selection_key_sha256"], item["source_sha256"]))
    selected = []
    selected_passages = set()
    for item in ranked:
        passage_sha256 = item["passage_sha256"]
        if passage_sha256 in selected_passages:
            continue
        selected.append(item)
        selected_passages.add(passage_sha256)
        if len(selected) == count:
            return selected
    raise ValueError("BoolQ selection has too few unique, eligible passages")


__all__ = [
    "canonical_hash",
    "prepare_record",
    "process_source",
    "select_source_rows",
    "source_row_payload",
]
