"""Canonical encoding used by the task record hashes."""
import json


def canonical_json_bytes(payload, *, default=None):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False, default=default).encode('utf-8')
