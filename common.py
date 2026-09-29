"""Content-addressed JSON and paired evaluation summaries."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    checksum = value.pop('sha256')
    if digest(value) != checksum:
        raise ValueError(f'Content hash mismatch: {Path(path).name}')
    return value


def publish(path, value):
    """Publish without replacing an existing result, including concurrent writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({**value, 'sha256': digest(value)}, indent=2, allow_nan=False) + '\n'
    fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def metrics(records, intact):
    ids = [r['sample_id'] for r in records]
    if not ids or len(set(ids)) != len(ids) or ids != [r['sample_id'] for r in intact]:
        raise ValueError('Paired source identities differ or contain duplicates')
    rescued = sum(r['correct'] and not b['correct'] for r, b in zip(records, intact))
    damaged = sum(b['correct'] and not r['correct'] for r, b in zip(records, intact))
    correct = sum(r['correct'] for r in records)
    baseline = sum(r['correct'] for r in intact)
    if correct != baseline + rescued - damaged:
        raise ValueError('Paired identity failed')
    return dict(n=len(ids), correct=correct, intact_correct=baseline,
                accuracy=correct/len(ids), intact_accuracy=baseline/len(ids),
                rescued=rescued, damaged=damaged, gain=rescued-damaged,
                gain_pp=100*(rescued-damaged)/len(ids))


def condition_id(heads):
    return '+'.join(f'L{layer}H{head}' for layer, head in heads) or 'intact'


def rank_key(result):
    m = result['metrics']
    heads = result['heads']
    return (-m['gain'], m['damaged'], -m['correct'], len(heads), heads)


def code_digest():
    paths = [ROOT/'requirements.txt', ROOT/'sandbox.def'] + sorted(ROOT.glob('*.py')) + sorted(ROOT.glob('*.c')) + sorted((ROOT/'configs').glob('*.json'))
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
