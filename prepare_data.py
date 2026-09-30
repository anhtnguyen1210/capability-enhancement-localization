"""Download pinned public sources and reconstruct the frozen panels."""
import argparse
import json
from collections import Counter
from pathlib import Path
from common import ROOT, digest, publish

DATASETS = {
    'arithmetic': ('EleutherAI/arithmetic', '14413db3567723ff76bc468508333b5c7a9dcf5d'),
    'hellaswag': ('Rowan/hellaswag', '218ec52e09a7e7462a5400043bb9a69a41d06b76'),
    'boolq': ('aps/super_glue', '3de24cf8022e94f4ee4b9d55a6f539891524d646'),
    'mbpp': ('google-research-datasets/mbpp', '4bb6404fdc6cacfda99d4ac4205087b89d32030c'),
}
COUNTS = {'arithmetic': (2000, 1000), 'hellaswag': (2000, 10042),
          'boolq': (2000, 3270), 'mbpp': (374, 500)}


def validate(task, panels):
    if set(panels) != {'discovery','heldout'}:
        raise ValueError('Unexpected panel roles')
    discovery, heldout = panels['discovery'], panels['heldout']
    if (len(discovery), len(heldout)) != COUNTS[task]:
        raise ValueError('Panel counts differ from the release protocol')
    for rows in (discovery, heldout):
        if len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate source identity')
    if {r['sample_id'] for r in discovery} & {r['sample_id'] for r in heldout}:
        raise ValueError('Source overlap')
    if task == 'arithmetic':
        for rows, count in [(discovery, 200), (heldout, 100)]:
            counts = Counter(r['config'] for r in rows)
            if len(counts) != 10 or set(counts.values()) != {count}:
                raise ValueError('Arithmetic must be balanced across ten subtasks')
        keys = lambda rows: {' '.join(r['doc']['context'].split()) for r in rows}
        if keys(discovery) & keys(heldout):
            raise ValueError('Arithmetic problem overlap')
    if task == 'boolq':
        train = [r['doc']['passage'] for r in discovery]
        test = {r['doc']['passage'] for r in heldout}
        if len(set(train)) != len(train) or set(train) & test:
            raise ValueError('BoolQ discovery passages overlap or repeat')
    if task == 'mbpp':
        train = {r['doc']['task_id'] for r in discovery}
        test = {r['doc']['task_id'] for r in heldout}
        if len(train) != len(discovery) or len(test) != len(heldout):
            raise ValueError('Duplicate MBPP task IDs')
        if train & test or (train | test) & set(range(1, 11)):
            raise ValueError('MBPP train/test/prompt overlap')
        if {r['split'] for r in discovery} != {'train'} or {r['split'] for r in heldout} != {'test'}:
            raise ValueError('MBPP must use training only for discovery')


def reconstruct(task, entries, loader):
    panels = {}
    for role, selected in entries.items():
        rows = []
        for ref in selected:
            doc = dict(loader(ref['config'], ref['split'])[ref['index']])
            if digest(doc) != ref['source_sha256']:
                raise ValueError(f'Source mismatch: {task}/{ref["split"]}/{ref["index"]}')
            rows.append({'sample_id': f'{ref["config"]}:{ref["split"]}:{ref["index"]}',
                         'config': ref['config'], 'split': ref['split'], 'index': ref['index'], 'doc': doc})
        panels[role] = rows
    validate(task, panels)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--task', choices=list(DATASETS)+['all'], default='all')
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    from datasets import load_dataset
    manifest = json.loads((ROOT/'configs/panels.json').read_text())
    tasks = list(DATASETS) if args.task == 'all' else [args.task]
    for task in tasks:
        cache = {}
        name, revision = DATASETS[task]
        def loader(config, split):
            key = (config, split)
            if key not in cache:
                cache[key] = load_dataset(name, config, split=split, revision=revision,
                                          cache_dir=args.cache_dir, trust_remote_code=False)
            return cache[key]
        panels = reconstruct(task, manifest['panels'][task], loader)
        publish(args.output/f'{task}.json', {'task':task, 'dataset':name, 'revision':revision,
                'membership_sha256':digest(manifest['panels'][task]), 'panels':panels})
        print(task, {k:len(v) for k,v in panels.items()})


if __name__ == '__main__':
    main()
