"""Run discovery screening, frozen selection, confirmation, and held-out evaluation."""
import argparse
import fcntl
import itertools
import hashlib
import json
import time
import uuid
from pathlib import Path
from common import ROOT, code_digest, condition_id, digest, metrics, publish, rank_key, read
from prepare_data import validate, DATASETS

STAGES = ('prepare', 'baseline', 'singleton', 'search', 'confirmation', 'heldout')


def singleton_conditions(spec):
    return [[[layer, head]] for layer in range(spec['layers']) for head in range(spec['heads'])]


def checked_result(path, manifest):
    result = read(path)
    if result['manifest_sha256'] != digest(manifest):
        raise ValueError('Result belongs to a different experiment')
    if result['metrics'] != metrics(result['records'], result['intact']):
        raise ValueError('Paired summary mismatch')
    if len(result['records']) != manifest['counts'][result['role']]:
        raise ValueError('Incomplete result')
    if [r['sample_id'] for r in result['records']] != manifest['sample_ids'][result['role']]:
        raise ValueError('Result source identities differ from frozen panel')
    return result


def checked_summary(path, manifest):
    # Audit one condition at a time; never retain an entire singleton scan in RAM.
    result = checked_result(path, manifest)
    return {'heads':result['heads'], 'metrics':result['metrics'],
            'role':result['role'], 'result_sha256':digest(result)}


def prepare(args):
    if args.task == 'mbpp' and (args.sandbox_image is None or not args.sandbox_image.is_file()):
        raise ValueError('MBPP requires a readable --sandbox-image, including during preparation')
    spec = json.loads((ROOT/'configs/models.json').read_text())[args.model]
    data = read(args.data/f'{args.task}.json')
    validate(args.task, data['panels'])
    membership = json.loads((ROOT/'configs/panels.json').read_text())['panels'][args.task]
    if (data['dataset'], data['revision']) != DATASETS[args.task]:
        raise ValueError('Dataset revision differs from frozen release')
    if data['task'] != args.task or data['membership_sha256'] != digest(membership):
        raise ValueError('Prepared data membership differs')
    for role, rows in data['panels'].items():
        refs = membership[role]
        if any(r['index'] != ref['index'] or r['split'] != ref['split'] or r['config'] != ref['config']
               or digest(r['doc']) != ref['source_sha256'] for r,ref in zip(rows,refs)):
            raise ValueError('Prepared source content differs')
    manifest = {'model_key':args.model, 'model':spec, 'task':args.task, 'data_sha256':digest(data),
                'code_sha256':code_digest(),
                'sandbox_sha256':hashlib.sha256(args.sandbox_image.read_bytes()).hexdigest() if args.task == 'mbpp' else None,
                'batch_size':8, 'max_heads':3, 'nominees':10, 'finalists':10,
                'counts':{k:len(v) for k,v in data['panels'].items()},
                'sample_ids':{k:[r['sample_id'] for r in v] for k,v in data['panels'].items()}}
    if (args.run/'manifest.json').exists():
        if read(args.run/'manifest.json') != manifest:
            raise ValueError('Existing run differs; choose a fresh run directory')
    else:
        publish(args.run/'manifest.json',manifest)
    return manifest, data


def freeze_search(run, manifest):
    results = [checked_summary(run/'singleton'/f'{condition_id(h)}.json',manifest)
               for h in singleton_conditions(manifest['model'])]
    results.sort(key=rank_key)
    nominees = sorted(r['heads'][0] for r in results[:10])
    combinations = [list(c) for size in (2,3) for c in itertools.combinations(nominees,size)]
    design = {'manifest_sha256':digest(manifest), 'nominees':nominees, 'conditions':combinations,
              'singleton_results_sha256':digest(results)}
    path = run/'search_design.json'
    if path.exists():
        if read(path) != design:
            raise ValueError('Discovery nominees changed')
    else:
        publish(path,design)
    return design, results[:10]


def freeze_finalists(run, manifest):
    design, singles = freeze_search(run,manifest)
    pairs = [checked_summary(run/'search'/f'{condition_id(h)}.json',manifest) for h in design['conditions']]
    candidates = sorted(singles+pairs,key=rank_key)
    frozen = {'manifest_sha256':digest(manifest), 'candidate_count':len(candidates),
              'candidate_results_sha256':digest(candidates),
              'conditions':[{'discovery_rank':i+1,'heads':r['heads'],'discovery_metrics':r['metrics']}
                            for i,r in enumerate(candidates[:10])]}
    path = run/'finalists.json'
    if path.exists():
        if read(path) != frozen:
            raise ValueError('Discovery finalist freeze changed')
    else:
        publish(path,frozen)
    return frozen


def execute(args, evaluator_class=None):
    manifest, data = prepare(args)
    if args.stage == 'prepare':
        return
    role = 'heldout' if args.stage == 'heldout' else 'discovery'
    if args.stage == 'baseline':
        if (args.run/'baseline.json').exists():
            baseline = read(args.run/'baseline.json')
            if baseline['manifest_sha256'] != digest(manifest) or not baseline['repeat_agreement']:
                raise ValueError('Existing baseline did not pass; use a fresh run after investigating')
            if baseline['records'] != baseline['repeat']:
                raise ValueError('Baseline repeat records differ')
            return
        conditions = [[], []]
    else:
        baseline = read(args.run/'baseline.json')
        if baseline['manifest_sha256'] != digest(manifest) or not baseline['repeat_agreement']:
            raise ValueError('Intact repeat validation is required')
        if args.stage == 'singleton':
            conditions = singleton_conditions(manifest['model'])
        elif args.stage == 'search':
            conditions = freeze_search(args.run,manifest)[0]['conditions']
        else:
            frozen = freeze_finalists(args.run,manifest)
            conditions = [r['heads'] for r in frozen['conditions']]
            if args.stage == 'heldout':
                for heads in conditions:
                    checked_result(args.run/'confirmation'/f'{condition_id(heads)}.json',manifest)
    if args.task == 'mbpp' and (args.sandbox_image is None or not args.sandbox_image.is_file()):
        raise ValueError('MBPP requires --sandbox-image; no host-execution fallback exists')
    stage_dir = args.run/args.stage
    stage_dir.mkdir(exist_ok=True)
    remaining = []
    for heads in conditions:
        path = stage_dir/f'{condition_id(heads)}.json'
        if args.stage != 'baseline' and path.exists():
            if checked_result(path,manifest)['heads'] != heads:
                raise ValueError('Condition mismatch')
        else:
            remaining.append(heads)
    if not remaining:
        if args.stage == 'search':
            freeze_finalists(args.run,manifest)
        if args.stage == 'heldout':
            report(args.run,manifest)
        return
    attempt = stage_dir/'attempts'/uuid.uuid4().hex
    attempt.mkdir(parents=True)
    if evaluator_class is None:
        from runtime import Evaluator
        evaluator_class = Evaluator
    if args.task == 'mbpp' and evaluator_class.__module__ == 'runtime':
        from sandbox_check import check
        check(args.sandbox_image, attempt/'sandbox-check')
    evaluator = evaluator_class(args.model, manifest['model'], args.task, data['panels'][role], args.sandbox_image)
    intact, _ = evaluator.evaluate([], attempt/'intact')
    if args.stage == 'baseline':
        repeat, _ = evaluator.evaluate([], attempt/'repeat')
        agreement = intact == repeat
        publish(args.run/'baseline.json', {'manifest_sha256':digest(manifest), 'repeat_agreement':agreement,
                 'hardware':evaluator.hardware, 'records':intact, 'repeat':repeat})
        if not agreement:
            raise RuntimeError('Repeated intact evaluation differed')
        return
    for heads in remaining:
        start = time.monotonic()
        records, activity = evaluator.evaluate(heads, attempt/condition_id(heads))
        result = {'manifest_sha256':digest(manifest), 'role':role, 'heads':heads,
                  'records':records, 'intact':intact, 'metrics':metrics(records,intact),
                  'activity':activity, 'hardware':evaluator.hardware, 'seconds':time.monotonic()-start}
        publish(stage_dir/f'{condition_id(heads)}.json',result)
        print(condition_id(heads),result['metrics'],flush=True)
    if args.stage == 'search':
        freeze_finalists(args.run,manifest)
    if args.stage == 'heldout':
        report(args.run,manifest)


def report(run, manifest):
    frozen = freeze_finalists(run,manifest)
    rows = []
    for c in frozen['conditions']:
        name = condition_id(c['heads'])+'.json'
        confirmation = checked_result(run/'confirmation'/name,manifest)
        heldout = checked_result(run/'heldout'/name,manifest)
        rows.append({**c,'confirmation_metrics':confirmation['metrics'],
                     'heldout_metrics':heldout['metrics'],'hardware':heldout['hardware']})
    result = {'manifest_sha256':digest(manifest), 'primary_discovery_rank':1, 'rows':rows}
    path = run/'report.json'
    if path.exists():
        if read(path) != result:
            raise ValueError('Report content changed')
    else:
        publish(path,result)
    # All rows retain their discovery order, including negative held-out gains.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['qwen','llama','smollm3'])
    parser.add_argument('--task', required=True, choices=['arithmetic','hellaswag','boolq','mbpp'])
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--sandbox-image', type=Path)
    args = parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=True)
    with (args.run/'.lock').open('a+') as lock:
        # A whole-run lock prevents two processes from duplicating conditions.
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(args)


if __name__ == '__main__':
    main()
