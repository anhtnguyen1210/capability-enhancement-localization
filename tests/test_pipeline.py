"""CPU fixture verifies selection gates and rank preservation end to end."""
import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from common import digest, read
from experiment import execute, freeze_search, checked_result


class FakeEvaluator:
    def __init__(self, model, spec, task, rows, image):
        self.rows=rows
        self.hardware={'gpu':'CPU test fixture'}
    def evaluate(self, heads, destination):
        good = bool(heads) and sum(h for _,h in heads)%2 == 0
        if self.rows[0]['sample_id'].startswith('heldout'):
            good = not good
        return [{'sample_id':r['sample_id'], 'correct':good} for r in self.rows],{}


class PipelineTests(unittest.TestCase):
    def test_frozen_selection_confirmation_and_heldout(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            data={'panels':{r:[{'sample_id':f'{r}:{i}'} for i in range(2)] for r in ['discovery','heldout']}}
            manifest={'model':{'layers':1,'heads':10}, 'counts':{'discovery':2,'heldout':2},
                      'sample_ids':{r:[x['sample_id'] for x in v] for r,v in data['panels'].items()}}
            args=Namespace(run=run,model='qwen',task='boolq',sandbox_image=None,stage='search')
            with patch('experiment.prepare',return_value=(manifest,data)),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):execute(args,FakeEvaluator)
                args.stage='baseline';execute(args,FakeEvaluator)
                with self.assertRaises(FileNotFoundError):freeze_search(run,manifest)
                args.stage='singleton';execute(args,FakeEvaluator)
                args.stage='search';execute(args,FakeEvaluator)
                frozen=read(run/'finalists.json')
                self.assertEqual(frozen['candidate_count'],175)
                self.assertEqual(len(frozen['conditions']),10)
                self.assertEqual(len(read(run/'search_design.json')['conditions']),165)
                args.stage='heldout'
                with self.assertRaises(FileNotFoundError):execute(args,FakeEvaluator)
                args.stage='confirmation';execute(args,FakeEvaluator)
                args.stage='heldout';execute(args,FakeEvaluator)
                report=read(run/'report.json')
                self.assertEqual([x['heads'] for x in report['rows']], [x['heads'] for x in frozen['conditions']])
                self.assertTrue(all(x['heldout_metrics']['gain']<0 for x in report['rows']))
                # A completed held-out resume neither reselects nor adds conditions.
                execute(args,FakeEvaluator)
                self.assertEqual(read(run/'finalists.json'),frozen)
                self.assertEqual(read(run/'report.json'),report)
                first=next((run/'heldout').glob('*.json'))
                value=read(first)
                value['records'][0]['sample_id']='wrong-source'
                import json
                first.write_text(json.dumps({**value,'sha256':digest(value)}))
                with self.assertRaises(ValueError):checked_result(first,manifest)


if __name__=='__main__':unittest.main()
