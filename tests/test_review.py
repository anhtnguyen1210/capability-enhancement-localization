"""Regression checks for release review findings; no generated code runs on host."""
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from common import digest, metrics, publish
from experiment import checked_result, checked_baseline
from runtime import decode_completion, Evaluator
from secure_execute import _secure_humaneval_execution


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'model':{'layers':2,'heads':4},'counts':{'discovery':1,'heldout':1},
                       'sample_ids':{'discovery':['d:0'],'heldout':['h:0']}}
        rows=[{'sample_id':'d:0','correct':True,'input_ids':[1,2]}]
        self.result={'manifest_sha256':digest(self.manifest),'stage':'singleton','role':'discovery',
                     'heads':[[0,1]],'records':rows,'intact':rows,'metrics':metrics(rows,rows)}

    def test_wrong_stage_or_condition_is_rejected_even_with_valid_checksum(self):
        for field,value in [('stage','heldout'),('role','heldout'),('heads',[[0,2]]),
                            ('heads',[[0,1],[0,1]])]:
            with self.subTest(field=field,value=value),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'singleton/L0H1.json'
                publish(path,{**self.result,field:value})
                with self.assertRaises(ValueError):checked_result(path,self.manifest)

    def test_paired_input_tokens_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'singleton/L0H1.json'
            value={**self.result,'intact':[{'sample_id':'d:0','correct':True,'input_ids':[1,3]}]}
            publish(path,value)
            with self.assertRaises(ValueError):checked_result(path,self.manifest)

    def test_baseline_flag_alone_cannot_satisfy_gate(self):
        for rows,repeat in [([],[]),([{'sample_id':'wrong','correct':True}],[{'sample_id':'wrong','correct':True}]),
                             ([{'sample_id':'d:0','correct':True}],[{'sample_id':'d:0','correct':False}])]:
            with tempfile.TemporaryDirectory() as tmp:
                publish(Path(tmp)/'baseline.json',{'manifest_sha256':digest(self.manifest),
                        'repeat_agreement':True,'records':rows,'repeat':repeat})
                with self.assertRaises(ValueError):checked_baseline(Path(tmp),self.manifest)

    def test_non_boolean_correctness_is_rejected(self):
        with self.assertRaises(ValueError):metrics([{'sample_id':'a','correct':1}],[{'sample_id':'a','correct':True}])


class SandboxClassifierTests(unittest.TestCase):
    def test_setup_failure_is_not_a_model_error(self):
        env={'LLMCOMP_HUMANEVAL_SANDBOX':'1','LLMCOMP_HUMANEVAL_HELPER':'mock-helper'}
        with patch.dict(os.environ,env),patch('os.path.isfile',return_value=True),patch('os.access',return_value=True):
            with patch('secure_execute.subprocess.run',return_value=subprocess.CompletedProcess([],125,b'',b'filter failed')):
                result=_secure_humaneval_execution('unused','unused')
                self.assertEqual(result['execution_outcome'],'sandbox_error')
            with patch('secure_execute.subprocess.run',return_value=subprocess.CompletedProcess([],1,b'',b'assertion failed')):
                result=_secure_humaneval_execution('unused','unused')
                self.assertEqual(result['execution_outcome'],'failed')

    def test_no_host_execution_without_sandbox(self):
        with patch.dict(os.environ,{},clear=True),patch('secure_execute.subprocess.run') as run:
            with self.assertRaises(RuntimeError):_secure_humaneval_execution('unused','unused')
            run.assert_not_called()


class StopTests(unittest.TestCase):
    class Tokenizer:
        def decode(self,ids,skip_special_tokens=True):
            return ''.join({0:'',1:'return 1',2:'',3:'[DO',4:'NE]',5:'extra'}[x] for x in ids)
    def test_marker_removes_later_batch_padding(self):
        ids,text,stopped,ended=decode_completion(self.Tokenizer(),[1,3,4,0,0,2],{2},'[DONE]')
        self.assertEqual(ids,[1,3,4]);self.assertEqual(text,'return 1[DONE]')
        self.assertTrue(stopped);self.assertFalse(ended)
    def test_eos_wins_over_later_content(self):
        ids,text,stopped,ended=decode_completion(self.Tokenizer(),[1,2,3,4],{2},'[DONE]')
        self.assertEqual(ids,[1,2]);self.assertFalse(stopped);self.assertTrue(ended)


class NumericalScorerTests(unittest.TestCase):
    def record(self,index,choices,continuations):
        from hellaswag_tasks import canonical_hash
        tokens=[]
        for choice,ids in zip(choices,continuations):
            tokens.append({'context_ids':[1,2],'continuation_ids':ids,'context_token_count':2,
                           'continuation_token_count':len(ids),'choice_character_length':len(choice)})
        value={'sample_id':str(index),'dataset_split':'train','dataset_row_index':index,
               'partition':'discovery','choice_tokenizations':tokens,'choices':choices,'gold':1}
        return {**value,'record_sha256':canonical_hash(value)}

    def test_causal_offsets_and_normalization_and_partial_batch(self):
        import torch
        from boolq_evaluation import score_examples as boolq
        from hellaswag_evaluation import score_examples as hellaswag
        class Model:
            def __init__(self):self.batches=[]
            def __call__(self,input_ids,attention_mask,use_cache):
                self.batches.append(input_ids.shape[0])
                logits=torch.arange(8).float()[None,None,:]*(torch.arange(input_ids.shape[1]).float()[None,:,None]+1)/10
                return SimpleNamespace(logits=logits.expand(input_ids.shape[0],-1,-1))
        for scorer,choices,continuations in [(boolq,['no','yes'],[[3,4],[5]]),
                                             (hellaswag,['a','bbbb','cc','ddd'],[[3],[4,5],[6],[7,3]])]:
            rows=[self.record(i,choices,continuations) for i in range(9)]
            model=Model();result=scorer(model,rows,condition={},pad_token_id=0,batch_size_examples=8,device='cpu')
            self.assertEqual(model.batches,[8*len(choices),len(choices)])
            self.assertEqual([r['sample_id'] for r in result],[str(i) for i in range(9)])
            expected=[]
            for ids in continuations:
                score=0
                for step,token in enumerate(ids):
                    factor=(step+2)/10  # First continuation uses context_length-1 = position 1.
                    score+=token*factor-math.log(sum(math.exp(v*factor) for v in range(8)))
                expected.append(score)
            for row in result:
                for i,choice in enumerate(row['choices']):
                    self.assertAlmostEqual(choice['sum_loglikelihood'],expected[i],places=5)
                    self.assertAlmostEqual(choice['normalized_loglikelihood'],expected[i]/len(choices[i]),places=5)
                scores=expected if scorer is boolq else [v/len(c) for v,c in zip(expected,choices)]
                self.assertEqual(row['predicted'],max(range(len(scores)),key=scores.__getitem__))


class NativeGenerationTests(unittest.TestCase):
    def test_arithmetic_generation_full_batch_and_tail_all_models(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import (PreTrainedTokenizerFast,LlamaConfig,LlamaForCausalLM,
                                  Qwen2Config,Qwen2ForCausalLM,SmolLM3Config,SmolLM3ForCausalLM)
        torch.set_num_threads(2)
        backend=Tokenizer(models.WordLevel({'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3,'x':4,'1':5},unk_token='[UNK]'))
        backend.pre_tokenizer=pre_tokenizers.Whitespace()
        tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',eos_token='[EOS]',bos_token='[BOS]',unk_token='[UNK]')
        tok.padding_side='left'
        for config_cls,model_cls in [(LlamaConfig,LlamaForCausalLM),(Qwen2Config,Qwen2ForCausalLM),(SmolLM3Config,SmolLM3ForCausalLM)]:
            with self.subTest(model=model_cls.__name__):
                torch.manual_seed(9)
                cfg=config_cls(vocab_size=8,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                               num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,
                               pad_token_id=0,bos_token_id=1,eos_token_id=2)
                if model_cls is SmolLM3ForCausalLM:cfg.no_rope_layers=[1,0]
                ev=Evaluator.__new__(Evaluator)
                ev.model=model_cls(cfg).eval();ev.torch=torch;ev.tokenizer=tok;ev.device='cpu';ev.task='arithmetic'
                ev.rows=[{'sample_id':str(i),'doc':{'completion':'1'}} for i in range(9)]
                ev.prepared=['x' if i%2 else 'x x' for i in range(9)]
                records=ev.generate()
                self.assertEqual(len(records),9)
                self.assertTrue(all(1<=len(r['output_ids'])<=32 for r in records))
                self.assertEqual(records[0]['input_ids'],[4,4]);self.assertEqual(records[1]['input_ids'],[4])
                self.assertTrue(all(type(r['correct']) is bool for r in records))


if __name__=='__main__':unittest.main()
