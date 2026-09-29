import json
import tempfile
import unittest
from pathlib import Path
from common import ROOT, metrics, publish, read, rank_key
from experiment import singleton_conditions
from runtime import integer_answer, mbpp_context
from prepare_data import COUNTS


class ProtocolTests(unittest.TestCase):
    def test_exact_membership_counts_and_disjoint_ids(self):
        panels=json.loads((ROOT/'configs/panels.json').read_text())['panels']
        for task,counts in COUNTS.items():
            self.assertEqual(tuple(len(panels[task][r]) for r in ['discovery','heldout']),counts)
            ids=[]
            for role in ['discovery','heldout']:
                rows=panels[task][role]
                keys={(r['config'],r['split'],r['index']) for r in rows}
                self.assertEqual(len(keys),len(rows))
                ids.append(keys)
            self.assertFalse(ids[0]&ids[1])
        self.assertEqual({r['split'] for r in panels['mbpp']['discovery']},{'train'})

    def test_geometry(self):
        models=json.loads((ROOT/'configs/models.json').read_text())
        self.assertEqual({k:len(singleton_conditions(v)) for k,v in models.items()},
                         {'qwen':576,'llama':672,'smollm3':576})

    def test_parser_and_prompt(self):
        self.assertEqual(integer_answer(' -42\n'),-42)
        for text in ['answer: 42','42.0','<think>42</think>','42\nextra','']:
            self.assertIsNone(integer_answer(text))
        context=mbpp_context({'text':'Add two numbers.', 'test_list':['assert add(1,2)==3']})
        self.assertEqual(context.count('[DONE]'),3)
        self.assertEqual(context.count('[BEGIN]'),4)
        self.assertTrue(context.endswith('[BEGIN]\n'))

    def test_paired_metrics_and_rank(self):
        b=[{'sample_id':str(i),'correct':x} for i,x in enumerate([True,False,False])]
        r=[{'sample_id':str(i),'correct':x} for i,x in enumerate([False,True,True])]
        m=metrics(r,b)
        self.assertEqual((m['rescued'],m['damaged'],m['gain']), (2,1,1))
        with self.assertRaises(ValueError):metrics(r,list(reversed(b)))
        a={'heads':[[0,0]],'metrics':{**m,'damaged':0}}
        c={'heads':[[0,1]],'metrics':m}
        self.assertLess(rank_key(a),rank_key(c))

    def test_results_do_not_clobber_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.json'
            publish(path,{'value':1})
            self.assertEqual(read(path),{'value':1})
            with self.assertRaises(FileExistsError):publish(path,{'value':2})
            raw=json.loads(path.read_text());raw['value']=2;path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):read(path)


class NativeArchitectureTests(unittest.TestCase):
    def test_hook_activity_slice_and_restore_all_architectures(self):
        import torch
        from transformers import (LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM,
                                  SmolLM3Config,SmolLM3ForCausalLM)
        from fixed_head_output_hook import attach_fixed_head_hooks
        torch.set_num_threads(2)
        for config_cls,model_cls in [(LlamaConfig,LlamaForCausalLM),(Qwen2Config,Qwen2ForCausalLM),
                                      (SmolLM3Config,SmolLM3ForCausalLM)]:
            with self.subTest(model=model_cls.__name__):
                torch.manual_seed(7)
                cfg=config_cls(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                               num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                               max_position_embeddings=64,attention_dropout=0.0,
                               pad_token_id=0,bos_token_id=1,eos_token_id=2)
                if model_cls is SmolLM3ForCausalLM:cfg.no_rope_layers=[1,0]
                model=model_cls(cfg).eval()
                ids=torch.tensor([[1,2,3,4]])
                with torch.inference_mode():
                    intact=model(ids).logits
                    handles,activity=attach_fixed_head_hooks(model.model.layers,[[1,2]],4,8)
                    changed=model(ids).logits
                    for h in handles:h.remove()
                    restored=model(ids).logits
                self.assertFalse(torch.equal(intact,changed))
                self.assertTrue(torch.equal(intact,restored))
                self.assertGreater(activity['1']['nonzero_before'],0)
                projection=model.model.layers[1].self_attn.o_proj
                x=torch.arange(64,dtype=torch.float32).reshape(1,2,32)
                seen=[]
                handles,_=attach_fixed_head_hooks(model.model.layers,[[1,2]],4,8)
                observer=projection.register_forward_pre_hook(lambda module,args:seen.append(args[0].clone()))
                projection(x)
                observer.remove()
                for h in handles:h.remove()
                expected=x.clone();expected[...,16:24]=0
                self.assertTrue(torch.equal(seen[0],expected))
                self.assertTrue(torch.equal(x,torch.arange(64,dtype=torch.float32).reshape(1,2,32)))


if __name__ == '__main__':unittest.main()
