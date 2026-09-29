"""Native Transformers inference and isolated MBPP scoring."""
import json
import re
import subprocess
import sys
from pathlib import Path
from common import ROOT, digest, read
from fewshot import list_fewshot_samples
from fixed_head_output_hook import attach_fixed_head_hooks


def integer_answer(text):
    text = text.strip()
    return int(text) if re.fullmatch(r'[+-]?\d+', text) else None


def mbpp_context(doc):
    def question(row):
        return ('You are an expert Python programmer, and here is your task: '
                + row['text'] + ' Your code should pass these tests:\n\n'
                + '\n'.join(row['test_list']) + '\n[BEGIN]\n')
    examples = [question(row)+row['code']+'\n[DONE]' for row in list_fewshot_samples()]
    return '\n\n'.join(examples+[question(doc)])


def render_prompt(tokenizer, model_key, task, doc):
    if task == 'arithmetic':
        user = ('Solve the arithmetic problem. Return only the final integer, '
                'with no words or explanation.\n\n'+doc['context'])
        suffix = ''
    else:
        user, suffix = mbpp_context(doc).rsplit('[BEGIN]\n', 1)
        if suffix.strip():
            raise ValueError('Nonempty MBPP completion prefill')
        suffix = '[BEGIN]\n'
    messages = [{'role':'user', 'content':user}]
    if model_key == 'smollm3':
        from smollm_runtime import render
        return render(tokenizer, messages)+suffix
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         date_string='24 Sep 2026')+suffix


def score_programs(records, destination, image):
    """Never execute model-generated Python in the host interpreter."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'programs.json').write_text(json.dumps(records)+'\n')
    command = [sys.executable, str(ROOT/'humaneval_sandbox.py'), '--image', str(Path(image).resolve()),
               '--repo', str(ROOT), '--timeout-seconds', str(5*len(records)+120),
               '--memory-bytes', '4294967296', '--pids-limit', '128',
               '--cpu-seconds', str(5*len(records)+120), '--file-bytes', '67108864',
               '--passthrough', '--preserve-returncode', '--bind', f'{ROOT}:/package:ro',
               '--bind', f'{destination}:/io:rw', '--', 'env', 'PYTHONDONTWRITEBYTECODE=1',
               'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'LLMCOMP_HUMANEVAL_SANDBOX=1',
               'LLMCOMP_HUMANEVAL_HELPER=/sandbox/seccomp_exec', 'python3', '/package/score.py',
               '/io/programs.json', '/io/scores.json']
    subprocess.run(command, check=True)
    scored = read(destination/'scores.json')['records']
    if len(scored) != len(records) or any(any(r[k] != s[k] for k in r) for r,s in zip(records,scored)):
        raise ValueError('Sandbox changed input records')
    return scored


class Evaluator:
    def __init__(self, model_key, model_spec, task, rows, image=None):
        import importlib.metadata
        for requirement in (ROOT/'requirements.txt').read_text().splitlines():
            package, expected = requirement.split('==')
            installed = importlib.metadata.version(package)
            if installed.split('+')[0] != expected:
                raise RuntimeError(f'{package} version differs: expected {expected}, found {installed}')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('A CUDA GPU with BF16 support is required')
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.torch, self.model_key, self.task, self.rows = torch, model_key, task, rows
        self.image = image
        self.tokenizer = AutoTokenizer.from_pretrained(model_spec['id'], revision=model_spec['revision'],
                                                       trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_spec['id'], revision=model_spec['revision'],
                     dtype=torch.bfloat16, attn_implementation='sdpa', trust_remote_code=False).to('cuda').eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        if (cfg.model_type, cfg.num_hidden_layers, cfg.num_attention_heads) != (
                model_spec['model_type'], model_spec['layers'], model_spec['heads']):
            raise ValueError('Model architecture differs from the frozen specification')
        tok = self.tokenizer
        tok.padding_side = 'left'
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        if model_key == 'smollm3':
            from smollm_runtime import configure_tokenizer
            configure_tokenizer(tok)
        self.hardware = {'gpu':torch.cuda.get_device_name(), 'cuda':torch.version.cuda,
                         'torch':torch.__version__, 'model_class':type(self.model).__name__,
                         'template_sha256':digest(tok.chat_template)}
        import importlib.metadata
        self.hardware['packages'] = {k:importlib.metadata.version(k) for k in ('transformers','tokenizers','datasets')}
        if task in ('boolq', 'hellaswag'):
            import boolq_tasks, hellaswag_tasks
            module = boolq_tasks if task == 'boolq' else hellaswag_tasks
            records = []
            for row in rows:
                r = module.prepare_record(tok, row['doc'], dataset_split=row['split'],
                        dataset_row_index=row['index'], partition='evaluation',
                        selection_key_sha256=digest(row), experiment_id='preliminary-release')
                r['sample_id'] = row['sample_id']
                r.pop('record_sha256')
                r['record_sha256'] = module.canonical_hash(r)
                records.append(r)
            self.prepared = sorted(records, key=lambda r:(max(len(c['context_ids'])+len(c['continuation_ids'])
                                   for c in r['choice_tokenizations']), r['sample_id']))
            if any(len(c['context_ids'])+len(c['continuation_ids']) > cfg.max_position_embeddings
                   for r in self.prepared for c in r['choice_tokenizations']):
                raise ValueError('Choice exceeds context window; truncation is forbidden')
        else:
            self.prepared = [render_prompt(tok, model_key, task, row['doc']) for row in rows]

    def evaluate(self, heads, destination):
        torch, tok, model = self.torch, self.tokenizer, self.model
        torch.manual_seed(0)
        cfg = model.config
        handles, activity = attach_fixed_head_hooks(model.model.layers, heads, cfg.num_attention_heads,
                                  getattr(cfg,'head_dim',None) or cfg.hidden_size//cfg.num_attention_heads)
        try:
            if self.task in ('boolq', 'hellaswag'):
                from boolq_evaluation import score_examples as boolq_score
                from hellaswag_evaluation import score_examples as hellaswag_score
                scorer = boolq_score if self.task == 'boolq' else hellaswag_score
                scored = scorer(model, self.prepared, condition={'head_pairs':heads},
                                pad_token_id=tok.pad_token_id, batch_size_examples=8, device='cuda')
                bank = {r['sample_id']:r for r in scored}
                records = [bank[r['sample_id']] for r in self.rows]
            else:
                records = self.generate()
        finally:
            for handle in handles:
                handle.remove()
        if heads and not all(v['calls'] > 0 and v['nonzero_before'] > 0 for v in activity.values()):
            raise RuntimeError('Head intervention was inactive')
        if self.task == 'mbpp':
            records = score_programs(records, destination, self.image)
        return records, activity

    def generate(self):
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        torch, tok, model = self.torch, self.tokenizer, self.model
        cap = 32 if self.task == 'arithmetic' else 256
        config = GenerationConfig.from_dict(model.generation_config.to_dict())
        config.update(do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=cap,
                      repetition_penalty=1.0, pad_token_id=tok.pad_token_id, use_cache=True)
        eos = config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos)
        records = []
        class MarkerStop(StoppingCriteria):
            def __init__(self, offset):
                self.offset = offset
            def __call__(self, ids, scores, **kwargs):
                texts = tok.batch_decode(ids[:,self.offset:], skip_special_tokens=False)
                return torch.tensor(['[DONE]' in text for text in texts], device=ids.device)
        for start in range(0, len(self.rows), 8):
            rows, prompts = self.rows[start:start+8], self.prepared[start:start+8]
            enc = tok(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to('cuda')
            width = enc['input_ids'].shape[1]
            if width+cap > model.config.max_position_embeddings:
                raise ValueError('Prompt exceeds context window; truncation is forbidden')
            stops = StoppingCriteriaList([MarkerStop(width)]) if self.task == 'mbpp' else StoppingCriteriaList()
            with torch.inference_mode():
                sequences = model.generate(**enc, generation_config=config, stopping_criteria=stops)
            for j,row in enumerate(rows):
                ids = sequences[j,width:].tolist()
                end = next((i for i,t in enumerate(ids) if t in eos), None)
                if end is not None:
                    ids = ids[:end+1]
                text = tok.decode(ids, skip_special_tokens=True)
                stopped = self.task == 'mbpp' and '[DONE]' in text
                item = {'sample_id':row['sample_id'], 'prompt':prompts[j],
                        'input_ids':enc['input_ids'][j][enc['attention_mask'][j].bool()].tolist(),
                        'output_ids':ids, 'text':text, 'capped':end is None and len(ids)>=cap and not stopped,
                        'thinking_marker_in_output':'<think>' in text or '</think>' in text}
                if self.task == 'arithmetic':
                    pred = integer_answer(text)
                    gold = int(row['doc']['completion'].strip())
                    item.update(prediction=pred, gold=gold, format_valid=pred is not None, correct=pred==gold)
                else:
                    program = text.split('[DONE]',1)[0].replace('\t','    ')
                    item.update(program=program, test_setup_code=row['doc'].get('test_setup_code',''),
                                test_list=row['doc']['test_list'])
                records.append(item)
        return records
