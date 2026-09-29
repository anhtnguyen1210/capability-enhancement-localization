"""Frozen non-thinking template adapter; no model or scorer changes."""
from pathlib import Path
import hashlib
DATE='25 September 2026'
def configure_tokenizer(tok):
    original=tok.chat_template
    assert isinstance(original,str)
    needle='strftime_now("%d %B %Y")'
    assert original.count(needle)==1
    template=original.replace(needle, '"'+DATE+'"')
    assert '{%- set enable_thinking = true -%}' in template
    template=template.replace('{%- set enable_thinking = true -%}', '{%- set enable_thinking = false -%}')
    tok.chat_template=template
    tok.padding_side='left'
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    return {'original_template_sha256':hashlib.sha256(original.encode()).hexdigest(),'effective_template_sha256':hashlib.sha256(template.encode()).hexdigest(),'fixed_date':DATE,'enable_thinking':False}
def render(tok,messages):
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert 'Reasoning Mode: /no_think' in text
    assert text.endswith('<|im_start|>assistant\n<think>\n\n</think>\n')
    return text
