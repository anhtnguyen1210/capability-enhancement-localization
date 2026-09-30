"""Frozen non-thinking template adapter; no model or scorer changes."""
import hashlib
DATE='25 September 2026'
def configure_tokenizer(tok):
    original=tok.chat_template
    if not isinstance(original,str):
        raise ValueError('SmolLM3 requires a string chat template')
    needle='strftime_now("%d %B %Y")'
    if original.count(needle) != 1:
        raise ValueError('SmolLM3 template date binding changed')
    template=original.replace(needle, '"'+DATE+'"')
    if '{%- set enable_thinking = true -%}' not in template:
        raise ValueError('SmolLM3 thinking-mode template binding changed')
    template=template.replace('{%- set enable_thinking = true -%}', '{%- set enable_thinking = false -%}')
    tok.chat_template=template
    tok.padding_side='left'
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    return {'original_template_sha256':hashlib.sha256(original.encode()).hexdigest(),'effective_template_sha256':hashlib.sha256(template.encode()).hexdigest(),'fixed_date':DATE,'enable_thinking':False}
def render(tok,messages):
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if 'Reasoning Mode: /no_think' not in text or not text.endswith('<|im_start|>assistant\n<think>\n\n</think>\n'):
        raise ValueError('SmolLM3 non-thinking prompt validation failed')
    return text
