import re

END_OF_TURN = "<|im_end|>"
IGNORE_INDEX = -100

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize_text(text):
    return _SPACE.sub(" ", _PUNCT.sub("", text.lower())).strip()


def clean_response(response):
    return response.lstrip("\n")


def render_prompt(tokenizer, query):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )


def end_of_turn_id(tokenizer):
    token_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    assert token_id is not None and token_id != tokenizer.unk_token_id
    return token_id


def encode_example(tokenizer, query, response):
    prompt_ids = tokenizer(render_prompt(tokenizer, query), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(clean_response(response), add_special_tokens=False)["input_ids"] + [end_of_turn_id(tokenizer)]
    return {
        "input_ids": prompt_ids + response_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + response_ids,
        "prompt_len": len(prompt_ids),
    }


def template_matches_manual_format(tokenizer, query, response):
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}, {"role": "assistant", "content": response}],
        tokenize=False,
        enable_thinking=False,
    )
    return full == render_prompt(tokenizer, query) + clean_response(response) + END_OF_TURN + "\n"
