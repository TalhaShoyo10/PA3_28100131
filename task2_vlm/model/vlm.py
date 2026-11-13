import torch
from torch import nn

IMAGE_MARK = "<<IMAGE>>"
QUESTION_MARK = "<<QUESTION>>"
CAPTION_INSTRUCTION = "Describe this image in one sentence."
QUESTION_CAP = 80
CAPTION_CAP = 64
ANSWER_CAP = 16
IGNORE = -100


class Connector(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.register_buffer("output_scale", torch.ones(()))

    def forward(self, features):
        return self.net(features) * self.output_scale


class PromptParts:
    def __init__(self, tokenizer):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": IMAGE_MARK + "\n" + QUESTION_MARK}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before_image, rest = rendered.split(IMAGE_MARK)
        between, after_question = rest.split(QUESTION_MARK)
        encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
        self.prefix = encode(before_image)
        self.between = encode(between)
        self.suffix = encode(after_question)
        self.end_of_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.tokenizer = tokenizer
        self.rendered_example = rendered

    def question_ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"][:QUESTION_CAP]

    def target_ids(self, text, cap):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"][:cap] + [self.end_of_turn]


def build_example(parts, question, target, target_cap, with_image):
    q = parts.question_ids(question)
    t = parts.target_ids(target, target_cap) if target is not None else []
    return {
        "prefix": parts.prefix,
        "with_image": with_image,
        "text_after_image": parts.between + q + parts.suffix,
        "target": t,
    }


def assemble(examples, features, embed, connector, pad_side, device, n_visual):
    hidden = embed.weight.shape[1]
    seqs, labels, lengths = [], [], []
    for ex, feat in zip(examples, features):
        pieces = [embed(torch.tensor(ex["prefix"], device=device))]
        lab = [IGNORE] * len(ex["prefix"])
        if ex["with_image"]:
            pieces.append(connector(feat.to(device).float()).to(pieces[0].dtype))
            lab += [IGNORE] * n_visual
        rest = ex["text_after_image"] + ex["target"]
        pieces.append(embed(torch.tensor(rest, device=device)))
        lab += [IGNORE] * len(ex["text_after_image"]) + ex["target"]
        seq = torch.cat(pieces, dim=0)
        seqs.append(seq)
        labels.append(torch.tensor(lab, device=device))
        lengths.append(seq.shape[0])
    width = max(lengths)
    batch = torch.zeros(len(seqs), width, hidden, dtype=seqs[0].dtype, device=device)
    label_t = torch.full((len(seqs), width), IGNORE, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), width, dtype=torch.long, device=device)
    for i, (s, lab) in enumerate(zip(seqs, labels)):
        n = s.shape[0]
        sl = slice(width - n, width) if pad_side == "left" else slice(0, n)
        batch[i, sl] = s
        label_t[i, sl] = lab
        mask[i, sl] = 1
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    return batch, label_t, mask, positions


def per_example_token_nll(model, embeds, labels, mask, positions, exclude_last_target=False):
    hidden = model.model(inputs_embeds=embeds, attention_mask=mask, position_ids=positions, use_cache=False).last_hidden_state
    targets = labels[:, 1:].clone()
    if exclude_last_target:
        for i in range(targets.shape[0]):
            idx = (targets[i] != IGNORE).nonzero()
            if len(idx):
                targets[i, idx[-1]] = IGNORE
    valid = targets != IGNORE
    rows = hidden[:, :-1][valid]
    token_nll = torch.nn.functional.cross_entropy(model.lm_head(rows).float(), targets[valid], reduction="none")
    example_index = valid.nonzero()[:, 0]
    sums = torch.zeros(targets.shape[0], dtype=torch.float32, device=hidden.device).index_add(0, example_index, token_nll)
    counts = valid.sum(1)
    assert (counts > 0).all(), "an example has no supervised token"
    return sums / counts, counts
