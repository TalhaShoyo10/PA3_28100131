import math

import torch
from torch import nn

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        assert base.bias is None
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        a = torch.empty(rank, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_A = nn.Parameter(a.to(base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))

    def effective_factors(self):
        return self.lora_B, self.lora_A

    def delta_weight(self):
        b, a = self.effective_factors()
        return self.scale * (b.float() @ a.float())

    def forward(self, x):
        b, a = self.effective_factors()
        update = (self.dropout(x) @ a.t().to(x.dtype)) @ b.t().to(x.dtype)
        return self.base(x) + self.scale * update


def target_modules(model):
    for layer_index, layer in enumerate(model.model.layers):
        for name in TARGETS:
            parent = layer.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
            yield layer_index, name, parent


def inject_adapters(model, make_module):
    created = {}
    for layer_index, name, parent in target_modules(model):
        module = make_module(getattr(parent, name))
        setattr(parent, name, module)
        created[f"layers.{layer_index}.{name}"] = module
    return created


def adapter_state(adapters, projected):
    state = {}
    for key, module in adapters.items():
        b, a = module.effective_factors() if projected else (module.lora_B, module.lora_A)
        state[f"{key}.B"] = b.detach().float().cpu().contiguous()
        state[f"{key}.A"] = a.detach().float().cpu().contiguous()
    return state
