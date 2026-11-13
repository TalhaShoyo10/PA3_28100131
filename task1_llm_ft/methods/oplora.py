import time

import torch

from task1_llm_ft.methods.lora import LoRALinear


class OPLoRALinear(LoRALinear):
    def __init__(self, base, rank, alpha, dropout, protect_rank):
        super().__init__(base, rank, alpha, dropout)
        u, _, vh = torch.linalg.svd(base.weight.detach().float(), full_matrices=False)
        self.register_buffer("U_k", u[:, :protect_rank].contiguous(), persistent=False)
        self.register_buffer("V_k", vh[:protect_rank].t().contiguous(), persistent=False)

    def effective_factors(self):
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            b = self.lora_B - self.U_k @ (self.U_k.t() @ self.lora_B)
            a = self.lora_A - (self.lora_A @ self.V_k) @ self.V_k.t()
        return b, a


def basis_bytes(adapters):
    return int(sum(m.U_k.numel() * m.U_k.element_size() + m.V_k.numel() * m.V_k.element_size() for m in adapters.values()))


def constraint_residuals(adapters):
    rows = {}
    for key, module in adapters.items():
        delta = module.delta_weight().detach()
        norm = torch.linalg.norm(delta)
        left = torch.linalg.norm(module.U_k.t() @ delta)
        right = torch.linalg.norm(delta @ module.V_k)
        rows[key] = {
            "delta_fro": float(norm),
            "UkT_delta_over_delta": float(left / norm) if norm > 0 else None,
            "delta_Vk_over_delta": float(right / norm) if norm > 0 else None,
        }
    return rows


def timed_export(adapters, export_fn):
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    state = export_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return state, time.perf_counter() - start
