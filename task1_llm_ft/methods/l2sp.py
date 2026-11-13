import torch


class L2SPReference:
    def __init__(self, named_params, device):
        self.reference = {}
        for name, p in named_params:
            ref = p.detach().to(device=device, dtype=torch.bfloat16)
            assert torch.equal(ref.to(p.device).float(), p.detach().float()), f"{name}: starting weights are not exactly bfloat16, reference would be lossy"
            self.reference[name] = ref
        self.device = device

    def storage_bytes(self):
        return int(sum(t.numel() * t.element_size() for t in self.reference.values()))

    def penalty_and_grad(self, named_params, lam):
        omega = torch.zeros((), dtype=torch.float64, device=self.device)
        for name, p in named_params:
            diff = p.detach() - self.reference[name].to(p.device, torch.float32)
            omega += 0.5 * diff.pow(2).sum().double()
            if lam != 0 and p.grad is not None:
                p.grad.add_(diff, alpha=lam)
        return float(omega)

    def displacement(self, named_params):
        total = torch.zeros((), dtype=torch.float64)
        for name, p in named_params:
            diff = p.detach().float() - self.reference[name].to(p.device, torch.float32)
            total += diff.pow(2).sum().double().cpu()
        return float(total.sqrt())
