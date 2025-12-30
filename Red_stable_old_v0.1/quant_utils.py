import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

# Optional deps
try:
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model, TaskType
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False


# ── Registry of quantization modes ───────────────────────────────

_QUANT_REGISTRY = {
    "none": lambda m: m,  # no quantization
    "qat_int8": lambda m: replace_linear_with_quant(m, method="int8"),
    "qat_int4": lambda m: replace_linear_with_quant(m, method="int4"),
    "qlora_int4": lambda m: apply_qlo_model(m),
}


def attach_quant_stub(model: nn.Module, mode: str = "none"):
    if mode not in _QUANT_REGISTRY:
        raise ValueError(f"Unknown quant mode: {mode}")
    print(f"[Quantize] Applying quant mode: {mode}")
    return _QUANT_REGISTRY[mode](model)


@contextmanager
def temp_quant_mode(model: nn.Module, mode: str):
    backup = {n: p.clone().detach().cpu() for n, p in model.named_parameters()}
    attach_quant_stub(model, mode)
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name].to(param.device))


# ── Fake quantization utils (QAT) ────────────────────────────────

def simulate_quantize(x: torch.Tensor, method: str = "int8", axis: int = -1):
    if method == "none":
        return x
    elif method == "int8":
        max_val = x.abs().amax(dim=axis, keepdim=True)
        scale = 127 / (max_val + 1e-6)
        qx = torch.clamp((x * scale).round(), -127, 127)
        return qx / scale
    elif method == "int4":
        max_val = x.abs().amax(dim=axis, keepdim=True)
        scale = 7 / (max_val + 1e-6)
        qx = torch.clamp((x * scale).round(), -7, 7)
        return qx / scale
    else:
        raise ValueError(f"Unknown quantization method: {method}")


class QuantLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, quant_method="none"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.quant_method = quant_method

    def forward(self, x):
        weight = simulate_quantize(self.linear.weight, self.quant_method)
        return F.linear(x, weight, self.linear.bias)


def replace_linear_with_quant(model: nn.Module, method: str = "int8"):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_dim  = module.in_features
            out_dim = module.out_features
            bias    = module.bias is not None

            new_lin = QuantLinear(in_dim, out_dim, bias, quant_method=method)

            # copy weights
            new_lin.linear.weight.data.copy_(module.weight.data)
            if bias:
                new_lin.linear.bias.data.copy_(module.bias.data)

            new_lin = new_lin.to(module.weight.device, dtype=module.weight.dtype)
            setattr(model, name, new_lin)
        else:
            replace_linear_with_quant(module, method)
    return model


# ── True QLoRA (bnb 4-bit + adapters) ────────────────────────────

def apply_qlo_model(model: nn.Module):
    if not _HAS_BNB:
        raise ImportError("bitsandbytes + peft required for qlora_int4")

    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # Replace Linear → Linear4bit
    def _convert(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                new_linear = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    quant_type="nf4",
                    compute_dtype=compute_dtype,
                    compress_statistics=True,
                    quant_storage=torch.uint8,
                )
                with torch.no_grad():
                    new_linear.weight.copy_(child.weight.data)
                    if child.bias is not None:
                        new_linear.bias.copy_(child.bias.data)
                setattr(module, name, new_linear)
            else:
                _convert(child)
    _convert(model)

    # Freeze base
    for p in model.parameters():
        p.requires_grad = False

    # Add LoRA adapters
    LORA_TARGETS = ["q", "k", "v", "o", "wi", "wo"]
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        inference_mode=False,
        bias="none"
    )
    model = get_peft_model(model, lora_cfg)
    print("[QLoRA] 4-bit base + LoRA adapters attached.")
    return model
