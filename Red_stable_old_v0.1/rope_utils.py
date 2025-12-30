import torch
import math
from config import Config

cfg = Config("D:/RED LLM/RED AI/RED/config/train_config.json")


def build_rope_cache(seq_len, dim, base=10000, device=None, dtype=None):
    """Build RoPE cache for rotary position encoding."""
    theta = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(seq_idx, theta)  # [seq_len, dim // 2]

    # Create sin and cos components
    cos_vals = freqs.cos()
    sin_vals = freqs.sin()

    # Stack them properly for RoPE application
    rope_cache = torch.stack([cos_vals, sin_vals], dim=-1)  # [seq_len, dim//2, 2]
    return rope_cache.view(seq_len, dim)  # [seq_len, dim]

def apply_rope(x, rope_cache):
    """
    Applies rotary positional encoding (RoPE) to tensor x using rope_cache.
    Handles shapes [T, D], [B, T, D], or [B, D] with padding/truncation safety.

    Args:
        x: [batch, heads, seq_len, head_dim]
        rope_cache: [T, D], [B, T, D], or [B, D]

    Returns:
        Tensor with RoPE applied (same shape as x)
    """
    if rope_cache is None:
        return x

    batch_size, num_heads, seq_len, head_dim = x.shape

    # Move rope_cache to correct device/dtype
    rope_cache = rope_cache.to(device=x.device, dtype=x.dtype)

    # ---- Auto-handle shapes ----
    if rope_cache.dim() == 3:  # [B, T, D]
        if rope_cache.shape[0] == batch_size and rope_cache.shape[2] == head_dim:
            rope = rope_cache[0, :seq_len, :]
        elif rope_cache.shape[1] == seq_len and rope_cache.shape[2] == head_dim:
            rope = rope_cache[0, :seq_len, :]
        else:
            raise ValueError(f"[❌] Unexpected rope_cache shape (3D): {rope_cache.shape}")

    elif rope_cache.dim() == 2:
        if rope_cache.shape[1] != head_dim:
            raise ValueError(f"[❌] rope_cache D mismatch: {rope_cache.shape[1]} vs head_dim {head_dim}")

        if rope_cache.shape[0] == seq_len:
            rope = rope_cache
        elif rope_cache.shape[0] == 1:  # [1, D] → expand to [T, D]
            rope = rope_cache.expand(seq_len, head_dim)
        elif rope_cache.shape[0] == batch_size:  # [B, D] → tile first row
            rope = rope_cache[0:1, :].expand(seq_len, head_dim)
        elif rope_cache.shape[0] < seq_len:  # pad
            pad_len = seq_len - rope_cache.shape[0]
            pad = torch.zeros(pad_len, head_dim, device=rope_cache.device, dtype=rope_cache.dtype)
            rope = torch.cat([rope_cache, pad], dim=0)[:seq_len, :]
        else:  # truncate
            rope = rope_cache[:seq_len, :]
    else:
        raise ValueError(f"[❌] Unsupported rope_cache ndim: {rope_cache.dim()}")

    # ---- Expand to match attention heads ----
    rope = rope.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]

    # ---- Split into even/odd and apply rotation ----
    cos_vals = rope[..., ::2]
    sin_vals = rope[..., 1::2]
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    x_rotated_even = x_even * cos_vals - x_odd * sin_vals
    x_rotated_odd = x_even * sin_vals + x_odd * cos_vals

    # ---- Merge back ----
    x_rotated = torch.zeros_like(x)
    x_rotated[..., ::2] = x_rotated_even
    x_rotated[..., 1::2] = x_rotated_odd

    return x_rotated
