import torch
import torch.nn as nn
import torch.nn.functional as F
from rope_utils import apply_rope
from config import Config

cfg = Config("D:/RED LLM/RED AI/RED/config/train_config.json")


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_rope=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope

        # 🔧 Separate projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv=None, mask=None, past_kv=None, use_cache=False, rope_cache=None):
        try:
            q_input = x  # Query input
            kv_input = kv if kv is not None else x  # Key-Value input

            # Ensure we're working with [B, T, D] format consistently
            original_q_shape = q_input.shape
            original_kv_shape = kv_input.shape

            # Convert to [B, T, D] if needed
            if q_input.dim() == 3 and q_input.shape[0] != q_input.shape[1]:
                # Likely in [T, B, D] format, transpose to [B, T, D]
                if q_input.shape[0] > q_input.shape[1] and q_input.shape[0] > 64:  # T > B heuristic
                    q_input = q_input.transpose(0, 1)
                    if cfg.debug_mode:
                        print(f"[DEBUG:ATTN] Transposed Q from {original_q_shape} to {q_input.shape}")

            if kv_input.dim() == 3 and kv_input.shape[0] != kv_input.shape[1]:
                if kv_input.shape[0] > kv_input.shape[1] and kv_input.shape[0] > 64:  # T > B heuristic
                    kv_input = kv_input.transpose(0, 1)
                    if cfg.debug_mode:
                        print(f"[DEBUG:ATTN] Transposed KV from {original_kv_shape} to {kv_input.shape}")

            B, Tq, _ = q_input.shape
            _, Tk, _ = kv_input.shape

            if cfg.debug_mode:
                print(f"[DEBUG:ATTN] After normalization - Q: {q_input.shape}, KV: {kv_input.shape}")
                print(f"[DEBUG:ATTN] Device: {q_input.device}, dtype: {q_input.dtype}")

            # QKV projection
            q_raw = self.q_proj(q_input)
            k_raw = self.k_proj(kv_input)
            v_raw = self.v_proj(kv_input)

            # Reshape for multi-head attention: [B, T, D] -> [B, num_heads, T, head_dim]
            q = q_raw.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
            k = k_raw.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
            v = v_raw.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

            # 🧠 Apply RoPE if enabled
            if self.use_rope and rope_cache is not None:
                if cfg.debug_mode:
                    print(f"[DEBUG:ROPE] Before RoPE - Q: {q.shape}, K: {k.shape}")
                    print(f"[DEBUG:ROPE] RoPE cache shape: {rope_cache.shape}")

                # Apply RoPE to Q and K
                q = apply_rope(q, rope_cache)
                k = apply_rope(k, rope_cache)

                if cfg.debug_mode:
                    print(f"[DEBUG:ROPE] After RoPE - Q: {q.shape}, K: {k.shape}")

            # Handle past KV cache (for decoder)
            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=2)  # Concat along sequence dimension
                v = torch.cat([past_v, v], dim=2)

            cache_to_return = (k, v) if use_cache else None

            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

            # Apply mask if provided
            if mask is not None:
                if mask.dtype != torch.bool:
                    mask = mask.bool()

                # Expand mask dimensions to match scores
                if mask.dim() == 2:  # [B, T]
                    mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                elif mask.dim() == 3:  # [B, T, T]
                    mask = mask.unsqueeze(1)  # [B, 1, T, T]

                # Apply mask
                scores = scores.masked_fill(~mask, float('-inf'))

            # 🔒 STABILITY PATCH
            # Replace NaNs and infs with safe values
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e4, neginf=-1e4)

            # Subtract max for softmax stability
            scores_max = scores.max(dim=-1, keepdim=True).values
            scores = scores - scores_max

            if cfg.debug_mode:
                print(
                    f"[🧪] Score stats after normalization → min: {scores.min()}, max: {scores.max()}, mean: {scores.mean()}")

            # Softmax and dropout
            attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            context = torch.matmul(attn_weights, v)

            # Reshape back: [B, num_heads, T, head_dim] -> [B, T, D]
            context = context.transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)

            # Final projection
            output = self.out_proj(context)

            if cfg.debug_mode:
                print(f"[DEBUG:ATTN] Final output shape: {output.shape}")

            return output, cache_to_return

        except Exception as e:
            print(f"[💥] ATTENTION CRASH → {type(e).__name__}: {e}")
            print(f"[💥] Shapes at crash - q_input: {q_input.shape if 'q_input' in locals() else 'N/A'}")
            print(f"[💥] Shapes at crash - kv_input: {kv_input.shape if 'kv_input' in locals() else 'N/A'}")
            if 'rope_cache' in locals() and rope_cache is not None:
                print(f"[💥] RoPE cache shape: {rope_cache.shape}")
            raise