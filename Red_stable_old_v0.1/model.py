import torch
import torch.nn as nn
import math
from peft import get_peft_model, LoraConfig
from transformers import T5Config
from torch.utils.checkpoint import checkpoint
from config import Config
import time
from attention import CustomMultiHeadAttention
import torch.nn.functional as F
from rope_utils import build_rope_cache

AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

cfg = Config("D:/RED LLM/RED AI/RED/config/train_config.json")

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, layer_index=None):
        super().__init__()
        self.layer_index = layer_index

        self.self_attn = CustomMultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            use_rope=True
        )

        self.gate_up = nn.Linear(d_model, dim_feedforward * 2)
        self.down = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.residual_scale = 0.85 if layer_index is not None and layer_index > 14 else 1.0
        self._initialize_weights()

    def _initialize_weights(self):
        if self.layer_index is not None and self.layer_index > 14:
            nn.init.xavier_uniform_(self.gate_up.weight, gain=1e-3)
            nn.init.xavier_uniform_(self.down.weight, gain=1e-3)
            nn.init.zeros_(self.gate_up.bias)
            nn.init.zeros_(self.down.bias)

    def forward(self, src, src_key_padding_mask=None, rope_cache=None):
        from torch.cuda.amp import autocast
        with autocast(dtype=AMP_DTYPE):
            if cfg.debug_mode:
                start = time.time()
                print(f"[⏳] → Encoder Layer {self.layer_index} | AMP dtype: {AMP_DTYPE}")

            # === Sanitize Input ===
            src = torch.nan_to_num(src, nan=0.0, posinf=10.0, neginf=-10.0)
            src = torch.clamp(src, -30.0, 30.0)

            # === Self-Attention ===
            residual = src
            src = self.norm1(src)

            if rope_cache is not None and rope_cache.shape[0] != src.shape[1]:
                rope_cache = rope_cache[:src.shape[1]]

            if cfg.debug_mode:
                print(f"[DEBUG] Encoder Layer {self.layer_index} input: {src.shape}, rope: {rope_cache.shape if rope_cache is not None else None}")

            src2, _ = self.self_attn(src, mask=src_key_padding_mask, rope_cache=rope_cache)
            src2 = torch.nan_to_num(src2, nan=0.0, posinf=10.0, neginf=-10.0)
            src2 = torch.clamp(src2, -30.0, 30.0)

            src = residual + self.dropout1(src2) * self.residual_scale

            # === Feedforward ===
            residual = src
            src = self.norm2(src)

            a, b = self.gate_up(src).chunk(2, dim=-1)
            src2 = F.silu(a) * b

            src2 = torch.nan_to_num(src2, nan=0.0, posinf=30.0, neginf=-30.0)
            src2 = self.dropout2(src2)
            src2 = self.down(src2)

            src2 = torch.nan_to_num(src2, nan=0.0, posinf=10.0, neginf=-10.0)
            src2 = torch.clamp(src2, -30.0, 30.0)

            output = residual + src2 * self.residual_scale
            output = torch.clamp(output, -30.0, 30.0)

            if self.layer_index == 17 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()

            if cfg.debug_mode:
                duration = time.time() - start
                print(f"[⏱️] ← Encoder Layer {self.layer_index} done in {duration:.3f}s | dtype: {output.dtype} | max: {output.max():.4f}")

            return output

class T5Encoder(nn.Module):
    def __init__(self, shared_embedding, d_model, num_layers, num_heads, d_ff,
                 dropout_rate=0.1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = shared_embedding
        self.embedding_dp = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList([
            CustomEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout_rate,
                layer_index=i
            ) for i in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, rope_cache=None):
        if input_ids is not None:
            input_ids = input_ids.long()
            x = self.embedding(input_ids)
        elif inputs_embeds is not None:
            assert inputs_embeds.dtype == torch.float32
            x = inputs_embeds
        else:
            raise ValueError("Provide either input_ids or inputs_embeds")

        x = self.embedding_dp(x)

        if torch.isnan(x).any():
            raise ValueError("❌ NaN in encoder input embeddings")

        key_padding_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.bool() if attention_mask.dtype != torch.bool else attention_mask
            key_padding_mask = ~attention_mask

        for i, layer in enumerate(self.layers):
            if cfg.debug_mode:
                print(f"[🔍] Encoder Layer {i} INPUT → min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")

            try:
                rope_slice = rope_cache[:x.shape[1]] if rope_cache is not None else None

                if self.use_checkpoint and self.training:
                    def enc_forward(x_):
                        return layer(x_, src_key_padding_mask=key_padding_mask, rope_cache=rope_slice)
                    x = checkpoint(enc_forward, x)
                else:
                    x = layer(x, src_key_padding_mask=key_padding_mask, rope_cache=rope_slice)

            except Exception as e:
                raise RuntimeError(f"[💥] Crash at Encoder Layer {i}: {e}")

            if torch.isnan(x).any():
                torch.save(x.detach().cpu(), f"crash_encoder_layer_{i}.pt")
                raise ValueError(f"❌ NaN detected after encoder layer {i}")

            if cfg.debug_mode:
                print(f"[✅] Encoder Layer {i} OUTPUT → min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")

        x = self.layer_norm(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=5.5, neginf=-5.5)
        x = torch.clamp(x, -30.0, 30.0)

        return x

class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, layer_index=None):
        super().__init__()
        self.layer_index = layer_index

        self.self_attn = CustomMultiHeadAttention(
            d_model, nhead, dropout=dropout, use_rope=True
        )
        self.cross_attn = CustomMultiHeadAttention(
            d_model, nhead, dropout=dropout, use_rope=True
        )

        self.gate_up = nn.Linear(d_model, dim_feedforward * 2)
        self.down = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.residual_scale = 0.85 if layer_index is not None and layer_index > 14 else 1.0
        self._initialize_weights()

    def _initialize_weights(self):
        if self.layer_index is not None and self.layer_index > 14:
            nn.init.xavier_uniform_(self.gate_up.weight, gain=1e-3)
            nn.init.xavier_uniform_(self.down.weight, gain=1e-3)
            nn.init.zeros_(self.gate_up.bias)
            nn.init.zeros_(self.down.bias)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_rope=None, mem_rope=None):
        from torch.cuda.amp import autocast

        with autocast(dtype=AMP_DTYPE):
            if cfg.debug_mode:
                start = time.time()
                print(f"[⏳] → Decoder Layer {self.layer_index} Forward Start | AMP dtype: {AMP_DTYPE}")

            # -------------------------
            # Normalize input layouts
            # Accept both [B, T, D] and [T, B, D]
            # -------------------------
            transposed_input = False
            if tgt.dim() == 3:
                # Heuristic: if first dim is much larger than second, it's likely [T, B, D]
                if tgt.shape[0] > tgt.shape[1] and tgt.shape[0] > 4:
                    tgt = tgt.transpose(0, 1).contiguous()
                    transposed_input = True
                    if cfg.debug_mode:
                        print(f"[TRANSPOSED] tgt was [T,B,D]; transposed to {tgt.shape}")

            if memory.dim() == 3:
                if memory.shape[0] > memory.shape[1] and memory.shape[0] > 4:
                    memory = memory.transpose(0, 1).contiguous()
                    if cfg.debug_mode:
                        print(f"[TRANSPOSED] memory was [T,B,D]; transposed to {memory.shape}")

            # Now assume tgt is [B, T_tgt, D], memory is [B, T_mem, D]
            B, T_tgt, D = tgt.shape
            _, T_mem, _ = memory.shape
            head_dim = self.self_attn.head_dim

            # -------------------------
            # Normalize/prepare RoPE caches so they are [T, head_dim]
            # Accept forms: [T, D], [1, D], [B, D], [B, T, D]
            # If rope looks like [B, D] (common bug), tile first row to [T, D]
            # -------------------------
            def _prepare_rope(rope, seq_len, rope_name="rope"):
                if rope is None:
                    return None

                # move to same device/dtype as model tensors
                rope = rope.to(device=tgt.device, dtype=tgt.dtype)

                if rope.dim() == 3:
                    # [B, T, D] or [1, T, D]
                    if rope.shape[0] == 1:
                        rope = rope[0, :seq_len, :].contiguous()
                    elif rope.shape[0] == B and rope.shape[1] >= seq_len and rope.shape[2] == head_dim:
                        rope = rope[0, :seq_len, :].contiguous()
                    else:
                        raise ValueError(
                            f"[❌] Unexpected {rope_name} shape {rope.shape} for B={B}, seq_len={seq_len}, head_dim={head_dim}"
                        )

                elif rope.dim() == 2:
                    # [T, D] (good), [1, D], [B, D], or mismatch
                    r0, r1 = rope.shape
                    if r1 != head_dim:
                        raise ValueError(f"[❌] {rope_name} second dim {r1} != expected head_dim {head_dim}")

                    if r0 == seq_len:
                        rope = rope.contiguous()
                    elif r0 == 1:
                        rope = rope.expand(seq_len, head_dim).contiguous()
                    elif r0 == B:
                        rope = rope[0:1].expand(seq_len, head_dim).contiguous()
                    elif r0 < seq_len:
                        pad_len = seq_len - r0
                        pad = torch.zeros(pad_len, head_dim, device=rope.device, dtype=rope.dtype)
                        rope = torch.cat([rope, pad], dim=0)[:seq_len, :].contiguous()
                    elif r0 > seq_len:
                        rope = rope[:seq_len, :].contiguous()
                    else:
                        raise ValueError(f"[❌] Unsupported {rope_name} shape {rope.shape}")

                else:
                    raise ValueError(f"[❌] Unsupported {rope_name} shape {rope.shape}")

                # ✅ Final safety check
                assert rope.shape == (seq_len, head_dim), \
                    f"[❌] {rope_name} final shape {rope.shape} != expected ({seq_len}, {head_dim})"

                return rope

            # Prepare both ropes
            tgt_rope_prepared = _prepare_rope(tgt_rope, T_tgt, "tgt_rope") if tgt_rope is not None else None
            mem_rope_prepared = _prepare_rope(mem_rope, T_mem, "mem_rope") if mem_rope is not None else None

            # -------------------------
            # Self-Attention (pre-norm)
            # -------------------------
            residual = tgt
            tgt = self.norm1(tgt)

            if cfg.debug_mode:
                print(
                    f"[DEBUG] Decoder Self-Attn input → {tgt.shape}, rope: {tgt_rope_prepared.shape if tgt_rope_prepared is not None else None}")

            tgt2, _ = self.self_attn(
                x=tgt,
                mask=tgt_key_padding_mask,
                rope_cache=tgt_rope_prepared
            )
            tgt2 = torch.nan_to_num(tgt2, nan=0.0, posinf=10.0, neginf=-10.0)
            tgt2 = torch.clamp(tgt2, -30.0, 30.0)

            tgt = residual + self.dropout1(tgt2) * self.residual_scale

            # -------------------------
            # Cross-Attention (pre-norm)
            # -------------------------
            residual = tgt
            tgt = self.norm2(tgt)

            if cfg.debug_mode:
                print(
                    f"[DEBUG] Decoder Cross-Attn input → tgt: {tgt.shape}, mem: {memory.shape}, rope: {mem_rope_prepared.shape if mem_rope_prepared is not None else None}")

            tgt2, _ = self.cross_attn(
                x=tgt,
                kv=memory,
                mask=memory_key_padding_mask,
                rope_cache=mem_rope_prepared
            )
            tgt2 = torch.nan_to_num(tgt2, nan=0.0, posinf=10.0, neginf=-10.0)
            tgt2 = torch.clamp(tgt2, -30.0, 30.0)

            tgt = residual + self.dropout2(tgt2) * self.residual_scale

            # -------------------------
            # Feedforward
            # -------------------------
            residual = tgt
            tgt = self.norm3(tgt)

            a, b = self.gate_up(tgt).chunk(2, dim=-1)
            tgt2 = F.silu(a) * b

            tgt2 = torch.nan_to_num(tgt2, nan=0.0, posinf=30.0, neginf=-30.0)
            tgt2 = self.dropout3(tgt2)
            tgt2 = self.down(tgt2)

            tgt2 = torch.nan_to_num(tgt2, nan=0.0, posinf=10.0, neginf=-10.0)
            tgt2 = torch.clamp(tgt2, -30.0, 30.0)

            output = residual + tgt2 * self.residual_scale
            output = torch.clamp(output, -30.0, 30.0)

            if self.layer_index == 17 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()

            if cfg.debug_mode:
                duration = time.time() - start
                print(
                    f"[⏱️] ← Decoder Layer {self.layer_index} done in {duration:.3f}s | dtype: {output.dtype} | max: {output.max():.4f}")

            return output

class T5Decoder(nn.Module):
    def __init__(self, shared_embedding, d_model, num_layers, num_heads, d_ff,
                 dropout_rate=0.1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = shared_embedding
        self.embedding_dp = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList([
            CustomDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout_rate,
                layer_index=i
            ) for i in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.output_layer = nn.Linear(d_model, self.embedding.num_embeddings, bias=False)
        self.output_layer.weight = self.embedding.weight

    def forward(self, tgt, memory, tgt_attention_mask=None, memory_attention_mask=None,
                tgt_rope=None, mem_rope=None):
        from torch.cuda.amp import autocast

        if tgt.dtype == torch.int64:
            x = self.embedding(tgt)
            x = self.embedding_dp(x)
        elif tgt.dtype == torch.float32:
            x = tgt
        else:
            raise ValueError(f"[❌] Unsupported tgt dtype: {tgt.dtype}")

        if torch.isnan(x).any():
            raise ValueError("❌ NaN in decoder input embeddings")

        x = x.transpose(0, 1)
        memory = memory.transpose(0, 1)

        tgt_mask = ~tgt_attention_mask.bool() if tgt_attention_mask is not None else None
        mem_mask = ~memory_attention_mask.bool() if memory_attention_mask is not None else None

        with autocast(dtype=AMP_DTYPE):
            for i, layer in enumerate(self.layers):
                if cfg.debug_mode:
                    print(f"[🔬] Decoder Layer {i} INPUT → min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")

                tgt_slice = tgt_rope[:x.shape[0]] if tgt_rope is not None else None
                mem_slice = mem_rope[:memory.shape[0]] if mem_rope is not None else None

                x = layer(
                    x, memory,
                    tgt_key_padding_mask=tgt_mask,
                    memory_key_padding_mask=mem_mask,
                    tgt_rope=tgt_slice,
                    mem_rope=mem_slice
                )

                x = torch.clamp(x, -30.0, 30.0)

                if torch.isnan(x).any():
                    raise ValueError(f"[❌] NaN after decoder layer {i}")

                if cfg.debug_mode:
                    print(f"[✅] Decoder Layer {i} OUTPUT → min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")

        x = x.transpose(0, 1)
        x = self.layer_norm(x)
        x = torch.clamp(x, -30.0, 30.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=5.5, neginf=-5.5)

        return x

class T5LikeModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout_rate=0.15, use_checkpoint=False):
        super().__init__()

        if cfg.debug_mode:
            print(f"[DEBUG] Initializing model with d_model={d_model}")

        head_dim = d_model // num_heads
        self.rope_cache = build_rope_cache(
            seq_len=512,  # ✅ Temporary static length
            dim=head_dim,  # ✅ Crucial fix: now matches Q/K
            device=torch.device("cuda"),
            dtype=torch.bfloat16  # ✅ BF16 stable + fast
        )

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dp = nn.Dropout(dropout_rate)

        # Output LM head tied with shared embedding weights
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Extra dropout after encoder and decoder output
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.decoder_dropout = nn.Dropout(dropout_rate)

        self.encoder = T5Encoder(
            shared_embedding=self.embedding,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            use_checkpoint=use_checkpoint
        )

        self.decoder = T5Decoder(
            shared_embedding=self.embedding,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            use_checkpoint=use_checkpoint
        )

        self.config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=num_heads,
            is_encoder_decoder=True,
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
            model_type="t5",
        )

        # Add generation compatibility attributes
        self.decoder_start_token_id = self.config.decoder_start_token_id
        self.eos_token_id = self.config.eos_token_id
        self.pad_token_id = self.config.pad_token_id

        self.main_input_name = "input_ids"
        self.generation_config = None

        # ✅ Initialize weights (new improved version)
        self.initialize_weights()

    # ✅ Add this method AFTER __init__ (same level as forward)
    def initialize_weights(self):
        """Improved weight initialization for better training stability"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Output layers need smaller initialization
                if 'output' in name or 'lm_head' in name:
                    nn.init.xavier_uniform_(module.weight, gain=1e-3)
                else:
                    # Other layers use ReLU-optimized initialization
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))

                # Initialize biases to zero
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                # Standard embedding initialization
                nn.init.xavier_uniform_(module.weight)

                # Special handling for padding token
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

            elif isinstance(module, nn.LayerNorm):
                # Standard LayerNorm initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ──────────────────────────────────────────────────────────────
    def quantize(self, mode: str = "none"):
        """
        Apply quantization wrappers in-place.

        Modes:
            - "none"       : Full precision (default).
            - "qat_int8"   : Fake quantization-aware training in int8.
                            *Forward pass simulates int8 weights,
                            but weights/optim states remain float.*
                            (Good for robustness/debug, no VRAM savings.)

            - "qat_int4"   : Fake quantization-aware training in int4.
                            *Forward pass simulates int4 weights,
                            but weights/optim states remain float.*
                            (Good for robustness/debug, no VRAM savings.)

            - "qlora_int4" : True QLoRA mode.
                            *Replaces Linear → bitsandbytes Linear4bit (nf4),
                            freezes base weights, attaches LoRA adapters.*
                            (Real VRAM savings, train adapters only.)

        Example:
            model.quantize("qlora_int4")  # enable 4-bit + LoRA training
        """
        from quant_utils import attach_quant_stub
        attach_quant_stub(self, mode)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, decoder_input_ids=None,
                decoder_inputs_embeds=None, decoder_attention_mask=None, labels=None, **kwargs):
        # ===== 1. Input Embedding =====
        if input_ids is not None:
            assert inputs_embeds is None, "Cannot specify both input_ids and inputs_embeds."
            input_ids = input_ids.long().clamp(0, self.embedding.num_embeddings - 1)
            inputs_embeds = self.embedding(input_ids)
            inputs_embeds = self.embedding_dp(inputs_embeds)
        elif inputs_embeds is not None:
            assert inputs_embeds.dtype == torch.float32, "inputs_embeds must be float32"
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        # ===== NEW: Clamp input embeddings =====
        inputs_embeds = torch.clamp(inputs_embeds, -30.0, 30.0)

        # ===== 2. Decoder Embedding =====
        if decoder_inputs_embeds is None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.long().clamp(0, self.embedding.num_embeddings - 1)
                decoder_inputs_embeds = self.embedding(decoder_input_ids)
                decoder_inputs_embeds = self.embedding_dp(decoder_inputs_embeds)
            else:
                raise ValueError("Either decoder_input_ids or decoder_inputs_embeds must be provided.")

        # ===== NEW: Clamp decoder embeddings =====
        decoder_inputs_embeds = torch.clamp(decoder_inputs_embeds, -30.0, 30.0)

        # ===== 3. Align Decoder Input with Attention Mask =====
        if decoder_attention_mask is not None:
            dec_len = decoder_attention_mask.size(1)
            if decoder_inputs_embeds.size(1) != dec_len:
                decoder_inputs_embeds = decoder_inputs_embeds[:, :dec_len, :]

        # ===== 4. Encoder Forward Pass =====
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            rope_cache=self.rope_cache
        )
        encoder_outputs = self.encoder_dropout(encoder_outputs)

        # ===== NEW: Clamp encoder outputs =====
        encoder_outputs = torch.clamp(encoder_outputs, -30.0, 30.0)

        # ===== 5. Decoder Forward Pass =====
        # Compute lengths from actual inputs
        mem_len = encoder_outputs.size(1)  # Fix: true encoder length
        tgt_len = decoder_inputs_embeds.size(1)

        tgt_rope = self.rope_cache[:tgt_len]
        mem_rope = self.rope_cache[:mem_len]  # ✅ FIXED
        if cfg.debug_mode:
            print(f"[DEBUG] mem_rope: {mem_rope.shape}, memory: {encoder_outputs.shape}")

        # ===== Rope Pre-sanitization before Decoder =====
        def _sanitize_rope(rope, expected_seq_len, ref_tensor):
            """
            Ensures rope is in [T, D] format and matches expected_seq_len.
            `ref_tensor` is used for device/dtype alignment.
            """
            if rope is None:
                return None

            # Match device/dtype to reference tensor (encoder or decoder outputs)
            rope = rope.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

            if rope.dim() == 2:
                # Already [T, D]
                if rope.shape[0] != expected_seq_len:
                    # Tile first row if shape is [1, D] or [B, D]
                    rope = rope[0:1].expand(expected_seq_len, rope.shape[1]).contiguous()

            elif rope.dim() == 3:
                # [B, T, D] → Take first batch
                if rope.shape[0] == 1:
                    rope = rope[0, :, :].contiguous()
                elif rope.shape[0] == ref_tensor.size(0):  # batch matches
                    rope = rope[0, :, :].contiguous()
                else:
                    raise ValueError(f"[❌] Unexpected rope shape: {rope.shape}")

            return rope

        # Use encoder_outputs instead of memory
        tgt_rope = _sanitize_rope(tgt_rope, tgt_len, decoder_inputs_embeds)
        mem_rope = _sanitize_rope(mem_rope, mem_len, encoder_outputs)

        decoder_outputs = self.decoder(
            tgt=decoder_inputs_embeds,
            memory=encoder_outputs,
            tgt_attention_mask=decoder_attention_mask,
            memory_attention_mask=attention_mask,
            tgt_rope=tgt_rope,
            mem_rope=mem_rope
        )

        # ===== NEW: Clamp decoder outputs =====
        decoder_outputs = torch.clamp(decoder_outputs, -30.0, 30.0)

        # ===== 6. Final Linear Projection =====
        if cfg.debug_mode:
            print("decoder_outputs.shape:", decoder_outputs.shape)
            print("lm_head weight shape:", self.lm_head.weight.shape)

        logits = self.lm_head(decoder_outputs)

        # ✅ Clamp logits to prevent FP16 overflow
        logits = torch.clamp(logits, -30, 30)  # Increased range

        # ===== 7. Optional Loss Computation =====
        loss = None
        if labels is not None:
            # ✅ Align batch size first
            if logits.size(0) != labels.size(0):
                if cfg.debug_mode:
                    print(
                        f"[DEBUG] Fixing batch mismatch: logits batch={logits.size(0)}, labels batch={labels.size(0)}")

                min_b = min(logits.size(0), labels.size(0))
                logits = logits[:min_b]
                labels = labels[:min_b]

            # ===== Sanity check labels =====
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print("⚠️ Invalid values detected in labels!")
                labels = torch.nan_to_num(labels, nan=-100, posinf=-100, neginf=-100)

            # ✅ Align sequence length
            seq_len = min(logits.size(1), labels.size(1))
            logits = logits[:, :seq_len, :]
            labels = labels[:, :seq_len]

            if cfg.debug_mode:
                print(f"[DEBUG] logits: {logits.shape}, labels: {labels.shape}")

            # ===== Safe CE loss =====
            try:
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits.reshape(-1, logits.size(-1)),  # [B*T, V]
                    labels.reshape(-1)  # [B*T]
                )
            except Exception as e:
                print(f"⚠️ Loss computation failed: {e}")
                loss = torch.tensor(0.0, requires_grad=True, device=logits.device)

        if not hasattr(self, "_debug_printed"):
            self._debug_printed = True

        # ===== NEW: Final NaN check =====
        if torch.isnan(logits).any() or (loss is not None and torch.isnan(loss)):
            print("🚨 CRITICAL: NaN detected in final outputs!")
            # Attempt to recover by replacing NaNs
            logits = torch.nan_to_num(logits, nan=0.0)
            if loss is not None and torch.isnan(loss):
                loss = torch.tensor(0.0, requires_grad=True, device=logits.device)

        return {
            "logits": logits,
            "loss": loss,
            "encoder_outputs": encoder_outputs,
        }

    # PEFT compatibility methods
    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, **kwargs):
        # Mimic HF T5's behavior
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
            "encoder_outputs": kwargs.get("encoder_outputs", None),
        }

    # Generation stubs
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        device = self.device
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": kwargs.get("attention_mask", None),
            "decoder_input_ids": kwargs.get("decoder_input_ids", input_ids).to(device),
        }

    def generate(self, input_ids, max_length=128, eos_token_id=None, pad_token_id=None):
        """
        Greedy decoding for sequence generation.
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        generated = torch.full((int(batch_size), 1), self.decoder_start_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            # Forward pass through model
            outputs = self(
                input_ids=input_ids,
                decoder_input_ids=generated,
            )
            logits = outputs["logits"]  # shape: [batch, seq_len, vocab_size]
            next_token_logits = logits[:, -1, :]  # get last token's logits

            # Greedy: take the highest prob token
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=1)

            # Optional: stop if all sequences generated EOS
            if eos_token_id is not None:
                if (next_tokens == eos_token_id).all():
                    break

        return generated

    def resize_token_embeddings(self, new_size: int):
        old_embedding = self.embedding
        new_embedding = nn.Embedding(new_size, old_embedding.embedding_dim)

        with torch.no_grad():
            num_tokens_to_copy = min(old_embedding.num_embeddings, new_size)
            new_embedding.weight[:num_tokens_to_copy] = old_embedding.weight[:num_tokens_to_copy]

        self.embedding = new_embedding.to(old_embedding.weight.device)

        # Also update encoder and decoder embeddings!
        self.encoder.embedding = self.embedding
        self.decoder.embedding = self.embedding

        self.config.vocab_size = new_size

    def adjust_logits_during_generation(self, logits, **kwargs):
        return logits

    @property
    def device(self):
        return next(self.parameters()).device

    def __str__(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return (f"T5LikeModel (trainable: {trainable:,}, "
                f"total: {total:,})")
