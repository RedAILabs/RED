import os
import random
import torch
import torch.nn as nn
from transformers import T5Tokenizer
from model import T5LikeModel
from dataset import prepare_dataloaders, prepare_streaming_dataloader
from utils import (
    log_training_stats,
    handle_training_interrupt,
    handle_training_error,
    verify_training_readiness,
    evaluate_model,
    finalize_training,
    clean_gpu_memory,
    print_model_parameters,
    build_optimizer,
    gpu_memory_summary
)
from checkpoint import save_checkpoint, load_checkpoint
from config import Config
from tqdm import tqdm
from torch.amp import GradScaler
import json
from bitsandbytes.optim import AdamW8bit
from torch.optim import Adafactor
from muon import MuonWithAuxAdam
from transformers import get_cosine_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
from rope_utils import build_rope_cache

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ─── Load Config ────────────────────────────────────────────────────────────
cfg = Config("D:/RED LLM/RED AI/RED/config/train_config.json")
print("[*] Training Configuration Loaded")
if cfg.debug_mode:
    print(cfg)

# ─── Main Training Loop ──────────────────────────────────────────────────────
def main():
    clean_gpu_memory()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Set random seeds
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(cfg.model_path, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if cfg.debug_mode:
        print(f"[✓] Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Initialize Model
    model = T5LikeModel(
        vocab_size=len(tokenizer),
        d_model=768,
        d_ff=3072,
        num_layers=18,
        num_heads=12,
        dropout_rate=0.1,
        use_checkpoint=True
    )

    rope_cache = build_rope_cache(seq_len=512, dim=64, device=device, dtype=torch.float32)

    # === [AMP Precision Setup] ===
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if cfg.debug_mode:
        print(f"[DEBUG] Using model from: {model.__class__}")
        print(f"🔧 AMP will use: {amp_dtype}")

    scaler = GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # ======================== Embedding Resize & Safe Reinit ========================
    new_vocab_size = len(tokenizer)
    old_vocab_size = model.embedding.num_embeddings

    if new_vocab_size != old_vocab_size:
        print(f"[⚠️] Resizing model embeddings: {old_vocab_size} → {new_vocab_size}")

        # Resize shared embedding
        model.embedding = nn.Embedding(new_vocab_size, model.embedding.embedding_dim).to(device)

        # Resize encoder embedding
        model.encoder.embedding = nn.Embedding(new_vocab_size, model.encoder.embedding.embedding_dim).to(device)

        # Resize decoder embedding
        model.decoder.embedding = nn.Embedding(new_vocab_size, model.decoder.embedding.embedding_dim).to(device)

        # Re-tie embedding weights
        model.lm_head = nn.Linear(model.encoder.embedding.embedding_dim, new_vocab_size, bias=False).to(device)
        model.lm_head.weight = model.embedding.weight  # tie with new shared embedding

        # ✨ Optional: initialize new embedding weights
        with torch.no_grad():
            model.embedding.weight.normal_(mean=0.0, std=0.005)
            model.encoder.embedding.weight = model.embedding.weight
            model.decoder.embedding.weight = model.embedding.weight

    # Encoder and Decoder embeddings safety check
    encoder_old_vocab_size = model.encoder.embedding.num_embeddings
    decoder_old_vocab_size = model.decoder.embedding.num_embeddings

    if encoder_old_vocab_size != new_vocab_size or decoder_old_vocab_size != new_vocab_size:
        print(f"[⚡] Resizing encoder and decoder embeddings to match tokenizer ({new_vocab_size})")

        # Create new embeddings
        new_encoder_embed = nn.Embedding(new_vocab_size, model.encoder.embedding.embedding_dim)
        new_decoder_embed = nn.Embedding(new_vocab_size, model.decoder.embedding.embedding_dim)

        with torch.no_grad():
            # Copy over existing weights safely
            tokens_to_copy = min(encoder_old_vocab_size, new_vocab_size)
            new_encoder_embed.weight[:tokens_to_copy] = model.encoder.embedding.weight[:tokens_to_copy]
            tokens_to_copy = min(decoder_old_vocab_size, new_vocab_size)
            new_decoder_embed.weight[:tokens_to_copy] = model.decoder.embedding.weight[:tokens_to_copy]

            # ✨ Initialize newly added tokens
            if new_vocab_size > encoder_old_vocab_size:
                new_encoder_embed.weight[encoder_old_vocab_size:].normal_(mean=0.0, std=0.005)
            if new_vocab_size > decoder_old_vocab_size:
                new_decoder_embed.weight[decoder_old_vocab_size:].normal_(mean=0.0, std=0.005)

        model.encoder.embedding = new_encoder_embed
        model.decoder.embedding = new_decoder_embed

    # Move model to device AFTER resizing
    # Optional: Compile model with torch.compile() (Linux-only due to Triton)
    if getattr(cfg, "use_torch_compile", False):
        try:
            model = torch.compile(model)
            print("[🧠] Model compiled with torch.compile()")
        except Exception as e:
            print(f"[⚠️] torch.compile() skipped: {e}")

    model = model.to(device)

    print_model_parameters(model)

    # Final check
    verify_training_readiness(model)

    # Prepare Dataset
    print("[*] Preparing Datasets...")
    train_loader, val_loader = prepare_dataloaders(
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        cache_path=cfg.cache_dataset,
        dataset_path=cfg.train_dataset_path
    )
    print(f"[✓] Train batches: {len(train_loader)}")
    print(f"[✓] Validation batches: {len(val_loader)}")

    # ─── Safety Check: Inspect first batch ───
    first_batch = next(iter(train_loader))
    input_ids = first_batch["input_ids"]
    decoder_input_ids = first_batch["decoder_input_ids"]

    if cfg.debug_mode:
        print(f"[DEBUG] input_ids max: {input_ids.max().item()}, vocab_size: {len(tokenizer)}")
        print(f"[DEBUG] decoder_input_ids max: {decoder_input_ids.max().item()}, vocab_size: {len(tokenizer)}")

    assert input_ids.max().item() < len(tokenizer), "❌ Error: input_ids contain values outside tokenizer vocab!"
    assert decoder_input_ids.max().item() < len(
        tokenizer), "❌ Error: decoder_input_ids contain values outside tokenizer vocab!"
    print("[✓] Safety Check Passed: Inputs are clean.")

    # --- OPTIMIZER SETUP: Deduplicated safe parameter groups ---

    # ✅ Track already assigned parameters
    seen_param_ids = set()

    # Helper to safely add a param group
    def safe_group(params, lr):
        unique_params = []
        for p in params:
            if id(p) not in seen_param_ids:
                unique_params.append(p)
                seen_param_ids.add(id(p))
        return {"params": unique_params, "lr": lr}

    # Grouped with non-overlapping sets
    #params = [
    #    safe_group(model.encoder.parameters(), cfg.learning_rate),
    #    safe_group(
    #        [p for n, p in model.decoder.named_parameters() if not n.startswith("layers.17")],
    #        cfg.learning_rate
    #    ),
    #    safe_group(model.decoder.layers[17].parameters(), cfg.learning_rate * 0.05),
    #]

    # Optional: Add embedding or lm_head params if you want finer control (ensure not duplicated)
    # safe_group([model.embedding.parameters()], cfg.learning_rate)

    # ✅ Quantization (QAT mode, fake int4 training)
    if getattr(cfg, "quantize_on_train", False):
        model.quantize("qat_int4")

    # ✅ Optimizer (8-bit AdamW)
    #optimizer = AdamW8bit(
    #    model.parameters(),
    #    lr=cfg.learning_rate,
    #    betas=(0.9, 0.999),
    #    eps=1e-8,
    #    weight_decay=0.01
    #)

    #optimizer = Adafactor(
    #    model.parameters(),
    #    lr=cfg.learning_rate,  # you can also set to None if using relative_step
    #    scale_parameter=True,  # allows LR scaling by parameter norm
    #    relative_step=False,  # False = fixed LR like AdamW, True = self-scaling
    #    warmup_init=True,  # good for transformers
    #    eps=(1e-30, 1e-3),  # Adafactor defaults (different from AdamW)
    #    clip_threshold=1.0,  # helps stability
    #    weight_decay=0.01
    #)

    # ✅ Optimizer (Muon)
    #hidden_weights = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'body' in n]
    #hidden_gains_biases = [p for n, p in model.named_parameters() if p.ndim < 2 and 'body' in n]
    #nonhidden_params = [p for n, p in model.named_parameters() if 'embed' in n or 'head' in n]

    #param_groups = [
    #    dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
    #    dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
    #         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
    #]

    #optimizer = MuonWithAuxAdam(param_groups)

    optimizer = build_optimizer(model, cfg)

    # --- Scheduler with warmup + cosine decay --- # Temporarily removed for phase 2
    total_training_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(0.05 * total_training_steps)  # You can adjust % (e.g., 5–10%)

    #scheduler = get_cosine_schedule_with_warmup(
    #    optimizer,
    #    num_warmup_steps=warmup_steps,
    #    num_training_steps=total_training_steps,
    #    num_cycles=0.5,  # Default cosine shape
    #)

    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps
    )

    log_path = os.path.join(cfg.output_dir, "training_log.csv")
    checkpoint_path = cfg.output_dir
    best_val_loss = float("inf")

    # ─── Load Checkpoint if Exists ───
    model, start_epoch, global_step, resume_step_within_epoch, lora_config, last_loss = load_checkpoint(
        model=model,
        checkpoint_path=cfg.output_dir,
        device=device
    )

    # Ensure start_step is explicitly defined for clarity
    start_step = global_step

    # Step the scheduler if resuming from checkpoint
    #for _ in range(global_step): # Temporarily removed for phase 2
    #    scheduler.step()

    print(f"🔄 Resuming from Step {start_step}, Epoch {start_epoch}")

    # ─── Training Loop ───
    try:
        num_epochs = cfg.epochs
        global_step = start_step
        epoch = -1
        step = -1
        debug_printed = False

        for epoch in range(start_epoch, num_epochs):
            print(f"\n[🚀] Starting Epoch {epoch + 1}/{num_epochs}")
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)

            for step, batch in enumerate(progress_bar):
                if epoch == start_epoch and resume_step_within_epoch is not None and step < resume_step_within_epoch:
                    continue  # Skip previously completed steps in the current epoch

                batch = {k: v.to(device) for k, v in batch.items()}

                # === 🧪 First Batch Debug Logs ===
                if cfg.debug_mode:
                    print(f"[DEBUG] First Batch Shapes:")
                    print(f" - input_ids: {batch['input_ids'].shape}")
                    print(f" - attention_mask: {batch['attention_mask'].shape}")
                    print(f" - decoder_input_ids: {batch['decoder_input_ids'].shape}")
                    print(f" - decoder_attention_mask: {batch['decoder_attention_mask'].shape}")
                    print(f" - labels: {batch['labels'].shape}")
                # Always check shapes (safety-critical)
                if batch["input_ids"].size(1) != batch["decoder_input_ids"].size(1) + 1:
                    print(
                        f"[WARN] Unusual seq lens: enc={batch['input_ids'].size(1)}, dec={batch['decoder_input_ids'].size(1)}")

                if batch["labels"].size(1) != batch["decoder_input_ids"].size(1):
                    print(
                        f"[WARN] labels/decoder mismatch: labels={batch['labels'].size(1)}, dec={batch['decoder_input_ids'].size(1)}")
                if cfg.debug_mode:
                    if not debug_printed:
                        print("[✓] First Batch Passed All Shape Checks.\n")
                        debug_printed = True

                vocab_size = model.embedding.num_embeddings
                for key in ("input_ids", "decoder_input_ids", "labels"):
                    if key in batch:
                        batch[key] = batch[key].clamp(0, vocab_size - 1)

                optimizer.zero_grad(set_to_none=True)

                try:
                    # forward pass inside autocast (only forward)
                    with torch.autocast("cuda", dtype=amp_dtype):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            decoder_input_ids=batch["decoder_input_ids"],
                            decoder_attention_mask=batch["decoder_attention_mask"],
                            labels=batch["labels"],
                        )

                    # robust loss extraction
                    if isinstance(outputs, dict):
                        loss = outputs.get("loss", None)
                    else:
                        loss = getattr(outputs, "loss", None)

                    # ✅ NaN/inf Loss Protection (always reported)
                    if loss is None or not torch.isfinite(loss):
                        print("🚨 Invalid loss detected (NaN or inf)! Skipping batch...")
                        if cfg.debug_mode:
                            print("Input IDs:", batch["input_ids"][0])
                            print("Labels:", batch["labels"][0])
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    # 🔍 Logit safety check (always raise on NaNs, but sample printed only in debug)
                    if "logits" in outputs and torch.isnan(outputs["logits"]).any():
                        print("❌ NaN detected in logits!")
                        if cfg.debug_mode:
                            print("Logits sample:", outputs["logits"][0][:5])
                        raise ValueError("NaNs found in logits")

                    # 🧪 AMP Debug (gated)
                    if cfg.debug_mode and step == 0:  # first step only to avoid spam
                        if scaler.is_enabled():
                            print(f"[AMP] GradScaler.enabled -> FP16 path (scaler active).")
                        else:
                            print(f"[AMP] GradScaler.disabled -> BF16 path (no scaler).")

                    # -------------------------
                    # FP16 path (GradScaler enabled)
                    # -------------------------
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()

                        if step % 2000 == 0:
                            print(f"\n[📊] VRAM Summary at step {step}:")
                            print(gpu_memory_summary())  # one-line summary
                            if cfg.debug_mode:
                                print(gpu_memory_summary(detailed=True))  # full dump

                        # Unscale before clipping
                        scaler.unscale_(optimizer)

                        # clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        if getattr(cfg, "grad_noise", False):
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad.add_(torch.randn_like(p.grad) * 1e-4)

                        # step + update scaler
                        scaler.step(optimizer)
                        scaler.update()

                    # -------------------------
                    # BF16 path (no scaler)
                    # -------------------------
                    else:
                        loss.backward()

                        if step % 2000 == 0:
                            print(f"\n[📊] VRAM Summary at step {step}:")
                            print(gpu_memory_summary())  # one-line summary
                            if cfg.debug_mode:
                                print(gpu_memory_summary(detailed=True))  # full dump

                        # clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        if getattr(cfg, "grad_noise", False):
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad.add_(torch.randn_like(p.grad) * 1e-4)

                        # normal optimizer step
                        optimizer.step()

                    # scheduler step (keep outside branches unless you require warmup-step-per-optimizer)
                    scheduler.step()

                    # 📉 Track loss
                    running_loss += loss.item()
                    global_step += 1
                    avg_train_loss = running_loss / (step + 1)

                    del loss, outputs
                    clean_gpu_memory()

                    current_lr = optimizer.param_groups[0].get("lr", cfg.learning_rate)
                    progress_bar.set_postfix(
                        train_loss=f"{avg_train_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                        step=global_step
                    )

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("⚠️ OOM caught — flushing VRAM and skipping batch")
                        clean_gpu_memory()
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    else:
                        raise  # re-raise if it's not OOM

            # ─── Validation ───
            avg_val_loss = evaluate_model(
                model, val_loader, tokenizer, device, epoch, avg_train_loss, amp_dtype, cfg
            )

            log_training_stats(
                log_path=log_path,
                step=global_step,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss
            )

            # ─── 🧠 Auto-Save Checkpoint Every N Steps ───
            if global_step % cfg.save_steps == 0 and torch.isfinite(torch.tensor(avg_train_loss)):
                epoch_ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_{global_step}")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    loss=torch.tensor(avg_train_loss) if 'avg_train_loss' in locals() else torch.tensor(0.0),
                    checkpoint_path=epoch_ckpt_path,
                    scaler=scaler,
                    resume_step_within_epoch=step  # 👈 mid-epoch progress
                )

            # ─── Save final checkpoint at the end of epoch ───
            epoch_ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_{global_step}")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                loss=torch.tensor(avg_train_loss),
                checkpoint_path=epoch_ckpt_path,
                scaler=scaler,
                resume_step_within_epoch=0  # 👈 reset within-epoch step at epoch end
            )

    except KeyboardInterrupt:
        handle_training_interrupt(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            lora_config=None,
            resume_step_within_epoch=step,
            loss = torch.tensor(avg_train_loss) if 'avg_train_loss' in locals() else torch.tensor(0.0)
        )

    except Exception as e:
        handle_training_error(
            e,
            model=model,
            optimizer=optimizer,
            start_epoch=epoch,
            start_step=global_step,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            lora_config=None,
            resume_step_within_epoch=step,
            loss = torch.tensor(avg_train_loss) if 'avg_train_loss' in locals() else torch.tensor(0.0)
        )

    finally:
        finalize_training(global_step)

    # ✅ Save Final Model
    save_path = os.path.join(cfg.output_dir, "final_model")
    os.makedirs(save_path, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_path, "model_weights.pt"))

    # Save tokenizer
    tokenizer.save_pretrained(save_path)

    # Save model config (if it exists)
    if hasattr(model, "config"):
        config_dict = model.config.to_dict() if hasattr(model.config, "to_dict") else dict(model.config)
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

    # Save final stats (loss, epoch, etc.)
    final_stats = {
        "final_epoch": epoch if 'epoch' in locals() else -1,
        "final_step": global_step if 'global_step' in locals() else -1,
        "final_train_loss": avg_train_loss if 'avg_train_loss' in locals() else None,
        "final_val_loss": avg_val_loss if 'avg_val_loss' in locals() else None,
        "learning_rate": getattr(cfg, "learning_rate", "unknown")
    }

    with open(os.path.join(save_path, "final_stats.json"), "w") as f:
        json.dump(final_stats, f, indent=4)

    print("✅ Final model, tokenizer, config, and stats saved to:", save_path)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
