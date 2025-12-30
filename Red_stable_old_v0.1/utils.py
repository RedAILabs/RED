import os
import csv
import torch
from torch import nn
from checkpoint import save_checkpoint
import gc
from torch.cuda.amp import autocast
from config import Config
cfg = Config("D:/RED LLM/RED AI/RED/config/train_config.json")
print("[*] Training Configuration Loaded:")

# ─── Logging ────────────────────────────────────────────────────────────────

def log_training_stats(log_path, step, epoch, train_loss, val_loss=None):
    """Log training and validation losses to CSV with header and safe formatting."""
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Ensure values are native floats
        train_loss = train_loss.item() if torch.is_tensor(train_loss) else float(train_loss)
        val_loss_str = ""
        if val_loss is not None:
            val_loss = val_loss.item() if torch.is_tensor(val_loss) else float(val_loss)
            val_loss_str = f"{val_loss:.4f}"

        # Check if header needs to be written
        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

        with open(log_path, "a", newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Step", "Epoch", "TrainLoss", "ValLoss"])
            writer.writerow([step, epoch, f"{train_loss:.4f}", val_loss_str])

        # Console log
        msg = f"[LOG] Epoch {epoch + 1} | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        print(msg)

    except Exception as e:
        print(f"[!] Failed to log training stats: {e}")

def log_interrupted_state(log_path, step, epoch, loss):
    """Log interrupted training state to CSV."""
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, loss.item() if torch.is_tensor(loss) else loss, "INTERRUPTED"])
        print(f"[LOG] Training interrupted at Epoch {epoch + 1}, Step {step}")
    except Exception as e:
        print(f"[!] Failed to log interrupted state: {e}")

# ─── Error and Interrupt Handling ───────────────────────────────────────────

def handle_training_interrupt(
    model,
    optimizer,
    epoch,
    global_step,
    checkpoint_path,
    log_path,
    lora_config=None,
    resume_step_within_epoch=None,
    loss=None
):
    """Handle manual training interrupt (e.g., Ctrl+C)."""
    print("\n[⚠️] Training interrupted. Attempting to save checkpoint...")
    try:
        fallback_loss = torch.tensor(0.0)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            loss=loss if loss is not None else fallback_loss,
            checkpoint_path=checkpoint_path,
            lora_config=lora_config,
            resume_step_within_epoch=resume_step_within_epoch
        )
        log_interrupted_state(
            log_path=log_path,
            step=global_step,
            epoch=epoch,
            loss=loss if loss is not None else fallback_loss
        )
    except Exception as e:
        print(f"[!] Error while saving interrupted checkpoint: {e}")


def handle_training_error(
    e,
    model,
    optimizer,
    start_epoch,
    start_step,
    checkpoint_path,
    log_path,
    lora_config=None,
    resume_step_within_epoch=None,
    loss=None
):
    """Handle unexpected training crash."""
    print(f"\n[❌] Training error encountered: {e}")
    try:
        fallback_loss = torch.tensor(0.0)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=start_epoch,
            step=start_step,
            loss=loss if loss is not None else fallback_loss,
            checkpoint_path=checkpoint_path,
            lora_config=lora_config,
            resume_step_within_epoch=resume_step_within_epoch
        )
        log_interrupted_state(
            log_path=log_path,
            step=start_step,
            epoch=start_epoch,
            loss=loss if loss is not None else fallback_loss
        )
    except Exception as save_error:
        print(f"[!] Failed to save error checkpoint: {save_error}")
    finally:
        print("[✋] Training aborted due to critical error.")

# ─── Readiness and Finalization ─────────────────────────────────────────────

def verify_training_readiness(model):
    """Verify model and device are ready for training."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mem = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            )
            print(f"\n[CUDA] Available VRAM: {free_mem / 1e9:.2f} GB")
            if free_mem < 2e9:
                raise RuntimeError("Insufficient VRAM for training")

        model.train()
        print("✅ Model is now in training mode.")

        if sum(p.numel() for p in model.parameters() if p.requires_grad) == 0:
            raise RuntimeError("No trainable parameters found.")

    except Exception as e:
        print(f"\n[!] Training readiness check failed: {e}")
        raise

def evaluate_model(model, val_loader, tokenizer, device, epoch, avg_train_loss, amp_dtype, cfg):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            vocab_size = model.embedding.num_embeddings
            for key in ("input_ids", "decoder_input_ids", "labels"):
                if key in batch:
                    batch[key] = batch[key].clamp(0, vocab_size - 1)

            # ✅ AMP context uses the same dtype as training
            with autocast(dtype=amp_dtype):
                outputs = model(**batch)
                val_loss_tensor = outputs.get("loss")
                if val_loss_tensor is None:
                    raise ValueError("Validation loss is None — ensure labels are passed.")
                val_loss += val_loss_tensor.item()

            if step == 0 and cfg.debug_mode:  # only print once and under debug mode
                input_sample = batch["input_ids"][0].unsqueeze(0)
                generated_ids = model.generate(input_sample, max_length=128)

                decoded_input = tokenizer.decode(input_sample[0], skip_special_tokens=True)
                decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                decoded_target = tokenizer.decode(batch["labels"][0], skip_special_tokens=True)

                print("🔍 Sample Input:", decoded_input)
                print("🔮 Predicted Output:", decoded_output)
                print("🎯 Target Response:", decoded_target)

    avg_val_loss = val_loss / len(val_loader)
    print(f"📈 Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")


    print(f"[📈] End of Epoch {epoch}:")
    print(torch.cuda.memory_summary())

    clean_gpu_memory()
    return avg_val_loss

def finalize_training(global_step: int):
    """Final cleanup and stats after training finishes."""
    print("\n🎯 Training completed successfully!")

    if torch.cuda.is_available():
        print("\n[CUDA] Final VRAM summary:")
        # one-line summary
        print(gpu_memory_summary())
        # detailed summary
        if cfg.debug_mode:
            print(gpu_memory_summary(detailed=True))
    else:
        print("[CPU-ONLY] Training done (no CUDA available).")

    print(f"[STATS] Total training steps completed: {global_step}")
    clean_gpu_memory()

def clean_gpu_memory():
    """
    Force garbage collection + clear CUDA caches + reset memory stats.
    Safe to call anytime to reduce fragmentation.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        print("[🧹] Cleared CUDA cache, reset memory stats, and ran garbage collector.")
    else:
        print("[🧹] CUDA not available, only ran garbage collector.")

def gpu_memory_summary(device: int = 0, detailed: bool = False) -> str:
    """
    Return a human-readable summary of GPU memory usage.
    - allocated: actively used by tensors
    - reserved: held by allocator (may include fragmentation)
    - peak: max allocated since reset
    - detailed: also include torch.cuda.memory_summary()
    """
    if not torch.cuda.is_available():
        return "CUDA not available."

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    summary = (f"[📊 GPU {device}] "
               f"Allocated: {allocated:.2f} MB | "
               f"Reserved: {reserved:.2f} MB | "
               f"Peak: {peak:.2f} MB")

    if detailed:
        try:
            summary += "\n\n" + torch.cuda.memory_summary(device=device, abbreviated=False)
        except Exception:
            summary += "\n(torch.cuda.memory_summary() unavailable)"

    return summary

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model Parameters Summary:")
    print(f" - Total Parameters:     {total_params:,}")
    print(f" - Trainable Parameters: {trainable_params:,}")
    print(f" - Non-trainable Params: {total_params - trainable_params:,}")

def make_param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = n.endswith(".bias")
        is_norm = any(x in n.lower() for x in ["norm", "layernorm", "ln"])
        is_embed = "embedding" in n.lower()
        if is_bias or is_norm or is_embed:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def build_optimizer(model, cfg):
    """
    Build optimizer with sane param groups:
      - no weight decay for bias, norm stats, embeddings
      - per-group weight_decay so it works across AdamW8bit / Muon / Adafactor
    """

    # -------- param grouping (decoupled WD like AdamW style) --------
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if not param.requires_grad:
                continue
            if param_name.endswith("bias"):
                no_decay.add(full_name)
            elif param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                decay.add(full_name)
            elif param_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                no_decay.add(full_name)
            else:
                # fallback: if 1D parameter (e.g., norm scales), do no_decay
                (no_decay if param.ndim == 1 else decay).add(full_name)

    # sanity: every param should be in exactly one set
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    inter = decay & no_decay
    union = decay | no_decay
    assert len(inter) == 0, f"parameters {str(inter)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union) == 0, \
        f"parameters {str(param_dict.keys() - union)} were not separated into decay/no_decay!"

    wd = getattr(cfg, "weight_decay", 0.01)
    lr = getattr(cfg, "learning_rate", 1e-4)

    decay_names = sorted(list(decay))
    no_decay_names = sorted(list(no_decay))

    param_groups = [
        {"params": [param_dict[n] for n in decay_names if n in param_dict], "weight_decay": wd},
        {"params": [param_dict[n] for n in no_decay_names if n in param_dict], "weight_decay": 0.0},
    ]

    # ---------- build optimizer ----------
    try:
        opt_name = cfg.optimizer.lower()

        if opt_name == "muon":
            from muon import Muon
            # Muon doesn’t consume per-group weight_decay. Flatten groups.
            muon_params = [p for g in param_groups for p in g["params"]]
            optimizer = Muon(
                muon_params,
                lr=lr,
                momentum=getattr(cfg, "muon_momentum", 0.95),
            )
            if cfg.debug_mode:
                print(f"[🔧] Using Muon → lr={lr}, momentum={getattr(cfg, 'muon_momentum', 0.95)} "
                      f"(decoupled weight decay NOT applied by optimizer)")

        elif opt_name == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit
            betas = tuple(cfg.betas) if isinstance(cfg.betas, (list, tuple)) else (0.9, 0.999)
            # Per-group WD is set in param_groups; no global weight_decay arg needed.
            optimizer = AdamW8bit(
                param_groups,
                lr=lr,
                betas=betas,
                eps=getattr(cfg, "eps", 1e-8),
            )
            if cfg.debug_mode:
                print(f"[🔧] Using AdamW8bit → lr={lr}, betas={betas}, eps={getattr(cfg, 'eps', 1e-8)}, "
                      f"wd(decay group)={wd}")

        elif opt_name == "adafactor":
            from transformers import Adafactor
            optimizer = Adafactor(
                param_groups,
                lr=lr,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False,
            )
            if cfg.debug_mode:
                print(f"[🔧] Using Adafactor → lr={lr}, wd(decay group)={wd}")

        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    except Exception as e:
        print(f"[⚠️] Optimizer {cfg.optimizer} failed: {e}. Falling back to Adafactor.")
        from transformers import Adafactor
        optimizer = Adafactor(
            param_groups,
            lr=lr,
            scale_parameter=True,
            relative_step=False,
            warmup_init=False,
        )
        if cfg.debug_mode:
            print(f"[🔧] Fallback → Adafactor optimizer with lr={lr}, wd(decay group)={wd}")

    if cfg.debug_mode:
        num_decay_tensors   = len(param_groups[0]["params"])
        num_nodecay_tensors = len(param_groups[1]["params"])
        num_decay_params    = sum(p.numel() for p in param_groups[0]["params"])
        num_nodecay_params  = sum(p.numel() for p in param_groups[1]["params"])
        print(f"[ℹ️] Param groups → decay: {num_decay_tensors} tensors / {num_decay_params:,} params | "
              f"no_decay: {num_nodecay_tensors} tensors / {num_nodecay_params:,} params")

    return optimizer
