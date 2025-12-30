import os
import json
import torch
import warnings
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from config import Config

cfg = Config("D:/RED LLM/RED AI/RED/config/train_config.json")

# --- Checkpoint Saving & Loading ---

def transform_state_dict_keys(state_dict, mode='load', strict=True):
    """
    Add/remove common prefixes when saving/loading checkpoints.
    Handles DDP/module wrappers, base_model prefixes, etc.

    Args:
        state_dict: checkpoint state_dict
        mode: 'load' or 'save'
        strict: if False, mismatched keys are skipped instead of failing
    """
    new_dict = {}
    prefix_map = [
        ("base_model.model.", ""),  # our custom prefix
        ("module.", ""),  # DDP-style
        ("model.", "")  # generic wrapper
    ]

    for key, val in state_dict.items():
        try:
            new_key = key
            if mode == "load":
                for prefix, repl in prefix_map:
                    if new_key.startswith(prefix):
                        new_key = new_key.replace(prefix, repl, 1)
            elif mode == "save":
                # Always enforce our preferred prefix
                if not new_key.startswith("base_model.model."):
                    new_key = f"base_model.model.{new_key}"

            new_dict[new_key] = val

        except Exception as e:
            if strict:
                raise
            else:
                warnings.warn(f"[transform_state_dict_keys] Skipped {key}: {e}")
                continue

    return new_dict

# ---------- SAVE ----------

def save_checkpoint(
    model,
    optimizer,
    epoch,
    step,
    loss,
    checkpoint_path,
    resume_step_within_epoch=0,
    save_optimizer=True,
    **kwargs,
):
    """
    Save model (and optionally optimizer) with strong NaN/Inf barriers and dual-file redundancy.
    Returns True/False for success.
    """
    try:
        # ===== NaN & Inf Protection =====
        if loss is None or (torch.is_tensor(loss) and not torch.isfinite(loss)) or (not torch.is_tensor(loss) and not torch.isfinite(torch.tensor(loss))):
            print(f"[❌] Checkpoint NOT saved — loss is invalid: {loss}")
            return False

        for name, param in model.named_parameters():
            # Skip tensors that require grad None or are empty
            if param is None or param.data.numel() == 0:
                continue
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                print(f"[❌] Checkpoint NOT saved — NaN/Inf in param: {name}")
                return False

        # ===== Construct file paths (accept file or dir) =====
        if checkpoint_path.endswith(".pt"):
            dir_path = os.path.dirname(checkpoint_path) or "."
            base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        else:
            dir_path = checkpoint_path
            base_name = f"checkpoint_{step}"
            checkpoint_path = os.path.join(dir_path, f"{base_name}.pt")

        os.makedirs(dir_path, exist_ok=True)
        safe_path = checkpoint_path.replace(".pt", "_safe.pt")

        # ===== Safe loss scalar =====
        try:
            loss_val = float(loss.item()) if torch.is_tensor(loss) else float(loss)
        except Exception:
            loss_val = None  # do not crash saving

        checkpoint = {
            "model_state_dict": transform_state_dict_keys(model.state_dict(), mode="save"),
            "epoch": int(epoch),
            "step": int(step),
            "loss": loss_val,
            "resume_step_within_epoch": int(resume_step_within_epoch),
            "model_signature": {
                "num_params": sum(p.numel() for p in model.parameters()),
                "architecture": model.__class__.__name__,
                "config": getattr(model, "config", {}),
            },
        }
        if save_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # ===== Dual save (standard + _safe) =====
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, safe_path, _use_new_zipfile_serialization=True)

        print(f"[✓] Saved checkpoint at {checkpoint_path} (and {os.path.basename(safe_path)})")
        return True

    except Exception as e:
        print(f"[X] Failed to save checkpoint: {e}")
        return False


# ---------- LOAD (directory OR single file) ----------

def load_checkpoint(model, checkpoint_path, device="cpu", load_optimizer=True, strict=False):
    """
    Load latest available checkpoint.
    - If `checkpoint_path` is a file: load that file (fallback to *_safe.pt).
    - If `checkpoint_path` is a dir: find latest `checkpoint_*.pt`, try that then its *_safe.pt, then next latest...
    Returns: (model, epoch, global_step, resume_step_within_epoch, optimizer_state_dict_or_None, last_loss)
             Always 6 values.
    """
    # Case A: direct file load
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pt"):
        return _try_load_file_then_safe(model, checkpoint_path, device, load_optimizer, strict)

    # Case B: directory search for latest
    if not os.path.isdir(checkpoint_path):
        print(f"[!] No checkpoint directory found: {checkpoint_path}")
        return model, 0, 0, None, None, None

    candidates = []
    for fn in os.listdir(checkpoint_path):
        if fn.startswith("checkpoint_") and fn.endswith(".pt") and not fn.endswith("_safe.pt"):
            try:
                step = int(fn.split("_")[1].split(".")[0])
                if step > 0:
                    candidates.append((step, fn))
            except ValueError:
                continue

    if not candidates:
        print("[!] No valid checkpoint files found.")
        return model, 0, 0, None, None, None

    candidates.sort(key=lambda x: x[0], reverse=True)

    for step, fn in candidates:
        main_path = os.path.join(checkpoint_path, fn)
        res = _try_load_file_then_safe(model, main_path, device, load_optimizer, strict, known_step=step)
        if res is not None:
            return res  # success

    print("[!] All checkpoint load attempts failed.")
    return model, 0, 0, None, None, None


def _try_load_file_then_safe(model, main_path, device, load_optimizer, strict, known_step=None):
    """Try loading main .pt, then its *_safe.pt counterpart."""
    base, ext = os.path.splitext(main_path)
    safe_path = f"{base}_safe{ext}"

    # Try main
    try:
        print(f"[*] Trying to load checkpoint: {os.path.basename(main_path)}")
        checkpoint = torch.load(main_path, map_location=device, weights_only=False)
        return _load_checkpoint_contents(model, checkpoint, load_optimizer, strict, fallback_name=os.path.basename(main_path), known_step=known_step)
    except Exception as e:
        print(f"[!] Error loading {os.path.basename(main_path)}: {str(e)[:200]}")

    # Try safe
    if os.path.exists(safe_path):
        try:
            print(f"[*] Trying safe checkpoint: {os.path.basename(safe_path)}")
            checkpoint = torch.load(safe_path, map_location=device, weights_only=False)
            return _load_checkpoint_contents(model, checkpoint, load_optimizer, strict, fallback_name=os.path.basename(safe_path), known_step=known_step)
        except Exception as e:
            print(f"[!] Error loading {os.path.basename(safe_path)}: {str(e)[:200]}")

    return None


def _load_checkpoint_contents(model, checkpoint, load_optimizer, strict, fallback_name="", known_step=None):
    """
    Internal: load model state (with key transform + strict control),
    return standardized tuple.
    """
    # ---- Model state ----
    model_state = checkpoint.get("model_state_dict", None)
    if model_state is None:
        print(f"[!] No model_state_dict in checkpoint {fallback_name}")
        return model, 0, 0, None, None, None

    model_state = transform_state_dict_keys(model_state, mode="load")

    try:
        load_res = model.load_state_dict(model_state, strict=strict)
        # load_res can be a NamedTuple in newer PyTorch; handle both
        missing = getattr(load_res, "missing_keys", []) if not isinstance(load_res, tuple) else load_res[0]
        unexpected = getattr(load_res, "unexpected_keys", []) if not isinstance(load_res, tuple) else load_res[1]
    except RuntimeError as e:
        if strict:
            print(f"[!] Strict load failed: {e}. Retrying with strict=False.")
            load_res = model.load_state_dict(model_state, strict=False)
            missing = getattr(load_res, "missing_keys", []) if not isinstance(load_res, tuple) else load_res[0]
            unexpected = getattr(load_res, "unexpected_keys", []) if not isinstance(load_res, tuple) else load_res[1]
        else:
            print(f"[!] Load failed: {e}")
            return model, 0, 0, None, None, None

    if missing:
        warnings.warn(f"[⚠️] Missing keys when loading checkpoint: {missing[:10]}{' ...' if len(missing)>10 else ''}")
    if unexpected:
        warnings.warn(f"[⚠️] Unexpected keys in checkpoint: {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")

    # ---- Metadata ----
    epoch = int(checkpoint.get("epoch", 0) or 0)
    step = int(checkpoint.get("step", known_step or 0) or 0)
    resume_step_within_epoch = checkpoint.get("resume_step_within_epoch", None)
    loss = checkpoint.get("loss", None)

    # ---- Optimizer state ----
    opt_state = checkpoint.get("optimizer_state_dict", None) if load_optimizer else None

    print(f"[📂] Loaded checkpoint {fallback_name or ''} (epoch {epoch}, step {step})")
    return model, epoch, step, resume_step_within_epoch, opt_state, loss

# --- Dataset Preprocessing & Loading ---

def extract_instruction_pairs(tree):
    """Extract prompt-response pairs from a conversation tree."""
    pairs = []
    stack = [(tree, None)]

    while stack:
        node, prom_text = stack.pop()
        role = node.get("role")
        text = node.get("text", "").strip()

        if role == "assistant" and prom_text:
            pairs.append({
                "instruction": prom_text,
                "context": "",
                "response": text
            })

        for child in reversed(node.get("children", [])):
            stack.append((child, text if role == "prompter" else prom_text))

    return pairs

def prepare_dataloaders(tokenizer: PreTrainedTokenizer, batch_size=8,
                        cache_path="RED_Trained_data/preprocessed_instruction_dataset"):
    """Prepare training/validation dataloaders from instruction dataset."""
    try:
        if os.path.isdir(cache_path) and os.listdir(cache_path):
            print("🔁 Loading cached dataset...")
            split_dataset = load_from_disk(cache_path)
        else:
            print("🛠️ Building dataset from scratch...")
            oasst_path = "D:/Red_LLM/openassistant/oasst1_full.json"
            with open(oasst_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            trees = raw.get("conversation_trees", [])
            all_examples = []
            for tree in trees:
                all_examples.extend(extract_instruction_pairs(tree))

            combined_dataset = Dataset.from_list(all_examples)

            def preprocess(batch):
                input_ids, attention_mask = [], []
                decoder_input_ids, decoder_attention_mask, labels = [], [], []

                for instr, ctx, resp in zip(batch["instruction"], batch["context"], batch["response"]):
                    inp = instr.strip() + (f" {ctx.strip()}" if ctx else "")
                    tgt = resp.strip()
                    if not inp or not tgt:
                        continue

                    enc_in = tokenizer(inp, padding="max_length", truncation=True,
                                       max_length=512, return_tensors="pt")
                    if cfg.debug_mode:
                        print(f"[DEBUG] Input: {inp[:50]}...")  # Print first 50 chars of input
                        print(f"[DEBUG] Token count: {enc_in.input_ids.size(1)}")

                    enc_tg = tokenizer(tgt, padding="max_length", truncation=True,
                                       max_length=512, return_tensors="pt")
                    if cfg.debug_mode:
                        print(f"[DEBUG] Target: {tgt[:50]}...")
                        print(f"[DEBUG] Target token count: {enc_tg.input_ids.size(1)}")

                    # Get target tokens and attention mask
                    target_ids = enc_tg.input_ids.squeeze(0)
                    target_mask = enc_tg.attention_mask.squeeze(0).bool()

                    # Create labels with padding masked (set to -100)
                    # Shift left: predict next token at each position
                    label_ids = target_ids[1:].clone()
                    label_mask = target_mask[1:]  # Mask for next tokens

                    # Set padding positions to -100 (ignored in loss)
                    label_ids[~label_mask] = -100

                    input_ids.append(enc_in.input_ids.squeeze())
                    attention_mask.append(enc_in.attention_mask.squeeze())

                    # Decoder input is target without last token
                    decoder_input_ids.append(target_ids[:-1])
                    decoder_attention_mask.append(enc_tg.attention_mask.squeeze()[:-1])
                    labels.append(label_ids)  # Use masked labels

                if not input_ids:
                    return {
                        "input_ids": [], "attention_mask": [],
                        "decoder_input_ids": [], "labels": [], "decoder_attention_mask": []
                    }

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "labels": labels,
                    "decoder_attention_mask": decoder_attention_mask,
                }

            processed = combined_dataset.map(
                preprocess,
                batched=True,
                remove_columns=combined_dataset.column_names,
                desc="Tokenizing dataset"
            )

            split_dataset = processed.train_test_split(test_size=0.1)
            split_dataset.save_to_disk(cache_path)

        for split in ["train", "test"]:
            split_dataset[split].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "decoder_input_ids", "labels", "decoder_attention_mask"]
            )

        train_loader = DataLoader(
            split_dataset["train"], batch_size=batch_size,
            shuffle=True, pin_memory=True, num_workers=0,
            persistent_workers=False, prefetch_factor=None
        )

        val_loader = DataLoader(
            split_dataset["test"], batch_size=batch_size,
            pin_memory=True, num_workers=0
        )

        return train_loader, val_loader

    except Exception as e:
        print(f"[❌] Dataset preparation failed: {e}")
        raise
