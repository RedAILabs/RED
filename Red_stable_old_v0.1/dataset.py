import os
import json
import torch
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

# --- Load OpenAssistant-style dataset ---

def load_manual_dataset(file_path: str) -> Dataset:
    """Load full OpenAssistant dataset into HuggingFace Dataset."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            full_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in {file_path}")

    if "conversation_trees" not in full_data:
        raise ValueError("Missing 'conversation_trees' in dataset.")

    for tree in full_data["conversation_trees"]:
        current_node = tree
        conversation_history = []

        try:
            while current_node:
                msg = {
                    "role": current_node.get("role", ""),
                    "text": current_node.get("text", "").strip(),
                    "message_id": current_node.get("message_id", "")
                }
                conversation_history.append(msg)

                if len(conversation_history) >= 2:
                    prev_msg = conversation_history[-2]
                    curr_msg = conversation_history[-1]
                    if prev_msg["role"] == "prompter" and curr_msg["role"] == "assistant":
                        data.append({
                            "instruction": prev_msg["text"],
                            "context": "\n".join([m["text"] for m in conversation_history[:-2]]),
                            "response": curr_msg["text"]
                        })

                current_node = current_node["children"][0] if current_node.get("children") else None

        except Exception as e:
            print(f"⚠️ Skipped broken tree {tree.get('message_id')}: {e}")

    if not data:
        raise ValueError("No valid conversations extracted.")

    print(f"[✓] Extracted {len(data)} examples from {file_path}")
    return Dataset.from_list(data)

# --- Extract instruction-response pairs ---

def extract_instruction_pairs(tree: dict) -> list:
    """Extract simple prompt-response pairs from a conversation tree."""
    pairs = []
    stack = [(tree, None)]  # (current_node, last_prompter_text)

    while stack:
        node, last_prompt = stack.pop()
        role = node.get("role")
        text = node.get("text", "").strip()

        if role == "assistant" and last_prompt:
            pairs.append({
                "instruction": last_prompt,
                "context": "",
                "response": text
            })

        for child in reversed(node.get("children", [])):
            stack.append((child, text if role == "prompter" else last_prompt))

    return pairs

# --- Prepare dataloaders (train/test splits) ---
def prepare_dataloaders(tokenizer: PreTrainedTokenizer, batch_size=8,
                        cache_path="RED_Trained_data/preprocessed_instruction_dataset",
                        dataset_path="D:/Red_LLM/openassistant/oasst1_full.json"):
    """Load or preprocess dataset, return PyTorch DataLoaders."""
    try:
        if os.path.isdir(cache_path) and os.listdir(cache_path):
            print("🔁 Loading cached dataset from disk...")
            split_dataset = load_from_disk(cache_path)
        else:
            print("🛠️ Preprocessing dataset from scratch...")
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            trees = raw.get("conversation_trees", [])
            print(f"🌳 Found {len(trees)} trees.")

            all_examples = []
            for tree in trees:
                all_examples.extend(extract_instruction_pairs(tree))

            if not all_examples:
                raise ValueError("No instruction–response pairs extracted.")

            combined_dataset = Dataset.from_list(all_examples)

            def preprocess(batch):
                input_ids, attention_mask = [], []
                decoder_input_ids, decoder_attention_mask, labels = [], [], []

                for instr, ctx, resp in zip(batch["instruction"], batch["context"], batch["response"]):
                    inp = instr.strip() + (f" {ctx.strip()}" if ctx else "")
                    tgt = resp.strip()

                    if not inp or not tgt:
                        continue  # 🛡️ Skip empty samples

                    enc_in = tokenizer(inp, padding="max_length", truncation=True,
                                       max_length=512, return_tensors="pt")
                    enc_tg = tokenizer(tgt, padding="max_length", truncation=True,
                                       max_length=512, return_tensors="pt")

                    src_ids = enc_in.input_ids.squeeze(0)
                    src_mask = enc_in.attention_mask.squeeze(0)

                    tgt_ids = enc_tg.input_ids.squeeze(0)
                    tgt_mask = enc_tg.attention_mask.squeeze(0)

                    # 🧠 Shift decoder inputs and labels
                    decoder_input = tgt_ids[:-1]
                    label = tgt_ids[1:]

                    # Skip bad sequences (e.g., entirely padding or all -100)
                    if decoder_input.sum() == 0 or label.sum() == 0:
                        continue

                    # Clamp to avoid invalid token IDs
                    decoder_input = decoder_input.clamp(0, tokenizer.vocab_size - 1)
                    label = label.clamp(0, tokenizer.vocab_size - 1)

                    # Pad to 511 for decoder (max_length - 1)
                    pad_len = 511 - decoder_input.size(0)
                    if pad_len > 0:
                        decoder_input = torch.cat(
                            [decoder_input, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
                        label = torch.cat([label, torch.full((pad_len,), -100, dtype=torch.long)])

                    decoder_mask = tgt_mask[:-1]
                    if decoder_mask.size(0) < 511:
                        decoder_mask = torch.cat(
                            [decoder_mask, torch.zeros(511 - decoder_mask.size(0), dtype=torch.long)])

                    input_ids.append(src_ids)
                    attention_mask.append(src_mask)
                    decoder_input_ids.append(decoder_input)
                    decoder_attention_mask.append(decoder_mask)
                    labels.append(label)

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
                desc="Tokenizing dataset..."
            )

            split_dataset = processed.train_test_split(test_size=0.1)
            split_dataset.save_to_disk(cache_path)
            print("💾 Saved preprocessed dataset.")

        for split in ["train", "test"]:
            split_dataset[split].set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "decoder_input_ids", "labels", "decoder_attention_mask"]
            )

        train_loader = DataLoader(
            split_dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False
        )

        val_loader = DataLoader(
            split_dataset["test"],
            batch_size=batch_size,
            pin_memory=True,
            num_workers=0
        )

        print("[✓] DataLoaders prepared.")
        return train_loader, val_loader

    except Exception as e:
        print(f"[❌] Dataset preparation failed: {e}")
        raise

def prepare_streaming_dataloader(tokenizer, batch_size=8):
    print("🚀 Loading streaming dataset (OpenAssistant)...")

    # Load streaming dataset
    dataset = load_dataset("openassistant/oasst1", split="train", streaming=True)

    def tokenize_example(example):
        instr = example.get("instruction", "")
        ctx = example.get("context", "")
        resp = example.get("response", "")

        if not instr or not resp:
            return None  # Skip

        inp = instr.strip() + (f" {ctx.strip()}" if ctx else "")
        tgt = resp.strip()

        enc_in = tokenizer(inp, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        enc_tg = tokenizer(tgt, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

        decoder_input = enc_tg.input_ids.squeeze()[:-1]
        label = enc_tg.input_ids.squeeze()[1:]

        if decoder_input.size(0) < 511:
            pad_len = 511 - decoder_input.size(0)
            decoder_input = torch.cat([decoder_input, torch.full((pad_len,), tokenizer.pad_token_id)])
            label = torch.cat([label, torch.full((pad_len,), -100)])

        return {
            "input_ids": enc_in.input_ids.squeeze(),
            "attention_mask": enc_in.attention_mask.squeeze(),
            "decoder_input_ids": decoder_input,
            "decoder_attention_mask": enc_tg.attention_mask.squeeze()[:-1],
            "labels": label
        }

    class OpenAssistantStreamDataset(IterableDataset):
        def __iter__(self):
            for example in dataset:
                tokenized = tokenize_example(example)
                if tokenized:
                    yield tokenized

    dataloader = DataLoader(
        OpenAssistantStreamDataset(),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    print("[✓] Streaming DataLoader ready.")
    return dataloader
