# smoke_train.py - Hybrid test for VRAM + checkpoints
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from model import T5LikeModel
from utils import clean_gpu_memory
from checkpoint import save_checkpoint, load_checkpoint

# ===== Config =====
B, T_enc, T_dec = 2, 32, 16
steps = 20
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dummy dataset =====
input_ids = torch.randint(0, 32000, (100, T_enc))
decoder_ids = torch.randint(0, 32000, (100, T_dec))
labels = torch.randint(0, 32000, (100, T_dec))
dataset = TensorDataset(input_ids, decoder_ids, labels)
loader = DataLoader(dataset, batch_size=B)

# ===== Model =====
model = T5LikeModel(vocab_size=32000, d_model=256, num_layers=2, num_heads=8, d_ff=512)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler(device="cuda", enabled=(amp_dtype == torch.float16))

# ===== Training loop (tiny smoke run) =====
model.train()
for step, (x, y, z) in enumerate(loader):
    if step >= steps:
        break
    x, y, z = x.to(device), y.to(device), z.to(device)
    optimizer.zero_grad(set_to_none=True)

    with autocast(device_type="cuda", dtype=amp_dtype):
        out = model(input_ids=x, decoder_input_ids=y, labels=z)
        loss = out["loss"]

    if amp_dtype == torch.float16:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    if step % 5 == 0:
        print(f"Step {step}: loss={loss.item():.4f}, dtype={amp_dtype}")
        print(torch.cuda.memory_summary())

print("✅ Smoke training loop passed")

# ===== Checkpoint Save/Load Test =====
ok = save_checkpoint(model, optimizer, epoch=0, step=step, loss=loss, checkpoint_path="./checkpoints")
if ok:
    model, epoch, step, *_ , success = load_checkpoint(model, "./checkpoints", device=device)
    if success:
        print("✅ Checkpoint reload passed")
    else:
        print("❌ Checkpoint reload failed")
else:
    print("❌ Checkpoint save failed, skipping reload")

# ===== Clean up =====
clean_gpu_memory()
