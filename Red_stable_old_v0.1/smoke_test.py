# minimal local smoke test
import torch
from model import T5LikeModel

model = T5LikeModel(vocab_size=32000, d_model=256, num_layers=2, num_heads=4, d_ff=512)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

B, T_enc, T_dec = 2, 16, 8
input_ids = torch.randint(0, model.embedding.num_embeddings, (B, T_enc), device=device)
decoder_ids = torch.randint(0, model.embedding.num_embeddings, (B, T_dec), device=device)
attention_mask = torch.ones(B, T_enc, dtype=torch.bool, device=device)
dec_mask = torch.ones(B, T_dec, dtype=torch.bool, device=device)

with torch.no_grad():
    out = model(input_ids=input_ids,
                decoder_input_ids=decoder_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=dec_mask)

print("logits shape:", out["logits"].shape)
assert not torch.isnan(out["logits"]).any()
print("smoke test passed ✅")
