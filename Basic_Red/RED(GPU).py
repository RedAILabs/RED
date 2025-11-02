import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
import os

# Print available GPUs
print("Available GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

secure_model_path = "D:\\RED LLM\\RED AI\\Red_Trained_data\\Red.pth"
os.makedirs(os.path.dirname(secure_model_path), exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_length, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + torch.randn_like(self.embedding(x)) * 0.005
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)  # (batch_size, seq_len, vocab_size)

def pad_sequence(seq, pad_token=0):
    return seq[:20] + [pad_token] * (20 - len(seq))

# Model Parameters
d_model = 256
num_layers = 10
num_heads = 8
d_ff = 512

word_to_index = {
    "hello": 0, "world": 1, "goodbye": 2, "morning": 3, "night": 4,
    "happy": 5, "sad": 6, "fast": 7, "slow": 8, "big": 9, "small": 10,
    "sun": 11, "moon": 12, "star": 13, "ocean": 14, "sky": 15,
    "bright": 16, "dark": 17, "cold": 18, "hot": 19, "wind": 20
}
index_to_word = {v: k for k, v in word_to_index.items()}
vocab_size = len(word_to_index)

# Move model to GPU
model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff).to(device)
optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

training_data = [
    [0, 1, 2, 3, 4, 0],        # "hello world goodbye morning night hello"
    [1, 2, 3, 4, 0, 1],        # "world goodbye morning night hello world"
    [2, 3, 4, 0, 1, 2],        # "goodbye morning night hello world goodbye"
    [3, 4, 0, 1, 2, 3],        # "morning night hello world goodbye morning"
    [4, 0, 1, 2, 3, 4],        # "night hello world goodbye morning night"
    [5, 6, 7, 8, 9, 10],       # "happy sad fast slow big small"
    [10, 9, 8, 7, 6, 5],       # "small big slow fast sad happy"
    [0, 5, 7, 3, 9, 4],        # "hello happy fast morning big night"
    [1, 6, 8, 2, 10, 0],       # "world sad slow goodbye small hello"
    [11, 12, 13, 14, 15, 11],  # "sun moon star ocean sky sun"
    [12, 13, 14, 15, 11, 12],  # "moon star ocean sky sun moon"
    [13, 14, 15, 11, 12, 13],  # "star ocean sky sun moon star"
    [14, 15, 11, 12, 13, 14],  # "ocean sky sun moon star ocean"
    [15, 11, 12, 13, 14, 15],  # "sky sun moon star ocean sky"
    [16, 17, 18, 19, 20, 16],  # "bright dark cold hot wind bright"
    [17, 18, 19, 20, 16, 17],  # "dark cold hot wind bright dark"
    [18, 19, 20, 16, 17, 18],  # "cold hot wind bright dark cold"
    [19, 20, 16, 17, 18, 19],  # "hot wind bright dark cold hot"
    [20, 16, 17, 18, 19, 20],  # "wind bright dark cold hot wind"
]

training_labels = [
    [1, 2, 3, 4, 0, 1],        # Next words after "hello world goodbye morning night hello"
    [2, 3, 4, 0, 1, 2],        # Next words after "world goodbye morning night hello world"
    [3, 4, 0, 1, 2, 3],        # Next words after "goodbye morning night hello world goodbye"
    [4, 0, 1, 2, 3, 4],        # Next words after "morning night hello world goodbye morning"
    [0, 1, 2, 3, 4, 0],        # Next words after "night hello world goodbye morning night"
    [6, 7, 8, 9, 10, 5],       # Next words after "happy sad fast slow big small"
    [9, 8, 7, 6, 5, 10],       # Next words after "small big slow fast sad happy"
    [5, 7, 3, 9, 4, 0],        # Next words after "hello happy fast morning big night"
    [6, 8, 2, 10, 0, 1],       # Next words after "world sad slow goodbye small hello"
    [12, 13, 14, 15, 11, 12],  # Next words after "sun moon star ocean sky sun"
    [13, 14, 15, 11, 12, 13],  # Next words after "moon star ocean sky sun moon"
    [14, 15, 11, 12, 13, 14],  # Next words after "star ocean sky sun moon star"
    [15, 11, 12, 13, 14, 15],  # Next words after "ocean sky sun moon star ocean"
    [11, 12, 13, 14, 15, 11],  # Next words after "sky sun moon star ocean sky"
    [17, 18, 19, 20, 16, 17],  # Next words after "bright dark cold hot wind bright"
    [18, 19, 20, 16, 17, 18],  # Next words after "dark cold hot wind bright dark"
    [19, 20, 16, 17, 18, 19],  # Next words after "cold hot wind bright dark cold"
    [20, 16, 17, 18, 19, 20],  # Next words after "hot wind bright dark cold hot"
    [16, 17, 18, 19, 20, 16],  # Next words after "wind bright dark cold hot wind"
]

training_data = [pad_sequence(seq) for seq in training_data]
training_labels = [pad_sequence(seq) for seq in training_labels]

if os.path.exists(secure_model_path):
    model.load_state_dict(torch.load(secure_model_path, map_location=device))
    print("Red loaded for inference!")

else:
    print("No pre-trained model found. Training now...")
    epochs = 2000
    for epoch in range(epochs):
        inputs = torch.tensor(training_data, dtype=torch.long).to(device)
        targets = torch.tensor(training_labels, dtype=torch.long).to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save Model
    torch.save(model.state_dict(), secure_model_path)
    print(f"Red trained and saved successfully at {secure_model_path}!")

# Inference
with torch.no_grad():
    test_input = torch.tensor([[18, 2, 18, 10, 19]], dtype=torch.long).to(device)
    test_output = model(test_input)
    predicted_index = torch.argmax(test_output[:, -1, :].detach(), dim=-1).item()
    print("Predicted next word:", index_to_word.get(predicted_index, "UNKNOWN"))
