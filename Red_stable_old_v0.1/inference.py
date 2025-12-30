import torch
from transformers import T5Tokenizer
from model import T5LikeModel
import os
import glob


class RedInference:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Initialize with 16 layers/32 heads
        self.model = T5LikeModel(
            vocab_size=32128,
            d_model=512,
            num_layers=16,  # Your new architecture
            num_heads=32,  # Your new architecture
            d_ff=2048
        ).to(self.device)

        self._load_weights(model_path)
        self.model.eval()

    def _load_weights(self, model_path: str):
        """Safe weight loading for mixed architectures"""
        if model_path is None:
            checkpoints = glob.glob("RED_Trained_data/checkpoints/checkpoint_9500.pt")
            model_path = max(checkpoints, key=os.path.getctime)

        print(f"Loading: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Dimension-aware loading
        model_state = self.model.state_dict()
        filtered = {}

        for name, param in model_state.items():
            if name in state_dict:
                # Reshape if needed (for attention heads change 8->32)
                if param.dim() == state_dict[name].dim():
                    if param.shape == state_dict[name].shape:
                        filtered[name] = state_dict[name]
                    elif 'attention' in name and param.size(0) == 32 * 64:  # Example for head reshaping
                        # Repeat existing heads to fill new architecture
                        filtered[name] = self._adapt_heads(state_dict[name], new_heads=32)
                    else:
                        print(f"Skipping {name} (shape mismatch)")
                else:
                    print(f"Skipping {name} (dimension mismatch)")

        self.model.load_state_dict(filtered, strict=False)

    def _adapt_heads(self, tensor, new_heads=32):
        """Helper to adapt attention heads from 8 to 32"""
        old_heads = 8
        if tensor.dim() == 2:
            # Linear layer weights (dim 0 is heads*dim_per_head)
            dim_per_head = tensor.size(0) // old_heads
            return tensor.repeat_interleave(new_heads // old_heads, dim=0)
        return tensor

    def generate_response(self, prompt: str, max_length: int = 100):
        """Robust generation with dimension checks"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding='longest',  # More flexible than max_length
            truncation=True,
            max_length=512
        ).to(self.device)

        # Initialize decoder with proper dimensions [1,1]
        decoder_input = torch.full((1, 1), self.tokenizer.pad_token_id,
                                   dtype=torch.long, device=self.device)

        for _ in range(max_length):
            with torch.no_grad():
                try:
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        decoder_input_ids=decoder_input
                    )
                    logits = outputs["logits"]
                except RuntimeError as e:
                    print(f"Dimension error: {str(e)}")
                    print(f"Shapes - input: {inputs.input_ids.shape} decoder: {decoder_input.shape}")
                    break

            # Ensure we're working with 3D logits [1, seq_len, vocab]
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)

            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            decoder_input = torch.cat([
                decoder_input,
                next_token.unsqueeze(0).unsqueeze(0)  # Keep as [1,1]
            ], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(decoder_input[0], skip_special_tokens=True)


if __name__ == "__main__":
    try:
        red = RedInference()
        print("Test:", red.generate_response("Hello world"))
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Common fixes:")
        print("1. Verify model architecture matches checkpoint")
        print("2. Check tokenizer special tokens")
        print("3. Ensure consistent tensor shapes")