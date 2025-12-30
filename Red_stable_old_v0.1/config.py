import json

class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            d = json.load(f)

        # Assign all fields with defaults
        self.model_path = d.get("model_path", "t5-small")
        self.output_dir = d.get("output_dir", "RED_Trained_data/logs")
        self.epochs = d.get("epochs", 8)
        self.batch_size = d.get("batch_size", 8)
        self.learning_rate = d.get("learning_rate", 1e-4)
        self.seed = d.get("seed", 42)
        self.eval_steps = d.get("eval_steps", 500)
        self.save_steps = d.get("save_steps", 1000)
        self.logging_steps = d.get("logging_steps", 100)
        self.gradient_checkpointing = d.get("gradient_checkpointing", False)
        self.use_torch_compile = d.get("use_torch_compile", False)
        self.debug_mode = d.get("debug_mode", True)
        self.use_fp32_last_layer = d.get("use_fp32_last_layer", True)
        self.optimizer = d.get("optimizer", "muon")    #options: "muon", "adamw8bit", "adafactor"
        self.weight_decay = d.get("weight_decay", 0.01)
        self.betas = tuple(d.get("betas", [0.9, 0.999]))   #used only if optimizer=adamw8bit
        self.eps = d.get("eps", 1e-8)  #used only if optimizer=adamw8bit
        self.muon_momentum = d.get("muon_momentum", 0.95)  #used only if optimizer=muon
        self.grad_noise = d.get("grad_noise", True)
        # Dataset related
        self.cache_dataset = d.get("cache_dataset", "RED_Trained_data/preprocessed_instruction_dataset")
        self.train_dataset_path = d.get("train_dataset_path", "D:/Red_LLM/openassistant/oasst1_full.json")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)
