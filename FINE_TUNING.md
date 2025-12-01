# Fine-Tuning Guide for AgentFoundry

This guide explains how to fine-tune local models (Llama 3.2, etc.) for specific agents (Planner, Worker, Verifier) to improve performance and specialization.

## Overview

The goal is to create specialized models for each agent role:
- **Planner**: Specialized in decomposing complex tasks into subtasks.
- **Worker**: Specialized in tool selection and execution.
- **Verifier**: Specialized in validating outputs and summarizing results.

## Workflow

1. **Data Collection**: Log agent interactions.
2. **Dataset Preparation**: Format data for training.
3. **Fine-Tuning**: Train adapters using Unsloth/Axolotl.
4. **Export & Quantize**: Convert to GGUF format.
5. **Deploy**: Run with Ollama.

---

## 1. Data Collection

To train a model, you need examples of "good" behavior.

### Enable Logging
Ensure your agents are running and logging interactions. You can extract logs from the `MessageBus` or by saving agent traces.

### Required Format (Alpaca/ShareGPT)
For instruction tuning, format your data like this:

```json
[
  {
    "instruction": "You are the Planner Agent. Break down this task.",
    "input": "Research the latest renewable energy trends.",
    "output": "{\"subtasks\": [\"Search for renewable energy trends 2024\", \"Summarize key findings\"]}"
  }
]
```

---

## 2. Fine-Tuning with Unsloth (Recommended)

We recommend using [Unsloth](https://github.com/unslothai/unsloth) for 2x faster training and 60% less memory usage.

### Setup (Google Colab / Local GPU)

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)
```

### Training Loop
Load your dataset and train:

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)
trainer.train()
```

---

## 3. Export to GGUF (for Ollama)

After training, export the model to GGUF format which Ollama uses.

```python
model.save_pretrained_gguf("model_name", tokenizer, quantization_method = "q4_k_m")
```

This will generate a `.gguf` file (e.g., `model_name-unsloth.Q4_K_M.gguf`).

---

## 4. Deploy with Ollama

### Create a Modelfile
Create a file named `Modelfile` in the same directory as your GGUF file:

```dockerfile
FROM ./model_name-unsloth.Q4_K_M.gguf

SYSTEM """
You are the Planner Agent. Your goal is to decompose tasks into subtasks.
"""

PARAMETER temperature 0.1
PARAMETER stop "Observation:"
```

### Create the Model in Ollama
Run this command in your terminal:

```bash
ollama create planner-ft -f Modelfile
```

### Update AgentFoundry Config
Update your `.env` file to use the new model:

```bash
PLANNER_MODEL=planner-ft
```

---

## Recommended Base Models

| Agent | Base Model | Quantization | Reason |
|-------|------------|--------------|--------|
| **Planner** | Llama-3.2-1B-Instruct | q8_0 | High precision for logic, very fast |
| **Worker** | Llama-3.2-3B-Instruct | q4_k_m | Balanced reasoning and speed |
| **Verifier** | Llama-3.2-1B-Instruct | q8_0 | Fast validation |
