import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# === CONFIG ===
MODEL_DIR = "models/medical_llm_final_dataset"       # merged model
DATA_PATH = "data/medical_qa.jsonl"
SAVE_ADAPTER_PATH = "models/adapter_after_extra_training"

# === Load full model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    return_dict=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# === Apply fresh LoRA adapter ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# === Load and format dataset ===
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_instruction(example):
    return {
        "text": f"### Question:\n{example['instruction']} {example['input']}\n\n### Answer:\n{example['output']}"
    }

dataset = dataset.map(format_instruction)

def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

# === Training ===
training_args = TrainingArguments(
    output_dir=SAVE_ADAPTER_PATH,
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

trainer.train()

# === Save updated adapter ===
model.save_pretrained(SAVE_ADAPTER_PATH)
tokenizer.save_pretrained(SAVE_ADAPTER_PATH)

print("✅ Training complete. Adapter saved to:", SAVE_ADAPTER_PATH)
