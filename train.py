import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Modelo base
MODEL_NAME = "pablocosta/bert-base-portuguese-cased-financial-news"

# Carregar dataset local
dataset = load_dataset("csv", data_files={"train": "dataset.csv", "test": "dataset.csv"})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["texto"], padding=True, truncation=True, max_length=256)


dataset = dataset.map(tokenize, batched=True)

# Modelo base
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Configuração LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(base_model, lora_config)

# Argumentos de treino
training_args = TrainingArguments(
    output_dir="./lora-results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

# Salvar modelo final
model.save_pretrained("./lora-model")
tokenizer.save_pretrained("./lora-model")

print("Fine-tuning concluído! Modelo salvo em ./lora-model")
