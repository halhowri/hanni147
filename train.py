"""
Step 2 - fine-tune Qwen3.5-0.8B with LoRA on the TensorFlow -> PyTorch pairs.

Settings (as agreed): 16-bit LoRA, rank 16, batch size 1,
gradient accumulation 4, max sequence length 512.

Run:   ../llm_env/bin/python train.py
Result: outputs/lora/  (the small trained add-on)  and  outputs/train_log.json
"""
import json
import os

from unsloth import FastLanguageModel          # must be imported before transformers
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 512
EPOCHS = 3

# ---- 1. load the pretrained model in 16-bit ------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,        # Qwen3.5 is sensitive to 4-bit, so we do not use it
    load_in_16bit=True,
    full_finetuning=False,
)

# ---- 2. attach the LoRA add-on (only these small matrices are trained) ----
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=MAX_SEQ_LENGTH,
)

# ---- 3. turn each (TensorFlow, PyTorch) pair into one training text -------
def to_text(example):
    # Same prompt the model will see at inference time, followed by the answer.
    prompt = tokenizer.apply_chat_template(example["messages"][:2], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    return {"text": prompt + example["messages"][2]["content"] + "<|im_end|>\n"}

files = {"train": os.path.join(HERE, "data", "train.jsonl"), "val": os.path.join(HERE, "data", "val.jsonl")}
data = load_dataset("json", data_files=files)
keep = ["text"]
train_ds = data["train"].map(to_text).select_columns(keep)
val_ds = data["val"].map(to_text).select_columns(keep)
print(f"train examples: {len(train_ds)}   validation examples: {len(val_ds)}")

# ---- 4. train -------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        weight_decay=0.01,
        optim="adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        output_dir=os.path.join(HERE, "outputs", "checkpoints"),
        report_to="none",
        seed=3407,
        dataset_num_proc=1,
    ),
)
# Only the assistant's answer (the PyTorch code) counts towards the loss.
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                  response_part="<|im_start|>assistant\n")
result = trainer.train()

# ---- 5. save the LoRA add-on and the loss history --------------------------
lora_dir = os.path.join(HERE, "outputs", "lora")
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
with open(os.path.join(HERE, "outputs", "train_log.json"), "w") as f:
    json.dump({"metrics": result.metrics, "history": trainer.state.log_history}, f, indent=1)
print("saved LoRA adapter to", lora_dir)
for row in trainer.state.log_history:
    if "eval_loss" in row:
        print(f"epoch {row['epoch']:.0f}: validation loss {row['eval_loss']:.4f}")
