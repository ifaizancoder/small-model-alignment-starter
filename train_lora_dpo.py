import argparse
import json
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def to_conversational(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--output_dir", default="outputs/lora_dpo")
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.data) as f:
        rows = json.load(f)
    random.Random(args.seed).shuffle(rows)
    val_rows, train_rows = rows[:args.n_val], rows[args.n_val:]
    train_ds = Dataset.from_list([to_conversational(r) for r in train_rows])
    val_ds = Dataset.from_list([to_conversational(r) for r in val_rows])
    print(f"train pairs: {len(train_ds)} | val pairs: {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    print("Model initialized.")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    print("LoRA config initialized.")
    config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=512,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )

    print("DPO config initialized.")
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()

    trainer.train()
    final_eval = trainer.evaluate()
    print("final eval:", final_eval)

    trainer.save_model(f"{args.output_dir}/adapter")      # saves only the LoRA adapter
    tokenizer.save_pretrained(f"{args.output_dir}/adapter")
    with open(f"{args.output_dir}/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    with open(f"{args.output_dir}/run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("saved adapter, log_history.json and run_config.json")


if __name__ == "__main__":
    main()