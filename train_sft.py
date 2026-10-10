"""SFT training loop for Qwen2.5-0.5B (no HF Trainer).

Same skeleton as my PA1 training script: one outer iteration = one optimizer
step, with an inner gradient-accumulation loop. What is new for PA2 is marked
with "PA2:" comments.

Pilot:
    python train_sft.py --n_train 320 --n_val 64 --eval_interval 5 --log_interval 1 --output_dir outputs/sft_pilot
Full run:
    python train_sft.py
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from my_implementation import (
    build_sft_batch,
    compute_sft_loss,
    extract_pair_from_example,
    response_logprobs,
)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, cosine_steps):
    """Same schedule as PA1: linear warmup, cosine decay to min_lr, then flat."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step <= cosine_steps:
        progress = (step - warmup_steps) / max(1, cosine_steps - warmup_steps)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)
    return min_lr


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs          # list of (prompt, response) tuples

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def assemble(batch):
    """Turn a list of (prompt, response) tuples into two lists of strings."""
    prompts, responses = zip(*batch)
    return list(prompts), list(responses)


def make_dataloader(pairs, batch_size=2, shuffle=True, generator=None):
    return DataLoader(
        PairDataset(pairs),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=assemble,
        generator=generator,
    )


def load_pairs(tokenizer, n_train, n_val, seed, max_prompt_tokens):
    """Seeded sample of single-turn (prompt, response) pairs from UltraChat."""
    from datasets import load_dataset

    data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=seed)
    pairs = []
    for example in data:
        messages = example["messages"]
        if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            continue
        messages[0]["content"] = messages[0]["content"].strip()
        messages[1]["content"] = messages[1]["content"].strip()
        if not messages[0]["content"] or not messages[1]["content"]:
            continue
        prompt, response = extract_pair_from_example(example)
        # skip long prompts so the 512-token window has room for the response
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_tokens:
            continue
        pairs.append((prompt, response))
        if len(pairs) == n_train + n_val:
            break
    return pairs[:n_train], pairs[n_train:]


@torch.inference_mode()
def evaluate_validation(model, tokenizer, val_loader, max_length, device, use_amp):
    """Cross-entropy per response token over the whole validation set (no gradients)."""
    was_training = model.training
    model.eval()
    total_logp, total_tokens = 0.0, 0
    for prompts, responses in val_loader:
        batch = build_sft_batch(tokenizer, prompts, responses, max_length)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logps = response_logprobs(model, batch)
        total_logp += logps.sum().item()
        total_tokens += (batch["labels"][:, 1:] != -100).sum().item()
    model.train(was_training)
    return -total_logp / max(total_tokens, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--output_dir", default="outputs/sft")
    p.add_argument("--n_train", type=int, default=30000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--max_prompt_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)        # 2 x 16 = effective batch 32
    p.add_argument("--val_batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_lr", type=float, default=2e-5)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--train_seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(args.device)
    use_amp = device.type == "cuda"
    print(f"device: {device}")
    torch.manual_seed(args.train_seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # PA2: pretrained model instead of my own transformer. Weights stay float32.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)

    train_pairs, val_pairs = load_pairs(
        tokenizer, args.n_train, args.n_val, args.train_seed, args.max_prompt_tokens
    )
    print(f"train pairs: {len(train_pairs)} | val pairs: {len(val_pairs)}")
    train_gen = torch.Generator().manual_seed(args.train_seed)
    train_loader = make_dataloader(train_pairs, args.batch_size, shuffle=True, generator=train_gen)
    val_loader = make_dataloader(val_pairs, args.val_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    # PA2: fp16 forward pass needs a gradient scaler
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # PA2: the number of steps comes from the dataset size (one pass per epoch)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    num_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * num_steps))
    cosine_steps = num_steps - 1
    print(f"optimizer steps: {num_steps} | warmup steps: {warmup_steps}")

    def save_model(name):
        # PA2: save in Hugging Face format so DPO and generation can load it
        model.save_pretrained(out_dir / name)
        tokenizer.save_pretrained(out_dir / name)

    best_val = float("inf")
    start = time.time()
    batches = iter(train_loader)
    for step in range(num_steps):
        model.train()
        lr = get_lr_cosine_schedule(step, args.max_lr, args.min_lr, warmup_steps, cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        train_loss = 0.0
        seen = 0
        for _ in range(args.grad_accum):
            try:
                prompts, responses = next(batches)
            except StopIteration:                      # end of an epoch
                if step == num_steps - 1:
                    break                              # last step: partial window
                batches = iter(train_loader)
                prompts, responses = next(batches)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, args.context_length)
            scaler.scale(loss / args.grad_accum).backward()
            train_loss += loss.detach().item()
            seen += 1
        train_loss /= max(seen, 1)
        scaler.unscale_(optimizer)                     # so clipping sees the true gradient size
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        completed = step + 1
        final = completed == num_steps
        do_val = final or completed % args.eval_interval == 0
        do_log = do_val or completed % args.log_interval == 0
        val_loss = None
        is_best = False
        if do_val:
            val_loss = evaluate_validation(
                model, tokenizer, val_loader, args.context_length, device, use_amp
            )
            is_best = val_loss < best_val
            if is_best:                                # PA2: keep only the best checkpoint
                best_val = val_loss
                save_model("best")
        if do_log:
            msg = f"step {completed:5d} | lr {lr:.2e} | train {train_loss:.4f} | grad_norm {grad_norm:.3f}"
            if val_loss is not None:
                msg += f" | val {val_loss:.4f}" + (" (best)" if is_best else "")
            msg += f" | {time.time() - start:.0f}s"
            print(msg, flush=True)
            record = {"step": completed, "lr": lr, "train_loss": train_loss,
                      "grad_norm": float(grad_norm), "seconds": time.time() - start}
            if val_loss is not None:
                record.update({"val_loss": val_loss, "best": is_best})
            with open(metrics_path, "a") as f:         # survives a Kaggle disconnect
                f.write(json.dumps(record) + "\n")

    print("training done")
    save_model("final")
    print(f"best val loss {best_val:.4f} | saved {out_dir}/best and {out_dir}/final")


if __name__ == "__main__":
    main()