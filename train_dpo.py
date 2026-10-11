import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
 
from my_implementation import compute_dpo_loss, compute_response_logprobs, split_hh_example
from train_sft import PairDataset

def load_hh_triples(tokenizer, n_train, n_val, seed, max_prompt_tokens):
    from datasets import load_dataset
    data = load_dataset("Anthropic/hh-rlhf", split="train").shuffle(seed=seed)
    triples = []
    for example in data:
        triple = split_hh_example(example)
        if triple is None:
            continue
        instruction, chosen, rejected = triple
        if len(tokenizer.encode(instruction, add_special_tokens=False)) > max_prompt_tokens:
            continue
        triples.append(triple)
        if len(triples) >= n_train + n_val:
            break
    if len(triples) < n_train + n_val:
        print(f"Warning: only found {len(triples)} valid examples, fewer than requested {n_train+n_val}")
    return triples[:n_train], triples[n_train:n_train+n_val]

def load_policy_and_reference_models(sft_path, policy_device, reference_device):
    from transformers import AutoModelForCausalLM
    policy = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=torch.float32).to(policy_device)
    reference_model = AutoModelForCausalLM.from_pretrained(sft_path, torch_dtype=torch.float16).to(reference_device)
    reference_model.eval()  # reference model is frozen
    for param in reference_model.parameters():
        param.requires_grad = False
    return policy, reference_model

def assemble_triples(batch):
    prompts, chosen, rejected = zip(*batch)
    return list(prompts), list(chosen), list(rejected)
 
 
def make_triple_dataloader(triples, batch_size=1, shuffle=True, generator=None):
    return DataLoader(
        PairDataset(triples),          # the dataset class only stores and returns items
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=assemble_triples,
        generator=generator,
    )
 
@torch.inference_mode()
def eval_dpo(policy, reference, tokenizer, val_loader, beta, max_length, use_amp):
    was_training = policy.training
    policy.eval()
    total_loss, total_correct, total_margin, count = 0.0, 0, 0.0, 0

    for prompts, chosen, rejected in val_loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pc = compute_response_logprobs(policy, tokenizer, prompts, chosen, max_length)
            pr = compute_response_logprobs(policy, tokenizer, prompts, rejected, max_length)
            rc = compute_response_logprobs(reference, tokenizer, prompts, chosen, max_length)
            rr = compute_response_logprobs(reference, tokenizer, prompts, rejected, max_length)
        pc, pr = pc.float(), pr.float()
        rc, rr = rc.float().to(pc.device), rr.float().to(pc.device)
        reward_chosen = beta * (pc - rc)
        reward_rejected = beta * (pr - rr)
        diff = reward_chosen - reward_rejected
        total_loss += (-F.logsigmoid(diff)).sum().item()
        total_correct += (diff > 0).sum().item()
        total_margin += diff.sum().item()
        count += len(prompts)
    policy.train(was_training)
    return {
        "val_loss": total_loss / count,
        "reward_accuracy": total_correct / count,
        "reward_margin": total_margin / count,
    }

def pick_devices():
    """DPO: policy on cuda:0, frozen reference on cuda:1 when two GPUs are available."""
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        return torch.device("cuda:0"), torch.device("cuda:1")
    if n_gpus == 1:
        return torch.device("cuda:0"), torch.device("cuda:0")
    return torch.device("cpu"), torch.device("cpu")
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", required=True)                  # DPO: start from the SFT best checkpoint
    p.add_argument("--output_dir", default="outputs/dpo_hh")
    p.add_argument("--n_train", type=int, default=15000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--max_prompt_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=32)         # 1 x 32 = effective batch 32
    p.add_argument("--val_batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)             # DPO: small fixed learning rate
    p.add_argument("--beta", type=float, default=0.1)            # DPO
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--train_seed", type=int, default=42)
    args = p.parse_args()
 
    from transformers import AutoTokenizer
 
    policy_device, reference_device = pick_devices()
    use_amp = policy_device.type == "cuda"
    print(f"policy device: {policy_device} | reference device: {reference_device}")
    torch.manual_seed(args.train_seed)
 
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")
 
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
    # DPO: two copies of the SFT model, a trainable policy and a frozen reference
    policy, reference = load_policy_and_reference_models(args.sft_path, policy_device, reference_device)
 
    train_triples, val_triples = load_hh_triples(
        tokenizer, args.n_train, args.n_val, args.train_seed, args.max_prompt_tokens
    )
    print(f"train triples: {len(train_triples)} | val triples: {len(val_triples)}")
    train_gen = torch.Generator().manual_seed(args.train_seed)
    train_loader = make_triple_dataloader(train_triples, args.batch_size, shuffle=True, generator=train_gen)
    val_loader = make_triple_dataloader(val_triples, args.val_batch_size, shuffle=False)
 
    # DPO: the optimizer only ever sees the policy's weights
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(policy_device.type, enabled=use_amp)
 
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    num_steps = steps_per_epoch * args.epochs
    print(f"optimizer steps: {num_steps} | fixed lr: {args.lr:.1e} | beta: {args.beta}")
 
    def save_model(name):
        policy.save_pretrained(out_dir / name)
        tokenizer.save_pretrained(out_dir / name)
 
    def write_record(record):
        with open(metrics_path, "a") as f:             # survives a Kaggle disconnect
            f.write(json.dumps(record) + "\n")
 
    # DPO: sanity check before training. Policy == reference, so expect
    # accuracy 0, margin 0 and loss log(2) = 0.6931.
    start = time.time()
    val = eval_dpo(policy, reference, tokenizer, val_loader, args.beta, args.context_length, use_amp)
    print(
        f"step     0 | val {val['val_loss']:.4f} | acc {val['reward_accuracy']:.4f} "
        f"| margin {val['reward_margin']:.4f} | {time.time() - start:.0f}s",
        flush=True,
    )
    write_record({"step": 0, "seconds": time.time() - start, **val})
 
    best_acc = -1.0                                    # DPO: best = highest validation reward accuracy
    best_loss = float("inf")                           # tie-breaker when accuracy is equal
    start = time.time()
    batches = iter(train_loader)
    for step in range(num_steps):
        policy.train()
        optimizer.zero_grad()
        train_loss = 0.0
        seen = 0
        for _ in range(args.grad_accum):
            try:
                prompts, chosen, rejected = next(batches)
            except StopIteration:                      # end of an epoch
                if step == num_steps - 1:
                    break                              # last step: partial window
                batches = iter(train_loader)
                prompts, chosen, rejected = next(batches)
            with torch.autocast(device_type=policy_device.type, dtype=torch.float16, enabled=use_amp):
                loss = compute_dpo_loss(
                    policy, reference, tokenizer, prompts, chosen, rejected,
                    args.beta, args.context_length,
                )
            scaler.scale(loss / args.grad_accum).backward()
            train_loss += loss.detach().item()
            seen += 1
        train_loss /= max(seen, 1)
        scaler.unscale_(optimizer)                     # so clipping sees the true gradient size
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
 
        completed = step + 1
        final = completed == num_steps
        do_val = final or completed % args.eval_interval == 0
        do_log = do_val or completed % args.log_interval == 0
        val = None
        is_best = False
        if do_val:
            val = eval_dpo(policy, reference, tokenizer, val_loader, args.beta, args.context_length, use_amp)
            acc, vloss = val["reward_accuracy"], val["val_loss"]
            is_best = acc > best_acc or (acc == best_acc and vloss < best_loss)
            if is_best:
                best_acc, best_loss = acc, vloss
                save_model("best")
        if do_log:
            msg = f"step {completed:5d} | train {train_loss:.4f} | grad_norm {grad_norm:.3f}"
            if val is not None:
                msg += (
                    f" | val {val['val_loss']:.4f} | acc {val['reward_accuracy']:.4f} "
                    f"| margin {val['reward_margin']:.4f}" + (" (best)" if is_best else "")
                )
            msg += f" | {time.time() - start:.0f}s"
            print(msg, flush=True)
            record = {"step": completed, "lr": args.lr, "train_loss": train_loss,
                      "grad_norm": float(grad_norm), "seconds": time.time() - start}
            if val is not None:
                record.update({**val, "best": is_best})
            write_record(record)
 
    print("training done")
    save_model("final")
    print(f"best val reward accuracy {best_acc:.4f} | saved {out_dir}/best and {out_dir}/final")
 
 
if __name__ == "__main__":
    main()
