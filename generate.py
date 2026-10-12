import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from my_implementation import generate_responses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)          
    p.add_argument("--generator", required=True)     
    p.add_argument("--eval_set", default="eval_sets/alpaca_eval_300_seed42.json")
    p.add_argument("--dataset", default="alpaca_eval")
    p.add_argument("--output", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)  
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    args = p.parse_args()
    if Path(args.output).exists():
        print(f"{args.output} already exists, skipping")
        return
    with open(args.eval_set) as f:
        prompts = json.load(f)
    if args.limit:
        prompts = prompts[:args.limit]
    print(f"{len(prompts)} prompts from {args.eval_set}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")

    instructions = [item["instruction"] for item in prompts]
    responses = generate_responses(
        model, tokenizer, instructions,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        repetition_penalty=args.repetition_penalty
    )

    results = []
    for item, response in zip(prompts, responses):
        results.append({
            **item,
            "output": response,
            "generator": args.generator,
            "dataset": args.dataset,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {len(results)} responses to {args.output}")


if __name__ == "__main__":
    main()