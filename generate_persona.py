import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--eval_set", default="eval_sets/persona_eval_prompts.json")
    p.add_argument("--output", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    with open(args.eval_set) as f:
        prompts = json.load(f)

    results = []
    for item in prompts:
        messages = [{"role": "user", "content": item["instruction"]}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda:0")
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append({**item, "output": text,
                        "generator": "smollm2-360m-lora-dpo" if args.adapter else "smollm2-360m-instruct"})
        print(f"{len(results)}/{len(prompts)} [{item['category']}]", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()