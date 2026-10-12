import json
import random

from huggingface_hub import hf_hub_download

path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
with open(path) as f:
    data = json.load(f)
print("total:", len(data))
rng = random.Random(42)
chosen = rng.sample(range(len(data)), 300)       

subset = []
for i in sorted(chosen):
    subset.append({
        "id": f"alpaca_{i}",
        "instruction": data[i]["instruction"],             
    })

with open("eval_sets/alpaca_eval_300_seed42.json", "w") as f:
    json.dump(subset, f, indent=2)
print("saved", len(subset), "prompts")