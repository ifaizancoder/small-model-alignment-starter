import argparse
import json
import os
import random

# (comparison name, SFT file, DPO file, DPO model label, output file)
COMPARISONS = [
    ("dpo_uf_vs_sft_alpaca",
     "generations/sft_alpaca.json", "generations/dpo_uf_alpaca.json", "dpo_uf",
     "eval_pairs/dpo_uf_vs_sft_alpaca_pairs.jsonl"),
    ("dpo_hh_vs_sft_hh",
     "generations/sft_hh.json", "generations/dpo_hh_hh.json", "dpo_hh",
     "eval_pairs/dpo_hh_vs_sft_hh_pairs.jsonl"),
]


def load_by_id(path):
    with open(path) as f:
        rows = json.load(f)
    by_id = {}
    for r in rows:
        if r["id"] in by_id:
            raise ValueError(f"duplicate id {r['id']} in {path}")
        by_id[r["id"]] = r
    return rows, by_id


def build(name, sft_path, dpo_path, dpo_label, out_path, seed):
    sft_rows, sft = load_by_id(sft_path)
    _, dpo = load_by_id(dpo_path)

    if set(sft) != set(dpo):
        missing = set(sft) ^ set(dpo)
        raise ValueError(f"{name}: SFT and DPO files cover different ids, e.g. {list(missing)[:5]}")

    rng = random.Random(seed)
    pairs, dpo_is_a, empty = [], 0, 0
    for r in sft_rows:  # keep the eval-set order
        pid = r["id"]
        s, d = sft[pid], dpo[pid]
        if s["instruction"] != d["instruction"]:
            raise ValueError(f"{name}: instruction mismatch for id {pid}")
        if not s["output"].strip() or not d["output"].strip():
            empty += 1

        if rng.random() < 0.5:
            a_model, a_out, b_model, b_out = dpo_label, d["output"], "sft", s["output"]
            dpo_is_a += 1
        else:
            a_model, a_out, b_model, b_out = "sft", s["output"], dpo_label, d["output"]

        pair = {
            "pair_id": f"{name}-{pid}",
            "id": pid,
            "dataset": s.get("dataset", ""),
            "instruction": s["instruction"],
            "response_a": a_out,
            "response_b": b_out,
            "hidden": {"a_model": a_model, "b_model": b_model},
        }
        if "category" in s:  # the HH eval set may carry a category; useful for analysis
            pair["category"] = s["category"]
        pairs.append(pair)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"{name}: {len(pairs)} pairs -> {out_path}")
    print(f"   DPO in position A: {dpo_is_a} | position B: {len(pairs) - dpo_is_a}")
    if empty:
        print(f"   warning: {empty} pairs have an empty response")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for i, comp in enumerate(COMPARISONS):
        build(*comp, seed=args.seed + i)  # different coin flips per file, still reproducible


if __name__ == "__main__":
    main()