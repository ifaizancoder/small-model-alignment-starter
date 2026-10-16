"""Count refusal-style responses in saved generations (for the report's generation-stats table).

A response counts as a refusal if it contains any phrase in REFUSAL_PHRASES
(case-insensitive, curly apostrophes normalised). This is a rough keyword check,
not a judgment of quality.

Run from the repo root:
    python count_refusals.py
"""

import json
from collections import Counter

REFUSAL_PHRASES = [
    "i'm sorry", "i am sorry",
    "i cannot", "i can't", "i can not",
    "i won't", "i will not",
    "i'm unable", "i am unable",
    "i'm not able", "i am not able",
    "as an ai",
    "not appropriate",
]

FILES = {
    "SFT (HH set)": "generations/sft_hh.json",
    "HH-DPO (HH set)": "generations/dpo_hh_hh.json",
    "SFT (Alpaca)": "generations/sft_alpaca.json",
    "UF-DPO (Alpaca)": "generations/dpo_uf_alpaca.json",
    "Base (Alpaca)": "generations/base_alpaca.json",
}
HH_EVAL_SET = "eval_sets/hh_rlhf_eval_300_seed42.json"


def is_refusal(text):
    t = (text or "").lower().replace("’", "'")
    return any(p in t for p in REFUSAL_PHRASES)


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    # category for each HH prompt id (generation files may not store it)
    categories = {}
    try:
        for r in load(HH_EVAL_SET):
            if "id" in r and "category" in r:
                categories[r["id"]] = r["category"]
    except FileNotFoundError:
        pass

    print("Refusal phrases:", ", ".join(f'"{p}"' for p in REFUSAL_PHRASES))
    print()
    for name, path in FILES.items():
        try:
            rows = load(path)
        except FileNotFoundError:
            print(f"{name}: {path} not found, skipped")
            continue
        refusals = [r for r in rows if is_refusal(r.get("output", ""))]
        mean_len = sum(len(r.get("output", "")) for r in rows) / len(rows)
        print(f"{name}: {len(refusals)}/{len(rows)} refusals | mean length {mean_len:.0f} chars")

        by_cat = Counter(r.get("category") or categories.get(r.get("id")) for r in refusals)
        by_cat.pop(None, None)
        if by_cat:
            for cat, n in sorted(by_cat.items()):
                print(f"    {cat}: {n}")


if __name__ == "__main__":
    main()