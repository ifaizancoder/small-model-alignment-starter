import json

def load(path):
    with open(path) as f:
        return {item["id"]: item for item in json.load(f)}

for name, sft_path, dpo_path in [
    ("UltraFeedback", "generations/sft_alpaca.json", "generations/dpo_uf_alpaca.json"),
    ("HH-RLHF", "generations/sft_hh.json", "generations/dpo_hh_hh.json"),
]:
    sft, dpo = load(sft_path), load(dpo_path)
    sft_len = sum(len(x["output"]) for x in sft.values()) / len(sft)
    dpo_len = sum(len(x["output"]) for x in dpo.values()) / len(dpo)
    print(f"\n######## {name}: average length SFT {sft_len:.0f} chars, DPO {dpo_len:.0f} chars")
    for pid in list(sft)[:2]:
        print("=" * 80)
        print("Q:", sft[pid]["instruction"])
        print("--- SFT ---")
        print(sft[pid]["output"][:600])
        print("--- DPO ---")
        print(dpo[pid]["output"][:600])