import json

with open("data/persona_lora/ramsay_style_preferences.json") as f:
    data = json.load(f)

print("rows:", len(data))
print("keys:", list(data[0].keys()))
for row in data[:2]:
    print("=" * 80)
    print("PROMPT:  ", row["prompt"])
    print("CHOSEN:  ", row["chosen"][:400])
    print("REJECTED:", row["rejected"][:400])