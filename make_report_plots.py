"""Make every figure for the PA2 report from the saved logs and judge summaries.

Run from the repo root:
    python make_report_plots.py

Reads:
    logs/sft_metrics.jsonl
    logs/dpo_hh_metrics.jsonl
    logs/dpo_uf_metrics.jsonl
    logs/lora_dpo_log_history.json
    judgments/dpo_hh_vs_sft_hh_judged_summary.json
Writes (PDF, vector, ready for LaTeX):
    figures/sft_loss.pdf
    figures/dpo_curves.pdf
    figures/lora_dpo_run1.pdf
    figures/hh_categories.pdf
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # no window needed, just save files
import matplotlib.pyplot as plt

LOGS = "logs"
JUDGMENTS = "judgments"
OUT = "figures"

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 8,
})


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def moving_avg(xs, k):
    """Trailing moving average over the last k points (shorter at the start)."""
    out = []
    for i in range(len(xs)):
        window = xs[max(0, i - k + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


def selected_checkpoint(rows):
    """The trainer marks each new best with "best": true; the last one is the saved best checkpoint."""
    marked = [r for r in rows if r.get("best")]
    return marked[-1] if marked else rows[-1]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------- SFT
def plot_sft():
    rows = read_jsonl(f"{LOGS}/sft_metrics.jsonl")
    train = [r for r in rows if "train_loss" in r]
    val = [r for r in rows if "val_loss" in r]

    steps = [r["step"] for r in train]
    loss = [r["train_loss"] for r in train]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(steps, loss, color="tab:blue", alpha=0.25, lw=0.8, label="train loss (logged every 10 steps)")
    ax.plot(steps, moving_avg(loss, 5), color="tab:blue", lw=1.6, label="train loss, moving average (50 steps)")
    ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-",
            color="tab:orange", ms=4, label="validation loss (every 100 steps)")

    best = selected_checkpoint(val)
    ax.plot(best["step"], best["val_loss"], "*", color="red", ms=12,
            label=f"best checkpoint = final (step {best['step']}, {best['val_loss']:.4f})")

    ax.set_xlabel("optimizer step (effective batch 32)")
    ax.set_ylabel("response-token loss (nats/token)")
    ax.set_title("SFT on UltraChat (Qwen2.5-0.5B)")
    ax.legend(loc="upper right")
    save(fig, "sft_loss.pdf")


# ---------------------------------------------------------------- DPO
def plot_dpo():
    runs = [
        ("HH-DPO", f"{LOGS}/dpo_hh_metrics.jsonl", "tab:red"),
        ("UF-DPO", f"{LOGS}/dpo_uf_metrics.jsonl", "tab:green"),
    ]
    panels = [
        ("val_loss", "validation DPO loss"),
        ("reward_accuracy", "validation reward accuracy"),
        ("reward_margin", "validation reward margin"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for name, path, color in runs:
        val = [r for r in read_jsonl(path) if "val_loss" in r]
        best = selected_checkpoint(val)
        steps = [r["step"] for r in val]
        for ax, (key, _) in zip(axes, panels):
            ax.plot(steps, [r[key] for r in val], "o-", color=color, ms=3, lw=1.3, label=name)
            ax.plot(best["step"], best[key], "*", color=color, ms=12, markeredgecolor="black",
                    markeredgewidth=0.6, label=f"{name} selected (step {best['step']})")

    axes[0].axhline(0.6931, color="gray", ls=":", lw=1, label="ln 2 (start)")
    for ax, (_, title) in zip(axes, panels):
        ax.set_xlabel("optimizer step (effective batch 32)")
        ax.set_title(title)
    axes[0].set_ylabel("loss")
    axes[1].set_ylabel("fraction with $r_w > r_l$")
    axes[2].set_ylabel("mean $r_w - r_l$")
    # one shared legend under the three panels so it never covers a curve
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "dpo_curves.pdf")


# ---------------------------------------------------------------- LoRA persona
def plot_lora():
    with open(f"{LOGS}/lora_dpo_log_history.json") as f:
        logs = json.load(f)
    train = [r for r in logs if "loss" in r and "rewards/chosen" in r]
    evals, seen = [], set()
    for r in logs:  # TRL repeats the final eval once more after training; keep one per step
        if "eval_loss" in r and r["step"] not in seen:
            seen.add(r["step"])
            evals.append(r)

    ts = [r["step"] for r in train]
    es = [r["step"] for r in evals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3))
    ax1.plot(ts, [r["loss"] for r in train], label="train loss")
    ax1.plot(es, [r["eval_loss"] for r in evals], "o--", label="validation loss")
    ax1.axhline(0.6931, color="gray", lw=0.8, ls=":", label="ln 2 (start)")
    ax1.set_xlabel("optimizer step (25 steps = 1 epoch)")
    ax1.set_ylabel("DPO loss")
    ax1.set_title("LoRA-DPO persona: loss")
    ax1.legend()

    lines = []
    for key, label, color in [("rewards/chosen", "chosen (Ramsay)", "tab:blue"),
                              ("rewards/rejected", "rejected (polite)", "tab:orange"),
                              ("rewards/margins", "margin", "tab:green")]:
        line, = ax2.plot(ts, [r[key] for r in train], color=color, label=label)
        ax2.plot(es, [r["eval_" + key] for r in evals], "o", color=color)
        lines.append(line)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xlabel("optimizer step (25 steps = 1 epoch)")
    ax2.set_ylabel(r"implicit reward  $\beta\log(\pi/\pi_{ref})$")
    ax2.set_title("Implicit rewards (lines = train, dots = validation)")
    ax2.legend(handles=lines, fontsize=7, loc="upper left")
    save(fig, "lora_dpo_run1.pdf")


# ---------------------------------------------------------------- HH judge by category
def plot_hh_categories():
    with open(f"{JUDGMENTS}/dpo_hh_vs_sft_hh_judged_summary.json") as f:
        summary = json.load(f)
    cats = summary["by_category"]
    # sort so the categories where DPO did best are at the top
    order = sorted(cats, key=lambda c: cats[c]["dpo_wins"] - cats[c]["sft_wins"])
    names = [c.replace("_", " ") for c in order]
    dpo = [cats[c]["dpo_wins"] for c in order]
    tie = [cats[c]["ties"] + cats[c].get("unknown", 0) for c in order]
    sft = [cats[c]["sft_wins"] for c in order]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = range(len(order))
    ax.barh(y, dpo, color="tab:green", label="DPO wins")
    ax.barh(y, tie, left=dpo, color="lightgray", label="ties")
    ax.barh(y, sft, left=[a + b for a, b in zip(dpo, tie)], color="tab:red", label="SFT wins")
    for i in y:
        if dpo[i]:
            ax.text(dpo[i] / 2, i, str(dpo[i]), ha="center", va="center", fontsize=7, color="white")
        if sft[i]:
            ax.text(dpo[i] + tie[i] + sft[i] / 2, i, str(sft[i]), ha="center", va="center",
                    fontsize=7, color="white")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel("number of prompts (30 per category)")
    ax.set_title("HH-DPO vs SFT by category (judge: gemini-3.6-flash)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=3, bbox_to_anchor=(1.0, 1.08), frameon=False)
    save(fig, "hh_categories.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_sft()
    plot_dpo()
    plot_lora()
    plot_hh_categories()