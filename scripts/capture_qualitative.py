#!/usr/bin/env python3
"""Capture the per-example predictions the report's qualitative section needs.

**Why this exists.** NB5's `qualitative.json` records only the fine-tune's own score and
a 90-character preview of its prediction. The rubric asks for a table comparing each
example against baseline (b) -- including at least two cases where the fine-tune LOSES --
and neither (b)'s predictions nor the full (c) outputs are saved anywhere. The comparison
is required and the data for it was never written.

Nothing here trains. It runs two generation sweeps per model, so on a T4 it is ~10
minutes rather than the ~50 of a full pipeline.

    python scripts/capture_qualitative.py

Writes `results/qualitative_full.json`.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate, report          # noqa: E402
from labkit.config import get_tier                            # noqa: E402


def load_jsonl(p: pathlib.Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sweep(model, tok, target, regression, label):
    """Both eval groups for one model. Returns (target_preds, regression_preds)."""
    tpreds, _ = generate.generate_batch(
        model, tok, [r["input"] for r in target], system=generate.NAIVE_PROMPT,
        label=f"{label}/target")
    rpreds, _ = generate.generate_batch(
        model, tok, [r["instruction"] for r in regression], system=None, max_new_tokens=96,
        label=f"{label}/regression")
    return tpreds, rpreds


def main() -> int:
    tier = get_tier()
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    regression = load_jsonl(ROOT / "data" / "eval_regression.jsonl")
    print(f"tier={tier.name}  target={len(target)}  regression={len(regression)}")

    # --- (b) base + optimized prompt -----------------------------------------
    print("\n=== (b) base + optimized prompt ===")
    model, tok = generate.load_base(tier)
    b_t, b_r = sweep(model, tok, target, regression, "b")
    del model
    generate.free_memory()

    # --- (c) base + the trained adapter --------------------------------------
    # Scored the way NB5 scores it: the NAIVE prompt, because the behaviour is supposed
    # to have moved into the weights.
    print("\n=== (c) LoRA fine-tune ===")
    from peft import PeftModel

    model, tok = generate.load_base(tier)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / "correct"))
    model.eval()
    c_t, c_r = sweep(model, tok, target, regression, "c")
    del model
    generate.free_memory()

    # --- assemble -------------------------------------------------------------
    rows = []
    for i, (rec, bt, ct) in enumerate(zip(target, b_t, c_t)):
        sb = ev.triage_field_accuracy(bt, rec["label"])
        sc = ev.triage_field_accuracy(ct, rec["label"])
        rows.append({
            "i": i,
            "ticket": rec["input"],
            "label": rec["label"],
            "b_pred": bt.strip(),
            "c_pred": ct.strip(),
            "b_score": round(sb, 4),
            "c_score": round(sc, 4),
            "delta": round(sc - sb, 4),
            "ft_loses": sc < sb,
        })

    reg_rows = []
    for i, (rec, br, cr) in enumerate(zip(regression, b_r, c_r)):
        kb = ev.keyword_recall(br, rec["keywords"])
        kc = ev.keyword_recall(cr, rec["keywords"])
        reg_rows.append({
            "i": i,
            "instruction": rec["instruction"],
            "keywords": rec["keywords"],
            "b_pred": br.strip()[:400],
            "c_pred": cr.strip()[:400],
            "b_recall": round(kb, 4),
            "c_recall": round(kc, 4),
            "ft_loses": kc < kb,
        })

    losses = [r for r in rows if r["ft_loses"]]
    reg_losses = [r for r in reg_rows if r["ft_loses"]]
    out = {
        "n_target": len(rows),
        "n_regression": len(reg_rows),
        "b_target_mean": round(sum(r["b_score"] for r in rows) / len(rows), 4),
        "c_target_mean": round(sum(r["c_score"] for r in rows) / len(rows), 4),
        "n_ft_loses_target": len(losses),
        "n_ft_loses_regression": len(reg_losses),
        "target": rows,
        "regression": reg_rows,
    }
    report.write_json(out, "qualitative_full.json", results_dir=ROOT / "results")

    print(f"\n(b) target mean {out['b_target_mean']:.4f}   "
          f"(c) target mean {out['c_target_mean']:.4f}")
    print(f"fine-tune LOSES on {len(losses)}/{len(rows)} target examples")
    print(f"fine-tune LOSES on {len(reg_losses)}/{len(reg_rows)} regression examples")
    print("-> results/qualitative_full.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
