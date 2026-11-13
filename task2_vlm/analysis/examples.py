import pandas as pd

from common.io import ROOT

PER_CATEGORY = 2


def complementary_example_rows(preds):
    rows = {}
    for r in preds.to_dict("records"):
        rows.setdefault((r["suite"], r["pair_id"], r["slot"]), {})[r["condition"]] = r
    pair_ids = sorted({(s, p) for s, p, _ in rows})
    chosen = []
    for suite in ["vqav2", "grounding"]:
        buckets = {"correct_both_and_swap_follows_partner": [], "correct_both_but_swap_keeps_own_answer": [], "wrong_under_correct_image": []}
        for s, pid in pair_ids:
            if s != suite:
                continue
            slots = [rows[(s, pid, k)] for k in (0, 1)]
            ok = [x["correct"]["exact_canonical"] for x in slots]
            follows = [x["complementary"]["equals_partner_reference"] for x in slots]
            if all(ok) and all(follows):
                buckets["correct_both_and_swap_follows_partner"].append(pid)
            elif all(ok):
                buckets["correct_both_but_swap_keeps_own_answer"].append(pid)
            elif not any(ok):
                buckets["wrong_under_correct_image"].append(pid)
        for rule, ids in buckets.items():
            for pid in ids[:PER_CATEGORY]:
                for k in (0, 1):
                    x = rows[(suite, pid, k)]
                    c = x["correct"]
                    chosen.append({
                        "suite": suite, "rule": rule, "pair_id": pid, "slot": k, "type": c["type"], "question": c["question"], "reference": c["reference"],
                        **{f"pred_{cond}": x[cond]["prediction"] for cond in ["correct", "complementary", "mismatched", "none"]},
                    })
    return pd.DataFrame(chosen)


def main():
    preds = pd.read_json(ROOT / "results" / "task2" / "predictions.jsonl", lines=True)
    out = complementary_example_rows(preds)
    out.to_csv(ROOT / "results" / "task2" / "examples_complementary.csv", index=False)
    print(out.groupby(["suite", "rule"]).size())


if __name__ == "__main__":
    main()
