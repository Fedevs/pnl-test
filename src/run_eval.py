"""Run classifiers on a split, cache predictions and compute metrics.

    python src/run_eval.py --limit 20                     # smoke test on 20 items
    python src/run_eval.py                                # all models x all variants
    python src/run_eval.py --models gemini --variants zs  # a subset
    python src/run_eval.py --from-cache                   # only recompute metrics/figures
    python src/run_eval.py --split perturbed              # robustness set (src/perturb.py)

Predictions are appended to results/<split>/preds/<provider>__<variant>.jsonl. Items
already predicted without error are skipped, so an interrupted run (e.g. a free-tier
daily limit) resumes where it stopped. TF-IDF is cheap and is always recomputed.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

import config
import metrics
import prompts
from classifiers import get_classifier

LLM_PROVIDERS = ["gemini", "groq"]


def cached_ok_ids(path):
    if not path.exists():
        return set()
    ok = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("label") and not rec.get("error"):
                ok.add(rec["id"])
    return ok


def run_one(clf, provider, variant, items, preds_dir):
    path = preds_dir / f"{provider}__{variant}.jsonl"
    mode = "a"
    if provider == "tfidf":
        mode = "w"
        todo = items
    else:
        done = cached_ok_ids(path)
        todo = items[~items["id"].isin(done)]
    print(f"\n{provider}:{variant} ({clf.model}) - {len(todo)} to predict, {len(items) - len(todo)} cached")
    if not len(todo):
        return
    errors = 0
    with path.open(mode, encoding="utf-8") as f:
        for row in tqdm(todo.itertuples(), total=len(todo), file=sys.stdout):
            out = clf.predict(row.texto)
            errors += bool(out["error"])
            rec = {"id": row.id, "provider": provider, "variant": variant, "model": clf.model, **out,
                   "created_at": datetime.now(timezone.utc).isoformat()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    if errors:
        print(f"  {errors} errors (re-run the same command to retry them)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["tfidf", *LLM_PROVIDERS], choices=["tfidf", *LLM_PROVIDERS])
    parser.add_argument("--variants", nargs="+", default=prompts.VARIANTS, choices=prompts.VARIANTS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    parser.add_argument("--from-cache", action="store_true")
    args = parser.parse_args()

    base, preds_dir, _ = metrics.split_paths(args.split)
    if args.limit:
        # Keep smoke-test results apart so they never mix with the full run.
        base = config.RESULTS_DIR / f"{args.split}_smoke{args.limit}"
        preds_dir = base / "preds"

    if not args.from_cache:
        gold = metrics.load_gold(args.split).reset_index()
        items = gold.head(args.limit) if args.limit else gold
        preds_dir.mkdir(parents=True, exist_ok=True)
        train_ids = set(pd.read_csv(config.TRAIN_PATH, dtype={"id": str})["id"])
        overlap = train_ids & set(gold["id"])
        if overlap:
            raise SystemExit(f"Leakage: {len(overlap)} {args.split} ids are also in train")

        meta = {
            "split": args.split,
            "n_items": len(items),
            "test_ids_sha256": hashlib.sha256("\n".join(sorted(items["id"])).encode()).hexdigest(),
            "models": {"gemini": config.GEMINI_MODEL, "groq": config.GROQ_MODEL},
            "prices_usd_per_m": config.PRICES_USD_PER_M,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        for provider in args.models:
            variants = ["default"] if provider == "tfidf" else args.variants
            for variant in variants:
                clf = get_classifier(provider, None if provider == "tfidf" else variant)
                if set(clf.few_shot_ids) & set(items["id"]):
                    raise SystemExit("Leakage: few-shot examples found in the evaluation set")
                if provider == "tfidf":
                    meta["tfidf"] = clf.describe()
                    print(f"TF-IDF best params: {clf.best_params} (CV macro-F1 {clf.cv_f1_macro})")
                run_one(clf, provider, variant, items, preds_dir)
        (base / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    metrics.run(args.split, base if args.limit else None, args.limit)


if __name__ == "__main__":
    main()
