"""Turn raw generations into a reviewed, split dataset.

Steps (run in order):
    python data/prepare.py build                         # raw -> data/dataset.csv
    python data/prepare.py review-sheets --members ana beto carla
    #   ...each member fills data/review/<member>_revision.csv and <member>_ciego.csv
    python data/prepare.py apply-review                  # -> data/dataset_revisado.csv + kappa
    python data/prepare.py split                         # -> data/splits/{train,test}.csv

The gold label is never typed by hand: annotators mark the intents present and the
label is derived with the precedence rule (docs/guia_etiquetado.md §2).

Review sheets use ';' as separator and UTF-8 with BOM so they open directly in a
Spanish-locale Excel; keep that format when saving.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

RAW_PATH = config.RAW_DIR / "generations.jsonl"
REVIEWED_PATH = config.DATA_DIR / "dataset_revisado.csv"
SHEET_KW = {"sep": ";", "encoding": "utf-8-sig", "index": False}
READ_KW = {"sep": ";", "encoding": "utf-8-sig", "dtype": str, "keep_default_na": False}


def normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def conversation_text(turns):
    return "\n".join(f"{t['rol'].strip().capitalize()}: {t['texto'].strip()}" for t in turns)


# --- build ---------------------------------------------------------------------------


def cmd_build(args):
    rows, discarded = [], []
    with RAW_PATH.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    for rec in records:
        if rec.get("error"):
            continue
        expected = set(rec["intents"])
        for conv in rec.get("conversations", []):
            reason = None
            turns = conv.get("turnos") or []
            intents = set(conv.get("intenciones_presentes") or [])
            if not turns or any(not isinstance(t, dict) or not t.get("texto") for t in turns):
                reason = "malformed_turns"
            elif turns[0].get("rol", "").lower() != "cliente":
                reason = "does_not_start_with_client"
            elif not intents <= set(config.CATEGORIES):
                reason = "unknown_intent"
            elif intents != expected:
                reason = "intents_mismatch"
            text = conversation_text(turns) if reason != "malformed_turns" else json.dumps(conv, ensure_ascii=False)
            if reason:
                discarded.append({"call_id": rec["call_id"], "reason": reason, "texto": text})
                continue
            rows.append(
                {
                    "texto": text,
                    "categoria_gold": config.apply_precedence(intents),
                    "dificultad": rec["difficulty"],
                    "intenciones_presentes": config.format_intents(intents),
                    "n_turnos": len(turns),
                    "generador": rec["model"],
                    "call_id": rec["call_id"],
                }
            )

    df = pd.DataFrame(rows)
    df["_norm"] = df["texto"].map(normalize)
    dup = df.duplicated("_norm")
    for text, call_id in df.loc[dup, ["texto", "call_id"]].itertuples(index=False):
        discarded.append({"call_id": call_id, "reason": "duplicate", "texto": text})
    df = df.loc[~dup].drop(columns="_norm")

    df = df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    df.insert(0, "id", [f"c{i:04d}" for i in range(len(df))])
    df.to_csv(config.DATASET_PATH, index=False, encoding="utf-8")
    pd.DataFrame(discarded).to_csv(config.RAW_DIR / "discarded.csv", index=False, encoding="utf-8")

    print(f"Kept {len(df)} conversations -> {config.DATASET_PATH}")
    print(f"Discarded {len(discarded)} -> {config.RAW_DIR / 'discarded.csv'}")
    if discarded:
        print(pd.Series([d["reason"] for d in discarded]).value_counts().to_string())
    print("\nBy category x difficulty:")
    print(pd.crosstab(df["categoria_gold"], df["dificultad"], margins=True).to_string())


# --- review sheets -------------------------------------------------------------------


def cmd_review_sheets(args):
    df = pd.read_csv(config.DATASET_PATH, dtype=str)
    ids = df["id"].tolist()
    random.Random(config.SEED).shuffle(ids)
    n_shared = round(len(ids) * args.shared_frac)
    shared, rest = ids[:n_shared], ids[n_shared:]
    by_id = df.set_index("id")

    config.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (config.REVIEW_DIR / "compartidos.json").write_text(json.dumps(shared), encoding="utf-8")

    for k, member in enumerate(args.members):
        own = rest[k :: len(args.members)]
        sheet = by_id.loc[own, ["texto", "intenciones_presentes", "dificultad"]].reset_index()
        sheet["accion"] = "ok"  # ok | descartar
        sheet["intenciones_corregidas"] = ""
        sheet["dificultad_corregida"] = ""
        sheet["comentario"] = ""
        sheet.to_csv(config.REVIEW_DIR / f"{member}_revision.csv", **SHEET_KW)

        blind = by_id.loc[shared, ["texto"]].reset_index()
        blind = blind.sample(frac=1, random_state=k).reset_index(drop=True)  # order differs per member
        blind["intenciones"] = ""
        blind["descartar"] = ""
        blind.to_csv(config.REVIEW_DIR / f"{member}_ciego.csv", **SHEET_KW)
        print(f"{member}: {len(own)} to review + {len(shared)} blind")

    print(
        f"\nSheets in {config.REVIEW_DIR}. In 'intenciones' / 'intenciones_corregidas' write "
        f"codes separated by '|', e.g. 'pagos|envios'. Valid codes: {', '.join(config.CATEGORIES)}"
    )


# --- apply review --------------------------------------------------------------------


def _validated_intents(value, where):
    intents = config.parse_intents(value)
    bad = set(intents) - set(config.CATEGORIES)
    if bad:
        raise SystemExit(f"{where}: unknown intent code(s) {sorted(bad)}")
    return intents


def cmd_apply_review(args):
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

    df = pd.read_csv(config.DATASET_PATH, dtype=str).set_index("id")
    df["revisado_por"] = ""
    df["corregido"] = False
    drop_ids = set()

    revision_files = sorted(config.REVIEW_DIR.glob("*_revision.csv"))
    blind_files = sorted(config.REVIEW_DIR.glob("*_ciego.csv"))
    if not revision_files:
        raise SystemExit(f"No review sheets in {config.REVIEW_DIR}; run review-sheets first.")

    for path in revision_files:
        member = path.stem.removesuffix("_revision")
        sheet = pd.read_csv(path, **READ_KW)
        for row in sheet.itertuples(index=False):
            where = f"{path.name} id={row.id}"
            df.loc[row.id, "revisado_por"] = member
            if row.accion.strip().lower() == "descartar":
                drop_ids.add(row.id)
                continue
            if row.intenciones_corregidas.strip():
                intents = _validated_intents(row.intenciones_corregidas, where)
                df.loc[row.id, "intenciones_presentes"] = config.format_intents(intents)
                df.loc[row.id, "categoria_gold"] = config.apply_precedence(intents)
                df.loc[row.id, "corregido"] = True
                if len(intents) > 1:
                    df.loc[row.id, "dificultad"] = "multi_intencion"
                elif df.loc[row.id, "dificultad"] == "multi_intencion" and not row.dificultad_corregida.strip():
                    print(f"warning: {where} is now single-intent; set dificultad_corregida (defaulting to 'facil')")
                    df.loc[row.id, "dificultad"] = "facil"
            if row.dificultad_corregida.strip():
                if row.dificultad_corregida not in config.DIFFICULTIES:
                    raise SystemExit(f"{where}: unknown difficulty {row.dificultad_corregida!r}")
                df.loc[row.id, "dificultad"] = row.dificultad_corregida
                df.loc[row.id, "corregido"] = True

    # Blind shared subset: inter-annotator agreement and majority-vote labels.
    agreement = {}
    if len(blind_files) >= 2:
        labels, discard_votes = {}, {}
        for path in blind_files:
            member = path.stem.removesuffix("_ciego")
            sheet = pd.read_csv(path, **READ_KW)
            for row in sheet.itertuples(index=False):
                if row.descartar.strip():
                    discard_votes[row.id] = discard_votes.get(row.id, 0) + 1
                intents = _validated_intents(row.intenciones, f"{path.name} id={row.id}")
                if intents:
                    labels.setdefault(row.id, {})[member] = config.apply_precedence(intents)

        n_members = len(blind_files)
        complete = {i: v for i, v in labels.items() if len(v) == n_members}
        if complete:
            matrix = [[v[m] for m in sorted(v)] for v in complete.values()]
            codes = [[config.CATEGORIES.index(x) for x in row] for row in matrix]
            table, _ = aggregate_raters(codes, n_cat=len(config.CATEGORIES))
            disagreements = []
            gen_matches = 0
            for item_id, votes in complete.items():
                counts = pd.Series(list(votes.values())).value_counts()
                if counts.iloc[0] > n_members / 2:
                    majority = counts.index[0]
                    gen_matches += majority == df.loc[item_id, "categoria_gold"]
                    df.loc[item_id, "categoria_gold"] = majority
                    df.loc[item_id, "revisado_por"] = "ciego_mayoria"
                else:
                    disagreements.append({"id": item_id, **votes, "generador": df.loc[item_id, "categoria_gold"]})
            agreement = {
                "items": len(complete),
                "annotators": n_members,
                "fleiss_kappa": round(float(fleiss_kappa(table)), 3),
                "no_majority": len(disagreements),
                "generator_agrees_with_majority": round(gen_matches / max(len(complete) - len(disagreements), 1), 3),
            }
            pd.DataFrame(disagreements).to_csv(config.REVIEW_DIR / "desacuerdos.csv", **SHEET_KW)
        drop_ids |= {i for i, v in discard_votes.items() if v > n_members / 2}
        incomplete = len(labels) - len(complete)
        if incomplete:
            print(f"warning: {incomplete} blind items not labeled by every annotator (excluded from kappa)")
    else:
        print("warning: fewer than 2 blind sheets; skipping inter-annotator agreement")

    df = df.drop(index=sorted(drop_ids & set(df.index)))
    df.reset_index().to_csv(REVIEWED_PATH, index=False, encoding="utf-8")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "kept": len(df),
        "discarded_in_review": len(drop_ids),
        "corrected": int(df["corregido"].sum()),
        "agreement": agreement,
    }
    (config.RESULTS_DIR / "revision_humana.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if agreement.get("no_majority"):
        print(f"Discuss {agreement['no_majority']} items in {config.REVIEW_DIR / 'desacuerdos.csv'} "
              "(they keep the generator label) and update the labeling guide if a new criterion appears.")
    print(f"-> {REVIEWED_PATH}")


# --- split ---------------------------------------------------------------------------


def cmd_split(args):
    from sklearn.model_selection import train_test_split

    source = REVIEWED_PATH if REVIEWED_PATH.exists() else config.DATASET_PATH
    if source == config.DATASET_PATH:
        print("warning: no reviewed dataset found; splitting the unreviewed one")
    df = pd.read_csv(source, dtype={"id": str})
    strata = df["categoria_gold"] + "/" + df["dificultad"]
    train, test = train_test_split(df, test_size=args.test_size, stratify=strata, random_state=config.SEED)

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.sort_values("id").to_csv(config.TRAIN_PATH, index=False, encoding="utf-8")
    test.sort_values("id").to_csv(config.TEST_PATH, index=False, encoding="utf-8")
    digest = hashlib.sha256("\n".join(sorted(test["id"])).encode()).hexdigest()
    (config.SPLITS_DIR / "test_ids.sha256").write_text(digest + "\n", encoding="utf-8")

    print(f"Source: {source.name}; train={len(train)} test={len(test)}; test ids sha256={digest[:12]}")
    print(pd.crosstab(test["categoria_gold"], test["dificultad"], margins=True).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("review-sheets")
    p.add_argument("--members", nargs="+", required=True)
    p.add_argument("--shared-frac", type=float, default=0.12)
    sub.add_parser("apply-review")
    p = sub.add_parser("split")
    p.add_argument("--test-size", type=float, default=0.30)
    args = parser.parse_args()
    {"build": cmd_build, "review-sheets": cmd_review_sheets, "apply-review": cmd_apply_review, "split": cmd_split}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
