"""Metrics, statistical tests, cascade routing analysis and figures.

Reads cached predictions from results/<split>/preds/*.jsonl and writes tables (CSV) and
figures (PNG) to results/<split>/. Called by run_eval.py; can also be run directly:

    python src/metrics.py --split test
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import config

# Fixed color per provider (identity follows the entity, never its rank) and marker per
# prompt variant as secondary encoding. Palette validated (light): all checks pass;
# aqua/yellow are below 3:1 contrast, so every point is direct-labeled.
COLORS = {"tfidf": "#2a78d6", "gemini": "#eb6834", "groq": "#1baf7a", "cascade": "#eda100"}
MARKERS = {"default": "o", "zs": "o", "zs_rule": "s", "fs_rule": "D"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

N_BOOT = 1000


# --- loading -------------------------------------------------------------------------


def split_paths(split):
    base = config.RESULTS_DIR / split
    return base, base / "preds", base / "figures"


def load_gold(split):
    path = config.TEST_PATH if split == "test" else config.SPLITS_DIR / f"{split}.csv"
    return pd.read_csv(path, dtype={"id": str}).set_index("id")


def load_predictions(preds_dir, gold):
    """Return {system: DataFrame indexed like gold} keeping the last record per id."""
    systems = {}
    for path in sorted(preds_dir.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        df = pd.DataFrame(records).drop_duplicates("id", keep="last").set_index("id")
        df = df.reindex(gold.index)  # missing ids -> NaN (count as not predicted)
        systems[path.stem.replace("__", ":").removesuffix(":default")] = df
    return systems


def provider_of(system):
    return system.split(":")[0]


def variant_of(system):
    return system.split(":")[1] if ":" in system else "default"


# --- metrics -------------------------------------------------------------------------


def _labels(pred):
    # Invalid / missing predictions become a sentinel that never matches gold.
    return pred["label"].fillna("__invalid__").astype(str)


def bootstrap_ci(y_true, y_pred, fn, rng):
    n = len(y_true)
    stats = [fn(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return np.percentile(stats, [2.5, 97.5])


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=config.CATEGORIES, average="macro", zero_division=0)


def cost_per_1000(provider, tokens_in, tokens_out):
    p_in, p_out = config.PRICES_USD_PER_M.get(provider, (np.nan, np.nan))
    return (np.nanmean(tokens_in) * p_in + np.nanmean(tokens_out) * p_out) / 1e6 * 1000


def summary_table(gold, systems):
    rng = np.random.default_rng(config.SEED)
    y_true = gold["categoria_gold"].to_numpy()
    rows = []
    for system, pred in systems.items():
        y_pred = _labels(pred).to_numpy()
        acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy_score, rng)
        f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, macro_f1, rng)
        tokens_in = pd.to_numeric(pred["tokens_in"], errors="coerce")
        tokens_out = pd.to_numeric(pred["tokens_out"], errors="coerce")
        latency = pd.to_numeric(pred["latency_ms"], errors="coerce")
        rows.append(
            {
                "system": system,
                "provider": provider_of(system),
                "variant": variant_of(system),
                "model": pred["model"].dropna().iloc[0] if pred["model"].notna().any() else None,
                "n": len(y_true),
                "predicted": int(pred["label"].notna().sum()),
                "invalid_or_missing_rate": round(float(pred["label"].isna().mean()), 4),
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "accuracy_ci95": f"[{acc_lo:.3f}, {acc_hi:.3f}]",
                "macro_f1": round(macro_f1(y_true, y_pred), 4),
                "macro_f1_lo": round(f1_lo, 4),
                "macro_f1_hi": round(f1_hi, 4),
                "latency_p50_ms": round(float(np.nanpercentile(latency, 50)), 2),
                "latency_p95_ms": round(float(np.nanpercentile(latency, 95)), 2),
                "mean_tokens_in": round(float(np.nanmean(tokens_in)), 1),
                "mean_tokens_out": round(float(np.nanmean(tokens_out)), 1),
                "usd_per_1000_equiv": round(float(cost_per_1000(provider_of(system), tokens_in, tokens_out)), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def per_class_table(gold, systems):
    rows = []
    for system, pred in systems.items():
        p, r, f, s = precision_recall_fscore_support(
            gold["categoria_gold"], _labels(pred), labels=config.CATEGORIES, zero_division=0
        )
        for cat, *vals in zip(config.CATEGORIES, p, r, f, s):
            rows.append({"system": system, "categoria": cat, "precision": vals[0], "recall": vals[1], "f1": vals[2], "support": vals[3]})
    return pd.DataFrame(rows).round(4)


def by_difficulty_table(gold, systems):
    rows = []
    for system, pred in systems.items():
        y_pred = _labels(pred)
        for diff in config.DIFFICULTIES:
            mask = gold["dificultad"] == diff
            if not mask.any():
                continue
            rows.append(
                {
                    "system": system,
                    "dificultad": diff,
                    "n": int(mask.sum()),
                    "accuracy": round(accuracy_score(gold.loc[mask, "categoria_gold"], y_pred[mask]), 4),
                    "macro_f1": round(macro_f1(gold.loc[mask, "categoria_gold"].to_numpy(), y_pred[mask].to_numpy()), 4),
                }
            )
    return pd.DataFrame(rows)


def multi_intent_table(gold, systems):
    """On multi-intent items, separate 'picked another present intent' (ignored the
    precedence rule) from 'picked an intent that was not there'."""
    multi = gold[gold["dificultad"] == "multi_intencion"]
    rows = []
    for system, pred in systems.items():
        y_pred = _labels(pred).loc[multi.index]
        present = multi["intenciones_presentes"].map(config.parse_intents)
        correct = y_pred == multi["categoria_gold"]
        other_present = [(p in ints) and not c for p, ints, c in zip(y_pred, present, correct)]
        rows.append(
            {
                "system": system,
                "n_multi": len(multi),
                "correct": round(float(correct.mean()), 4) if len(multi) else np.nan,
                "precedence_violation": round(float(np.mean(other_present)), 4) if len(multi) else np.nan,
                "absent_intent": abs(round(float(1 - correct.mean() - np.mean(other_present)), 4)) if len(multi) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def mcnemar_table(gold, systems):
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests

    y_true = gold["categoria_gold"]
    hits = {s: (_labels(p) == y_true).to_numpy() for s, p in systems.items()}
    rows = []
    for a, b in itertools.combinations(hits, 2):
        both = np.sum(hits[a] & hits[b])
        only_a = np.sum(hits[a] & ~hits[b])
        only_b = np.sum(~hits[a] & hits[b])
        neither = np.sum(~hits[a] & ~hits[b])
        res = mcnemar([[both, only_a], [only_b, neither]], exact=True)
        rows.append({"system_a": a, "system_b": b, "only_a_correct": only_a, "only_b_correct": only_b, "p_value": res.pvalue})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = multipletests(df["p_value"], method="holm")[1]
        df["significant_0.05"] = df["p_holm"] < 0.05
    return df.round(5)


def cascade_table(gold, systems, thresholds=np.round(np.arange(0.0, 1.0001, 0.05), 2)):
    """TF-IDF answers when its top probability >= threshold, otherwise defer to the LLM."""
    if "tfidf" not in systems:
        return pd.DataFrame()
    tf = systems["tfidf"]
    tf_conf = tf["proba"].map(lambda p: max(p.values()) if isinstance(p, dict) else 0.0)
    tf_lat = pd.to_numeric(tf["latency_ms"], errors="coerce").fillna(0)
    y_true = gold["categoria_gold"].to_numpy()
    rows = []
    for system, llm in systems.items():
        if system == "tfidf":
            continue
        llm_cost = cost_per_1000(provider_of(system), pd.to_numeric(llm["tokens_in"], errors="coerce"),
                                 pd.to_numeric(llm["tokens_out"], errors="coerce"))
        llm_lat = pd.to_numeric(llm["latency_ms"], errors="coerce").fillna(0)
        for t in thresholds:
            use_tfidf = (tf_conf >= t).to_numpy()
            y_pred = np.where(use_tfidf, _labels(tf), _labels(llm))
            llm_share = 1 - use_tfidf.mean()
            rows.append(
                {
                    "llm_system": system,
                    "threshold": t,
                    "llm_share": round(float(llm_share), 4),
                    "accuracy": round(accuracy_score(y_true, y_pred), 4),
                    "macro_f1": round(macro_f1(y_true, y_pred), 4),
                    "usd_per_1000_equiv": round(float(llm_share * llm_cost), 4),
                    "mean_latency_ms": round(float(np.mean(tf_lat + np.where(use_tfidf, 0, llm_lat))), 1),
                }
            )
    return pd.DataFrame(rows)


def calibration_table(gold, pred, n_bins=10):
    conf = pred["proba"].map(lambda p: max(p.values()) if isinstance(p, dict) else np.nan)
    correct = (_labels(pred) == gold["categoria_gold"]).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin_lo": bins[b], "bin_hi": bins[b + 1], "n": int(mask.sum()),
                         "mean_confidence": float(conf[mask].mean()), "accuracy": float(correct[mask].mean())})
    df = pd.DataFrame(rows)
    ece = float((df["n"] * (df["mean_confidence"] - df["accuracy"]).abs()).sum() / df["n"].sum()) if len(df) else np.nan
    return df.round(4), round(ece, 4)


def error_sheets(gold, systems, out_dir):
    """One CSV of errors per system (with an empty 'tipo_error' column for the manual
    taxonomy) plus the items that every system gets wrong."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wrong_everywhere = pd.Series(True, index=gold.index)
    for system, pred in systems.items():
        y_pred = _labels(pred)
        wrong = y_pred != gold["categoria_gold"]
        wrong_everywhere &= wrong
        sheet = gold.loc[wrong, ["dificultad", "categoria_gold", "intenciones_presentes", "texto"]].copy()
        sheet.insert(2, "prediccion", y_pred[wrong])
        sheet["tipo_error"] = ""
        sheet.reset_index().to_csv(out_dir / f"{system.replace(':', '__')}.csv", sep=";", encoding="utf-8-sig", index=False)
    gold.loc[wrong_everywhere].reset_index().to_csv(out_dir / "fallan_todos.csv", sep=";", encoding="utf-8-sig", index=False)


# --- figures -------------------------------------------------------------------------


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
            "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
            "axes.spines.top": False, "axes.spines.right": False,
            "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
            "legend.frameon": False, "lines.linewidth": 2,
        }
    )


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def fig_confusion(gold, systems, path):
    import matplotlib.pyplot as plt

    n = len(systems)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    for ax, (system, pred) in zip(axes.flat, systems.items()):
        cm = confusion_matrix(gold["categoria_gold"], _labels(pred), labels=config.CATEGORIES)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap=_seq_cmap(), vmin=0, vmax=1)
        ax.grid(False)
        for i, j in itertools.product(range(len(config.CATEGORIES)), repeat=2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="#ffffff" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(config.CATEGORIES)), config.CATEGORIES, rotation=40, ha="right")
        ax.set_yticks(range(len(config.CATEGORIES)), config.CATEGORIES)
        ax.set_title(system)
        ax.set_xlabel("predicción")
        ax.set_ylabel("gold")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Matrices de confusión (color = proporción por fila, número = casos)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _scatter_systems(summary, x_col, x_label, title, path, log_x=True):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for row in summary.itertuples():
        color = COLORS.get(row.provider, MUTED)
        x = getattr(row, x_col)
        ax.errorbar(x, row.macro_f1, yerr=[[row.macro_f1 - row.macro_f1_lo], [row.macro_f1_hi - row.macro_f1]],
                    fmt=MARKERS.get(row.variant, "o"), color=color, markersize=8, elinewidth=1.2, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(row.system, (x, row.macro_f1), xytext=(7, 4), textcoords="offset points", fontsize=8.5, color=INK_2)
    if log_x:
        ax.set_xscale("symlog", linthresh=1e-3 if x_col.startswith("usd") else 1)
    # Room on the right for the direct labels of the rightmost points.
    ax.set_xlim(right=max(summary[x_col].max(), 1e-3) * 6)
    ax.set_xlabel(x_label)
    ax.set_ylabel("macro-F1 (IC 95% bootstrap)")
    ax.set_title(title, loc="left")
    handles = [Line2D([], [], color=COLORS[p], marker="o", linestyle="", markersize=8, label=p)
               for p in summary["provider"].unique() if p in COLORS]
    handles += [Line2D([], [], color=MUTED, marker=MARKERS[v], linestyle="", markersize=7, label=v)
                for v in summary["variant"].unique() if v in MARKERS and v != "default"]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_by_difficulty(table, path):
    import matplotlib.pyplot as plt

    pivot = table.pivot(index="system", columns="dificultad", values="accuracy")
    pivot = pivot[[d for d in config.DIFFICULTIES if d in pivot.columns]]
    # Scores cluster near the top, so the color scale starts at the floor of the data
    # (at most 0.5) instead of 0; the numbers carry the exact values.
    vmin = min(0.5, np.floor(np.nanmin(pivot.to_numpy()) * 10) / 10)
    fig, ax = plt.subplots(figsize=(1.6 * pivot.shape[1] + 3.2, 0.5 * pivot.shape[0] + 1.6))
    im = ax.imshow(pivot.to_numpy(), cmap=_seq_cmap(), vmin=vmin, vmax=1, aspect="auto")
    ax.grid(False)
    for i, j in itertools.product(range(pivot.shape[0]), range(pivot.shape[1])):
        v = pivot.iat[i, j]
        dark = (v - vmin) / (1 - vmin) > 0.55
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#ffffff" if dark else INK)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED)
    ax.set_title("Accuracy por nivel de dificultad (failure modes)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_cascade(cascade, summary, path):
    import matplotlib.pyplot as plt

    # One curve per LLM provider, using its best variant, to keep the figure readable.
    llm = summary[summary["provider"] != "tfidf"]
    best = llm.sort_values("macro_f1", ascending=False).drop_duplicates("provider")["system"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for system in best:
        c = cascade[cascade["llm_system"] == system].sort_values("llm_share")
        color = COLORS.get(provider_of(system), MUTED)
        ax.plot(c["llm_share"] * 100, c["macro_f1"], color=color, marker="o", markersize=4, label=f"TF-IDF → {system}")
        last = c.iloc[-1]
        # Direct label at the right end (100% = pure LLM), outside the plotted lines.
        ax.annotate(system, (last["llm_share"] * 100, last["macro_f1"]), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(-3, 125)
    ax.set_xticks(range(0, 101, 20))
    ax.legend(loc="upper right", fontsize=8.5)
    if "tfidf" in set(summary["system"]):
        tf = summary.set_index("system").loc["tfidf", "macro_f1"]
        ax.axhline(tf, color=COLORS["tfidf"], linewidth=1, linestyle="--")
        ax.annotate("solo TF-IDF", (0, tf), xytext=(4, -12), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xlabel("% de consultas derivadas al LLM")
    ax.set_ylabel("macro-F1")
    ax.set_title("Routing en cascada: TF-IDF resuelve si está seguro, si no deriva al LLM", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_calibration(calib, ece, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("calibración perfecta", (0.62, 0.55), fontsize=8.5, color=MUTED, rotation=38)
    ax.plot(calib["mean_confidence"], calib["accuracy"], color=COLORS["tfidf"], marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("confianza media (max predict_proba)")
    ax.set_ylabel("accuracy observada")
    ax.set_title(f"Calibración TF-IDF (ECE = {ece:.3f})", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --- entry point ---------------------------------------------------------------------


def run(split="test", base=None, limit=None):
    """Compute everything for a split. `base`/`limit` point at a smoke-test subset."""
    if base is None:
        base, preds_dir, fig_dir = split_paths(split)
    else:
        preds_dir, fig_dir = base / "preds", base / "figures"
    gold = load_gold(split)
    if limit:
        gold = gold.head(limit)
    systems = load_predictions(preds_dir, gold)
    if not systems:
        raise SystemExit(f"No predictions in {preds_dir}; run src/run_eval.py first.")
    fig_dir.mkdir(parents=True, exist_ok=True)
    _style()

    summary = summary_table(gold, systems)
    summary.to_csv(base / "summary.csv", index=False)
    per_class_table(gold, systems).to_csv(base / "per_class.csv", index=False)
    diff = by_difficulty_table(gold, systems)
    diff.to_csv(base / "by_difficulty.csv", index=False)
    multi_intent_table(gold, systems).to_csv(base / "multi_intent.csv", index=False)
    if len(systems) > 1:
        mcnemar_table(gold, systems).to_csv(base / "mcnemar.csv", index=False)
    error_sheets(gold, systems, base / "errores")

    fig_confusion(gold, systems, fig_dir / "confusion.png")
    fig_by_difficulty(diff, fig_dir / "by_difficulty.png")
    _scatter_systems(summary, "usd_per_1000_equiv", "USD equivalentes por 1.000 consultas (escala log)",
                     "Calidad vs costo", fig_dir / "quality_vs_cost.png")
    _scatter_systems(summary, "latency_p50_ms", "latencia p50 en ms (escala log)",
                     "Calidad vs latencia", fig_dir / "quality_vs_latency.png")

    if "tfidf" in systems:
        calib, ece = calibration_table(gold, systems["tfidf"])
        calib.to_csv(base / "calibration_tfidf.csv", index=False)
        fig_calibration(calib, ece, fig_dir / "calibration_tfidf.png")
        cascade = cascade_table(gold, systems)
        if len(cascade):
            cascade.to_csv(base / "cascade.csv", index=False)
            fig_cascade(cascade, summary, fig_dir / "cascade.png")

    cols = ["system", "accuracy", "macro_f1", "invalid_or_missing_rate", "latency_p50_ms", "usd_per_1000_equiv"]
    print(summary[cols].to_string(index=False))
    print(f"\nTables -> {base}\nFigures -> {fig_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    run(parser.parse_args().split)
