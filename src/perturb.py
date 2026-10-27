"""Optional robustness test: controlled noise on the easy test items.

Takes the `facil` items of the test split and applies, with a fixed seed: lowercasing,
accent removal, dropped punctuation and random character typos (swap / drop / duplicate)
in the client's turns. Labels are unchanged, so any accuracy drop is due to the noise.

    python src/perturb.py                    # -> data/splits/perturbed.csv
    python src/run_eval.py --split perturbed
"""

import argparse
import random
import re
import unicodedata

import pandas as pd

import config


def strip_accents(text):
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def add_typos(text, rate, rng):
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch.isalpha() and rng.random() < rate:
            op = rng.choice(["swap", "drop", "dup"])
            if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                out += [chars[i + 1], ch]
                i += 2
                continue
            if op == "dup":
                out += [ch, ch]
            # "drop": skip the char
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def perturb_client_turns(text, rate, rng):
    lines = []
    for line in text.split("\n"):
        if line.startswith("Cliente:"):
            body = line[len("Cliente:"):]
            body = strip_accents(body.lower())
            body = re.sub(r"[¿¡.,;:!?]", "", body)
            body = add_typos(body, rate, rng)
            line = "Cliente:" + body
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--typo-rate", type=float, default=0.04, help="per-letter probability of a typo")
    args = parser.parse_args()

    rng = random.Random(config.SEED)
    test = pd.read_csv(config.TEST_PATH, dtype={"id": str})
    easy = test[test["dificultad"] == "facil"].copy()
    easy["texto_original"] = easy["texto"]
    easy["texto"] = [perturb_client_turns(t, args.typo_rate, rng) for t in easy["texto"]]
    out = config.SPLITS_DIR / "perturbed.csv"
    easy.to_csv(out, index=False, encoding="utf-8")
    print(f"{len(easy)} perturbed items -> {out}\n\nExample:\n{easy['texto'].iloc[0]}")


if __name__ == "__main__":
    main()
