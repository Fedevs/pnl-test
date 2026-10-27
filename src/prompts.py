"""Classification prompt variants shared by every LLM classifier.

    zs       zero-shot with category descriptions
    zs_rule  zs + explicit business precedence rule for multi-intent queries
    fs_rule  zs_rule + k labeled examples taken from the train split
"""

import random

import pandas as pd

import config

VARIANTS = ["zs", "zs_rule", "fs_rule"]

BASE = """Sos un sistema que clasifica consultas de atención al cliente de un e-commerce. Vas a recibir una conversación entre un cliente y un agente. Tenés que asignarle exactamente UNA categoría según el motivo de contacto del cliente.

Categorías:
{categories}
"""

MAIN_INTENT = "Elegí la categoría que mejor represente el motivo principal de contacto del cliente.\n"

RULE = """Si el cliente plantea más de un pedido o pregunta concreta, asigná la categoría de mayor prioridad según este orden de negocio:
devoluciones > pagos > cuenta > envios > productos
Las menciones de contexto que no son un pedido (por ejemplo "lo pagué con tarjeta") no cuentan como intención.
"""

OUTPUT = 'Respondé SOLO con un objeto JSON: {{"categoria": "<código>"}}, donde <código> es uno de: {codes}.'


def _categories_block():
    return "\n".join(f"- {c}: {config.CATEGORY_DESCRIPTIONS[c]}" for c in config.CATEGORIES)


def select_examples(train_df, k_per_class=1, seed=config.SEED):
    """Pick few-shot examples: k per class plus one multi-intent case, fixed by seed."""
    rng = random.Random(seed)
    picked = []
    for cat in config.CATEGORIES:
        pool = train_df[(train_df["categoria_gold"] == cat) & (train_df["dificultad"] != "multi_intencion")]
        picked += rng.sample(list(pool.index), min(k_per_class, len(pool)))
    multi = train_df[train_df["dificultad"] == "multi_intencion"]
    if len(multi):
        picked.append(rng.choice(list(multi.index)))
    rng.shuffle(picked)
    return train_df.loc[picked]


def system_prompt(variant, examples=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    parts = [BASE.format(categories=_categories_block())]
    parts.append(RULE if variant in ("zs_rule", "fs_rule") else MAIN_INTENT)
    if variant == "fs_rule":
        if examples is None or not len(examples):
            raise ValueError("fs_rule needs examples from the train split")
        shots = "\n\n".join(
            f"Conversación:\n{row.texto}\nRespuesta: {{\"categoria\": \"{row.categoria_gold}\"}}"
            for row in examples.itertuples()
        )
        parts.append("Ejemplos:\n\n" + shots + "\n")
    parts.append(OUTPUT.format(codes=", ".join(config.CATEGORIES)))
    return "\n".join(parts)


def user_prompt(text):
    return f"Conversación:\n{text}"


def load_examples():
    train = pd.read_csv(config.TRAIN_PATH, dtype={"id": str})
    return select_examples(train)
