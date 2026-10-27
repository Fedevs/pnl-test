"""Generate the synthetic conversation dataset with Mistral (free tier).

The generator is from a different model family than the classifiers (Gemini, Groq
open-weights) so that we measure the task and not model-to-model affinity.

Usage:
    python data/generate.py --dry-run         # show the call plan and one prompt
    python data/generate.py                   # run (resumable: skips finished calls)
    python data/generate.py --limit-calls 3   # quick test with a few calls

Output: data/raw/generations.jsonl (one line per API call). Then run
`python data/prepare.py build` to turn it into data/dataset.csv.
"""

import argparse
import itertools
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from llm_utils import RateLimiter, api_retry, parse_json  # noqa: E402

RAW_PATH = config.RAW_DIR / "generations.jsonl"

CHANNELS = ["chat de la web", "WhatsApp", "email", "chat de la app", "mensaje directo en redes sociales"]
TONES = ["neutral", "apurado", "enojado", "amable", "confundido", "irónico", "preocupado"]
DOMAINS = [
    "celulares", "notebooks", "electrodomésticos", "ropa", "zapatillas", "muebles",
    "juguetes", "herramientas", "cosmética", "repuestos de auto", "libros", "bicicletas",
    "videojuegos", "artículos de cocina", "suplementos deportivos",
]
VARIETIES = ["rioplatense (voseo)"] * 7 + ["español neutro"] * 3

DIFFICULTY_INSTRUCTIONS = {
    "facil": (
        "El cliente tiene UNA sola intención, expresada de forma clara y explícita, con "
        "lenguaje estándar."
    ),
    "ambigua": (
        "El cliente tiene UNA sola intención real, pero la expresa de forma vaga o "
        "indirecta (\"tengo un problema con mi compra\", \"esto no puede ser\"), o usando "
        "palabras que suenan a otra categoría. La intención debe poder deducirse leyendo "
        "toda la conversación, pero no a partir de palabras clave obvias."
    ),
    "disfluente": (
        "El cliente tiene UNA sola intención, pero escribe con ruido: errores de tipeo, sin "
        "tildes, abreviaturas (q, xq, x favor, tmb), todo en minúscula o en mayúscula, "
        "mensajes cortados en varias partes, o estilo de audio transcrito con muletillas."
    ),
    "multi_intencion": (
        "El cliente plantea DOS intenciones distintas y ambas son pedidos o preguntas "
        "concretas (no simple contexto). Variá el orden: a veces menciona primero y con más "
        "énfasis la intención menos grave."
    ),
}

SYSTEM_PROMPT = """Sos un generador de datos sintéticos para entrenar y evaluar clasificadores de consultas de atención al cliente de un e-commerce argentino (marketplace con compradores y vendedores).

Categorías posibles de intención del cliente:
{categories}

Reglas para TODAS las conversaciones:
- Son conversaciones escritas entre un cliente y un agente humano de atención, con la cantidad de turnos indicada, empezando siempre por el cliente.
- El agente NUNCA nombra áreas, categorías ni derivaciones ("te paso con devoluciones") y no resuelve el caso: solo saluda, pide datos (número de pedido, etc.), repregunta o confirma que entendió.
- No incluyas datos personales reales: inventá nombres, números de pedido y montos verosímiles (en pesos argentinos).
- Cada conversación tiene que ser distinta de las demás en situación, vocabulario y estructura. Evitá fórmulas repetidas y empezar siempre con "Hola".
- `intenciones_presentes` lista TODAS las intenciones accionables del cliente, usando exactamente los códigos de las categorías.

Respondé SOLO con un objeto JSON con esta forma:
{{"conversaciones": [{{"spec": <número de spec>, "turnos": [{{"rol": "cliente" | "agente", "texto": "..."}}], "intenciones_presentes": ["<código>", ...]}}]}}"""

USER_PROMPT = """Generá {n} conversaciones.

Intención(es) que debe tener el cliente: {intents}
Nivel de dificultad: {difficulty}. {difficulty_instructions}

Especificación de cada conversación:
{specs}"""


def build_plan(per_class, multi_share, overshoot, batch_size, seed):
    """Return the list of generation calls needed for a class-balanced dataset."""
    rng = random.Random(seed)
    pairs = list(itertools.combinations(config.CATEGORIES, 2))
    total = per_class * len(config.CATEGORIES)
    per_pair = round(total * multi_share / len(pairs))

    cells = []  # (difficulty, intents, n)
    multi_wins = {c: 0 for c in config.CATEGORIES}
    for pair in pairs:
        cells.append(("multi_intencion", list(pair), per_pair))
        multi_wins[config.apply_precedence(pair)] += per_pair

    # Fill each class up to per_class with single-intent items, split 50/15/15 like the
    # overall target proportions (rescaled to the single-intent share).
    single_split = {"facil": 50, "ambigua": 15, "disfluente": 15}
    weight_total = sum(single_split.values())
    for cat in config.CATEGORIES:
        remaining = per_class - multi_wins[cat]
        for diff, w in single_split.items():
            cells.append((diff, [cat], round(remaining * w / weight_total)))

    calls = []
    for difficulty, intents, n in cells:
        n = round(n * overshoot)
        for i, start in enumerate(range(0, n, batch_size)):
            size = min(batch_size, n - start)
            specs = [
                {
                    "spec": j + 1,
                    "turnos": rng.randint(2, 4),
                    "canal": rng.choice(CHANNELS),
                    "tono": rng.choice(TONES),
                    "rubro": rng.choice(DOMAINS),
                    "variedad": rng.choice(VARIETIES),
                }
                for j in range(size)
            ]
            calls.append(
                {
                    "call_id": f"{difficulty}-{'+'.join(intents)}-{i:02d}",
                    "difficulty": difficulty,
                    "intents": intents,
                    "specs": specs,
                }
            )
    return calls


def render_prompts(call):
    categories = "\n".join(
        f"- {c}: {config.CATEGORY_NAMES[c]}. {config.CATEGORY_DESCRIPTIONS[c]}" for c in config.CATEGORIES
    )
    specs = "\n".join(
        f"{s['spec']}. {s['turnos']} turnos en total; canal: {s['canal']}; tono del cliente: "
        f"{s['tono']}; rubro: {s['rubro']}; variedad del español: {s['variedad']}."
        for s in call["specs"]
    )
    system = SYSTEM_PROMPT.format(categories=categories)
    user = USER_PROMPT.format(
        n=len(call["specs"]),
        intents=" y ".join(call["intents"]),
        difficulty=call["difficulty"],
        difficulty_instructions=DIFFICULTY_INSTRUCTIONS[call["difficulty"]],
        specs=specs,
    )
    return system, user


def done_call_ids():
    if not RAW_PATH.exists():
        return set()
    done = set()
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("error"):
                done.add(rec["call_id"])
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--multi-share", type=float, default=0.20)
    parser.add_argument("--overshoot", type=float, default=1.15, help="extra to absorb discards in review")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--limit-calls", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args.per_class, args.multi_share, args.overshoot, args.batch_size, config.SEED)
    n_convs = sum(len(c["specs"]) for c in plan)
    print(f"Plan: {len(plan)} calls, {n_convs} conversations requested")

    if args.dry_run:
        by_diff = {}
        for c in plan:
            by_diff[c["difficulty"]] = by_diff.get(c["difficulty"], 0) + len(c["specs"])
        print("By difficulty:", by_diff)
        system, user = render_prompts(plan[0])
        print("\n--- SYSTEM ---\n" + system + "\n\n--- USER ---\n" + user)
        return

    from mistralai.client import Mistral

    client = Mistral(api_key=config.require_env("MISTRAL_API_KEY"))
    limiter = RateLimiter(config.RPM["mistral"])

    @api_retry
    def call_api(system, user):
        limiter.wait()
        return client.chat.complete(
            model=config.MISTRAL_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=args.temperature,
        )

    done = done_call_ids()
    pending = [c for c in plan if c["call_id"] not in done]
    if args.limit_calls:
        pending = pending[: args.limit_calls]
    print(f"Already done: {len(done)}; running now: {len(pending)}")

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for k, call in enumerate(pending, 1):
        system, user = render_prompts(call)
        record = {**call, "model": config.MISTRAL_MODEL, "temperature": args.temperature}
        t0 = time.perf_counter()
        try:
            resp = call_api(system, user)
            content = resp.choices[0].message.content
            record["raw"] = content
            record["conversations"] = parse_json(content).get("conversaciones", [])
        except Exception as exc:  # keep going; failed calls are retried on the next run
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.perf_counter() - t0, 2)
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        with RAW_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        status = record.get("error") or f"{len(record['conversations'])} convs"
        print(f"[{k}/{len(pending)}] {call['call_id']}: {status}")


if __name__ == "__main__":
    main()
