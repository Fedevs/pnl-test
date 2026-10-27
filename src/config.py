"""Shared configuration: label set, precedence rule, paths and model settings."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows consoles default to a legacy code page; print accents correctly.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REVIEW_DIR = DATA_DIR / "review"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DATASET_PATH = DATA_DIR / "dataset.csv"
TRAIN_PATH = SPLITS_DIR / "train.csv"
TEST_PATH = SPLITS_DIR / "test.csv"

load_dotenv(ROOT / ".env")

SEED = 42

# Ordered by business precedence (highest first). See docs/guia_etiquetado.md §2.
CATEGORIES = ["devoluciones", "pagos", "cuenta", "envios", "productos"]

CATEGORY_NAMES = {
    "devoluciones": "Devoluciones y reembolsos",
    "pagos": "Pagos y cobros",
    "cuenta": "Cuenta y acceso",
    "envios": "Envíos y entregas",
    "productos": "Productos y publicaciones",
}

# Short descriptions used in prompts (generator and classifiers).
CATEGORY_DESCRIPTIONS = {
    "devoluciones": (
        "Devolver, cambiar o cancelar una compra y el reembolso asociado: producto "
        "fallado, roto o equivocado, talle incorrecto, arrepentimiento, estado de un "
        "reembolso, cancelación de pedido."
    ),
    "pagos": (
        "Problemas o dudas con el dinero de una compra que no se devuelve: medios de "
        "pago, pago rechazado, cobro duplicado o no reconocido, cuotas e intereses, "
        "factura o comprobante."
    ),
    "cuenta": (
        "Acceso y datos de la cuenta: login, contraseña, código de verificación, cuenta "
        "bloqueada o suspendida, cambio de email o teléfono, datos personales, acceso no "
        "autorizado."
    ),
    "envios": (
        "Desde que se confirma la compra hasta que llega: plazos, seguimiento, demoras, "
        "dirección de entrega, costo de envío, retiro en sucursal, paquete marcado como "
        "entregado que no llegó."
    ),
    "productos": (
        "Información sobre productos antes o fuera de una compra concreta: "
        "características, stock, talles, compatibilidad, qué cubre la garantía, precio "
        "publicado, preguntas sobre una publicación, problemas al publicar (vendedores)."
    ),
}

DIFFICULTIES = ["facil", "ambigua", "disfluente", "multi_intencion"]


def apply_precedence(intents):
    """Return the gold label for a set of intents using the business precedence rule."""
    intents = set(intents)
    for cat in CATEGORIES:
        if cat in intents:
            return cat
    raise ValueError(f"No valid category in intents: {intents}")


def parse_intents(value):
    """Parse a pipe-separated intents string ('pagos|envios') into a sorted list."""
    if not isinstance(value, str) or not value.strip():
        return []
    return sorted({v.strip() for v in value.split("|") if v.strip()})


def format_intents(intents):
    return "|".join(sorted(set(intents)))


# --- Models (names are configurable: free-tier rosters change often) ---------------

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# Requests per minute we allow ourselves (kept below each free tier's limit).
RPM = {
    "gemini": int(os.getenv("GEMINI_RPM", "8")),
    "groq": int(os.getenv("GROQ_RPM", "25")),
    "mistral": int(os.getenv("MISTRAL_RPM", "20")),
}

# Paid-tier list prices in USD per 1M tokens (input, output), used only to report the
# "equivalent cost" of the free usage. VERIFY against each provider's pricing page and
# update before reporting; record the date checked in the report.
PRICES_USD_PER_M = {
    "gemini": (float(os.getenv("GEMINI_PRICE_IN", "0.30")), float(os.getenv("GEMINI_PRICE_OUT", "2.50"))),
    "groq": (float(os.getenv("GROQ_PRICE_IN", "0.15")), float(os.getenv("GROQ_PRICE_OUT", "0.60"))),
    "tfidf": (0.0, 0.0),
}


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing {name}. Copy .env.example to .env and fill it in.")
    return value
