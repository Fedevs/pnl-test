# Clasificación de consultas de atención al cliente — TP NLP Grupo 2

Comparación de enfoques para clasificar consultas de e-commerce en castellano (rioplatense, informal) en 5 categorías:

| Código | Categoría |
|---|---|
| `devoluciones` | Devoluciones y reembolsos |
| `pagos` | Pagos y cobros |
| `cuenta` | Cuenta y acceso |
| `envios` | Envíos y entregas |
| `productos` | Productos y publicaciones |

**Enfoques comparados** (todos con herramientas gratuitas):

| Rol | Modelo |
|---|---|
| Generador del dataset sintético | Mistral (free tier), de otra familia que los clasificadores |
| Clasificador LLM 1 | Gemini Flash (Google AI Studio, free tier), salida estructurada con enum |
| Clasificador LLM 2 | Modelo open-weights vía Groq (free tier), JSON mode |
| Baseline clásica | TF-IDF (palabras 1-2 + caracteres 3-5) + regresión logística |
| Routing en cascada | TF-IDF responde si su confianza ≥ umbral; si no, deriva al LLM |

Variantes de prompt para los LLM: `zs` (zero-shot con descripciones), `zs_rule` (+ regla de prelación) y `fs_rule` (+ 6 ejemplos del train).

Regla multi-intención (**prelación de negocio**): `devoluciones > pagos > cuenta > envios > productos`. Ver [docs/guia_etiquetado.md](docs/guia_etiquetado.md).

## Estructura

```
data/generate.py          generación con Mistral (reanudable)       -> data/raw/generations.jsonl
data/prepare.py           build / review-sheets / apply-review / split
docs/guia_etiquetado.md   criterios de etiquetado y revisión humana
src/config.py             categorías, prelación, modelos, precios
src/prompts.py            variantes de prompt
src/classifiers/          tfidf.py, gemini.py, groq_llm.py (interfaz común predict)
src/run_eval.py           corre los clasificadores con caché y calcula métricas
src/metrics.py            métricas, IC bootstrap, McNemar, cascada, calibración, figuras
src/perturb.py            (opcional) test de robustez con ruido controlado
notebooks/analisis.ipynb  tablas y figuras para el informe
results/<split>/          predicciones cacheadas, tablas CSV, figuras PNG, errores
```

## Setup

```bash
py -m venv .venv                  # Windows (en Linux/Mac/Colab: python -m venv .venv)
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Linux/Mac: cp .env.example .env  -> completar las API keys
```

API keys gratuitas:
- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys
- Mistral: https://console.mistral.ai/api-keys

Los límites y los modelos de los free tiers cambian seguido. Hay que verificarlos en la consola de cada proveedor y, si hace falta, ajustar `*_MODEL` / `*_RPM` en `.env`. En el informe va la versión exacta de cada modelo (queda en `results/<split>/run_meta.json`).

## Flujo completo

```bash
# 1. Dataset
python data/generate.py --dry-run                 # ver el plan (~150 llamadas, ~1150 conversaciones)
python data/generate.py --limit-calls 2           # probar con 2 llamadas y revisar la salida
python data/generate.py                           # generar todo (si se corta, re-ejecutar: retoma)
python data/prepare.py build                      # validar, deduplicar -> data/dataset.csv
python data/prepare.py review-sheets --members ana beto carla
#    ...cada integrante completa sus planillas en data/review/ (ver la guía, §5)
python data/prepare.py apply-review               # -> data/dataset_revisado.csv + kappa de Fleiss
python data/prepare.py split                      # 70/30 estratificado -> data/splits/

# 2. Evaluación
python src/run_eval.py --limit 20                 # smoke test (resultados aparte en results/test_smoke20/)
python src/run_eval.py                            # todo: tfidf + gemini/groq x 3 variantes
python src/run_eval.py --from-cache               # regenerar métricas y figuras sin llamar a las APIs

# 3. (Opcional) robustez
python src/perturb.py
python src/run_eval.py --split perturbed
```

`run_eval.py` guarda cada predicción en `results/<split>/preds/*.jsonl` apenas la recibe. Si se alcanza el límite diario del free tier, se vuelve a ejecutar el mismo comando al día siguiente y solo se predice lo que falta (unas 300 × 3 = 900 llamadas por proveedor).

## Salidas (`results/test/`)

| Archivo | Contenido |
|---|---|
| `summary.csv` | accuracy, macro-F1 (IC 95%), tasa de inválidos, latencia p50/p95, tokens, USD equivalentes / 1.000 consultas |
| `per_class.csv` | precision / recall / F1 por categoría |
| `by_difficulty.csv` | accuracy y macro-F1 por dificultad (failure modes) |
| `multi_intent.csv` | en multi-intención: aciertos, violaciones de la prelación, intención inexistente |
| `mcnemar.csv` | test de McNemar exacto entre cada par de sistemas (corrección de Holm) |
| `cascade.csv` | cascada TF-IDF → LLM para cada umbral: calidad, % derivado, costo, latencia |
| `calibration_tfidf.csv` | diagrama de confiabilidad y ECE de TF-IDF |
| `errores/*.csv` | errores por sistema, con la columna `tipo_error` para la taxonomía manual, y `fallan_todos.csv` |
| `figures/*.png` | matrices de confusión, calidad vs costo, calidad vs latencia, dificultad, cascada, calibración |
