"""Classic baseline: TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression."""

import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

import config


def build_pipeline():
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), strip_accents="unicode", lowercase=True, min_df=1)),
            # Character n-grams within word boundaries: robust to typos and missing accents.
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode", lowercase=True, min_df=2)),
        ]
    )
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


PARAM_GRID = {
    "features__word__sublinear_tf": [True, False],
    "features__char__sublinear_tf": [True, False],
    "clf__C": [0.3, 1, 3, 10, 30],
}


class TfidfClassifier:
    provider = "tfidf"
    model = "tfidf+logreg"
    variant = "default"
    few_shot_ids = []

    def __init__(self):
        train = pd.read_csv(config.TRAIN_PATH, dtype={"id": str})
        search = GridSearchCV(
            build_pipeline(),
            PARAM_GRID,
            scoring="f1_macro",
            cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED),
            n_jobs=-1,
        )
        t0 = time.perf_counter()
        search.fit(train["texto"], train["categoria_gold"])
        self.train_seconds = round(time.perf_counter() - t0, 2)
        self.pipeline = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_f1_macro = round(search.best_score_, 4)

    def predict(self, text):
        t0 = time.perf_counter()
        probs = self.pipeline.predict_proba([text])[0]
        latency = round((time.perf_counter() - t0) * 1000, 3)
        classes = self.pipeline.classes_
        proba = {c: round(float(p), 5) for c, p in zip(classes, probs)}
        label = max(proba, key=proba.get)
        return {"label": label, "proba": proba, "raw": None, "error": None,
                "tokens_in": 0, "tokens_out": 0, "latency_ms": latency}

    def describe(self):
        return {"best_params": self.best_params, "cv_f1_macro": self.cv_f1_macro, "train_seconds": self.train_seconds}
