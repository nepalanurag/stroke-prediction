"""Threshold selection under a validation protocol (stroke-prediction follow-up).

Follow-up robustness check on the threshold analysis in threshold_calibration.py.
That script tunes the decision threshold by sweeping F1 on the held-out test
set, so the reported F1 is subject to test-set selection optimism. Here the
threshold is tuned on a separate validation split and reported exactly once on
a locked test split:

    60% train / 20% validation / 20% test, seed 42, stratified.

- Train: fit the logistic model (the best-AUC model in the original analysis).
- Validation: sweep thresholds, pick the F1-maximizing threshold.
- Test: report precision/recall/F1 once at the chosen threshold, plus at 0.5
  for reference.

Run: python3 analysis/threshold_validation_protocol.py   (from the repo root)
Writes: analysis/threshold_validation.json, docs/figs/stroke_threshold_validation.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

df = pd.read_csv(os.path.join(ROOT, "healthcare-dataset-stroke-data.csv"))
df = df.drop(columns=["id"])
y = df["stroke"].values
X = df.drop(columns=["stroke"])

num_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# 60/20/20: first carve off 40% as (validation + test), then split that half
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)

pipe = Pipeline([("pre", pre),
                 ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
pipe.fit(X_train, y_train)
p_val = pipe.predict_proba(X_val)[:, 1]
p_test = pipe.predict_proba(X_test)[:, 1]


def prf(y_true, p, t):
    yp = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
    return {"threshold": float(t),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(f1_score(y_true, yp)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


ths = np.linspace(0.05, 0.95, 37)
val_rows = [prf(y_val, p_val, t) for t in ths]
val_f1 = np.array([r["f1"] for r in val_rows])
t_star = float(ths[int(np.argmax(val_f1))])

out = {
    "protocol": "60/20/20 train/validation/test, seed 42, stratified; "
                "threshold tuned on validation only, reported once on locked test",
    "seed": SEED,
    "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    "chosen_threshold": t_star,
    "validation_best": next(r for r in val_rows if r["threshold"] == t_star),
    "test_at_chosen": prf(y_test, p_test, t_star),
    "test_at_0.5": prf(y_test, p_test, 0.5),
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ths, val_f1, label="F1 (validation)")
ax.axvline(t_star, color="C3", linestyle=":", label=f"chosen @ {t_star:.2f}")
ax.axvline(0.5, color="k", linestyle="--", alpha=0.5, label="0.5 default")
ax.set_xlabel("decision threshold")
ax.set_ylabel("F1")
ax.set_title("Threshold selection on the validation split (locked test untouched)")
ax.legend(fontsize=9)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, "docs", "figs"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "docs", "figs", "stroke_threshold_validation.png"), dpi=110)
plt.close(fig)

with open(os.path.join(HERE, "threshold_validation.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
