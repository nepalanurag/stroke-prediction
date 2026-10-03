"""Expansion analysis for the stroke-prediction project.

New questions the original 8-model bake-off did not answer:
1. Are the predicted probabilities calibrated? (calibration curves + Brier score)
2. What decision threshold should a clinician actually use? (threshold sweep)
3. Which features drive the predictions? (permutation importance + logistic coefs)

Data: healthcare-dataset-stroke-data.csv (5,110 rows, 249 strokes, 4.9% positive).
All numbers below are computed on a held-out 20% test split (seed 42).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, brier_score_loss, precision_recall_curve,
                             confusion_matrix, f1_score)
from sklearn.inspection import permutation_importance

SEED = 42
df = pd.read_csv("/home/hatch/workspace/expand-work/stroke.csv")
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

models = {
    "logistic": LogisticRegression(max_iter=2000, random_state=SEED),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "grad_boosting": HistGradientBoostingClassifier(random_state=SEED),
}

out = {"seed": SEED, "n_train": len(y_train), "n_test": len(y_test),
       "test_positive_rate": float(y_test.mean()), "models": {}}
probs = {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    probs[name] = p
    out["models"][name] = {
        "auc": float(roc_auc_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
    }

# 1. Calibration curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, p in probs.items():
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10)
    ax.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier {out['models'][name]['brier']:.4f})")
ax.plot([0, 1], [0, 1], "k--", label="perfect")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("fraction of positives")
ax.set_title("Calibration curves (held-out test set)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/expand-work/figs/stroke_calibration.png", dpi=110)
plt.close(fig)

# 2. Threshold sweep on the best-AUC model
best = max(out["models"], key=lambda m: out["models"][m]["auc"])
p = probs[best]
ths = np.linspace(0.05, 0.95, 37)
rows = []
for t in ths:
    yp = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
    rows.append({"threshold": float(t),
                 "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                 "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                 "f1": float(f1_score(y_test, yp)),
                 "youden": float(tp / (tp + fn) - fp / (fp + tn)) if (tp + fn) and (fp + tn) else 0.0})
thr = pd.DataFrame(rows)
t_f1 = thr.loc[thr["f1"].idxmax()]
t_youden = thr.loc[thr["youden"].idxmax()]
t05 = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-9)
out["threshold_analysis"] = {
    "model": best,
    "at_0.5": t05,
    "best_f1": {"threshold": float(t_f1["threshold"]), "f1": float(t_f1["f1"]),
                "precision": float(t_f1["precision"]), "recall": float(t_f1["recall"])},
    "youden_j": {"threshold": float(t_youden["threshold"]), "j": float(t_youden["youden"]),
                 "precision": float(t_youden["precision"]), "recall": float(t_youden["recall"])},
}
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thr["threshold"], thr["precision"], label="precision")
ax.plot(thr["threshold"], thr["recall"], label="recall")
ax.plot(thr["threshold"], thr["f1"], label="F1", linewidth=2)
ax.axvline(0.5, color="k", linestyle="--", alpha=0.5, label="0.5 default")
ax.axvline(float(t_f1["threshold"]), color="C3", linestyle=":", label=f'best F1 @ {t_f1["threshold"]:.2f}')
ax.set_xlabel("decision threshold")
ax.set_ylabel("score")
ax.set_title(f"Threshold trade-off ({best}, held-out test)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("/home/hatch/workspace/expand-work/figs/stroke_threshold.png", dpi=110)
plt.close(fig)

# 3. Feature importance: permutation importance on the RF pipeline + logistic coefs
pipe_rf = Pipeline([("pre", pre),
                    ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))])
pipe_rf.fit(X_train, y_train)
r = permutation_importance(pipe_rf, X_test, y_test, n_repeats=10, random_state=SEED,
                           n_jobs=-1, scoring="roc_auc")
raw_names = list(X.columns)  # permutation acts on the 10 raw input columns
imp = pd.Series(r.importances_mean, index=raw_names).sort_values(ascending=False)
out["permutation_importance_rf"] = {k: float(v) for k, v in imp.head(10).items()}
feat_names = (num_cols + list(pipe_rf.named_steps["pre"].named_transformers_["cat"]
                             .named_steps["oh"].get_feature_names_out(cat_cols)))

pipe_lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
pipe_lr.fit(X_train, y_train)
coefs = pd.Series(pipe_lr.named_steps["clf"].coef_[0], index=feat_names).sort_values(key=abs, ascending=False)
out["logistic_top_coefs"] = {k: float(v) for k, v in coefs.head(12).items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
imp.head(12)[::-1].plot.barh(ax=axes[0])
axes[0].set_title("Permutation importance (random forest)")
axes[0].set_xlabel("mean AUC drop")
coefs.head(12)[::-1].plot.barh(ax=axes[1])
axes[1].set_title("Logistic regression coefficients (top |coef|)")
axes[1].set_xlabel("coefficient")
fig.tight_layout()
fig.savefig("/home/hatch/workspace/expand-work/figs/stroke_importance.png", dpi=110)
plt.close(fig)

with open("/home/hatch/workspace/expand-work/stroke_metrics.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)[:2500])
