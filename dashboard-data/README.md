# Dashboard data: Stroke Prediction

Results from `stroke-prediction.Rmd`. Values are transcribed from the rendered
PDF, the archived output of the completed analysis. Eight models x four
scenarios (SMOTE yes/no crossed with PCA yes/no), tuned with 5-fold CV on ROC
and scored on a held-out 30% test split.

## Files

- `model_results.csv` — one row per model/scenario (30 rows; the two QDA rows
  without PCA are absent because QDA failed to fit those folds: rank
  deficiency). Columns: `model`, `scenario`, `auc`, `accuracy`, `sensitivity`,
  `specificity`.
- `class_distribution.csv` — target imbalance in the full data. Columns:
  `stroke` (0/1), `count`, `percentage`. Only 4.87% of patients had a stroke,
  which is why sensitivity/specificity matter more than accuracy here.

Headline: no-SMOTE models often predict the majority class (sensitivity near
zero, specificity near one); SMOTE trades accuracy for real stroke detection.
