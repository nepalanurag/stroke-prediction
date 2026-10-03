# Stroke Prediction

Walkthrough with all plots: https://nepalanurag.github.io/stroke-prediction/

I predicted stroke occurrence from a healthcare dataset of patient demographics and health indicators.

I compared eight models, logistic regression, LDA, QDA, KNN, decision tree, random forest, bagging, and boosting, across four preprocessing scenarios: with and without SMOTE for the class imbalance, and with and without PCA for dimensionality reduction. Each model-scenario combination was evaluated on AUC, accuracy, sensitivity, and specificity, with ROC curves to compare trade-offs.

## Files

- `stroke-prediction.Rmd` - the analysis
- `stroke-prediction.pdf` / `stroke-prediction.html` - rendered versions
- `stroke-prediction-report.pdf` - the written report
- `healthcare-dataset-stroke-data.csv` - the dataset

## Follow-up: calibration, thresholds, and feature importance

The bake-off left three questions open, so I ran a follow-up in Python (scikit-learn, 80/20 split, seed 42): [expansion analysis](https://nepalanurag.github.io/stroke-prediction/expansion.html). Logistic regression turned out best on both AUC (0.842) and calibration (Brier 0.041); at the default 0.5 threshold it catches almost no strokes (recall 0.02), while 0.125 gives F1 0.35 with recall 0.66; age dominates the predictions, then glucose, smoking, hypertension, and heart disease. Code in `analysis/`.
