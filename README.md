# Stroke Prediction

Walkthrough with all plots: https://nepalanurag.github.io/stroke-prediction/

I predicted stroke occurrence from a healthcare dataset of patient demographics and health indicators.

I compared eight models, logistic regression, LDA, QDA, KNN, decision tree, random forest, bagging, and boosting, across four preprocessing scenarios: with and without SMOTE for the class imbalance, and with and without PCA for dimensionality reduction. Each model-scenario combination was evaluated on AUC, accuracy, sensitivity, and specificity, with ROC curves to compare trade-offs.

## Files

- `stroke-prediction.Rmd` - the analysis
- `stroke-prediction.pdf` / `stroke-prediction.html` - rendered versions
- `stroke-prediction-report.pdf` - the written report
- `healthcare-dataset-stroke-data.csv` - the dataset

The model comparison is followed by calibration, threshold, and feature-importance analysis in the Results section of the walkthrough (Python, scikit-learn, 80/20 split, seed 42). Logistic regression turned out best on both AUC (0.842) and calibration (Brier 0.041); at the default 0.5 threshold it catches almost no strokes (recall 0.02), while 0.125 gives F1 0.35 with recall 0.66; age dominates the predictions, then glucose, smoking, hypertension, and heart disease. Code in `analysis/`.

### Follow-up robustness checks

Two extensions of the original analysis, each kept separate so the original runs stand as reported:

- **Protocol-consistent KNN.** In the bake-off, KNN runs without scaling in the no-PCA scenarios while the PCA scenarios get center/scale, which confounds the PCA comparison for KNN. The new final section of `stroke-prediction.Rmd` re-runs the two no-PCA KNN models with `preProcess = c("center", "scale")` and reports them next to the original runs: on the test split the scaled runs score 0.697/0.714 AUC vs 0.744/0.773 for the originals (without/with SMOTE), a descriptive preprocessing sensitivity, not a change to the original results. Refresh the numbers with `Rscript analysis/knn_scaling_followup.R` (writes `analysis/knn_scaling_comparison.csv`).
- **Threshold selection under a validation protocol.** The original threshold sweep tunes the threshold on the test set, which adds selection optimism. `analysis/threshold_validation_protocol.py` re-runs the selection under a 60/20/20 train/validation/test split (seed 42): the validation split picks 0.15, and the locked test set is reported once, F1 0.256 (precision 0.18, recall 0.42). The tuned threshold still beats 0.5 (test F1 0.077) by a wide margin, and the lower estimate relative to the test-tuned 0.35 is what you expect once selection can no longer peek at the test set. Results in `analysis/threshold_validation.json`, plot in `docs/figs/stroke_threshold_validation.png`. Run: `python3 analysis/threshold_validation_protocol.py`.

Reproducibility: `session_info.txt` records the R session that produced the follow-up runs.
