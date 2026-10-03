# Stroke Prediction

I predicted stroke occurrence from a healthcare dataset of patient demographics and health indicators.

I compared logistic regression, LDA, QDA, and KNN across four preprocessing scenarios: with and without SMOTE for the class imbalance, and with and without PCA for dimensionality reduction. Each model-scenario combination was evaluated on AUC and accuracy, with ROC curves to compare trade-offs.

## Files

- `stroke-prediction.Rmd` - the analysis
- `stroke-prediction.pdf` / `stroke-prediction.html` - rendered versions
- `stroke-prediction-report.pdf` - the written report
- `healthcare-dataset-stroke-data.csv` - the dataset
