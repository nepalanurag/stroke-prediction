# Protocol-consistent KNN comparison for stroke-prediction (follow-up).
#
# The original analysis runs KNN without scaling in the two no-PCA scenarios
# while the PCA scenarios get center/scale, so distances in the no-PCA runs
# are dominated by age and glucose. This follow-up re-runs the two no-PCA KNN
# models with preProcess = c("center", "scale") and reports them alongside
# the original runs, so the PCA comparison can be read on equal footing.
#
# Data prep mirrors stroke-prediction.Rmd exactly (EDA + leakage-free BMI
# imputation chunks, 70/30 split with set.seed(123)); the four runs below use
# the same train() calls as the Rmd, plus the two new scaled runs.
#
# Run: Rscript analysis/knn_scaling_followup.R   (from the repo root)
# Writes: analysis/knn_scaling_comparison.csv
suppressMessages({
  library(caret)
  library(pROC)
  library(dplyr)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
HERE <- if (length(file_arg)) dirname(normalizePath(file_arg)) else getwd()
ROOT <- dirname(HERE)

stroke_data <- read.csv(file.path(ROOT, "healthcare-dataset-stroke-data.csv"))
stroke_data$bmi <- as.double(stroke_data$bmi)
stroke_data <- stroke_data[stroke_data$gender != "Other", ]

categorize_age <- function(age) {
  if (age >= 0 & age <= 1) return('Infant')
  else if (age > 1 & age <= 12) return('Children')
  else if (age > 12 & age < 20) return('Teenager')
  else if (age >= 20 & age <= 39) return('Adult')
  else if (age > 39 & age <= 59) return('Middle Aged')
  else return('Senior')
}
stroke_data$age_group <- sapply(stroke_data$age, categorize_age)
avg_bmi_by_group <- aggregate(bmi ~ age_group + gender, data = stroke_data, FUN = mean)
colnames(avg_bmi_by_group) <- c("age_group", "gender", "avg_bmi")
stroke_data$bmi_raw <- stroke_data$bmi
stroke_data <- merge(stroke_data, avg_bmi_by_group, by = c("age_group", "gender"), all.x = TRUE)
stroke_data$bmi <- ifelse(is.na(stroke_data$bmi), stroke_data$avg_bmi, stroke_data$bmi)
stroke_data$avg_bmi <- NULL

bmi_lookup <- stroke_data[, c("age_group", "gender", "bmi_raw")]
stroke_data <- stroke_data %>%
  select(-id, -age_group, -Residence_type, -gender, -bmi_raw) %>%
  mutate(
    ever_married = as.factor(ever_married),
    work_type = as.factor(work_type),
    smoking_status = as.factor(smoking_status),
    hypertension = as.factor(hypertension),
    heart_disease = as.factor(heart_disease),
    stroke = as.factor(stroke)
  )
stroke_data <- stroke_data %>%
  mutate(stroke = factor(stroke, levels = c(1, 0), labels = c("YES", "NO")))

set.seed(123)
trainIndex <- createDataPartition(stroke_data$stroke, p = 0.7, list = FALSE)
train_data <- stroke_data[trainIndex, ]
test_data  <- stroke_data[-trainIndex, ]

train_lookup <- bmi_lookup[trainIndex, ]
test_lookup  <- bmi_lookup[-trainIndex, ]
train_bmi_means <- aggregate(bmi_raw ~ age_group + gender, data = train_lookup,
                             FUN = function(x) mean(x, na.rm = TRUE))
colnames(train_bmi_means) <- c("age_group", "gender", "avg_bmi")
overall_bmi <- mean(train_lookup$bmi_raw, na.rm = TRUE)
train_bmi_means$avg_bmi[is.nan(train_bmi_means$avg_bmi)] <- overall_bmi
fill_bmi <- function(lookup) {
  key_l <- paste(lookup$age_group, lookup$gender)
  key_m <- paste(train_bmi_means$age_group, train_bmi_means$gender)
  fill <- train_bmi_means$avg_bmi[match(key_l, key_m)]
  fill[is.na(fill)] <- overall_bmi
  ifelse(is.na(lookup$bmi_raw), fill, lookup$bmi_raw)
}
train_data$bmi <- fill_bmi(train_lookup)
test_data$bmi  <- fill_bmi(test_lookup)
stopifnot(sum(is.na(train_data$bmi)) == 0, sum(is.na(test_data$bmi)) == 0)

ctrl_base <- trainControl(
  method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary
)
ctrl_smote <- trainControl(
  method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary,
  savePredictions = "final", sampling = "smote"
)

run_knn <- function(label, trControl, preProcess = NULL) {
  set.seed(123)
  fit <- train(stroke ~ ., data = train_data, method = "knn",
               trControl = trControl, preProcess = preProcess, metric = "ROC")
  prob <- predict(fit, newdata = test_data, type = "prob")[, "YES"]
  cls  <- predict(fit, newdata = test_data, type = "raw")
  data.frame(
    run = label,
    k = fit$bestTune$k,
    test_AUC = round(as.numeric(roc(test_data$stroke, prob, quiet = TRUE)$auc), 4),
    test_accuracy = round(as.numeric(confusionMatrix(cls, test_data$stroke)$overall["Accuracy"]), 4)
  )
}

out <- rbind(
  run_knn("no SMOTE, no PCA (original)", ctrl_base),
  run_knn("SMOTE, no PCA (original)", ctrl_smote),
  run_knn("no SMOTE, no PCA + center/scale (new)", ctrl_base, c("center", "scale")),
  run_knn("SMOTE, no PCA + center/scale (new)", ctrl_smote, c("center", "scale"))
)
rownames(out) <- NULL
write.csv(out, file.path(HERE, "knn_scaling_comparison.csv"), row.names = FALSE)
print(out, row.names = FALSE)
