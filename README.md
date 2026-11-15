
# Santander Customer Satisfaction

A machine learning project that estimates the probability that a bank customer is dissatisfied using 369 anonymized predictors from the [Santander Customer Satisfaction Kaggle competition](https://www.kaggle.com/competitions/santander-customer-satisfaction).

## Results

Both models were evaluated on the same five stratified folds with shuffling and a random seed of `42`. The primary metric was ROC-AUC, which measures how well the model ranks dissatisfied customers above satisfied customers.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ROC-AUC | Fold SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7836 | 0.8073 | 0.8053 | 0.7860 | 0.7888 | 0.7942 | 0.0112 |
| Histogram gradient boosting | 0.8300 | 0.8401 | 0.8391 | 0.8315 | 0.8402 | **0.8362** | 0.0050 |

Histogram gradient boosting improved mean ROC-AUC by **0.0420** and scored higher on every fold. It was selected for the first submission. The scores above are cross-validation results.

## Data

- **Training set:** 76,020 rows, including 3,008 dissatisfied customers (3.96%).
- **Test set:** 75,818 rows.
- **Target:** `1` = dissatisfied, `0` = satisfied.
- **Predictors:** 369 columns after excluding `ID` and `TARGET`. The train and test predictor names match. The training data had no missing predictor values and contained 34 constant predictors.

## Modeling workflow

1. **Inspect the data.** Check class balance, predictor names, missing values, and constant columns.
2. **Fit the logistic regression baseline.** Apply zero-variance filtering and standardization inside each training fold using a scikit-learn pipeline.
3. **Evaluate histogram gradient boosting.** Fit the model on the original predictors with a learning rate of 0.05, up to 15 leaves per tree, and 200 boosting iterations. Reuse the baseline folds.
4. **Generate the submission.** Refit the selected model on all training rows, align the test predictors to the training column order, and export the probability of `TARGET = 1` for each test ID.

## Repository structure

| Path | Purpose |
| --- | --- |
| `data/raw/` | Local Kaggle CSVs |
| `notebooks/01_data_inspection.ipynb` | Dataset checks and target distribution |
| `notebooks/02_baseline_logistic_regression.ipynb` | Baseline pipeline and validation |
| `notebooks/03_model_comparison_and_submission.ipynb` | Gradient boosting validation and CSV generation |
| `outputs/submission.csv` | Generated Kaggle submission |

## Reproduce the results

1. Set up a Python environment with pandas, scikit-learn, and Jupyter.

   ```bash
   python -m pip install pandas scikit-learn jupyter
   ```

2. Sign in to Kaggle, accept the competition rules, and download the files from the [competition data page](https://www.kaggle.com/competitions/santander-customer-satisfaction/data). Extract `train.csv` and `test.csv` directly into `data/raw/`.
3. Run the notebooks in numerical order. The final notebook saves `outputs/submission.csv` with columns `ID,TARGET` and one probability per test customer.

Keep the raw data and generated submission files local using the repository's `.gitignore`.

## Author

**Yaw Asante** - Data Scientist, Statistician and Financial Economist

Email: vkpoasante@gmail.com

## License
**MIT**


