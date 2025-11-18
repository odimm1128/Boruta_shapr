# Boruta_shapr

## Boruta + SHAP Feature Selection Pipeline

This repository demonstrates how to combine `tuneRanger`, `ranger`, `treeshap`, and `Boruta` in R to perform automated feature selection with SHAP-based importance scores. The code is designed so you can reuse the pipeline on any classification or regression task with minimal changes.

## Repository Structure

| File | Description |
| --- | --- |
| `boruta_shap_pipeline.R` | Core helpers and `run_boruta_shap()` wrapper that tunes a `ranger` model, computes SHAP importance with `treeshap`, and runs Boruta with the custom importance. |
| `boruta_shap_tests.R` | Example/test script that exercises the pipeline on both a classification task (`iris`) and a regression task (`mtcars`). |
| `.Rproj`, `.gitignore` | Standard RStudio project files (optional). |

## Requirements

Install the following R packages (CRAN):

```r
install.packages(c(
  "ranger",
  "tuneRanger",
  "treeshap",
  "Boruta",
  "mlr",
  "data.table"
))
```

## Using the Pipeline

1. **Source the pipeline** (or run it directly as part of your script):
   ```r
   source("boruta_shap_pipeline.R")
   result <- run_boruta_shap(
     df = your_dataframe,
     target_col = "target_column_name",
     num_trees = 600,       # adjust as needed
     max_runs = 75,         # Boruta iterations
     seed = 123
   )
   ```

2. **Inspect outputs**:
   ```r
   result$selected_attributes    # Final Boruta decisions
   result$importance_table       # Mean SHAP importance and decision per feature
   result$boruta_final           # Boruta object after TentativeRoughFix
   result$base_ranger_args       # Tuned ranger parameters used for SHAP
   ```

3. **Customize tuning or Boruta behaviour** via optional arguments:
   ```r
   run_boruta_shap(
     df,
     target_col,
     tune_extra_args = list(iters = 80, iters.warmup = 30),
     boruta_extra_args = list(pValue = 0.01),
     num_threads = 4,
     max_runs = 100
   )
   ```

## Running the Sample Tests

`boruta_shap_tests.R` acts as both documentation and a quick regression test suite. It runs:

- **Classification** example on the `iris` dataset (predicting `Species`).
- **Regression** example on the `mtcars` dataset (predicting `mpg`).

Run them with:

```bash
Rscript boruta_shap_tests.R
```

## Adapting to New Tasks

1. **Prepare your data frame** with the response column specified in `target_col` (factor for classification, numeric for regression).
2. **Adjust tuning scope**:
   - Increase `num_trees`, `iters`, or `max_runs` for larger datasets.
   - Override `tune_parameters` or `tune_extra_args` if you want to expose additional `ranger` hyperparameters.
3. **Leverage the outputs**:
   - Feed `importance_table` into downstream reporting.
   - Reuse `base_ranger_args` if you want to retrain the tuned model outside Boruta.

## Troubleshooting

- **Package warnings**: If you see “package was built under R version …”, reinstall that package with your current R version.
- **treeshap errors about predictions**: Ensure the dataset used at SHAP time matches the feature columns used during training, and that no unsupported data types (e.g., raw characters) remain.
- **Slow tuning**: Reduce `num_trees`, `max_runs`, or `tune_extra_args$iters` while prototyping; increase them for the final run.
