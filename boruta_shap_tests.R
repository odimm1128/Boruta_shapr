source("boruta_shap_pipeline.R")

print_section <- function(title) {
  cat("\n", strrep("=", 60), "\n", title, "\n", strrep("=", 60), "\n", sep = "")
}

assert_valid_result <- function(result, task_name) {
  if (!inherits(result$boruta, "Boruta")) {
    stop(sprintf("[%s] Result does not contain a Boruta object.", task_name))
  }
  if (nrow(result$importance_table) == 0L) {
    stop(sprintf("[%s] Importance table is empty.", task_name))
  }
  required_cols <- c("feature", "meanImp", "decision")
  if (!all(required_cols %in% names(result$importance_table))) {
    stop(sprintf("[%s] Importance table missing expected columns.", task_name))
  }
  invisible(result)
}

run_classification_example <- function() {
  data(iris)
  print_section("Classification Example: iris")
  result <- run_boruta_shap(
    df = iris,
    target_col = "Species",
    num_trees = 300,
    max_runs = 35,
    seed = 42,
    num_threads = 1,
    tune_extra_args = list(iters = 30, iters.warmup = 10)
  )
  assert_valid_result(result, "classification")
  print(result$boruta)
  cat("\nSelected attributes (including tentative):\n")
  print(result$selected_attributes)
  print(result$importance_table)
  invisible(result)
}

run_regression_example <- function() {
  data(mtcars)
  print_section("Regression Example: mtcars (mpg)")
  result <- run_boruta_shap(
    df = mtcars,
    target_col = "mpg",
    num_trees = 300,
    max_runs = 35,
    seed = 99,
    num_threads = 1,
    tune_extra_args = list(iters = 30, iters.warmup = 10)
  )
  assert_valid_result(result, "regression")
  print(result$boruta)
  cat("\nSelected attributes (including tentative):\n")
  print(result$selected_attributes)
  print(result$importance_table)
  invisible(result)
}

run_all_examples <- function() {
  classification_result <- run_classification_example()
  regression_result <- run_regression_example()
  invisible(list(classification = classification_result, regression = regression_result))
}

if (sys.nframe() == 0) {
  run_all_examples()
}
