library(ranger)
library(tuneRanger)
library(treeshap)
library(Boruta)
library(mlr)
library(data.table)

# Utility helpers --------------------------------------------------------------
resolve_threads <- function(num_threads = NULL) {
  if (is.null(num_threads)) {
    return(max(1, parallel::detectCores() - 1))
  }
  num_threads
}

prepare_boruta_inputs <- function(df, target_col) {
  if (!target_col %in% names(df)) {
    stop(sprintf("Column '%s' not found in data.", target_col))
  }
  y <- df[[target_col]]
  feature_cols <- setdiff(names(df), target_col)
  X <- df[feature_cols]
  is_classification <- is.factor(y)
  mlr_task <- if (is_classification) {
    mlr::makeClassifTask(data = df, target = target_col)
  } else {
    mlr::makeRegrTask(data = df, target = target_col)
  }
  measure_list <- if (is_classification) {
    list(mlr::multiclass.brier)
  } else {
    list(mlr::rmse)
  }
  list(
    X = X,
    y = y,
    feature_cols = feature_cols,
    target_col = target_col,
    is_classification = is_classification,
    mlr_task = mlr_task,
    measure_list = measure_list
  )
}

tune_ranger_hyperparameters <- function(task_bundle,
                                        num_trees,
                                        num_threads,
                                        tune_parameters,
                                        extra_args = list()) {
  base_args <- list(
    task = task_bundle$mlr_task,
    measure = task_bundle$measure_list,
    num.trees = num_trees,
    num.threads = num_threads,
    parameters = tune_parameters,
    show.info = TRUE
  )
  tune_call <- modifyList(base_args, extra_args)
  ans <- do.call(tuneRanger, tune_call)
  print(ans$recommended.pars)
  ans
}

extract_ranger_parameters <- function(tune_result,
                                      num_trees,
                                      num_threads,
                                      is_classification) {
  valid_ranger_params <- names(formals(ranger))
  tuned_param_candidates <- list()
  if (!is.null(tune_result$model) && !is.null(tune_result$model$learner$par.vals)) {
    tuned_param_candidates <- tune_result$model$learner$par.vals
  } else if (!is.null(tune_result$recommended.pars)) {
    rec <- tune_result$recommended.pars
    if (is.data.frame(rec)) {
      rec <- rec[1, , drop = FALSE]
    }
    tuned_param_candidates <- as.list(rec)
  }
  if (length(tuned_param_candidates) > 0 && !is.null(names(tuned_param_candidates))) {
    tuned_param_candidates <- tuned_param_candidates[names(tuned_param_candidates) %in% valid_ranger_params]
  }
  default_args <- list(
    num.trees = num_trees,
    num.threads = num_threads,
    respect.unordered.factors = "order",
    write.forest = TRUE,
    probability = is_classification
  )
  modifyList(default_args, tuned_param_candidates)
}

prepare_tree_predictions <- function(tree_data, target_class = NULL) {
  if ("prediction" %in% names(tree_data)) {
    if (!is.numeric(tree_data[["prediction"]])) {
      tree_data[, prediction := as.numeric(as.character(prediction))]
    }
    return(tree_data)
  }
  prob_cols <- grep("^pred\\.", names(tree_data), value = TRUE)
  if (length(prob_cols) == 0) {
    stop("ranger::treeInfo output does not contain prediction data required for treeshap.")
  }
  chosen_col <- prob_cols[1]
  if (!is.null(target_class)) {
    candidate <- paste0("pred.", target_class)
    if (candidate %in% prob_cols) {
      chosen_col <- candidate
    }
  }
  tree_data[, prediction := as.numeric(get(chosen_col))]
  tree_data
}

unify_ranger_for_shap <- function(rf_model, data, target_class = NULL) {
  n <- rf_model$num.trees
  tree_list <- lapply(seq_len(n), function(tree_idx) {
    tree_data <- data.table::as.data.table(ranger::treeInfo(rf_model, tree = tree_idx))
    tree_data <- prepare_tree_predictions(tree_data, target_class)
    tree_data[, c("nodeID", "leftChild", "rightChild", "splitvarName", "splitval", "prediction")]
  })
  treeshap:::ranger_unify.common(
    x = tree_list,
    n = n,
    data = data,
    feature_names = rf_model$forest$independent.variable.names
  )
}

compute_shap_importance <- function(x, y, base_ranger_args) {
  train_data <- data.frame(.target = y, x, check.names = FALSE)
  ranger_args <- modifyList(
    base_ranger_args,
    list(
      data = train_data,
      dependent.variable.name = ".target",
      classification = is.factor(y),
      probability = is.factor(y)
    )
  )
  rf_model <- do.call(ranger, ranger_args)
  if (is.factor(y)) {
    class_levels <- levels(y)
    shap_mats <- lapply(class_levels, function(cls) {
      unified_model <- unify_ranger_for_shap(rf_model, x, target_class = cls)
      treeshap_result <- treeshap::treeshap(unified_model, x)
      abs(as.matrix(treeshap_result$shaps))
    })
    shap_matrix <- Reduce(`+`, shap_mats) / length(shap_mats)
  } else {
    unified_model <- unify_ranger_for_shap(rf_model, x)
    treeshap_result <- treeshap::treeshap(unified_model, x)
    shap_matrix <- abs(as.matrix(treeshap_result$shaps))
  }
  colMeans(shap_matrix)
}

make_shap_importance_function <- function(base_ranger_args) {
  function(x, y) compute_shap_importance(x, y, base_ranger_args)
}

# Public pipeline --------------------------------------------------------------
run_boruta_shap <- function(df,
                            target_col,
                            num_trees = 600,
                            max_runs = 50,
                            seed = NULL,
                            num_threads = NULL,
                            boruta_trace = 1,
                            tune_parameters = list(replace = FALSE, respect.unordered.factors = "order"),
                            tune_extra_args = list(),
                            boruta_extra_args = list()) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  task_bundle <- prepare_boruta_inputs(df, target_col)
  threads <- resolve_threads(num_threads)
  tune_result <- tune_ranger_hyperparameters(
    task_bundle = task_bundle,
    num_trees = num_trees,
    num_threads = threads,
    tune_parameters = tune_parameters,
    extra_args = tune_extra_args
  )
  base_ranger_args <- extract_ranger_parameters(
    tune_result = tune_result,
    num_trees = num_trees,
    num_threads = threads,
    is_classification = task_bundle$is_classification
  )
  shap_importance_fun <- make_shap_importance_function(base_ranger_args)
  boruta_args <- modifyList(
    list(
      x = task_bundle$X,
      y = task_bundle$y,
      getImp = shap_importance_fun,
      maxRuns = max_runs,
      doTrace = boruta_trace
    ),
    boruta_extra_args
  )
  boruta_result <- do.call(Boruta, boruta_args)
  boruta_final <- TentativeRoughFix(boruta_result)
  selected <- getSelectedAttributes(boruta_final, withTentative = TRUE)
  importance_table <- attStats(boruta_final)
  importance_table$feature <- rownames(importance_table)
  importance_table <- importance_table[order(-importance_table$meanImp), c("feature", "meanImp", "decision")]
  list(
    boruta = boruta_result,
    boruta_final = boruta_final,
    selected_attributes = selected,
    importance_table = importance_table,
    base_ranger_args = base_ranger_args,
    tune_result = tune_result,
    task_bundle = task_bundle
  )
}

# (Example and test usage moved to boruta_shap_tests.R)
