import pandas as pd
import numpy as np
import joblib
import os
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import argparse
import logging
import time
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    roc_curve,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import keras_tuner as kt
    from scikeras.wrappers import KerasClassifier, KerasRegressor
except ImportError as e:
    print(
        f"ImportError: {e}. Please ensure all required libraries (TensorFlow, Keras Tuner, Scikit-learn, Scikeras) are installed."
    )
    print("You can typically install them using pip:")
    print(
        "pip install pandas numpy joblib matplotlib scikit-learn tensorflow keras-tuner scikeras"
    )
    raise

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Custom Metric for Regression ---
def mean_absolute_relative_error(y_true, y_pred):
    epsilon = 1e-9
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    valid_indices = np.abs(y_true_np) > (epsilon * 1000)
    if not np.any(valid_indices):
        return np.nan

    y_true_filt = y_true_np[valid_indices]
    y_pred_filt = y_pred_np[valid_indices]

    if len(y_true_filt) == 0:
        return np.nan

    relative_error = (y_pred_filt - y_true_filt) / (y_true_filt + epsilon)
    return np.mean(np.abs(relative_error))


# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Simplified NN pipelines with Keras Tuner and SelectKBest, adhering to submission guidelines."
    )

    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Base directory for all outputs (plots, models, submission CSVs).",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="./data/AppML_InitialProject_train.h5",
        help="Training data HDF5 path.",
    )

    parser.add_argument(
        "--firstname",
        type=str,
        required=True,
        help="Your first name for submission file naming.",
    )
    parser.add_argument(
        "--lastname",
        type=str,
        required=True,
        help="Your last name for submission file naming.",
    )
    parser.add_argument(
        "--cls_solution_name",
        type=str,
        default="KerasNN_Cls",
        help="A descriptive name for your classification solution.",
    )
    parser.add_argument(
        "--reg_solution_name",
        type=str,
        default="KerasNN_Reg",
        help="A descriptive name for your regression solution.",
    )

    parser.add_argument(
        "--cls_test_path",
        type=str,
        default="./data/AppML_InitialProject_test_classification.h5",
        help="Classification test data HDF5 path.",
    )
    parser.add_argument(
        "--cls_target_column",
        type=str,
        default="p_Truth_isElectron",
        help="Classification target column name.",
    )
    parser.add_argument(
        "--cls_max_features",
        type=int,
        default=25,
        help="Max features for classification (k for SelectKBest).",
    )
    parser.add_argument(
        "--cls_kt_max_trials",
        type=int,
        default=10,
        help="Keras Tuner max trials for classification.",
    )
    parser.add_argument(
        "--cls_kt_epochs",
        type=int,
        default=50,
        help="Keras Tuner epochs per trial for classification.",
    )
    parser.add_argument(
        "--cls_selected_features_list",
        type=str,
        default=None,
        help="Pre-selected classification features (comma-separated string). Overrides SelectKBest if provided.",
    )
    parser.add_argument(
        "--skip_cls_tuning",
        action="store_true",
        help="Skip Keras Tuner for classification and use default model architecture/hyperparameters.",
    )
    parser.add_argument(
        "--disable_cls_feature_selection",
        action="store_true",
        help="Disable SelectKBest for classification (uses all initial features up to max_features if no list provided).",
    )

    parser.add_argument(
        "--reg_test_path",
        type=str,
        default="./data/AppML_InitialProject_test_regression.h5",
        help="Regression test data HDF5 path.",
    )
    parser.add_argument(
        "--reg_target_column",
        type=str,
        default="p_Truth_Energy",
        help="Regression target column name.",
    )
    parser.add_argument(
        "--electron_flag_column",
        type=str,
        default="p_Truth_isElectron",
        help="Column name for electron flag (for filtering in regression).",
    )
    parser.add_argument(
        "--reg_max_features",
        type=int,
        default=12,
        help="Max features for regression (k for SelectKBest).",
    )
    parser.add_argument(
        "--reg_kt_max_trials",
        type=int,
        default=10,
        help="Keras Tuner max trials for regression.",
    )
    parser.add_argument(
        "--reg_kt_epochs",
        type=int,
        default=60,
        help="Keras Tuner epochs per trial for regression.",
    )
    parser.add_argument(
        "--reg_selected_features_list",
        type=str,
        default=None,
        help="Pre-selected regression features (comma-separated string). Overrides SelectKBest if provided.",
    )
    parser.add_argument(
        "--skip_reg_tuning",
        action="store_true",
        help="Skip Keras Tuner for regression and use default model architecture/hyperparameters.",
    )
    parser.add_argument(
        "--disable_reg_feature_selection",
        action="store_true",
        help="Disable SelectKBest for regression (uses all initial features up to max_features if no list provided).",
    )

    parser.add_argument(
        "--exclude_columns_common",
        nargs="*",
        default=["p_Truth_Energy", "p_Truth_isElectron"],
        help="List of columns to always exclude from features.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.20,
        help="Proportion of training data to use for validation set.",
    )
    parser.add_argument(
        "--dev_test_size",
        type=float,
        default=0.15,
        help="Proportion of original data to use for development test set.",
    )
    parser.add_argument(
        "--final_model_epochs",
        type=int,
        default=100,
        help="Epochs for training the final model after tuning (or if tuning is skipped).",
    )

    return parser.parse_args()


# --- Keras Tuner Model Building Functions ---
def build_classification_model_kt(hp, num_features):
    model = Sequential(name="classification_nn_kt")
    model.add(Input(shape=(num_features,), name="input_layer_cls"))
    hp_layers = hp.Int("num_layers", 1, 2, default=1)
    for i in range(hp_layers):
        model.add(
            Dense(
                units=hp.Int(
                    f"units_{i}", min_value=32, max_value=128, step=32, default=64
                ),
                activation="relu",
                name=f"cls_hidden_{i+1}",
            )
        )
        if hp.Boolean(f"dropout_enabled_{i}", default=True):
            model.add(
                Dropout(
                    hp.Float(f"dropout_rate_{i}", 0.1, 0.4, step=0.1, default=0.2),
                    name=f"cls_dropout_{i+1}",
                )
            )
    model.add(Dense(1, activation="sigmoid", name="output_layer_cls"))
    hp_learning_rate = hp.Float(
        "lr", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3
    )
    optimizer = Adam(learning_rate=hp_learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_regression_model_kt(hp, num_features):
    model = Sequential(name="regression_nn_kt")
    model.add(Input(shape=(num_features,), name="input_layer_reg"))
    hp_layers = hp.Int("num_layers", 1, 2, default=1)
    for i in range(hp_layers):
        model.add(
            Dense(
                units=hp.Int(
                    f"units_{i}", min_value=32, max_value=128, step=32, default=64
                ),
                activation="relu",
                name=f"reg_hidden_{i+1}",
            )
        )
        if hp.Boolean(f"dropout_enabled_{i}", default=True):
            model.add(
                Dropout(
                    hp.Float(f"dropout_rate_{i}", 0.0, 0.3, step=0.1, default=0.1),
                    name=f"reg_dropout_{i+1}",
                )
            )
    model.add(Dense(1, activation="linear", name="output_layer_reg"))
    hp_learning_rate = hp.Float(
        "lr", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3
    )
    optimizer = Adam(learning_rate=hp_learning_rate)
    model.compile(
        optimizer=optimizer, loss="mean_absolute_error", metrics=["mae", "mse"]
    )
    return model


# --- Helper: Prepare Data ---
def prepare_data(
    df,
    target_column,
    feature_columns_initial,
    selected_features_list_arg,
    max_features,
    random_seed,
    dev_test_size,
    validation_size,
    task_type="classification",
    disable_feature_selection=False,
):
    logger.info(f"Preparing data for {task_type} task...")
    if df.empty:
        logger.error(f"Input DataFrame for {task_type} is empty. Cannot prepare data.")
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype="float64")
        return (
            empty_df,
            empty_df,
            empty_df,
            empty_series,
            empty_series,
            empty_series,
            [],
        )

    X_original_all_features = df[feature_columns_initial]
    y_original = df[target_column]

    stratify_split_dev = (
        y_original
        if task_type == "classification" and y_original.nunique() > 1
        else None
    )
    X_train_val_all_features, X_test_dev_all_features, y_train_val, y_test_dev = (
        train_test_split(
            X_original_all_features,
            y_original,
            test_size=dev_test_size,
            random_state=random_seed,
            stratify=(
                stratify_split_dev
                if stratify_split_dev is not None and len(stratify_split_dev) > 0
                else None
            ),
        )
    )

    final_selected_features = []
    if selected_features_list_arg:
        final_selected_features = [
            f.strip()
            for f in selected_features_list_arg.split(",")
            if f.strip() in X_train_val_all_features.columns
        ]
        if len(final_selected_features) > max_features:
            logger.warning(
                f"Provided feature list for {task_type} has {len(final_selected_features)} features, "
                f"but max_features is {max_features}. Truncating to first {max_features}."
            )
            final_selected_features = final_selected_features[:max_features]
        logger.info(
            f"Using pre-selected features for {task_type} ({len(final_selected_features)}): {final_selected_features}"
        )
    elif disable_feature_selection:
        final_selected_features = feature_columns_initial[:]
        logger.info(
            f"Feature selection disabled for {task_type}. Using all {len(final_selected_features)} initial features."
        )
    else:
        k_to_select = min(max_features, X_train_val_all_features.shape[1])
        logger.info(
            f"Performing SelectKBest for {task_type} to select top {k_to_select} features..."
        )
        X_train_val_all_features_cleaned = X_train_val_all_features.replace(
            [np.inf, -np.inf], np.nan
        )
        for col in X_train_val_all_features_cleaned.columns:
            if X_train_val_all_features_cleaned[col].isnull().any():
                X_train_val_all_features_cleaned[col] = (
                    X_train_val_all_features_cleaned[col].fillna(
                        X_train_val_all_features_cleaned[col].median()
                    )
                )
        X_train_val_all_features_cleaned = X_train_val_all_features_cleaned.fillna(0)

        score_func = f_classif if task_type == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=k_to_select)
        try:
            selector.fit(X_train_val_all_features_cleaned, y_train_val)
            selected_indices = selector.get_support(indices=True)
            final_selected_features = X_train_val_all_features_cleaned.columns[
                selected_indices
            ].tolist()
            logger.info(
                f"Selected top {len(final_selected_features)} features for {task_type} using SelectKBest: {final_selected_features}"
            )
        except Exception as e_skb:
            logger.error(
                f"Error during SelectKBest for {task_type}: {e_skb}. Falling back to using first {k_to_select} initial features."
            )
            final_selected_features = feature_columns_initial[:k_to_select]

    if not final_selected_features:
        logger.warning(
            f"No features were selected for {task_type}. Attempting to use all initial features (up to max_features)."
        )
        final_selected_features = feature_columns_initial[:max_features]
        if not final_selected_features:
            raise ValueError(
                f"CRITICAL: No features selected or available for {task_type} even after fallback."
            )

    X_train_val_subset = X_train_val_all_features[final_selected_features]
    X_test_dev_subset = X_test_dev_all_features[final_selected_features]

    val_size_adjusted = (
        validation_size / (1 - dev_test_size)
        if (1 - dev_test_size) > 0 and validation_size > 0
        else 0
    )

    if val_size_adjusted == 0 or len(y_train_val) == 0:
        X_train_subset, X_val_subset, y_train, y_val = (
            X_train_val_subset,
            pd.DataFrame(),
            y_train_val,
            pd.Series(dtype="float64"),
        )
    else:
        stratify_split_train_val = (
            y_train_val
            if task_type == "classification" and y_train_val.nunique() > 1
            else None
        )
        X_train_subset, X_val_subset, y_train, y_val = train_test_split(
            X_train_val_subset,
            y_train_val,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=(
                stratify_split_train_val
                if stratify_split_train_val is not None
                and len(stratify_split_train_val) > 0
                else None
            ),
        )

    logger.info(
        f"  {task_type.upper()} X_train shape: {X_train_subset.shape}, y_train shape: {y_train.shape}"
    )
    logger.info(
        f"  {task_type.upper()} X_val shape: {X_val_subset.shape}, y_val shape: {y_val.shape}"
    )
    logger.info(
        f"  {task_type.upper()} X_test_dev shape: {X_test_dev_subset.shape}, y_test_dev shape: {y_test_dev.shape}"
    )

    return (
        X_train_subset,
        X_val_subset,
        X_test_dev_subset,
        y_train,
        y_val,
        y_test_dev,
        final_selected_features,
    )


# --- Helper: Save Submission Files ---
def save_submission_files(
    predictions_df, variable_list, output_path_predictions, output_path_variables
):
    try:
        predictions_df.to_csv(output_path_predictions, index=False, header=False)
        logger.info(f"Saved predictions to: {output_path_predictions}")
        with open(output_path_predictions, "r") as f:
            logger.info(
                f"  First 3 lines of predictions file:\n  "
                + "".join(
                    [next(f) for _ in range(min(3, len(predictions_df)))]
                ).replace("\n", "\n  ")
            )
    except IOError as e:
        logger.error(f"Error saving predictions to {output_path_predictions}: {e}")

    try:
        with open(output_path_variables, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for feature in variable_list:
                writer.writerow([feature])
        logger.info(f"Saved variable list to: {output_path_variables}")
        with open(output_path_variables, "r") as f:
            logger.info(
                f"  First 3 lines of variable list file:\n  "
                + "".join([next(f) for _ in range(min(3, len(variable_list)))]).replace(
                    "\n", "\n  "
                )
            )
    except IOError as e:
        logger.error(f"Error saving variable list to {output_path_variables}: {e}")


# --- Main Execution ---
def main(args):
    os.makedirs(args.output_base_dir, exist_ok=True)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    logger.info(f"Global Output Base Directory: {args.output_base_dir}")
    logger.info(f"Loading training data from: {args.train_path}")
    try:
        original_train_df = pd.read_hdf(args.train_path)
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise
    logger.info(f"Original training data shape: {original_train_df.shape}")
    all_original_cols = original_train_df.columns.tolist()

    # ========================== CLASSIFICATION ==========================
    logger.info("\n" + "=" * 30 + " CLASSIFICATION PIPELINE " + "=" * 30)
    CLS_OUTPUT_DIR = os.path.join(args.output_base_dir, "classification_nn_kt")
    os.makedirs(CLS_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Classification outputs will be in: {CLS_OUTPUT_DIR}")

    cls_submission_base_name = (
        f"Classification_{args.firstname}{args.lastname}_{args.cls_solution_name}"
    )
    CLS_PREDICTIONS_SUBMISSION_PATH = os.path.join(
        CLS_OUTPUT_DIR, f"{cls_submission_base_name}.csv"
    )
    CLS_VARIABLELIST_SUBMISSION_PATH = os.path.join(
        CLS_OUTPUT_DIR, f"{cls_submission_base_name}_VariableList.csv"
    )
    logger.info(
        f"Classification predictions submission CSV will be: {CLS_PREDICTIONS_SUBMISSION_PATH}"
    )
    logger.info(
        f"Classification variable list submission CSV will be: {CLS_VARIABLELIST_SUBMISSION_PATH}"
    )

    exclude_cols_cls = list(set(args.exclude_columns_common + [args.cls_target_column]))
    feature_cols_cls_initial = [
        col for col in all_original_cols if col not in exclude_cols_cls
    ]

    (
        X_train_cls,
        X_val_cls,
        X_test_dev_cls,
        y_train_cls,
        y_val_cls,
        y_test_dev_cls,
        final_selected_features_cls,
    ) = prepare_data(
        original_train_df,
        args.cls_target_column,
        feature_cols_cls_initial,
        args.cls_selected_features_list,
        args.cls_max_features,
        args.random_seed,
        args.dev_test_size,
        args.validation_size,
        task_type="classification",
        disable_feature_selection=args.disable_cls_feature_selection,
    )

    if X_train_cls.empty:
        logger.error(
            "Classification training data is empty after preparation. Skipping classification pipeline."
        )
    else:
        N_FEATURES_FINAL_CLS = len(final_selected_features_cls)
        logger.info(
            f"Number of final selected features for CLS: {N_FEATURES_FINAL_CLS}"
        )

        cls_scaler = StandardScaler()
        X_train_cls_scaled = cls_scaler.fit_transform(X_train_cls)
        # X_val_cls is a DataFrame. If it's empty, transform would fail or return empty array.
        # If not empty, transform returns a NumPy array.
        X_val_cls_scaled = (
            cls_scaler.transform(X_val_cls) if not X_val_cls.empty else np.array([])
        )  # Ensure NumPy array for consistency if empty

        joblib.dump(cls_scaler, os.path.join(CLS_OUTPUT_DIR, "cls_scaler.joblib"))

        best_hps_cls = None
        if not args.skip_cls_tuning:
            logger.info("NNC5: Hyperparameter Tuning for CLS NN using Keras Tuner...")

            class ClsHyperModel(kt.HyperModel):
                def __init__(self, num_features):
                    self.num_features = num_features

                def build(self, hp):
                    return build_classification_model_kt(hp, self.num_features)

            cls_hypermodel_instance = ClsHyperModel(num_features=N_FEATURES_FINAL_CLS)
            tuner_cls = kt.Hyperband(
                hypermodel=cls_hypermodel_instance,
                objective=kt.Objective("val_auc", direction="max"),
                max_epochs=args.cls_kt_epochs,
                factor=3,
                hyperband_iterations=1,
                directory=os.path.join(CLS_OUTPUT_DIR, "kt_cls_tuning_dir"),
                project_name="classification_tuning",
                seed=args.random_seed,
                overwrite=True,
            )
            early_stopping_tuner = EarlyStopping(
                monitor="val_auc",
                patience=5,
                mode="max",
                restore_best_weights=True,
                verbose=1,
            )
            logger.info(
                f"Starting Keras Tuner search for CLS (Max Trials: {args.cls_kt_max_trials}, Epochs per trial: {args.cls_kt_epochs})..."
            )

            # CORRECTED: Check original X_val_cls and y_val_cls for emptiness
            if X_val_cls.empty or y_val_cls.empty:
                logger.warning(
                    "CLS: Validation data for Keras Tuner is empty. Tuning will proceed without validation-based early stopping or metrics."
                )
                tuner_cls.search(
                    X_train_cls_scaled,
                    y_train_cls,
                    epochs=args.cls_kt_epochs,
                    verbose=2,
                )
            else:
                tuner_cls.search(
                    X_train_cls_scaled,
                    y_train_cls,
                    epochs=args.cls_kt_epochs,
                    validation_data=(
                        X_val_cls_scaled,
                        y_val_cls,
                    ),  # X_val_cls_scaled is numpy here
                    callbacks=[early_stopping_tuner],
                    verbose=2,
                )
            try:
                best_hps_cls = tuner_cls.get_best_hyperparameters(num_trials=1)[0]
                logger.info(f"Best CLS Hyperparameters found: {best_hps_cls.values}")
            except IndexError:
                logger.error(
                    "Keras Tuner for CLS found no best hyperparameters. Using default HPs."
                )
                best_hps_cls = None

        logger.info("NNC6: Training final CLS NN model...")
        if best_hps_cls:
            logger.info("Using Keras Tuner's best hyperparameters for final CLS model.")
        else:
            logger.info(
                "Using default hyperparameters for final CLS model (tuning skipped or failed)."
            )

        final_lr_cls = best_hps_cls.get("lr") if best_hps_cls else 0.001
        final_num_layers_cls = best_hps_cls.get("num_layers") if best_hps_cls else 1
        final_units_cls = []
        final_dropout_rates_cls = []
        final_dropout_enabled_per_layer_cls = []

        if best_hps_cls:
            for i in range(final_num_layers_cls):
                final_units_cls.append(best_hps_cls.get(f"units_{i}", 64))
                layer_dropout_enabled = best_hps_cls.get(f"dropout_enabled_{i}", True)
                final_dropout_enabled_per_layer_cls.append(layer_dropout_enabled)
                if layer_dropout_enabled:
                    final_dropout_rates_cls.append(
                        best_hps_cls.get(f"dropout_rate_{i}", 0.2)
                    )
                else:
                    final_dropout_rates_cls.append(0.0)
        else:
            final_units_cls = [64] * final_num_layers_cls
            final_dropout_enabled_per_layer_cls = [True] * final_num_layers_cls
            final_dropout_rates_cls = [0.2] * final_num_layers_cls

        def get_final_cls_model_for_scikeras(
            num_features,
            hidden_layer_sizes_tuple,
            dropout_rates_tuple,
            learning_rate,
            dropout_enabled_list_tuple,
        ):
            model = Sequential(name="final_classification_nn")
            model.add(Input(shape=(num_features,), name="input_layer_final_cls"))
            hidden_layer_sizes = list(hidden_layer_sizes_tuple)
            dropout_rates = list(dropout_rates_tuple)
            dropout_enabled_list = list(dropout_enabled_list_tuple)
            for i, neurons in enumerate(hidden_layer_sizes):
                model.add(
                    Dense(neurons, activation="relu", name=f"final_cls_hidden_{i+1}")
                )
                if (
                    i < len(dropout_enabled_list)
                    and dropout_enabled_list[i]
                    and i < len(dropout_rates)
                    and dropout_rates[i] > 0
                ):
                    model.add(
                        Dropout(dropout_rates[i], name=f"final_cls_dropout_{i+1}")
                    )
            model.add(Dense(1, activation="sigmoid", name="output_layer_final_cls"))
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
            )
            return model

        final_cls_pipeline_model = KerasClassifier(
            model=get_final_cls_model_for_scikeras,
            model__num_features=N_FEATURES_FINAL_CLS,
            model__hidden_layer_sizes_tuple=tuple(final_units_cls),
            model__dropout_rates_tuple=tuple(final_dropout_rates_cls),
            model__learning_rate=final_lr_cls,
            model__dropout_enabled_list_tuple=tuple(
                final_dropout_enabled_per_layer_cls
            ),
            epochs=args.final_model_epochs,
            batch_size=128,
            verbose=0,
            random_state=args.random_seed,
        )
        final_classification_pipeline = Pipeline(
            [("scaler", cls_scaler), ("nn", final_cls_pipeline_model)]
        )
        early_stopping_final = EarlyStopping(
            monitor="val_auc",
            patience=10,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        )
        logger.info(
            f"Fitting final CLS pipeline (Epochs: {args.final_model_epochs})..."
        )

        fit_params = {}
        # CORRECTED: Check original X_val_cls and y_val_cls for emptiness
        if not X_val_cls.empty and not y_val_cls.empty:
            # X_val_cls_scaled is already a NumPy array if X_val_cls was not empty
            fit_params["nn__validation_data"] = (X_val_cls_scaled, y_val_cls)
            fit_params["nn__callbacks"] = [early_stopping_final]
        else:
            logger.warning(
                "CLS: Final model training without validation data for early stopping."
            )

        final_classification_pipeline.fit(
            X_train_cls, y_train_cls, **fit_params
        )  # X_train_cls is DataFrame, scaler in pipeline handles it
        joblib.dump(
            final_classification_pipeline,
            os.path.join(CLS_OUTPUT_DIR, "final_classification_pipeline_kt.joblib"),
        )
        logger.info(f"Final CLS Pipeline (Keras Tuner) saved to {CLS_OUTPUT_DIR}")

        logger.info("NNC7: Evaluating CLS NN model on dev test set...")
        if not X_test_dev_cls.empty:
            dev_cls_pred_proba = final_classification_pipeline.predict_proba(
                X_test_dev_cls
            )[:, 1]
            dev_cls_pred_class = (dev_cls_pred_proba > 0.5).astype(int)
            logger.info(
                f"  CLS Dev LogLoss: {log_loss(y_test_dev_cls, dev_cls_pred_proba):.4f}"
            )
            logger.info(
                f"  CLS Dev ROC-AUC: {roc_auc_score(y_test_dev_cls, dev_cls_pred_proba):.4f}"
            )
            logger.info(
                "  CLS Dev Classification Report:\n"
                + classification_report(
                    y_test_dev_cls,
                    dev_cls_pred_class,
                    target_names=["NotElectron", "Electron"],
                )
            )
        else:
            logger.warning("CLS Dev test data is empty. Skipping evaluation.")

        logger.info("NNC8: Generating CLS predictions for submission file...")
        try:
            final_cls_test_df = pd.read_hdf(args.cls_test_path)
            X_final_cls_test_subset = final_cls_test_df[final_selected_features_cls]
            final_cls_pred_proba_test = final_classification_pipeline.predict_proba(
                X_final_cls_test_subset
            )[:, 1]
            ids_cls_submission = np.arange(len(final_cls_test_df))
            cls_submission_df_predictions = pd.DataFrame(
                {"id": ids_cls_submission, "prediction": final_cls_pred_proba_test}
            )
            save_submission_files(
                cls_submission_df_predictions,
                final_selected_features_cls,
                CLS_PREDICTIONS_SUBMISSION_PATH,
                CLS_VARIABLELIST_SUBMISSION_PATH,
            )
        except FileNotFoundError:
            logger.warning(
                f"CLS test file not found at {args.cls_test_path}. Skipping final CLS submission file generation."
            )
        except Exception as e:
            logger.error(
                f"Error during CLS final test prediction/submission file generation: {e}",
                exc_info=True,
            )

    # ========================== REGRESSION ==========================
    logger.info("\n" + "=" * 30 + " REGRESSION PIPELINE " + "=" * 30)
    REG_OUTPUT_DIR = os.path.join(args.output_base_dir, "regression_nn_kt")
    os.makedirs(REG_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Regression outputs will be in: {REG_OUTPUT_DIR}")

    reg_submission_base_name = (
        f"Regression_{args.firstname}{args.lastname}_{args.reg_solution_name}"
    )
    REG_PREDICTIONS_SUBMISSION_PATH = os.path.join(
        REG_OUTPUT_DIR, f"{reg_submission_base_name}.csv"
    )
    REG_VARIABLELIST_SUBMISSION_PATH = os.path.join(
        REG_OUTPUT_DIR, f"{reg_submission_base_name}_VariableList.csv"
    )
    logger.info(
        f"Regression predictions submission CSV will be: {REG_PREDICTIONS_SUBMISSION_PATH}"
    )
    logger.info(
        f"Regression variable list submission CSV will be: {REG_VARIABLELIST_SUBMISSION_PATH}"
    )

    electron_df_reg = original_train_df[
        original_train_df[args.electron_flag_column] == 1
    ].copy()
    if electron_df_reg.empty:
        logger.warning(
            "REG: No true electrons found in training data. Skipping regression pipeline."
        )
    else:
        exclude_cols_reg = list(
            set(
                args.exclude_columns_common
                + [args.reg_target_column, args.electron_flag_column]
            )
        )
        feature_cols_reg_initial = [
            col
            for col in electron_df_reg.columns.tolist()
            if col not in exclude_cols_reg
        ]

        (
            X_train_reg,
            X_val_reg,
            X_test_dev_reg,
            y_train_reg,
            y_val_reg,
            y_test_dev_reg,
            final_selected_features_reg,
        ) = prepare_data(
            electron_df_reg,
            args.reg_target_column,
            feature_cols_reg_initial,
            args.reg_selected_features_list,
            args.reg_max_features,
            args.random_seed,
            args.dev_test_size,
            args.validation_size,
            task_type="regression",
            disable_feature_selection=args.disable_reg_feature_selection,
        )

        if X_train_reg.empty:
            logger.error(
                "Regression training data is empty after preparation. Skipping regression pipeline."
            )
        else:
            N_FEATURES_FINAL_REG = len(final_selected_features_reg)
            logger.info(
                f"Number of final selected features for REG: {N_FEATURES_FINAL_REG}"
            )

            reg_scaler = StandardScaler()
            X_train_reg_scaled = reg_scaler.fit_transform(X_train_reg)
            X_val_reg_scaled = (
                reg_scaler.transform(X_val_reg) if not X_val_reg.empty else np.array([])
            )  # Ensure NumPy array
            joblib.dump(reg_scaler, os.path.join(REG_OUTPUT_DIR, "reg_scaler.joblib"))

            best_hps_reg = None
            if not args.skip_reg_tuning:
                logger.info(
                    "NNR5: Hyperparameter Tuning for REG NN using Keras Tuner..."
                )

                class RegHyperModel(kt.HyperModel):
                    def __init__(self, num_features):
                        self.num_features = num_features

                    def build(self, hp):
                        return build_regression_model_kt(hp, self.num_features)

                reg_hypermodel_instance = RegHyperModel(
                    num_features=N_FEATURES_FINAL_REG
                )
                tuner_reg = kt.Hyperband(
                    hypermodel=reg_hypermodel_instance,
                    objective=kt.Objective("val_loss", direction="min"),
                    max_epochs=args.reg_kt_epochs,
                    factor=3,
                    hyperband_iterations=1,
                    directory=os.path.join(REG_OUTPUT_DIR, "kt_reg_tuning_dir"),
                    project_name="regression_tuning",
                    seed=args.random_seed,
                    overwrite=True,
                )
                early_stopping_tuner_reg = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    mode="min",
                    restore_best_weights=True,
                    verbose=1,
                )
                logger.info(
                    f"Starting Keras Tuner search for REG (Max Trials: {args.reg_kt_max_trials}, Epochs per trial: {args.reg_kt_epochs})..."
                )

                # CORRECTED: Check original X_val_reg and y_val_reg for emptiness
                if X_val_reg.empty or y_val_reg.empty:
                    logger.warning(
                        "REG: Validation data for Keras Tuner is empty. Tuning will proceed without validation-based early stopping or metrics."
                    )
                    tuner_reg.search(
                        X_train_reg_scaled,
                        y_train_reg,
                        epochs=args.reg_kt_epochs,
                        verbose=2,
                    )
                else:
                    tuner_reg.search(
                        X_train_reg_scaled,
                        y_train_reg,
                        epochs=args.reg_kt_epochs,
                        validation_data=(
                            X_val_reg_scaled,
                            y_val_reg,
                        ),  # X_val_reg_scaled is numpy
                        callbacks=[early_stopping_tuner_reg],
                        verbose=2,
                    )
                try:
                    best_hps_reg = tuner_reg.get_best_hyperparameters(num_trials=1)[0]
                    logger.info(
                        f"Best REG Hyperparameters found: {best_hps_reg.values}"
                    )
                except IndexError:
                    logger.error(
                        "Keras Tuner for REG found no best HPs. Using defaults."
                    )
                    best_hps_reg = None

            logger.info("NNR6: Training final REG NN model...")
            if best_hps_reg:
                logger.info(
                    "Using Keras Tuner's best hyperparameters for final REG model."
                )
            else:
                logger.info(
                    "Using default hyperparameters for final REG model (tuning skipped or failed)."
                )

            final_lr_reg = best_hps_reg.get("lr") if best_hps_reg else 0.001
            final_num_layers_reg = best_hps_reg.get("num_layers") if best_hps_reg else 1
            final_units_reg = []
            final_dropout_rates_reg = []
            final_dropout_enabled_per_layer_reg = []

            if best_hps_reg:
                for i in range(final_num_layers_reg):
                    final_units_reg.append(best_hps_reg.get(f"units_{i}", 64))
                    layer_dropout_enabled = best_hps_reg.get(
                        f"dropout_enabled_{i}", True
                    )
                    final_dropout_enabled_per_layer_reg.append(layer_dropout_enabled)
                    if layer_dropout_enabled:
                        final_dropout_rates_reg.append(
                            best_hps_reg.get(f"dropout_rate_{i}", 0.1)
                        )
                    else:
                        final_dropout_rates_reg.append(0.0)
            else:
                final_units_reg = [64] * final_num_layers_reg
                final_dropout_enabled_per_layer_reg = [True] * final_num_layers_reg
                final_dropout_rates_reg = [0.1] * final_num_layers_reg

            def get_final_reg_model_for_scikeras(
                num_features,
                hidden_layer_sizes_tuple,
                dropout_rates_tuple,
                learning_rate,
                dropout_enabled_list_tuple,
            ):
                model = Sequential(name="final_regression_nn")
                model.add(Input(shape=(num_features,), name="input_layer_final_reg"))
                hidden_layer_sizes = list(hidden_layer_sizes_tuple)
                dropout_rates = list(dropout_rates_tuple)
                dropout_enabled_list = list(dropout_enabled_list_tuple)
                for i, neurons in enumerate(hidden_layer_sizes):
                    model.add(
                        Dense(
                            neurons, activation="relu", name=f"final_reg_hidden_{i+1}"
                        )
                    )
                    if (
                        i < len(dropout_enabled_list)
                        and dropout_enabled_list[i]
                        and i < len(dropout_rates)
                        and dropout_rates[i] > 0
                    ):
                        model.add(
                            Dropout(dropout_rates[i], name=f"final_reg_dropout_{i+1}")
                        )
                model.add(Dense(1, activation="linear", name="output_layer_final_reg"))
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(
                    optimizer=optimizer,
                    loss="mean_absolute_error",
                    metrics=["mae", "mse"],
                )
                return model

            final_reg_pipeline_model = KerasRegressor(
                model=get_final_reg_model_for_scikeras,
                model__num_features=N_FEATURES_FINAL_REG,
                model__hidden_layer_sizes_tuple=tuple(final_units_reg),
                model__dropout_rates_tuple=tuple(final_dropout_rates_reg),
                model__learning_rate=final_lr_reg,
                model__dropout_enabled_list_tuple=tuple(
                    final_dropout_enabled_per_layer_reg
                ),
                epochs=args.final_model_epochs,
                batch_size=128,
                verbose=0,
                random_state=args.random_seed,
            )
            final_regression_pipeline = Pipeline(
                [("scaler", reg_scaler), ("nn", final_reg_pipeline_model)]
            )
            early_stopping_final_reg = EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
                restore_best_weights=True,
                verbose=1,
            )
            logger.info(
                f"Fitting final REG pipeline (Epochs: {args.final_model_epochs})..."
            )

            fit_params_reg = {}
            # CORRECTED: Check original X_val_reg and y_val_reg for emptiness
            if not X_val_reg.empty and not y_val_reg.empty:
                # X_val_reg_scaled is already a NumPy array if X_val_reg was not empty
                fit_params_reg["nn__validation_data"] = (X_val_reg_scaled, y_val_reg)
                fit_params_reg["nn__callbacks"] = [early_stopping_final_reg]
            else:
                logger.warning(
                    "REG: Final model training without validation data for early stopping."
                )

            final_regression_pipeline.fit(
                X_train_reg, y_train_reg, **fit_params_reg
            )  # X_train_reg is DataFrame
            joblib.dump(
                final_regression_pipeline,
                os.path.join(REG_OUTPUT_DIR, "final_regression_pipeline_kt.joblib"),
            )
            logger.info(f"Final REG Pipeline (Keras Tuner) saved to {REG_OUTPUT_DIR}")

            logger.info("NNR7: Evaluating REG NN model on dev test set...")
            if not X_test_dev_reg.empty:
                dev_reg_pred = final_regression_pipeline.predict(X_test_dev_reg)
                logger.info(
                    f"  REG Dev MAE: {mean_absolute_error(y_test_dev_reg, dev_reg_pred):.4f} GeV"
                )
                logger.info(
                    f"  REG Dev R2 Score: {r2_score(y_test_dev_reg, dev_reg_pred):.4f}"
                )
                logger.info(
                    f"  REG Dev MARE: {mean_absolute_relative_error(y_test_dev_reg, dev_reg_pred):.4f}"
                )
            else:
                logger.warning("REG Dev test data is empty. Skipping evaluation.")

            logger.info("NNR8: Generating REG predictions for submission file...")
            try:
                final_reg_test_df = pd.read_hdf(args.reg_test_path)
                if args.electron_flag_column in final_reg_test_df.columns:
                    final_reg_test_df_electrons = final_reg_test_df[
                        final_reg_test_df[args.electron_flag_column] == 1
                    ].copy()
                    logger.info(
                        f"Filtered final REG test set for electrons. Shape: {final_reg_test_df_electrons.shape}"
                    )
                else:
                    logger.warning(
                        f"Electron flag column '{args.electron_flag_column}' not found in REG test set. Using all test data for regression predictions."
                    )
                    final_reg_test_df_electrons = final_reg_test_df.copy()

                if not final_reg_test_df_electrons.empty:
                    X_final_reg_test_subset = final_reg_test_df_electrons[
                        final_selected_features_reg
                    ]
                    final_reg_pred_test = final_regression_pipeline.predict(
                        X_final_reg_test_subset
                    )
                    ids_reg_submission = np.arange(len(final_reg_test_df_electrons))
                    reg_submission_df_predictions = pd.DataFrame(
                        {"id": ids_reg_submission, "prediction": final_reg_pred_test}
                    )
                    save_submission_files(
                        reg_submission_df_predictions,
                        final_selected_features_reg,
                        REG_PREDICTIONS_SUBMISSION_PATH,
                        REG_VARIABLELIST_SUBMISSION_PATH,
                    )
                else:
                    logger.info(
                        "REG: No data (or no electrons) in final test set after filtering. Skipping REG submission file generation."
                    )
            except FileNotFoundError:
                logger.warning(
                    f"REG test file not found at {args.reg_test_path}. Skipping final REG submission file generation."
                )
            except Exception as e:
                logger.error(
                    f"Error during REG final test prediction/submission file generation: {e}",
                    exc_info=True,
                )

    logger.info("\n" + "=" * 30 + " ALL PIPELINES FINISHED " + "=" * 30)


if __name__ == "__main__":
    cli_args = parse_arguments()
    try:
        main(cli_args)
        logger.info("Script finished successfully.")
    except Exception as e_main:
        logger.error(
            f"An unhandled error occurred in main execution: {e_main}", exc_info=True
        )
        import sys

        sys.exit(1)  # Exit with a non-zero status code on error
