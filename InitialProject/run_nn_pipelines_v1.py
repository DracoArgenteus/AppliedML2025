import pandas as pd
import numpy as np
import joblib 
import os
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import argparse 
import logging 
import time

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    log_loss, roc_auc_score, average_precision_score, f1_score, 
    confusion_matrix, classification_report, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam # Changed from tf.keras.optimizers.legacy to tf.keras.optimizers
    from scikeras.wrappers import KerasClassifier, KerasRegressor
except ImportError:
    print("TensorFlow/Keras/SciKeras not found. Please install these libraries for the NN pipeline.")
    # As a student, ensure your environment (e.g., Conda, venv) has these packages.
    # You can typically install them with:
    # pip install tensorflow scikeras scikit-learn pandas joblib matplotlib seaborn optuna
    # Or using Mamba/Conda:
    # mamba install tensorflow scikeras scikit-learn pandas joblib matplotlib seaborn optuna -c conda-forge (or appropriate channels)
    raise

import optuna 
try:
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_slice, plot_intermediate_values
    optuna_viz_available = True
except ImportError:
    print("Optuna visualization with Matplotlib backend not available. Skipping Optuna plots.")
    optuna_viz_available = False


# --- Setup Logging ---
# Consistent and informative logging is crucial for debugging, especially in complex pipelines.
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Custom Metric Function for Regression ---
def mean_absolute_relative_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Relative Error (MARE).
    Filters out y_true values that are zero or very close to zero to avoid division by zero.
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
    Returns:
        float: The MARE score, or np.nan if no valid y_true values are found.
    """
    epsilon = 1e-9 # Small constant to prevent division by zero if y_true is exactly zero
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    
    # Identify indices where y_true is substantially different from zero
    # This is important as division by a very small y_true can lead to huge relative errors.
    valid_indices = np.abs(y_true_np) > (epsilon * 1000) 
    
    if not np.any(valid_indices):
        logger.warning("mean_absolute_relative_error: All y_true values are too close to zero. MARE is undefined. Returning NaN.")
        return np.nan 
    
    y_true_filt = y_true_np[valid_indices]
    y_pred_filt = y_pred_np[valid_indices]

    # This check should ideally be redundant if the above `np.any(valid_indices)` is false,
    # but it's a good safeguard.
    if len(y_true_filt) == 0: 
        logger.warning("mean_absolute_relative_error: y_true_filt became empty after filtering (unexpected). Returning NaN.")
        return np.nan
        
    # Calculate relative error. Add epsilon to y_true_filt in the denominator for numeric stability,
    # though `valid_indices` filter should make y_true_filt reasonably large.
    relative_error = (y_pred_filt - y_true_filt) / (y_true_filt + epsilon) 
    return np.mean(np.abs(relative_error))

# --- Keras Model Creation Functions ---
# These functions define the architecture of the neural networks.
# They are passed to the SciKeras wrappers.

def create_classification_nn_model(num_features, hidden_layer_sizes=(64,32), dropout_rate=0.2, learning_rate=0.001):
    """
    Creates a Keras Sequential model for binary classification.
    Args:
        num_features (int): The number of input features.
        hidden_layer_sizes (tuple): Tuple of integers defining the number of neurons in each hidden layer.
        dropout_rate (float): Dropout rate to apply after each hidden layer.
        learning_rate (float): Learning rate for the Adam optimizer.
    Returns:
        tensorflow.keras.models.Sequential: Compiled Keras model.
    """
    model = Sequential(name="classification_nn")
    model.add(Input(shape=(num_features,), name="input_layer")) 
    for i, neurons in enumerate(hidden_layer_sizes):
        model.add(Dense(neurons, activation='relu', name=f"cls_hidden_{i+1}"))
        if dropout_rate > 0: # Apply dropout only if rate is positive
            model.add(Dropout(dropout_rate, name=f"cls_dropout_{i+1}"))
    model.add(Dense(1, activation='sigmoid', name="output_layer")) # Sigmoid for binary classification
    
    optimizer = Adam(learning_rate=learning_rate)
    # Key metrics for classification: accuracy and AUC (Area Under ROC Curve).
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# MODIFIED: Removed 'meta' parameter from the function signature in the original user code.
def create_regression_nn_model(num_features, hidden_layer_sizes=(64,32), learning_rate=0.001, dropout_rate=0.2):
    """
    Creates a Keras Sequential model for regression.
    Args:
        num_features (int): The number of input features.
        hidden_layer_sizes (tuple): Tuple of integers defining the number of neurons in each hidden layer.
        learning_rate (float): Learning rate for the Adam optimizer.
        dropout_rate (float): Dropout rate to apply after each hidden layer.
    Returns:
        tensorflow.keras.models.Sequential: Compiled Keras model.
    """
    model = Sequential(name="regression_nn")
    model.add(Input(shape=(num_features,), name="input_layer"))
    for i, neurons in enumerate(hidden_layer_sizes):
        model.add(Dense(neurons, activation='relu', name=f"reg_hidden_{i+1}"))
        if dropout_rate > 0: # Apply dropout only if rate is positive
            model.add(Dropout(dropout_rate, name=f"reg_dropout_{i+1}"))
    model.add(Dense(1, activation='linear', name="output_layer")) # Linear activation for regression output
    
    optimizer = Adam(learning_rate=learning_rate)
    # Key metrics for regression: MAE (Mean Absolute Error) and MSE (Mean Squared Error).
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae', 'mse'])
    return model

# --- Argument Parsing ---
# Defines command-line arguments for the script, allowing for flexible configuration.
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Neural Network pipelines for classification and regression.")
    # Paths and directories
    parser.add_argument('--output_base_dir', type=str, required=True, help='Base directory for all script outputs.')
    parser.add_argument('--train_path', type=str, default="./data/AppML_InitialProject_train.h5", help='Path to the training data HDF5 file.')
    
    # Classification specific arguments
    parser.add_argument('--cls_test_path', type=str, default="./data/AppML_InitialProject_test_classification.h5", help='Path to the CLASSIFICATION test data HDF5 file.')
    parser.add_argument('--cls_target_column', type=str, default='p_Truth_isElectron', help='Target column for classification.')
    parser.add_argument('--cls_max_features', type=int, default=25, help='Maximum number of features to select for classification.')
    parser.add_argument('--cls_n_optuna_trials', type=int, default=15, help='Number of Optuna trials for classification NN hyperparameter tuning.')
    parser.add_argument('--cls_selected_features_list', type=str, default=None, help='Comma-separated string of pre-selected feature names for classification. Highly recommended for reproducibility and targeted modeling.')
    parser.add_argument('--skip_cls_data_visualization', action='store_true', default=False, help="Skip CLS initial data/feature visualization steps.")
    parser.add_argument('--skip_cls_final_plots', action='store_true', default=False, help="Skip CLS final evaluation plots (e.g., ROC curve).")
    parser.add_argument('--skip_cls_optuna', action='store_true', default=False, help="Skip Optuna hyperparameter tuning for classification NN and use predefined defaults.")
    parser.add_argument('--skip_cls_optuna_plots', action='store_true', default=False, help="Skip generating Optuna visualization plots for classification.")

    # Regression specific arguments
    parser.add_argument('--reg_test_path', type=str, default="./data/AppML_InitialProject_test_regression.h5", help='Path to the REGRESSION test data HDF5 file.')
    parser.add_argument('--reg_target_column', type=str, default='p_Truth_Energy', help='Target column for regression (e.g., particle energy).')
    parser.add_argument('--electron_flag_column', type=str, default='p_Truth_isElectron', help='Column name for the flag indicating if a particle is an electron (used for filtering data for regression).')
    parser.add_argument('--reg_max_features', type=int, default=12, help='Maximum number of features to select for regression.')
    parser.add_argument('--reg_n_optuna_trials', type=int, default=15, help='Number of Optuna trials for regression NN hyperparameter tuning.')
    parser.add_argument('--reg_selected_features_list', type=str, default=None, help='Comma-separated string of pre-selected feature names for regression. Highly recommended.')
    parser.add_argument('--skip_reg_data_visualization', action='store_true', default=False, help="Skip REG initial data/feature/target visualization steps.")
    parser.add_argument('--skip_reg_final_plots', action='store_true', default=False, help="Skip REG final evaluation plots (e.g., true vs. predicted scatter plot).")
    parser.add_argument('--skip_reg_optuna', action='store_true', default=False, help="Skip Optuna hyperparameter tuning for regression NN and use predefined defaults.")
    parser.add_argument('--skip_reg_optuna_plots', action='store_true', default=False, help="Skip generating Optuna visualization plots for regression.")

    # Common arguments for both pipelines
    parser.add_argument('--exclude_columns_common', nargs='*', default=['p_Truth_Energy', 'p_Truth_isElectron'], help='List of columns to generally exclude from being features (e.g., target variables themselves).')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility of data splits and model initializations.')
    parser.add_argument('--validation_size', type=float, default=0.20, help='Proportion of the (training+validation) data to use for the validation set.')
    parser.add_argument('--dev_test_size', type=float, default=0.15, help='Proportion of the original training data to hold out as a development test set.')
    parser.add_argument('--cv_folds', type=int, default=3, help='Number of cross-validation folds for Optuna evaluation.')
    
    # SLURM/CPU related arguments for Optuna parallelism
    parser.add_argument('--slurm_cpus_allocated', type=int, default=os.cpu_count(), help='Total number of CPU cores allocated to the job (e.g., from SLURM_CPUS_PER_TASK). Defaults to os.cpu_count().')
    parser.add_argument('--nn_threads_per_trial', type=int, default=1, help='Expected/target number of main threads Keras/TensorFlow will use per Optuna trial. This helps calculate Optuna\'s n_jobs for parallel trials.')
    
    return parser.parse_args()

# --- Main Pipeline Function ---
def main(args):
    # --- Global Setup ---
    OUTPUT_BASE_DIR = args.output_base_dir
    TRAIN_PATH = args.train_path
    RANDOM_SEED = args.random_seed # Critical for reproducible research
    VALIDATION_SIZE = args.validation_size # For splitting train into train/val
    DEV_TEST_SIZE = args.dev_test_size     # For splitting original_train into a held-out dev test set
    CV_FOLDS = args.cv_folds               # For Optuna's internal cross-validation
    SLURM_CPUS_ALLOCATED = args.slurm_cpus_allocated
    NN_THREADS_PER_TRIAL = args.nn_threads_per_trial

    # Calculate Optuna n_jobs for parallel trials.
    # This aims to utilize available CPUs without oversubscribing, assuming each TF trial uses NN_THREADS_PER_TRIAL.
    if NN_THREADS_PER_TRIAL <= 0 or NN_THREADS_PER_TRIAL > SLURM_CPUS_ALLOCATED:
        # If config is invalid, run Optuna trials sequentially.
        actual_nn_threads_hint_for_optuna = SLURM_CPUS_ALLOCATED 
        OPTUNA_N_JOBS = 1 
        logger.info(f"NN_THREADS_PER_TRIAL ({NN_THREADS_PER_TRIAL}) is invalid or exceeds allocated CPUs ({SLURM_CPUS_ALLOCATED}). Optuna trials will run sequentially (n_jobs=1).")
    else:
        actual_nn_threads_hint_for_optuna = NN_THREADS_PER_TRIAL
        OPTUNA_N_JOBS = max(1, SLURM_CPUS_ALLOCATED // actual_nn_threads_hint_for_optuna)
        logger.info(f"Target NN threads per Optuna trial (for Optuna n_jobs calc): {actual_nn_threads_hint_for_optuna}")
        logger.info(f"Calculated Optuna n_jobs (parallel trials) for NN: {OPTUNA_N_JOBS}")
    
    if OPTUNA_N_JOBS > 1:
        logger.warning(f"Optuna running {OPTUNA_N_JOBS} NN trials in parallel. TensorFlow can be greedy with cores; monitor CPU usage to ensure it aligns with expectations and SLURM allocations.")

    # Create base output directory if it doesn't exist
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    logger.info(f"Global Output Base Directory: {OUTPUT_BASE_DIR}")

    # Set random seeds for all relevant libraries to ensure reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED) # For TensorFlow's own random operations
    # Note: Optuna studies also use a random_state, often implicitly through samplers or CV splits.
    
    logger.info(f"Loading original full training data from: {TRAIN_PATH}")
    try:
        original_train_df = pd.read_hdf(TRAIN_PATH) 
    except FileNotFoundError:
        logger.error(f"HDF5 training data file not found at {TRAIN_PATH}. Please check the path.")
        raise
    except Exception as e:
        logger.error(f"Error loading HDF5 training data from {TRAIN_PATH}: {e}.")
        raise
    logger.info(f"Original training data shape: {original_train_df.shape}")

    # ==============================================================================
    # Part 1: Neural Network for Classification
    # ==============================================================================
    logger.info("\n\n--- Starting Part 1: Neural Network for Classification ---")
    CLS_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "classification_nn")
    os.makedirs(CLS_OUTPUT_DIR, exist_ok=True) # Ensure classification-specific output dir exists
    logger.info(f"Classification NN outputs will be in: {CLS_OUTPUT_DIR}")
    
    logger.info("NNC1: Preparing data for classification...")
    all_cols_cls = original_train_df.columns.tolist()
    # Define columns to exclude: common exclusions + specific target for classification
    exclude_cols_cls = list(set(args.exclude_columns_common + [args.cls_target_column]))
    feature_cols_cls_initial = [col for col in all_cols_cls if col not in exclude_cols_cls]

    if not feature_cols_cls_initial:
        logger.error("CLS: No initial features found after exclusions. Check column names in data and --exclude_columns_common, --cls_target_column arguments.")
        raise ValueError("CLS: No initial features available for classification.")
    
    X_original_cls = original_train_df[feature_cols_cls_initial]
    y_original_cls = original_train_df[args.cls_target_column]
    
    logger.info(f"CLS: Splitting data (Original Train -> Dev Temp + Dev Test; Dev Temp -> Train + Val)...")
    # First split: Separate out a development test set from the original training data.
    # This dev test set is held out until the very end of this part of the pipeline.
    X_dev_temp_cls, X_test_dev_cls, y_dev_temp_cls, y_test_dev_cls = train_test_split(
        X_original_cls, y_original_cls, 
        test_size=DEV_TEST_SIZE, # Proportion for the dev test set
        random_state=RANDOM_SEED, 
        stratify=y_original_cls # Stratify by target for classification to maintain class proportions
    )
    # Second split: Create training and validation sets from the remaining data (X_dev_temp_cls).
    # The validation set is used for early stopping and hyperparameter tuning.
    # Adjust validation_size because it's a proportion of X_dev_temp_cls, not original_train_df.
    val_size_adjusted_cls = VALIDATION_SIZE / (1 - DEV_TEST_SIZE) if (1 - DEV_TEST_SIZE) > 0 else 0 
    X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
        X_dev_temp_cls, y_dev_temp_cls, 
        test_size=val_size_adjusted_cls, 
        random_state=RANDOM_SEED, 
        stratify=y_dev_temp_cls # Stratify again
    )
    logger.info(f"  CLS X_train shape: {X_train_cls.shape}, CLS y_train shape: {y_train_cls.shape}")
    logger.info(f"  CLS X_val shape: {X_val_cls.shape}, CLS y_val shape: {y_val_cls.shape}")
    logger.info(f"  CLS X_test_dev (development test) shape: {X_test_dev_cls.shape}, CLS y_test_dev shape: {y_test_dev_cls.shape}")

    logger.info("NNC2: Feature Selection for Classification...")
    if args.cls_selected_features_list:
        # Use pre-selected features if a list is provided via command line
        selected_features_from_arg = [f.strip() for f in args.cls_selected_features_list.split(',')]
        # Filter to ensure features actually exist in the training data columns and respect max_features
        final_selected_features_cls = [f for f in selected_features_from_arg if f in X_train_cls.columns][:args.cls_max_features]
        if len(final_selected_features_cls) < len(selected_features_from_arg):
            if len(selected_features_from_arg) > args.cls_max_features:
                 logger.warning(f"CLS: More features provided via --cls_selected_features_list ({len(selected_features_from_arg)}) than --cls_max_features ({args.cls_max_features}). Using the first {args.cls_max_features} valid features.")
            else:
                 logger.warning(f"CLS: Some features from --cls_selected_features_list were not found in the training data or were duplicates. Using {len(final_selected_features_cls)} valid features.")
    else:
        # Fallback: if no features are pre-selected, use the first N available features.
        # This is a placeholder and NOT recommended for rigorous scientific modeling.
        # Feature selection should be a deliberate process (e.g., based on domain knowledge, feature importance studies).
        logger.warning("CLS: No pre-selected features provided via --cls_selected_features_list. Using the first --cls_max_features available features as a placeholder. THIS IS STRONGLY NOT RECOMMENDED FOR ACCURATE OR REPRODUCIBLE MODELING. Perform proper feature selection.")
        final_selected_features_cls = X_train_cls.columns.tolist()[:args.cls_max_features]
    
    if not final_selected_features_cls: # Should not happen if X_train_cls.columns is not empty
        logger.error("CLS: No features were selected. This can happen if --cls_max_features is 0, or if X_train_cls has no columns after initial processing. Critical error.")
        raise ValueError("CLS: No features selected for classification. Pipeline cannot continue.")
    
    logger.info(f"CLS: Final {len(final_selected_features_cls)} features selected for use in the model: {final_selected_features_cls[:5]}..." + (f" (and {len(final_selected_features_cls)-5} more)" if len(final_selected_features_cls)>5 else ""))
    
    # Create subsets of data using only the selected features
    X_train_cls_subset = X_train_cls[final_selected_features_cls]
    X_val_cls_subset = X_val_cls[final_selected_features_cls]
    X_test_dev_cls_subset = X_test_dev_cls[final_selected_features_cls]
    N_FEATURES_FINAL_CLS = len(final_selected_features_cls) # Actual number of features input to the NN

    if not args.skip_cls_data_visualization:
        logger.info("NNC2.5: Visualizing CLS features (Simplified - placeholder for actual visualization code)...")
        # Placeholder: In a real scenario, add code here to visualize feature distributions, correlations, etc.
        # For example, using seaborn.pairplot (can be slow for many features/samples) or histograms.
        # try:
        #     sample_df_vis = X_train_cls_subset.sample(n=min(500, len(X_train_cls_subset)), random_state=RANDOM_SEED)
        #     # sns.pairplot(sample_df_vis) # This can be very resource-intensive
        #     # plt.savefig(os.path.join(CLS_OUTPUT_DIR, "sNNC2.5_cls_feature_pairplot_sample.png"))
        #     # plt.clf(); plt.close()
        #     logger.info("   (Actual feature visualization plot generation is commented out in this template)")
        # except Exception as e_vis:
        #     logger.warning(f"   Could not generate sample CLS feature visualization: {e_vis}")
    else:
        logger.info("Skipping CLS data visualization (NNC2.5) as per --skip_cls_data_visualization flag.")

    best_params_cls_nn_dict = {} 
    if args.skip_cls_optuna:
        logger.info("NNC5: Skipping Optuna for CLS NN. Using default NN parameters.")
        # Define default parameters if Optuna is skipped
        best_params_cls_nn_dict = {
            'hidden_layer_sizes': (64, 32), 
            'dropout_rate': 0.2, 
            'learning_rate': 0.001, 
            'batch_size': 128, # This is a Keras fit param, handled by KerasClassifier
            'epochs': 50      # This is a Keras fit param, handled by KerasClassifier
        }
    else:
        logger.info("NNC5: Hyperparameter Tuning for CLS NN using Optuna...")
        # Optuna objective function: defines how a single trial is evaluated.
        def optuna_objective_cls(trial):
            # Define search space for hyperparameters using trial.suggest_... methods
            hidden_layers = trial.suggest_int('cls_n_layers', 1, 2) # Number of hidden layers
            # Neurons per layer: using a loop allows for dynamic number of layers
            neurons_l = [trial.suggest_int(f'cls_neurons_l{i+1}', 32, 128, log=True) for i in range(hidden_layers)] 
            dropout = trial.suggest_float('cls_dropout', 0.1, 0.4) # Dropout rate
            lr = trial.suggest_float('cls_lr', 1e-4, 1e-2, log=True) # Learning rate (log scale is common)
            batch_s = trial.suggest_categorical('cls_batch_size', [64, 128, 256]) # Batch size for training
            epochs_s = trial.suggest_int('cls_epochs', 30, 80) # Number of epochs for training
            
            # Create KerasClassifier with current trial's parameters
            # SciKeras routes parameters: those matching the model function's signature are passed to it,
            # others (like epochs, batch_size) are used by the wrapper itself.
            cls_nn_optuna = KerasClassifier( 
                model=create_classification_nn_model, 
                num_features=N_FEATURES_FINAL_CLS, # This is a fixed model parameter
                # Parameters for create_classification_nn_model:
                hidden_layer_sizes=tuple(neurons_l), 
                dropout_rate=dropout, 
                learning_rate=lr,
                # Parameters for KerasClassifier.fit:
                epochs=epochs_s, 
                batch_size=batch_s,
                verbose=0, # Suppress Keras training logs during Optuna trials for cleaner output
                random_state=RANDOM_SEED # For reproducibility of model initialization within Optuna trial
            )
            # Create a scikit-learn pipeline: StandardScaler -> KerasClassifier
            # Scaling is crucial for neural network performance.
            pipeline_cls_optuna = Pipeline([('scaler', StandardScaler()), ('nn', cls_nn_optuna)])
            
            # Cross-validation setup within the Optuna trial
            cv_cls = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            logloss_scores = [] # Store score for each fold

            for fold, (train_idx, val_idx) in enumerate(cv_cls.split(X_train_cls_subset, y_train_cls)):
                X_f, X_v = X_train_cls_subset.iloc[train_idx], X_train_cls_subset.iloc[val_idx]
                y_f, y_v = y_train_cls.iloc[train_idx], y_train_cls.iloc[val_idx]
                try:
                    # Fit the pipeline on the current fold's training data
                    pipeline_cls_optuna.fit(X_f, y_f) 
                    # Predict probabilities on the validation data for this fold
                    preds_proba = pipeline_cls_optuna.predict_proba(X_v)[:, 1] # Probability of positive class
                    # Clip probabilities to avoid issues with log_loss (log(0) or log(1) are undefined)
                    epsilon=1e-15 # Small constant for clipping
                    preds_proba = np.clip(preds_proba, epsilon, 1 - epsilon)
                    score = log_loss(y_v, preds_proba) # LogLoss is a common metric for binary classification
                    logloss_scores.append(score)
                except Exception as e:
                    # If a trial fold fails (e.g., due to the __sklearn_tags__ error or model failing to train),
                    # log it and return a high error value to Optuna.
                    logger.warning(f"Optuna CLS Trial {trial.number} Fold {fold+1} encountered an error: {e}. Returning float('inf').")
                    return float('inf') # Indicate failure for this trial to Optuna
                
                # Optuna pruning: report intermediate value (score for this fold)
                # This allows Optuna to stop unpromising trials early.
                trial.report(score, step=fold)
                if trial.should_prune():
                    logger.info(f"Optuna CLS Trial {trial.number} Pruned at fold {fold+1} due to unpromising results.")
                    raise optuna.exceptions.TrialPruned()
            
            return np.mean(logloss_scores) # Return the average log loss over all CV folds for this trial

        optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce Optuna's own logging verbosity for cleaner output
        # Create Optuna study: direction='minimize' because we want to minimize log_loss.
        # A pruner is used to cut short trials that are not performing well.
        study_cls = optuna.create_study(direction='minimize', 
                                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1, CV_FOLDS-1), n_min_trials=max(5, CV_FOLDS))) 
        
        logger.info(f"Starting Optuna CLS NN ({args.cls_n_optuna_trials} trials) with OPTUNA_N_JOBS={OPTUNA_N_JOBS}...")
        # Run Optuna optimization process
        study_cls.optimize(optuna_objective_cls, n_trials=args.cls_n_optuna_trials, n_jobs=OPTUNA_N_JOBS, 
                           # Consider adding a timeout if trials can hang indefinitely
                           # timeout=3600 # Example: 1 hour timeout for the whole study
                           )
        
        # MODIFIED: Handle cases where Optuna study might not find any successful trials
        if study_cls.best_trial is None or (hasattr(study_cls.best_value, 'is_infinite') and study_cls.best_value.is_infinite()) or (isinstance(study_cls.best_value, float) and np.isinf(study_cls.best_value)):
            logger.error("Optuna CLS study failed to find any valid (non-infinite) parameters. This often happens if all trials encounter errors (e.g., model training failures, library incompatibilities like the '__sklearn_tags__' issue). Falling back to default parameters for CLS NN.")
            best_params_cls_nn_dict = {
                'hidden_layer_sizes': (64, 32), 
                'dropout_rate': 0.2, 
                'learning_rate': 0.001, 
                'batch_size': 128,
                'epochs': 50
            }
            logger.info(f"Using CLS default parameters: {best_params_cls_nn_dict}")
        else:
            optuna_best_params = study_cls.best_params
            best_logloss_cls_nn = study_cls.best_value
            logger.info(f"Optuna CLS study finished. Best LogLoss: {best_logloss_cls_nn:.4f}")
            logger.info(f"Best CLS parameters found by Optuna: {optuna_best_params}")
            
            # Store best parameters found by Optuna for the final model training
            best_params_cls_nn_dict = {
                'hidden_layer_sizes': tuple(optuna_best_params[f'cls_neurons_l{i+1}'] for i in range(optuna_best_params['cls_n_layers'])),
                'dropout_rate': optuna_best_params['cls_dropout'],
                'learning_rate': optuna_best_params['cls_lr'],
                'epochs': optuna_best_params['cls_epochs'], # Keras fit param
                'batch_size': optuna_best_params['cls_batch_size'] # Keras fit param
            }

        # Generate and save Optuna visualization plots if enabled and available
        if optuna_viz_available and not args.skip_cls_optuna_plots: 
            logger.info("NNC5: Generating Optuna plots for CLS...")
            # It's good practice to wrap plot generation in try-except blocks,
            # as plotting can sometimes fail with unusual study results (e.g., all trials failed).
            try:
                if study_cls.trials and any(t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and not np.isinf(t.value) for t in study_cls.trials):
                    plot_optimization_history(study_cls).figure.savefig(os.path.join(CLS_OUTPUT_DIR, "sNNC5_optuna_history_cls.pdf")); plt.clf(); plt.close()
                    plot_param_importances(study_cls).figure.savefig(os.path.join(CLS_OUTPUT_DIR, "sNNC5_optuna_param_importances_cls.pdf")); plt.clf(); plt.close()
                    plot_slice(study_cls).figure.savefig(os.path.join(CLS_OUTPUT_DIR, "sNNC5_optuna_slice_plot_cls.pdf")); plt.clf(); plt.close()
                    if CV_FOLDS > 1 : # Intermediate values plot is most useful with multiple steps (folds)
                         plot_intermediate_values(study_cls).figure.savefig(os.path.join(CLS_OUTPUT_DIR, "sNNC5_optuna_intermediate_values_cls.pdf")); plt.clf(); plt.close()
                    logger.info("   Saved Optuna plots for CLS.")
                else:
                    logger.warning("   Skipping Optuna plot generation for CLS as no successful trials were found or study is empty.")
            except Exception as e_opt_plot:
                logger.warning(f"Could not generate one or more Optuna plots for CLS: {e_opt_plot}")
            finally:
                plt.close('all') # Ensure all matplotlib figures are closed
        elif not optuna_viz_available:
            logger.warning("Optuna visualization (matplotlib backend) not available. Skipping Optuna plots for CLS.")
        else: # --skip_cls_optuna_plots is True
            logger.info("Skipping Optuna plots for CLS as per --skip_cls_optuna_plots flag.")

    logger.info(f"NNC6: Training final CLS NN model with parameters: {best_params_cls_nn_dict}")
    # Keras Callbacks for final model training, particularly EarlyStopping to prevent overfitting.
    keras_callbacks_cls = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',  # Monitor validation AUC (Area Under Curve)
            patience=10,        # Number of epochs with no improvement after which training will be stopped
            mode='max',         # For AUC, 'max' indicates we want to maximize it
            restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity
            verbose=1 # Log when early stopping is triggered
        )
    ]
    # Create final KerasClassifier instance with the (potentially default) best hyperparameters
    final_nn_classifier = KerasClassifier(
        model=create_classification_nn_model, 
        num_features=N_FEATURES_FINAL_CLS, 
        verbose=0, # Suppress Keras logs for final training unless debugging
        random_state=RANDOM_SEED,
        # Pass model construction parameters from best_params_cls_nn_dict
        **{k: v for k, v in best_params_cls_nn_dict.items() if k in ['hidden_layer_sizes', 'dropout_rate', 'learning_rate']},
        # Pass Keras fit parameters directly
        epochs=best_params_cls_nn_dict['epochs'],
        batch_size=best_params_cls_nn_dict['batch_size']
    )
    # Create the final scikit-learn pipeline for classification
    final_classification_pipeline = Pipeline([('scaler', StandardScaler()), ('nn', final_nn_classifier)])
    
    logger.info("Fitting final classification pipeline (NN) on X_train_cls_subset, validating on X_val_cls_subset...")
    # Fit the final pipeline. Pass validation data and callbacks to the KerasClassifier via 'nn__' prefix.
    # This allows Keras to use the validation set for early stopping.
    try:
        final_classification_pipeline.fit(
            X_train_cls_subset, y_train_cls, 
            nn__validation_data=(X_val_cls_subset, y_val_cls), 
            nn__callbacks=keras_callbacks_cls
        )
    except Exception as e_fit_final_cls:
        logger.error(f"Error during final CLS pipeline fitting: {e_fit_final_cls}. The '__sklearn_tags__' error might occur here if SciKeras/sklearn versions are incompatible.")
        logger.error("This script version includes robustness for Optuna failures, but the underlying library issue (if present) for '__sklearn_tags__' needs to be resolved by checking/updating SciKeras, scikit-learn, and TensorFlow versions.")
        raise # Re-raise the exception to halt execution, as the model is not trained.

    # Save the trained pipeline using joblib for later use/inference
    cls_pipeline_save_path = os.path.join(CLS_OUTPUT_DIR, "sNNC6_final_classification_nn_pipeline.joblib")
    joblib.dump(final_classification_pipeline, cls_pipeline_save_path)
    logger.info(f"CLS NN pipeline saved to: {cls_pipeline_save_path}")

    logger.info("NNC7: Evaluating CLS NN model on the development test set (X_test_dev_cls_subset)...")
    # Note: The __sklearn_tags__ error can also occur during predict_proba if the pipeline/estimator state is problematic.
    try:
        dev_cls_pred_proba = final_classification_pipeline.predict_proba(X_test_dev_cls_subset)[:, 1]
    except Exception as e_pred_cls:
        logger.error(f"Error during CLS pipeline predict_proba on dev test set: {e_pred_cls}.")
        logger.error("This could be due to the same '__sklearn_tags__' library incompatibility. Ensure SciKeras, scikit-learn, and TensorFlow are compatible and up-to-date.")
        # If prediction fails, we cannot calculate further metrics.
        logger.info("--- Part 1: Classification NN Pipeline Halted due to prediction error ---")
        # Depending on desired behavior, either raise e_pred_cls or try to continue to Part 2 if they are independent.
        # For now, assume we should halt if a core part fails.
        raise 

    dev_cls_pred_class = (dev_cls_pred_proba > 0.5).astype(int) # Convert probabilities to class labels (0 or 1)
    
    # Calculate and log various classification metrics
    dev_log_loss = log_loss(y_test_dev_cls, dev_cls_pred_proba)
    dev_roc_auc = roc_auc_score(y_test_dev_cls, dev_cls_pred_proba)
    dev_avg_precision = average_precision_score(y_test_dev_cls, dev_cls_pred_proba) # Also known as AUPRC
    dev_f1 = f1_score(y_test_dev_cls, dev_cls_pred_class)

    logger.info(f"    CLS Dev LogLoss: {dev_log_loss:.4f}")
    logger.info(f"    CLS Dev ROC-AUC: {dev_roc_auc:.4f}")
    logger.info(f"    CLS Dev Avg. Precision (AUPRC): {dev_avg_precision:.4f}")
    logger.info(f"    CLS Dev F1-Score (0.5 threshold): {dev_f1:.4f}")
    
    # Display classification report (precision, recall, F1-score per class) and confusion matrix
    logger.info("    CLS Dev Classification Report:\n" + classification_report(y_test_dev_cls, dev_cls_pred_class))
    conf_matrix_cls_dev = confusion_matrix(y_test_dev_cls, dev_cls_pred_class)
    logger.info(f"    CLS Dev Confusion Matrix:\n{conf_matrix_cls_dev}")

    if not args.skip_cls_final_plots:
        logger.info("  NNC7: Generating CLS Dev Test ROC Curve plot...")
        try:
            fpr, tpr, _ = roc_curve(y_test_dev_cls, dev_cls_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {dev_roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line for random chance
            plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]) # Standard limits for ROC
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title('CLS Dev Test ROC Curve'); plt.legend(loc="lower right"); plt.grid(True)
            roc_curve_save_path = os.path.join(CLS_OUTPUT_DIR, "sNNC7_cls_dev_roc_curve.pdf")
            plt.savefig(roc_curve_save_path); plt.clf(); plt.close()
            logger.info(f"     Saved CLS Dev Test ROC curve to {roc_curve_save_path}")
        except Exception as e_roc_plot:
            logger.warning(f"     Could not generate CLS Dev ROC plot: {e_roc_plot}")
        finally:
            plt.close('all')
    else:
        logger.info("Skipping CLS final plots for Dev Test as per --skip_cls_final_plots flag.")

    # Prediction on the final, unseen classification test set (for submission or final evaluation)
    logger.info(f"NNC8: Predicting on final CLS test set from {args.cls_test_path}...")
    try:
        final_cls_test_df_raw = pd.read_hdf(args.cls_test_path)
    except FileNotFoundError:
        logger.error(f"Final CLS test data file not found at {args.cls_test_path}. Skipping final prediction.")
        # Decide if this is a critical error or if the script can continue to Part 2.
        # For now, log and skip this step if file not found.
        final_cls_test_df_raw = None 
    except Exception as e:
        logger.error(f"Error loading final CLS test data from {args.cls_test_path}: {e}. Skipping final prediction.");
        final_cls_test_df_raw = None

    if final_cls_test_df_raw is not None:
        # Ensure all selected features are present in the test data
        missing_features_in_test = [f for f in final_selected_features_cls if f not in final_cls_test_df_raw.columns]
        if missing_features_in_test:
            logger.error(f"CLS: Final test data ({args.cls_test_path}) is missing the following required features: {missing_features_in_test}. Cannot proceed with prediction on this set.")
        else:
            X_final_cls_test_subset = final_cls_test_df_raw[final_selected_features_cls] 
            try:
                final_cls_pred_proba = final_classification_pipeline.predict_proba(X_final_cls_test_subset)[:, 1]
                
                # Prepare submission file
                # Use index from the loaded test dataframe if it exists and matches length, otherwise generate sequential IDs
                if hasattr(final_cls_test_df_raw, 'index') and len(final_cls_test_df_raw.index) == len(final_cls_pred_proba):
                    ids_cls_submission = final_cls_test_df_raw.index
                else:
                    logger.warning("CLS: Test data index from HDF5 file not suitable or mismatched length for submission file. Generating sequential IDs.")
                    ids_cls_submission = np.arange(len(final_cls_pred_proba))
                    
                cls_predictions_df = pd.DataFrame({'id': ids_cls_submission, args.cls_target_column: final_cls_pred_proba})
                cls_predictions_save_path = os.path.join(CLS_OUTPUT_DIR, "sNNC8_classification_nn_test_predictions.csv")
                cls_predictions_df.to_csv(cls_predictions_save_path, index=False)
                logger.info(f"CLS NN predictions for final test set saved to: {cls_predictions_save_path}")
            except Exception as e_final_pred_cls:
                 logger.error(f"Error during CLS pipeline predict_proba on final test set: {e_final_pred_cls}. This could be the '__sklearn_tags__' issue again.")

    logger.info("--- Part 1: Classification NN Pipeline Complete (or attempted) ---")

    # ==============================================================================
    # Part 2: Neural Network for Regression
    # ==============================================================================
    # This part is largely analogous to Part 1, but for regression.
    # Key differences: different target, different metrics (MAE, MSE, R2, MARE),
    # KFold instead of StratifiedKFold for Optuna CV (unless target distribution is highly skewed).
    logger.info("\n\n--- Starting Part 2: Neural Network for Regression ---")
    REG_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "regression_nn")
    os.makedirs(REG_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Regression NN outputs will be in: {REG_OUTPUT_DIR}")
    
    logger.info("NNR1: Preparing data for regression...")
    # Filter for true electrons for the regression task, as specified by problem (e.g., predict energy of electrons)
    electron_df_reg = original_train_df[original_train_df[args.electron_flag_column] == 1].copy()
    if electron_df_reg.empty:
        logger.error(f"REG: No true electrons found in the training data using flag column '{args.electron_flag_column}'. Cannot proceed with regression part of the pipeline.")
        # If regression is optional or independent, the script could continue.
        # For now, if no data, this part is skipped.
        logger.info("--- Part 2: Regression NN Pipeline Skipped (No data) ---")
        logger.info("\n === Both NN Pipelines Execution Finished (Regression Skipped) === ")
        return # Exit main function if no data for regression

    all_cols_reg = electron_df_reg.columns.tolist()
    # Exclude common columns, regression target, and the electron flag itself from features
    exclude_cols_reg = list(set(args.exclude_columns_common + [args.reg_target_column, args.electron_flag_column]))
    feature_cols_reg_initial = [col for col in all_cols_reg if col not in exclude_cols_reg]

    if not feature_cols_reg_initial:
        logger.error("REG: No initial features found for regression after exclusions. Check column names and exclusions.")
        raise ValueError("REG: No initial features available for regression.")
        
    X_original_reg = electron_df_reg[feature_cols_reg_initial]
    y_original_reg = electron_df_reg[args.reg_target_column]
    
    logger.info(f"REG: Splitting electron-only data (Original Train -> Dev Temp + Dev Test; Dev Temp -> Train + Val)...")
    X_dev_temp_reg, X_test_dev_reg, y_dev_temp_reg, y_test_dev_reg = train_test_split(
        X_original_reg, y_original_reg, 
        test_size=DEV_TEST_SIZE, 
        random_state=RANDOM_SEED
        # No stratification for regression target typically, unless it's binned.
    )
    val_size_adjusted_reg = VALIDATION_SIZE / (1 - DEV_TEST_SIZE) if (1 - DEV_TEST_SIZE) > 0 else 0
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
        X_dev_temp_reg, y_dev_temp_reg, 
        test_size=val_size_adjusted_reg, 
        random_state=RANDOM_SEED
    )
    logger.info(f"  REG X_train shape: {X_train_reg.shape}, REG y_train shape: {y_train_reg.shape}")
    logger.info(f"  REG X_val shape: {X_val_reg.shape}, REG y_val shape: {y_val_reg.shape}")
    logger.info(f"  REG X_test_dev shape: {X_test_dev_reg.shape}, REG y_test_dev shape: {y_test_dev_reg.shape}")

    logger.info("NNR2: Feature Selection for Regression...")
    # Similar feature selection logic as for classification
    if args.reg_selected_features_list:
        selected_features_from_arg_reg = [f.strip() for f in args.reg_selected_features_list.split(',')]
        final_selected_features_reg = [f for f in selected_features_from_arg_reg if f in X_train_reg.columns][:args.reg_max_features]
        if len(final_selected_features_reg) < len(selected_features_from_arg_reg):
             if len(selected_features_from_arg_reg) > args.reg_max_features:
                logger.warning(f"REG: More features provided ({len(selected_features_from_arg_reg)}) than reg_max_features ({args.reg_max_features}). Using the first {args.reg_max_features} valid features.")
             else:
                logger.warning(f"REG: Some provided features were not found in the training data. Using {len(final_selected_features_reg)} valid features.")
    else:
        logger.warning("REG: No pre-selected features provided via --reg_selected_features_list. Using the first N available features as a placeholder. THIS IS STRONGLY NOT RECOMMENDED.")
        final_selected_features_reg = X_train_reg.columns.tolist()[:args.reg_max_features]
    
    if not final_selected_features_reg:
        logger.error("REG: No features were selected. Check feature lists and data.")
        raise ValueError("REG: No features selected for regression.")
        
    logger.info(f"REG: Final {len(final_selected_features_reg)} features selected: {final_selected_features_reg[:5]}..." + (f" (and {len(final_selected_features_reg)-5} more)" if len(final_selected_features_reg)>5 else ""))
    
    X_train_reg_subset = X_train_reg[final_selected_features_reg]
    X_val_reg_subset = X_val_reg[final_selected_features_reg]
    X_test_dev_reg_subset = X_test_dev_reg[final_selected_features_reg]
    N_FEATURES_FINAL_REG = len(final_selected_features_reg)

    if not args.skip_reg_data_visualization:
        logger.info("NNR2.5: Visualizing REG features & target (Simplified - placeholder)...")
        # Placeholder for actual visualization code (e.g., histograms of target, scatter plots)
    else:
        logger.info("Skipping REG data visualization (NNR2.5).")

    best_params_reg_nn_dict = {}
    if args.skip_reg_optuna:
        logger.info("NNR5: Skipping Optuna for REG NN. Using default NN parameters.")
        best_params_reg_nn_dict = {
            'hidden_layer_sizes': (64, 32), 
            'dropout_rate': 0.1, 
            'learning_rate': 0.001, 
            'batch_size': 128, 
            'epochs': 60      
        }
    else:
        logger.info("NNR5: Hyperparameter Tuning for REG NN using Optuna...")
        def optuna_objective_reg(trial):
            hidden_layers_reg = trial.suggest_int('reg_n_layers', 1, 2)
            neurons_l_reg = [trial.suggest_int(f'reg_neurons_l{i+1}', 32, 128, log=True) for i in range(hidden_layers_reg)]
            dropout_reg = trial.suggest_float('reg_dropout', 0.0, 0.3) # Can include 0 for no dropout
            lr_reg = trial.suggest_float('reg_lr', 1e-4, 1e-2, log=True)
            batch_s_reg = trial.suggest_categorical('reg_batch_size', [64, 128, 256])
            epochs_s_reg = trial.suggest_int('reg_epochs', 30, 80)
            
            reg_nn_optuna = KerasRegressor(
                model=create_regression_nn_model, 
                num_features=N_FEATURES_FINAL_REG,
                hidden_layer_sizes=tuple(neurons_l_reg), 
                dropout_rate=dropout_reg, 
                learning_rate=lr_reg,
                epochs=epochs_s_reg, 
                batch_size=batch_s_reg, 
                verbose=0, 
                random_state=RANDOM_SEED
            )
            pipeline_reg_optuna = Pipeline([('scaler', StandardScaler()), ('nn', reg_nn_optuna)])
            
            # Use KFold for regression CV (StratifiedKFold is for classification)
            cv_reg_opt = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            # Using Mean Absolute Relative Error (MARE) as the optimization metric
            metric_scores = [] 
            for fold, (train_idx, val_idx) in enumerate(cv_reg_opt.split(X_train_reg_subset, y_train_reg)):
                X_f, X_v = X_train_reg_subset.iloc[train_idx], X_train_reg_subset.iloc[val_idx]
                y_f, y_v = y_train_reg.iloc[train_idx], y_train_reg.iloc[val_idx]
                try:
                    pipeline_reg_optuna.fit(X_f, y_f)
                    preds_fold = pipeline_reg_optuna.predict(X_v)
                    score = mean_absolute_relative_error(y_v, preds_fold) # Custom metric
                    if np.isnan(score): # Handle cases where MARE might be NaN
                        logger.warning(f"Optuna REG Trial {trial.number} Fold {fold+1} MARE is NaN. Returning inf.")
                        return float('inf') 
                    metric_scores.append(score)
                except Exception as e:
                    logger.warning(f"Optuna REG Trial {trial.number} Fold {fold+1} encountered an error: {e}. Returning float('inf').")
                    return float('inf')
                
                trial.report(score, step=fold)
                if trial.should_prune():
                    logger.info(f"Optuna REG Trial {trial.number} Pruned at fold {fold+1}.")
                    raise optuna.exceptions.TrialPruned()
            return np.mean(metric_scores)

        study_reg = optuna.create_study(direction='minimize', # Minimize MARE
                                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1,CV_FOLDS-1), n_min_trials=max(5,CV_FOLDS)))
        logger.info(f"Starting Optuna study for REG NN ({args.reg_n_optuna_trials} trials) with OPTUNA_N_JOBS={OPTUNA_N_JOBS}...")
        study_reg.optimize(optuna_objective_reg, n_trials=args.reg_n_optuna_trials, n_jobs=OPTUNA_N_JOBS)
        
        # MODIFIED: Handle Optuna failure for regression part
        if study_reg.best_trial is None or (hasattr(study_reg.best_value, 'is_infinite') and study_reg.best_value.is_infinite()) or (isinstance(study_reg.best_value, float) and np.isinf(study_reg.best_value)):
            logger.error("Optuna REG study failed to find any valid (non-infinite) parameters. Falling back to default parameters for REG NN.")
            best_params_reg_nn_dict = {
                'hidden_layer_sizes': (64, 32), 
                'dropout_rate': 0.1, 
                'learning_rate': 0.001, 
                'batch_size': 128,
                'epochs': 60
            }
            logger.info(f"Using REG default parameters: {best_params_reg_nn_dict}")
        else:
            optuna_best_params_reg = study_reg.best_params
            best_metric_reg_nn = study_reg.best_value # This would be best MARE
            logger.info(f"Optuna REG study finished. Best MARE (or chosen metric): {best_metric_reg_nn:.4f}")
            logger.info(f"Best REG parameters: {optuna_best_params_reg}")
            best_params_reg_nn_dict = {
                'hidden_layer_sizes': tuple(optuna_best_params_reg[f'reg_neurons_l{i+1}'] for i in range(optuna_best_params_reg['reg_n_layers'])),
                'dropout_rate': optuna_best_params_reg['reg_dropout'],
                'learning_rate': optuna_best_params_reg['reg_lr'],
                'epochs': optuna_best_params_reg['reg_epochs'],
                'batch_size': optuna_best_params_reg['reg_batch_size']
            }

        if optuna_viz_available and not args.skip_reg_optuna_plots:
            logger.info("NNR5: Generating Optuna plots for REG...")
            try:
                if study_reg.trials and any(t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and not np.isinf(t.value) for t in study_reg.trials):
                    plot_optimization_history(study_reg).figure.savefig(os.path.join(REG_OUTPUT_DIR, "sNNR5_optuna_history_reg.pdf")); plt.clf(); plt.close()
                    plot_param_importances(study_reg).figure.savefig(os.path.join(REG_OUTPUT_DIR, "sNNR5_optuna_param_importances_reg.pdf")); plt.clf(); plt.close()
                    plot_slice(study_reg).figure.savefig(os.path.join(REG_OUTPUT_DIR, "sNNR5_optuna_slice_plot_reg.pdf")); plt.clf(); plt.close()
                    if CV_FOLDS > 1:
                        plot_intermediate_values(study_reg).figure.savefig(os.path.join(REG_OUTPUT_DIR, "sNNR5_optuna_intermediate_values_reg.pdf")); plt.clf(); plt.close()
                    logger.info("   Saved all Optuna plots for REG.")
                else:
                    logger.warning("   Skipping Optuna plot generation for REG as no successful trials were found or study is empty.")
            except Exception as e_opt_plot_reg: 
                logger.warning(f"Could not generate one or more Optuna plots for REG: {e_opt_plot_reg}")
            finally: plt.close('all')
        elif not optuna_viz_available: logger.warning("Optuna viz (matplotlib) not available for REG.")
        else: logger.info("Skipping Optuna plots for REG as per --skip_reg_optuna_plots flag.")

    logger.info(f"NNR6: Training final REG NN model with parameters: {best_params_reg_nn_dict}")
    keras_callbacks_reg = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1)
    ] # For regression, 'val_loss' (e.g., val_mae) is common to monitor.
    final_nn_regressor = KerasRegressor(
        model=create_regression_nn_model, 
        num_features=N_FEATURES_FINAL_REG, 
        verbose=0, 
        random_state=RANDOM_SEED,
        **{k: v for k, v in best_params_reg_nn_dict.items() if k in ['hidden_layer_sizes', 'dropout_rate', 'learning_rate']},
        epochs=best_params_reg_nn_dict['epochs'],
        batch_size=best_params_reg_nn_dict['batch_size']
    )
    final_regression_pipeline = Pipeline([('scaler', StandardScaler()), ('nn', final_nn_regressor)])
    
    logger.info("Fitting final regression pipeline (NN) on X_train_reg_subset, validating on X_val_reg_subset...")
    try:
        final_regression_pipeline.fit(
            X_train_reg_subset, y_train_reg, 
            nn__validation_data=(X_val_reg_subset, y_val_reg), 
            nn__callbacks=keras_callbacks_reg
        )
    except Exception as e_fit_final_reg:
        logger.error(f"Error during final REG pipeline fitting: {e_fit_final_reg}. The '__sklearn_tags__' error might occur here if SciKeras/sklearn versions are incompatible.")
        raise # Halt if final model training fails

    reg_pipeline_save_path = os.path.join(REG_OUTPUT_DIR, "sNNR6_final_regression_nn_pipeline.joblib")
    joblib.dump(final_regression_pipeline, reg_pipeline_save_path)
    logger.info(f"REG NN pipeline saved to: {reg_pipeline_save_path}")

    logger.info("NNR7: Evaluating REG NN model on the development test set (X_test_dev_reg_subset)...")
    try:
        dev_reg_pred_energy = final_regression_pipeline.predict(X_test_dev_reg_subset)
    except Exception as e_pred_reg:
        logger.error(f"Error during REG pipeline predict on dev test set: {e_pred_reg}.")
        logger.error("This could be due to the same '__sklearn_tags__' library incompatibility. Ensure SciKeras, scikit-learn, and TensorFlow are compatible and up-to-date.")
        logger.info("--- Part 2: Regression NN Pipeline Halted due to prediction error ---")
        raise
        
    # Calculate standard regression metrics
    dev_mae = mean_absolute_error(y_test_dev_reg, dev_reg_pred_energy)
    dev_mse = mean_squared_error(y_test_dev_reg, dev_reg_pred_energy)
    dev_r2 = r2_score(y_test_dev_reg, dev_reg_pred_energy)
    # Custom metric: Mean Absolute Relative Error
    dev_rel_mae = mean_absolute_relative_error(y_test_dev_reg, dev_reg_pred_energy)

    logger.info(f"    REG Dev MAE: {dev_mae:.4f}")
    logger.info(f"    REG Dev MSE: {dev_mse:.4f}")
    logger.info(f"    REG Dev R2 Score: {dev_r2:.4f}")
    logger.info(f"    REG Dev Mean Absolute Relative Error (MARE): {dev_rel_mae:.4f}")

    if not args.skip_reg_final_plots:
        logger.info("  NNR7: Generating REG Dev Test True vs. Predicted Plot...")
        try:
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test_dev_reg, dev_reg_pred_energy, alpha=0.5, label="Predictions", s=10) # Smaller points for dense plots
            # Ideal line (y=x)
            min_val = min(y_test_dev_reg.min(), dev_reg_pred_energy.min())
            max_val = max(y_test_dev_reg.max(), dev_reg_pred_energy.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal")
            plt.xlabel("True Energy (Dev Test)"); plt.ylabel("Predicted Energy (Dev Test)"); plt.title("REG Dev: True vs. Predicted Energy")
            plt.legend(); plt.grid(True); plt.axis('equal') # Equal aspect ratio
            reg_dev_plot_path = os.path.join(REG_OUTPUT_DIR, "sNNR7_reg_dev_true_vs_pred.pdf")
            plt.savefig(reg_dev_plot_path); plt.clf(); plt.close()
            logger.info(f"     Saved REG Dev True vs. Predicted plot to {reg_dev_plot_path}")
        except Exception as e_reg_dev_plot:
            logger.warning(f"     Could not generate REG Dev True vs. Pred plot: {e_reg_dev_plot}")
        finally: plt.close('all')
    else:
        logger.info("Skipping REG final plots for Dev Test as per --skip_reg_final_plots flag.")

    logger.info(f"NNR8: Predicting on final REG test set from {args.reg_test_path}...")
    try:
        final_reg_test_df_raw = pd.read_hdf(args.reg_test_path)
    except FileNotFoundError:
        logger.error(f"Final REG test data file not found at {args.reg_test_path}. Skipping final prediction for regression.")
        final_reg_test_df_raw = None
    except Exception as e:
        logger.error(f"Error loading final REG test data from {args.reg_test_path}: {e}. Skipping final prediction for regression.");
        final_reg_test_df_raw = None
    
    final_test_electrons_df_reg = None # Initialize
    if final_reg_test_df_raw is not None:
        final_test_electrons_df_reg = final_reg_test_df_raw # Default to using all rows if no filter column
        if args.electron_flag_column in final_reg_test_df_raw.columns: 
            logger.info(f"Filtering final REG test data on '{args.electron_flag_column}' == 1.")
            final_test_electrons_df_reg = final_reg_test_df_raw[final_reg_test_df_raw[args.electron_flag_column] == 1].copy()
            if final_test_electrons_df_reg.empty:
                logger.warning(f"REG: No true electrons found in the final test file {args.reg_test_path} after filtering. No regression predictions will be made for this set.")
                final_test_electrons_df_reg = None 
        else:
            logger.info(f"REG: Electron flag column '{args.electron_flag_column}' not found in final REG test data. Assuming all rows are relevant or pre-filtered.")

    if final_test_electrons_df_reg is not None and not final_test_electrons_df_reg.empty:
        missing_features_in_test_reg = [f for f in final_selected_features_reg if f not in final_test_electrons_df_reg.columns]
        if missing_features_in_test_reg:
             logger.error(f"REG: Final test data (after potential filtering) from {args.reg_test_path} is missing required features: {missing_features_in_test_reg}. Cannot proceed with REG prediction.")
        else:
            X_final_reg_test_subset = final_test_electrons_df_reg[final_selected_features_reg] 
            try:
                final_reg_pred_energy = final_regression_pipeline.predict(X_final_reg_test_subset)
                
                if hasattr(final_test_electrons_df_reg, 'index') and len(final_test_electrons_df_reg.index) == len(final_reg_pred_energy):
                    ids_reg_submission = final_test_electrons_df_reg.index
                else:
                    logger.warning("REG: Test data index from HDF5 not suitable for submission file. Generating sequential IDs.")
                    ids_reg_submission = np.arange(len(final_reg_pred_energy))
                    
                reg_predictions_df = pd.DataFrame({'id': ids_reg_submission, args.reg_target_column: final_reg_pred_energy})
                reg_predictions_save_path = os.path.join(REG_OUTPUT_DIR, "sNNR8_regression_nn_test_predictions.csv")
                reg_predictions_df.to_csv(reg_predictions_save_path, index=False)
                logger.info(f"REG NN predictions for final test set saved to: {reg_predictions_save_path}")

                # If true energy is available in the final test set, calculate and log metrics
                if args.reg_target_column in final_test_electrons_df_reg.columns:
                    y_final_reg_test_true = final_test_electrons_df_reg[args.reg_target_column]
                    final_test_mae = mean_absolute_error(y_final_reg_test_true, final_reg_pred_energy)
                    final_test_rel_mae = mean_absolute_relative_error(y_final_reg_test_true, final_reg_pred_energy) # Custom metric
                    final_test_r2 = r2_score(y_final_reg_test_true, final_reg_pred_energy)
                    logger.info(f"    REG Final Test (if target known) MAE: {final_test_mae:.4f}")
                    logger.info(f"    REG Final Test (if target known) R2 Score: {final_test_r2:.4f}")
                    logger.info(f"    REG Final Test (if target known) Mean Absolute Relative Error (MARE): {final_test_rel_mae:.4f}")
                    if not args.skip_reg_final_plots:
                        logger.info("  NNR8: Generating REG Final Test True vs. Predicted Plot (if target known)...")
                        try:
                            plt.figure(figsize=(8, 8))
                            plt.scatter(y_final_reg_test_true, final_reg_pred_energy, alpha=0.5, label="Predictions", s=10)
                            min_val_final = min(y_final_reg_test_true.min(), final_reg_pred_energy.min())
                            max_val_final = max(y_final_reg_test_true.max(), final_reg_pred_energy.max())
                            plt.plot([min_val_final, max_val_final], [min_val_final, max_val_final], 'k--', lw=2, label="Ideal")
                            plt.xlabel("True Energy (Final Test)"); plt.ylabel("Predicted Energy (Final Test)"); plt.title("REG Final Test: True vs. Predicted Energy")
                            plt.legend(); plt.grid(True); plt.axis('equal')
                            reg_final_plot_path = os.path.join(REG_OUTPUT_DIR, "sNNR8_reg_final_test_true_vs_pred.pdf")
                            plt.savefig(reg_final_plot_path); plt.clf(); plt.close()
                            logger.info(f"     Saved REG Final Test True vs. Predicted plot to {reg_final_plot_path}")
                        except Exception as e_reg_final_plot:
                             logger.warning(f"     Could not generate REG Final Test True vs. Pred plot: {e_reg_final_plot}")
                        finally: plt.close('all')
                else: # Target column not in test data
                    logger.info(f"REG: Target column '{args.reg_target_column}' not found in the final REG test set. Skipping final test metric calculation and plotting.")
            except Exception as e_final_pred_reg:
                logger.error(f"Error during REG pipeline predict on final test set: {e_final_pred_reg}. This could be the '__sklearn_tags__' issue again.")
    else: # final_test_electrons_df_reg is None or empty
        logger.info("REG: Skipping prediction/saving for final regression test set due to no relevant data after filtering, or initial test file was not loaded/empty.")
    
    logger.info("--- Part 2: Regression NN Pipeline Complete (or attempted) ---")
    logger.info("\n === Both NN Pipelines Execution Finished === ")

if __name__ == "__main__":
    cli_args = parse_arguments()
    # It's good practice to wrap the main execution in a try-catch block at the highest level
    # to log any uncaught exceptions before the script exits.
    try:
        main(cli_args)
        logger.info("Script finished successfully.")
    except Exception as e_main:
        logger.error(f"An unhandled error occurred in main execution: {e_main}", exc_info=True)
        # Potentially exit with a non-zero status code if desired for SLURM or other job schedulers
        # import sys
        # sys.exit(1)

