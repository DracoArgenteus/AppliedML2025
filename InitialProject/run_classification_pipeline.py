import pandas as pd
import numpy as np
import joblib 
import os
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for Matplotlib BEFORE pyplot import
import matplotlib.pyplot as plt
import seaborn as sns # Import Seaborn
import argparse 
import logging 

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, \
                            confusion_matrix, classification_report, roc_curve, \
                            log_loss 

import lightgbm as lgb
import shap 
import optuna 

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    datefmt='%H:%M:%S') 
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parses command-line arguments for the pipeline.
    Allows for flexible configuration of paths, parameters, and execution toggles.
    """
    parser = argparse.ArgumentParser(description="Run the LightGBM classification pipeline for Dataset 1, adhering to submission guidelines.")
    
    # --- I/O Arguments ---
    parser.add_argument('--output_base_dir', type=str, required=True,
                        help='Base directory for all script outputs (plots, models, submission CSVs).')
    parser.add_argument('--train_path', type=str, default="./data/AppML_InitialProject_train.h5",
                        help='Path to the training data HDF5 file.')
    parser.add_argument('--test_path', type=str, default="./data/AppML_InitialProject_test_classification.h5",
                        help='Path to the classification test data HDF5 file (Dataset 1).')
    
    # --- Submission Naming Arguments (NEW) ---
    parser.add_argument('--firstname', type=str, required=True, help='Your first name for submission file naming.')
    parser.add_argument('--lastname', type=str, required=True, help='Your last name for submission file naming.')
    parser.add_argument('--solution_name', type=str, required=True, help='A descriptive name for your solution (e.g., LGBM_Optuna_TopFeatures).')

    # --- Data Configuration Arguments ---
    parser.add_argument('--target_column', type=str, default='p_Truth_isElectron',
                        help='Name of the target column.')
    parser.add_argument('--exclude_columns', nargs='*', default=['p_Truth_Energy'],
                        help='List of columns to exclude from features.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--validation_size', type=float, default=0.20, help='Proportion for validation set from main train split.')
    parser.add_argument('--dev_test_size', type=float, default=0.15, help='Proportion for development test set from main train split.')

    # --- Feature Analysis & Selection Arguments (Step 4) ---
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds for iterative analysis and Optuna.')
    parser.add_argument('--max_features_analysis', type=int, default=25, help='Max features for iterative performance analysis (Step 4).')
    parser.add_argument('--acceptable_roc_auc', type=float, default=0.98, help='Acceptable ROC-AUC threshold for N_FEATURES_FINAL determination.')
    parser.add_argument('--outer_loop_patience', type=int, default=3, help='Patience for early stopping in feature iteration loop (Step 4).')
    parser.add_argument('--min_roc_auc_improvement', type=float, default=0.0005, help='Min ROC-AUC improvement for feature iteration (Step 4).')
    
    # --- Optuna Configuration Arguments (Step 5) ---
    parser.add_argument('--n_optuna_trials', type=int, default=50, help='Number of Optuna trials (if not using predefined params).')
    parser.add_argument('--optuna_n_jobs', type=int, default=4, 
                        help='Number of parallel jobs for Optuna study.optimize(). Default: 4.')
    parser.add_argument('--use_predefined_params', action='store_true', default=False,
                        help='Skip Optuna (Step 5) and use predefined best hyperparameters.')

    # --- Checkpoint Control Arguments ---
    parser.add_argument('--save_checkpoint_s5_5', action='store_true', default=False,
                        help='Enable saving checkpoint after Step 5.5.')
    parser.add_argument('--load_checkpoint_s6', action='store_true', default=False,
                        help='Enable loading checkpoint before Step 6 (Final Model Training).')
    parser.add_argument('--load_checkpoint_s7', action='store_true', default=False,
                        help='Enable loading checkpoint before Step 7 (Evaluation).')
    
    # --- Plotting Control Arguments ---
    parser.add_argument('--skip_visualization_s2_5', action='store_true', default=False, help="Skip feature visualization in Step 2.5")
    parser.add_argument('--skip_shap_summary_plot_s3_5', action='store_true', default=False, help="Skip SHAP summary plot in Step 3.5")

    return parser.parse_args()

def main(args):
    """
    Main function to orchestrate the classification pipeline.
    """
    # --- Initialize variables from parsed arguments ---
    OUTPUT_BASE_DIR = args.output_base_dir
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path 
    
    # --- Submission Naming Variables (NEW) ---
    FIRST_NAME = args.firstname
    LAST_NAME = args.lastname
    SOLUTION_NAME = args.solution_name

    TARGET_COLUMN = args.target_column
    EXCLUDE_COLUMNS = args.exclude_columns
    RANDOM_SEED = args.random_seed
    VALIDATION_SIZE = args.validation_size
    DEV_TEST_SIZE = args.dev_test_size
    CV_FOLDS = args.cv_folds
    MAX_FEATURES_FOR_ANALYSIS = args.max_features_analysis
    ACCEPTABLE_ROC_AUC_THRESHOLD = args.acceptable_roc_auc
    OUTER_LOOP_EARLY_STOPPING_PATIENCE = args.outer_loop_patience
    MIN_ROC_AUC_IMPROVEMENT = args.min_roc_auc_improvement
    N_OPTUNA_TRIALS = args.n_optuna_trials
    OPTUNA_N_JOBS = args.optuna_n_jobs

    # --- Setup Output Directories (as per project guidelines) ---
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    plots_dir = os.path.join(OUTPUT_BASE_DIR, "plots")
    models_dir = os.path.join(OUTPUT_BASE_DIR, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info(f"Pipeline started. Output base directory: {OUTPUT_BASE_DIR}")
    logger.info(f"Submission files will use name: {FIRST_NAME}{LAST_NAME}_{SOLUTION_NAME}")
    logger.info(f"Plots will be saved to: {plots_dir}")
    logger.info(f"Models will be saved to: {models_dir}")
    logger.debug(f"Full arguments received: {args}")

    # --- Define paths for internal outputs (scaler, models, checkpoints) ---
    SCALER_SAVE_PATH = os.path.join(models_dir, "s2_scaler_classification.joblib")
    CHECKPOINT_DIR_S5_5 = os.path.join(OUTPUT_BASE_DIR, "s5_5_checkpoint_classification") 
    FINAL_MODEL_SAVE_PATH_NATIVE = os.path.join(models_dir, f"final_lgbm_model_{SOLUTION_NAME}_native.txt")
    FINAL_MODEL_SAVE_PATH_JOBLIB = os.path.join(models_dir, f"final_lgbm_model_{SOLUTION_NAME}_joblib.pkl")

    # --- Define submission file paths (NEW - will be used in Step 7.3) ---
    submission_file_base_name = f"Classification_{FIRST_NAME}{LAST_NAME}_{SOLUTION_NAME}"
    PREDICTIONS_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name}.csv")
    VARIABLELIST_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name}_VariableList.csv")
    logger.info(f"Predictions submission CSV will be: {PREDICTIONS_SUBMISSION_PATH}")
    logger.info(f"Variable list submission CSV will be: {VARIABLELIST_SUBMISSION_PATH}")


    # ==============================================================================
    # --- Step 1: Data Loading and Initial 3-Way Split ---
    # ==============================================================================
    logger.info("--- Step 1: Data Loading and Initial 3-Way Split ---")
    logger.info(f"Loading original training data from: {TRAIN_PATH}")
    try: 
        original_train_df = pd.read_hdf(TRAIN_PATH) 
    except FileNotFoundError: 
        logger.error(f"Training data file not found at {TRAIN_PATH}")
        raise
    except Exception as e: 
        logger.error(f"Error loading HDF5 file '{TRAIN_PATH}': {e}.")
        raise
    
    logger.debug(f"Original data shape: {original_train_df.shape}")
    all_cols = original_train_df.columns.tolist()
    # Ensure TARGET_COLUMN is correctly identified for exclusion from features
    feature_cols = [col for col in all_cols if col != TARGET_COLUMN and col not in EXCLUDE_COLUMNS]

    if not feature_cols: 
        logger.error("No feature columns identified after excluding target and other specified columns.")
        raise ValueError("No feature columns identified.")
    
    X_original = original_train_df[feature_cols]
    y_original = original_train_df[TARGET_COLUMN]
    logger.info(f"Number of original features: {len(feature_cols)}")
    logger.debug(f"Original feature names (first 5): {feature_cols[:5]}")

    logger.info(f"Splitting data (Validation size: {VALIDATION_SIZE*100:.0f}%, Dev Test size: {DEV_TEST_SIZE*100:.0f}%, Seed: {RANDOM_SEED})...")
    X_dev_temp, X_test_dev, y_dev_temp, y_test_dev = train_test_split(
        X_original, y_original, test_size=DEV_TEST_SIZE, random_state=RANDOM_SEED, stratify=y_original
    )
    # Adjust validation size based on the remaining data after dev_test split
    val_size_adjusted = VALIDATION_SIZE / (1 - DEV_TEST_SIZE) if (1 - DEV_TEST_SIZE) > 0 else 0
    if not (0 < val_size_adjusted < 1) and val_size_adjusted != 0 : # val_size_adjusted can be 0 if VALIDATION_SIZE is 0
        logger.error(f"Invalid adjusted validation size: {val_size_adjusted}. Check DEV_TEST_SIZE ({DEV_TEST_SIZE}) and VALIDATION_SIZE ({VALIDATION_SIZE}).")
        raise ValueError("Invalid data split sizes leading to invalid adjusted validation size.")
    
    if val_size_adjusted == 0: # Handle case where no validation set is needed from this split
        X_train, X_val, y_train, y_val = X_dev_temp, pd.DataFrame(), y_dev_temp, pd.Series()
        logger.info("No validation set created from train/dev_temp split as VALIDATION_SIZE is effectively zero.")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev_temp, y_dev_temp, test_size=val_size_adjusted, random_state=RANDOM_SEED, stratify=y_dev_temp
        )
    
    logger.info(f"  Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"  Shape of X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"  Shape of X_test_dev: {X_test_dev.shape}, y_test_dev: {y_test_dev.shape}")
    if not y_train.empty: logger.info(f"  Target distribution in y_train:\n{y_train.value_counts(normalize=True).to_string()}")
    if not y_val.empty: logger.info(f"  Target distribution in y_val:\n{y_val.value_counts(normalize=True).to_string()}")
    if not y_test_dev.empty: logger.info(f"  Target distribution in y_test_dev:\n{y_test_dev.value_counts(normalize=True).to_string()}")


    # ==============================================================================
    # --- Step 2: Pre-processing: Direct Scaling (Fitting on X_train) ---
    # ==============================================================================
    logger.info("\n--- Step 2: Pre-processing: Direct Scaling (Fitting on X_train) ---")
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy() if not X_val.empty else pd.DataFrame()
    X_test_dev_scaled = X_test_dev.copy() if not X_test_dev.empty else pd.DataFrame()
    
    numerical_cols_to_scale = X_train.select_dtypes(include=np.number).columns.tolist()
    # If all features were already selected into X_train, this list will be X_train.columns
    # If X_train could have non-numeric (e.g. pre-encoded categoricals as objects by mistake), this is safer.
    
    scaler = StandardScaler()
    if numerical_cols_to_scale and not X_train.empty :
        logger.info(f"Fitting StandardScaler on {len(numerical_cols_to_scale)} numerical features from X_train...")
        X_train_scaled[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
        logger.info("   StandardScaler fitted on X_train and X_train_scaled created.")
        
        if hasattr(scaler, 'mean_'): # Check if scaler was actually fitted
            if not X_val.empty:
                 X_val_scaled[numerical_cols_to_scale] = scaler.transform(X_val[numerical_cols_to_scale])
                 logger.info("   X_val scaled using fitted scaler.")
            if not X_test_dev.empty:
                X_test_dev_scaled[numerical_cols_to_scale] = scaler.transform(X_test_dev[numerical_cols_to_scale])
                logger.info("   X_test_dev scaled using fitted scaler.")
            joblib.dump(scaler, SCALER_SAVE_PATH)
            logger.info(f"   Fitted scaler saved to: {SCALER_SAVE_PATH}")
        else:
            logger.warning("   Scaler was not fitted (e.g., no numeric columns or empty data after selection). Skipping transform and save.")
            scaler = None # Ensure scaler is None if not fitted
    elif X_train.empty:
        logger.warning("   X_train is empty. Skipping scaling.")
        scaler = None
    else:
        logger.warning("   No numerical columns identified/selected to scale in X_train.")
        scaler = None 
    logger.info("--- Pre-processing with Direct Scaling complete ---")
    if not X_train_scaled.empty : logger.debug(f"X_train_scaled head (first 3 rows):\n{X_train_scaled.head(3).to_string()}")


    # ==============================================================================
    # --- Step 2.5 Visualizing Data: Input Distributions by Class ---
    # ==============================================================================
    if not args.skip_visualization_s2_5:
        logger.info("\n--- Step 2.5: Visualizing input feature distributions by class (Signal vs Background) ---")
        if X_train.empty or y_train.empty:
            logger.warning("   Skipping Step 2.5 visualization: X_train or y_train is empty.")
        else:
            example_feature_vis = 'pX_topoetcone20ptCorrection' 
            if example_feature_vis not in X_train.columns: 
                if len(X_train.columns) > 0: example_feature_vis = X_train.columns[0] 
                else: example_feature_vis = None
            
            features_to_visualize_list = []
            if example_feature_vis and example_feature_vis in X_train.columns:
                 features_to_visualize_list.append(example_feature_vis)
            
            if len(X_train.columns) > 1: 
                available_other_features = [col for col in X_train.columns if col != example_feature_vis]
                if available_other_features:
                    num_additional_plots = min(2, len(available_other_features)) 
                    other_features_vis_indices = np.random.choice(len(available_other_features), size=num_additional_plots, replace=False)
                    other_features_vis = [available_other_features[i] for i in other_features_vis_indices]
                    features_to_visualize_list.extend(other_features_vis)
            
            features_to_visualize_list = list(set(features_to_visualize_list))

            if not features_to_visualize_list: 
                logger.warning("   No features available or selected for visualization in Step 2.5.")
            else:
                logger.info(f"   Will visualize distributions for: {features_to_visualize_list}")
                for feature_name_to_vis in features_to_visualize_list:
                    if feature_name_to_vis in X_train.columns:
                        plt.figure(figsize=(10, 6))
                        signal_data = X_train.loc[y_train == 1, feature_name_to_vis]
                        background_data = X_train.loc[y_train == 0, feature_name_to_vis]

                        sns.histplot(signal_data, kde=True, color='red', label=f'Signal ({TARGET_COLUMN}=1)', bins=50, stat="density", common_norm=False, alpha=0.7)
                        sns.histplot(background_data, kde=True, color='blue', label=f'Background ({TARGET_COLUMN}=0)', bins=50, stat="density", common_norm=False, alpha=0.7)

                        plt.title(f'Distribution of {feature_name_to_vis} (Original Training Data by Class)')
                        plt.xlabel(f"{feature_name_to_vis} Value"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
                        
                        feature_vis_path = os.path.join(plots_dir, f"s2_5_distribution_{feature_name_to_vis}_by_class.pdf")
                        plt.savefig(feature_vis_path, bbox_inches='tight'); plt.clf(); plt.close()
                        logger.info(f"   Feature distribution by class for '{feature_name_to_vis}' saved to {feature_vis_path}")
                    else:
                        logger.warning(f"   Could not plot for '{feature_name_to_vis}', feature not in X_train.columns.")
                plt.close('all')
    else: 
        logger.info("Skipping Step 2.5: Feature Visualization as per CLI argument.")

    # --- Ensure data is available for subsequent steps ---
    if X_train_scaled.empty or y_train.empty:
        logger.error("X_train_scaled or y_train is empty. Cannot proceed with model training and SHAP analysis.")
        raise ValueError("Training data is empty, cannot proceed.")

    # ==============================================================================
    # --- Part 3: Fitting Preliminary LightGBM Model (for SHAP Analysis) ---
    # ==============================================================================
    logger.info("\n--- Step 3: Fitting Preliminary LightGBM Model (for SHAP Analysis) ---")
    if y_train.value_counts().get(1, 0) > 0 and y_train.value_counts().get(0, 0) > 0: 
        scale_pos_weight_shap = y_train.value_counts(normalize=True).loc[0] / y_train.value_counts(normalize=True).loc[1]
    else: 
        scale_pos_weight_shap = 1
        logger.warning("Could not calculate scale_pos_weight accurately for SHAP model due to missing classes or single class in y_train. Defaulting to 1.")
    logger.info(f"Scale_pos_weight for SHAP model: {scale_pos_weight_shap:.2f}")
    
    prelim_model_for_shap = lgb.LGBMClassifier(
        boosting_type='gbdt', objective='binary', random_state=RANDOM_SEED, 
        scale_pos_weight=scale_pos_weight_shap, num_leaves=11, n_jobs=-1, 
        verbosity=-1, n_estimators=500, learning_rate=0.5
    ) 
    logger.info("Fitting preliminary LightGBM model for SHAP analysis...")
    try: 
        eval_set_shap = []
        if not X_val_scaled.empty and not y_val.empty:
            eval_set_shap = [(X_val_scaled, y_val)]
            logger.info("Using X_val_scaled for early stopping in preliminary SHAP model.")
        else:
            logger.warning("X_val_scaled or y_val is empty. Preliminary SHAP model will train without early stopping based on a validation set.")

        prelim_model_for_shap.fit(
            X_train_scaled, y_train, 
            eval_set=eval_set_shap if eval_set_shap else None, 
            eval_metric='roc_auc', 
            callbacks=[lgb.early_stopping(20, verbose=False)] if eval_set_shap else []
        )
        logger.info("Preliminary LightGBM model fitted successfully.")
    except Exception as e: 
        logger.error(f"An unexpected error occurred during prelim_model_for_shap.fit(): {e}")
        raise

    # ==============================================================================
    # --- Part 3.5: SHAP Analysis & Feature Ranking ---
    # ==============================================================================
    logger.info("\n--- Step 3.5: SHAP Analysis & Feature Ranking ---")
    if not hasattr(prelim_model_for_shap, '_Booster') or prelim_model_for_shap._Booster is None: 
        logger.error("Preliminary SHAP model (prelim_model_for_shap) is not fitted.")
        raise RuntimeError("prelim_model_for_shap not fitted.")
    
    logger.info("Generating SHAP values...")
    explainer = shap.TreeExplainer(prelim_model_for_shap)
    # Use a subset of X_train_scaled for SHAP if it's very large, to speed up, e.g., X_train_scaled.sample(min(10000, len(X_train_scaled)), random_state=RANDOM_SEED)
    shap_values_raw = explainer.shap_values(X_train_scaled) 
    
    if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2: 
        shap_values_positive_class = shap_values_raw[1]
    else: 
        shap_values_positive_class = shap_values_raw
        
    mean_abs_shap = np.mean(np.abs(shap_values_positive_class), axis=0)
    shap_importance_df = pd.DataFrame({'feature': X_train_scaled.columns, 'importance': mean_abs_shap}).sort_values(by='importance', ascending=False)
    all_shap_ranked_features = shap_importance_df['feature'].tolist()
    logger.info(f"Top 10 SHAP ranked features:\n{all_shap_ranked_features[:10]}")
    
    if not args.skip_shap_summary_plot_s3_5:
        logger.info("Generating SHAP summary plot (feature ranking)...")
        plt.figure() 
        shap.summary_plot(shap_values_positive_class, X_train_scaled, show=False, plot_type="bar", max_display=min(30, len(X_train_scaled.columns)))
        plt.title("SHAP Feature Importance (Preliminary Model)")
        summary_plot_path = os.path.join(plots_dir, "s3_5_shap_feature_ranking.pdf")
        plt.savefig(summary_plot_path, bbox_inches='tight'); plt.clf(); plt.close('all')
        logger.info(f"SHAP summary plot (feature ranking) saved to {summary_plot_path}")
    else: 
        logger.info("Skipping Step 3.5: SHAP Summary Plot as per CLI argument.")

    # ==============================================================================
    # --- Step 4: Iterative Feature Performance Analysis ---
    # ==============================================================================
    logger.info("\n--- Step 4: Iterative Feature Performance Analysis ---")
    # ... (Step 4 code remains largely the same, ensure it uses y_train, X_train_scaled, X_val_scaled, y_val if available) ...
    # Ensure scale_pos_weight_iterative calculation is robust
    if not y_train.empty and y_train.value_counts().get(1,0)>0 and y_train.value_counts().get(0,0)>0: 
        scale_pos_weight_iterative = y_train.value_counts(normalize=True).loc[0]/y_train.value_counts(normalize=True).loc[1]
    else: 
        scale_pos_weight_iterative=1
        logger.warning("Could not calculate scale_pos_weight_iterative accurately for Step 4. Defaulting to 1.")
    logger.info(f"Scale_pos_weight for CV models (Step 4): {scale_pos_weight_iterative:.2f}")

    roc_auc_scores_vs_n_features = []; n_features_tested_list = [] 
    n_features_upper_bound = min(len(all_shap_ranked_features), MAX_FEATURES_FOR_ANALYSIS)
    if n_features_upper_bound < 1: 
        logger.warning("No features from SHAP for iterative analysis in Step 4. Using all original features if N_FEATURES_FINAL not determined otherwise."); 
        # Fallback: use all features if SHAP fails or yields no features
        all_shap_ranked_features = feature_cols 
        n_features_upper_bound = min(len(all_shap_ranked_features), MAX_FEATURES_FOR_ANALYSIS)
        if n_features_upper_bound < 1:
            logger.error("Still no features available for analysis in Step 4 even after fallback.")
            raise ValueError("No features for iterative analysis.")

    n_features_range_potential = range(1, n_features_upper_bound + 1)
    best_roc_auc_so_far = -np.inf; steps_without_improvement = 0

    for k_features in n_features_range_potential:
        current_top_k_features = all_shap_ranked_features[:k_features]
        X_subset_for_cv_train = X_train_scaled[current_top_k_features]
        # X_subset_for_cv_val will be used if X_val_scaled is available
        
        logger.info(f"  Step 4: Testing with top {k_features} features...")
        n_features_tested_list.append(k_features)
        
        cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_roc_auc_scores = [] 
        
        for fold_num, (train_idx, val_idx_cv) in enumerate(cv_splitter.split(X_subset_for_cv_train, y_train)):
            X_train_fold, X_val_fold_cv = X_subset_for_cv_train.iloc[train_idx], X_subset_for_cv_train.iloc[val_idx_cv]
            y_train_fold, y_val_fold_cv = y_train.iloc[train_idx], y_train.iloc[val_idx_cv]
            
            model_iterative = lgb.LGBMClassifier(
                boosting_type='gbdt', objective='binary', random_state=RANDOM_SEED + fold_num, 
                scale_pos_weight=scale_pos_weight_iterative, num_leaves=11, n_jobs=-1, 
                verbosity=-1, n_estimators=300, learning_rate=0.05
            )
            try: 
                model_iterative.fit(
                    X_train_fold, y_train_fold, 
                    eval_set=[(X_val_fold_cv, y_val_fold_cv)], 
                    eval_metric='roc_auc', 
                    callbacks=[lgb.early_stopping(10,verbose=False)]
                )
            except Exception as e: 
                logger.warning(f"Fold {fold_num+1} error ({k_features} features): {e}"); fold_roc_auc_scores.append(np.nan); continue
            
            preds_proba = model_iterative.predict_proba(X_val_fold_cv)[:,1]; roc_auc_val = roc_auc_score(y_val_fold_cv, preds_proba)
            fold_roc_auc_scores.append(roc_auc_val)
            
        mean_cv_roc_auc = np.nanmean(fold_roc_auc_scores) if fold_roc_auc_scores else np.nan
        roc_auc_scores_vs_n_features.append(mean_cv_roc_auc)
        logger.info(f"    Mean CV ROC-AUC for {k_features} features: {mean_cv_roc_auc:.4f}")
        
        if pd.notna(mean_cv_roc_auc) and mean_cv_roc_auc > best_roc_auc_so_far + MIN_ROC_AUC_IMPROVEMENT: 
            best_roc_auc_so_far = mean_cv_roc_auc; steps_without_improvement = 0
        else: steps_without_improvement += 1
            
        if steps_without_improvement >= OUTER_LOOP_EARLY_STOPPING_PATIENCE: 
            logger.info(f"Outer loop early stopping at {k_features} features."); break 
            
    plt.figure(figsize=(12,7)); 
    plt.plot(n_features_tested_list, roc_auc_scores_vs_n_features, marker='o', color='teal', label=f'Mean {CV_FOLDS}-Fold ROC-AUC')
    plt.xlabel("Number of Top SHAP Features"); plt.ylabel(f"Mean {CV_FOLDS}-Fold CV ROC-AUC"); plt.title("Model Performance vs. Number of Features")
    if n_features_tested_list:
        max_f = max(n_features_tested_list); min_f = min(n_features_tested_list)
        tick_s = 1 if max_f <=10 else 2 if max_f <=20 else int(np.ceil(max_f/10))
        plt.xticks(list(range(min_f, max_f + 1, tick_s)))
    plt.grid(True, alpha=0.6); plt.axhline(y=ACCEPTABLE_ROC_AUC_THRESHOLD, color='g', linestyle='--', label=f'Acceptable ROC-AUC ({ACCEPTABLE_ROC_AUC_THRESHOLD})')
    if roc_auc_scores_vs_n_features and not all(np.isnan(s) for s in roc_auc_scores_vs_n_features if s is not None):
        valid_scores_for_max = [s for s in roc_auc_scores_vs_n_features if pd.notna(s)]
        if valid_scores_for_max: plt.axhline(y=np.nanmax(valid_scores_for_max), color='purple', linestyle=':', label=f'Max Achieved ROC-AUC ({np.nanmax(valid_scores_for_max):.4f})')
    plt.legend(loc='lower right'); plt.tight_layout()
    performance_plot_path_s4 = os.path.join(plots_dir, "s4_roc_auc_vs_n_features.pdf"); 
    plt.savefig(performance_plot_path_s4); plt.clf(); plt.close('all') 
    logger.info(f"Performance vs. N Features plot saved to: {performance_plot_path_s4}")
    for k,s in zip(n_features_tested_list, roc_auc_scores_vs_n_features): logger.info(f"  Features: {k:2d}, ROC-AUC: {s:.4f}" if pd.notna(s) else f"  Features: {k:2d}, ROC-AUC: NaN")
    logger.info("--- Step 4 Complete ---")

    # ==============================================================================
    # --- Determine N_FEATURES_FINAL ---
    # ==============================================================================
    logger.info("\n--- Determining N_FEATURES_FINAL ---")
    # ... (N_FEATURES_FINAL determination logic remains largely the same) ...
    N_FEATURES_FINAL = 0; achieved_threshold_score = -1.0
    if not roc_auc_scores_vs_n_features or not n_features_tested_list: 
        logger.warning("Cannot determine N_FEATURES_FINAL from Step 4 results (empty). Will use all SHAP features or fallback.")
        N_FEATURES_FINAL = len(all_shap_ranked_features) if all_shap_ranked_features else 0
        if N_FEATURES_FINAL == 0 and feature_cols: # Fallback to original features if SHAP failed
             N_FEATURES_FINAL = len(feature_cols)
             all_shap_ranked_features = feature_cols # Use original features as ranked
             logger.warning(f"Using all {N_FEATURES_FINAL} original features as N_FEATURES_FINAL due to no Step 4 results.")
        elif N_FEATURES_FINAL == 0:
             logger.error("No features available at all. Cannot determine N_FEATURES_FINAL.")
             raise ValueError("No features available to select for the final model.")
    else: # Proceed with logic based on Step 4 results
        for i, score in enumerate(roc_auc_scores_vs_n_features):
            if np.isnan(score): continue 
            num_features_at_i = n_features_tested_list[i]
            if score >= ACCEPTABLE_ROC_AUC_THRESHOLD: 
                N_FEATURES_FINAL = num_features_at_i; achieved_threshold_score = score; break
        if N_FEATURES_FINAL == 0: 
            valid_scores = [(roc_auc_scores_vs_n_features[i], n_features_tested_list[i]) for i in range(len(roc_auc_scores_vs_n_features)) if not np.isnan(roc_auc_scores_vs_n_features[i])]
            if valid_scores:
                valid_scores.sort(key=lambda x: (-x[0], x[1])) 
                best_score_val, N_FEATURES_FINAL = valid_scores[0]; achieved_threshold_score = best_score_val
                logger.warning(f"Threshold not met. Using best score: {N_FEATURES_FINAL} features, ROC-AUC: {achieved_threshold_score:.4f}.")
            else: 
                N_FEATURES_FINAL = min(5, len(all_shap_ranked_features)) if all_shap_ranked_features else 0
                logger.error(f"All ROC AUC scores in Step 4 were NaN. Fallback N_FEATURES_FINAL to {N_FEATURES_FINAL}.")
    
    if N_FEATURES_FINAL > MAX_FEATURES_FOR_ANALYSIS: N_FEATURES_FINAL = MAX_FEATURES_FOR_ANALYSIS
    logger.info(f"Determined N_FEATURES_FINAL for Optuna: {N_FEATURES_FINAL}")
    if N_FEATURES_FINAL == 0 and len(all_shap_ranked_features) > 0: N_FEATURES_FINAL = min(5, len(all_shap_ranked_features))
    elif N_FEATURES_FINAL == 0 and not all_shap_ranked_features: logger.error("No features from SHAP for N_FEATURES_FINAL."); raise ValueError("No SHAP features for N_FEATURES_FINAL.")
        
    final_selected_features = all_shap_ranked_features[:N_FEATURES_FINAL] if N_FEATURES_FINAL > 0 else []
    if not final_selected_features: 
        logger.error("Final selected features list is empty!"); 
        if feature_cols:
            logger.warning("Falling back to using all original features as final_selected_features.")
            final_selected_features = feature_cols
            N_FEATURES_FINAL = len(feature_cols)
        else:
            raise ValueError("final_selected_features is empty and no fallback original features available.")
    logger.info(f"Final selected features to be used ({len(final_selected_features)}): {final_selected_features[:10]}...")


    # ==============================================================================
    # --- Step 5: Hyperparameter Optimization with Optuna (Optimizing for LogLoss) ---
    # ==============================================================================
    logger.info(f"\n--- Step 5: Optuna for Top {N_FEATURES_FINAL} Features (Minimizing LogLoss) ---")
    # ... (Optuna logic remains largely the same, ensure it uses y_train, X_train_selected_for_optuna which is X_train_scaled[final_selected_features]) ...
    # Ensure X_train_scaled has the final_selected_features
    if not all(f in X_train_scaled.columns for f in final_selected_features):
        missing_fs = [f for f in final_selected_features if f not in X_train_scaled.columns]
        logger.error(f"Critical error: Features selected for Optuna ({missing_fs}) are not in X_train_scaled. This should not happen.")
        raise ValueError(f"Selected features for Optuna not found in scaled training data: {missing_fs}")

    X_train_selected_for_optuna = X_train_scaled[final_selected_features]
    
    if not y_train.empty and y_train.value_counts().get(1,0)>0 and y_train.value_counts().get(0,0)>0: 
        global_scale_pos_weight = y_train.value_counts(normalize=True).loc[0]/y_train.value_counts(normalize=True).loc[1]
    else: global_scale_pos_weight = 1.0; logger.warning("Could not calculate global_scale_pos_weight for Optuna. Defaulting to 1.")
    best_lgbm_params = {} 
    best_logloss_optuna = float('inf') 

    if args.use_predefined_params:
        logger.info("Skipping Optuna study and using predefined hyperparameters.")
        best_lgbm_params = {'boosting_type': 'gbdt', 'n_estimators': 1000, 'learning_rate': 0.039, 'num_leaves': 20, 'max_depth': 5, 'min_child_samples': 25, 'subsample': 0.99, 'colsample_bytree': 0.57, 'reg_alpha': 1.5e-07, 'reg_lambda': 4.9e-06, 'objective': 'binary', 'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbosity': -1}
        # Manually set a representative best_logloss if using predefined, for logging consistency
        # This value would typically come from a previous Optuna run that found these params.
        # best_logloss_optuna = 0.1234 # Example value
        logger.info(f"Using predefined hyperparameters: {best_lgbm_params}")
        # If scale_pos_weight is beneficial and known for these params, add it:
        # best_lgbm_params['scale_pos_weight'] = global_scale_pos_weight # Or a specific tuned value
    else:
        logger.info(f"Starting Optuna study for {N_OPTUNA_TRIALS} trials (minimizing LogLoss)...")
        def optuna_objective_logloss(trial, X_data, y_data_obj): # Renamed y_data to y_data_obj to avoid scope issues
            lgbm_params = {
                'objective': 'binary', 'random_state': RANDOM_SEED, 'n_jobs': 4, 'verbosity': -1,
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 5, max(10, 50 if N_FEATURES_FINAL <=10 else 100)),
                'max_depth': trial.suggest_int('max_depth', 3, max(4, 8 if N_FEATURES_FINAL <=10 else 10)),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True)
            }
            imbalance_handling_method = trial.suggest_categorical('imbalance_handling', ['scale_pos_weight', 'is_unbalance', 'none'])
            if imbalance_handling_method == 'scale_pos_weight': lgbm_params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight_val', max(0.1, global_scale_pos_weight * 0.7), global_scale_pos_weight * 1.3)
            elif imbalance_handling_method == 'is_unbalance': lgbm_params['is_unbalance'] = True
            
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED); fold_logloss_scores = []
            for fold_num, (train_idx, val_idx_opt) in enumerate(cv.split(X_data, y_data_obj)):
                X_f,X_v=X_data.iloc[train_idx],X_data.iloc[val_idx_opt]; y_f,y_v=y_data_obj.iloc[train_idx],y_data_obj.iloc[val_idx_opt]
                model_optuna=lgb.LGBMClassifier(**lgbm_params)
                try: model_optuna.fit(X_f,y_f,eval_set=[(X_v,y_v)],eval_metric='logloss',callbacks=[lgb.early_stopping(25,verbose=False)])
                except Exception as e: logger.warning(f"Optuna Trial {trial.number} Fold {fold_num+1} Error: {e}"); return float('inf')
                preds_proba_val=model_optuna.predict_proba(X_v)[:,1];epsilon=1e-15;preds_proba_val_clipped=np.clip(preds_proba_val,epsilon,1-epsilon)
                logloss_val=log_loss(y_v,preds_proba_val_clipped);fold_logloss_scores.append(logloss_val)
                trial.report(logloss_val,step=fold_num); 
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            return np.mean(fold_logloss_scores) if fold_logloss_scores else float('inf')

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1, CV_FOLDS-2), n_min_trials=max(5, N_OPTUNA_TRIALS // 10)))
        study.optimize(lambda trial: optuna_objective_logloss(trial, X_train_selected_for_optuna, y_train), n_trials=N_OPTUNA_TRIALS, n_jobs=OPTUNA_N_JOBS, show_progress_bar=(N_OPTUNA_TRIALS > 10))
        best_lgbm_params = study.best_params; best_logloss_optuna = study.best_value
        logger.info(f"Optuna study finished. Best Mean CV LogLoss: {best_logloss_optuna:.4f}")
        imbalance_choice = best_lgbm_params.pop('imbalance_handling', None) # Clean up Optuna-specific trial param
        if imbalance_choice == 'scale_pos_weight' and 'scale_pos_weight_val' in best_lgbm_params: best_lgbm_params['scale_pos_weight'] = best_lgbm_params.pop('scale_pos_weight_val')
        elif imbalance_choice == 'is_unbalance': best_lgbm_params['is_unbalance'] = True
        if 'scale_pos_weight_val' in best_lgbm_params: best_lgbm_params.pop('scale_pos_weight_val') # Ensure removed
        best_lgbm_params.setdefault('objective', 'binary'); best_lgbm_params.setdefault('random_state', RANDOM_SEED); best_lgbm_params.setdefault('n_jobs', -1); best_lgbm_params.setdefault('verbosity', -1)


    logger.info("Best hyperparameters (either predefined or from Optuna):")
    for k,v in best_lgbm_params.items(): logger.info(f"    {k}: {v}")
    logger.info("--- Step 5 Complete ---")

    # ==============================================================================
    # --- Step 5.5: Saving Pipeline State for Checkpoint ---
    # ==============================================================================
    # ... (Checkpoint saving logic remains largely the same, ensure all necessary vars are captured) ...
    logger.info(f"\n--- Step 5.5: Saving Pipeline State for Checkpoint ---")
    if args.save_checkpoint_s5_5:
        os.makedirs(CHECKPOINT_DIR_S5_5, exist_ok=True) 
        vars_to_save_s5_5 = { 'X_train_scaled': X_train_scaled, 'y_train': y_train, 'X_val_scaled': X_val_scaled, 'y_val': y_val, 'X_test_dev_scaled': X_test_dev_scaled, 'y_test_dev': y_test_dev, 'final_selected_features': final_selected_features, 'best_lgbm_params': best_lgbm_params, 'scaler': scaler, 'RANDOM_SEED': RANDOM_SEED, 'TARGET_COLUMN': TARGET_COLUMN, 'EXCLUDE_COLUMNS': EXCLUDE_COLUMNS, 'TEST_PATH': TEST_PATH, 'feature_cols': feature_cols, 'N_FEATURES_FINAL': N_FEATURES_FINAL, 'ACCEPTABLE_ROC_AUC_THRESHOLD': ACCEPTABLE_ROC_AUC_THRESHOLD, 'FIRST_NAME': FIRST_NAME, 'LAST_NAME': LAST_NAME, 'SOLUTION_NAME': SOLUTION_NAME } # Added submission names
        for name, var_val in vars_to_save_s5_5.items(): 
            try: joblib.dump(var_val, os.path.join(CHECKPOINT_DIR_S5_5, f"s5_5_checkpoint_{name}.joblib")); logger.info(f"  Saved '{name}'") 
            except Exception as e: logger.error(f"  Error saving '{name}': {e}")
    else: logger.info("Skipping checkpoint saving for Step 5.5.")


    # ==============================================================================
    # --- Step 6: Final Model Training and Saving ---
    # ==============================================================================
    # ... (Final model training logic remains largely the same) ...
    logger.info(f"\n--- Step 6: Training and Saving Final LightGBM Model ---")
    if args.load_checkpoint_s6:
        logger.info(f"Attempting to load from checkpoint: {CHECKPOINT_DIR_S5_5}")
        try:
            # Reload all necessary variables from checkpoint_vars_map as in previous version
            # For brevity, assuming this part is robust. Ensure FIRST_NAME, LAST_NAME, SOLUTION_NAME are also loaded if they were checkpointed.
            # Example: FIRST_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_FIRST_NAME.joblib')) etc.
            # This example assumes they are re-declared if needed or passed through.
            # For this script, they are passed as args and set at the start of main().
            # If loading checkpoint, ensure these are consistent or reloaded.
            # The critical parts are X_train_scaled, y_train, X_val_scaled, y_val, final_selected_features, best_lgbm_params.
            X_train_scaled = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_X_train_scaled.joblib'))
            y_train = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_y_train.joblib'))
            X_val_scaled = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_X_val_scaled.joblib'))
            y_val = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_y_val.joblib'))
            final_selected_features = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_final_selected_features.joblib'))
            best_lgbm_params = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_best_lgbm_params.joblib'))
            # Other vars like RANDOM_SEED, scaler, etc.
            RANDOM_SEED = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_RANDOM_SEED.joblib'))
            scaler = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_scaler.joblib'))
            # And the new submission name related vars if they were checkpointed (though usually passed as args)
            FIRST_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_FIRST_NAME.joblib'))
            LAST_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_LAST_NAME.joblib'))
            SOLUTION_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_SOLUTION_NAME.joblib'))
            # Redefine submission paths if loaded from checkpoint
            submission_file_base_name = f"Classification_{FIRST_NAME}{LAST_NAME}_{SOLUTION_NAME}"
            PREDICTIONS_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name}.csv")
            VARIABLELIST_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name}_VariableList.csv")
            logger.info(f"Submission paths updated after loading checkpoint: {PREDICTIONS_SUBMISSION_PATH}, {VARIABLELIST_SUBMISSION_PATH}")


            logger.info("Core variables for Step 6 loaded from checkpoint.")
        except Exception as e: logger.error(f"Error loading core checkpoint for Step 6: {e}"); raise
    
    if X_train_scaled.empty or y_train.empty or not final_selected_features or not best_lgbm_params:
        logger.error("CRITICAL: Missing essential data for final model training (X_train_scaled, y_train, features, or params).")
        raise ValueError("Essential data for final model training is missing.")

    X_train_final_subset = X_train_scaled[final_selected_features]
    X_val_final_subset = X_val_scaled[final_selected_features] if not X_val_scaled.empty else None
    
    final_model = lgb.LGBMClassifier(**best_lgbm_params) # best_lgbm_params should include objective, random_state etc.
    logger.info("Training final model with optimized hyperparameters...")
    eval_set_final = []
    if X_val_final_subset is not None and not y_val.empty:
        eval_set_final = [(X_val_final_subset, y_val)]
        logger.info("Using X_val_final_subset for early stopping in final model training.")
    else:
        logger.warning("X_val_final_subset or y_val is empty/not available. Final model will train without early stopping on a validation set.")

    try:
        final_model.fit(X_train_final_subset, y_train, 
                        eval_set=eval_set_final if eval_set_final else None, 
                        eval_metric='logloss', 
                        callbacks=[lgb.early_stopping(30,verbose=True)] if eval_set_final else [])
        logger.info(f"Final model trained. Best iteration: {final_model.best_iteration_ if hasattr(final_model, 'best_iteration_') and final_model.best_iteration_ is not None else 'N/A'}")
    except Exception as e: logger.error(f"Error training final model: {e}"); raise
        
    try:
        final_model.booster_.save_model(FINAL_MODEL_SAVE_PATH_NATIVE)
        logger.info(f"Final model (native) saved: {FINAL_MODEL_SAVE_PATH_NATIVE}")
        joblib.dump(final_model, FINAL_MODEL_SAVE_PATH_JOBLIB)
        logger.info(f"Final model (joblib) saved: {FINAL_MODEL_SAVE_PATH_JOBLIB}")
    except Exception as e: logger.error(f"Error saving final model: {e}")
    logger.info("--- Step 6 Complete ---")


    # ==============================================================================
    # --- Step 7: Prediction and Validation on Test Sets & SUBMISSION FILE GENERATION ---
    # ==============================================================================
    logger.info(f"\n--- Step 7: Prediction, Validation, and Submission File Generation ---")
    # ... (Loading from checkpoint logic for S7, if applicable) ...
    if args.load_checkpoint_s7:
        logger.info(f"Attempting to load variables and model for Step 7 from checkpoint/saved paths...")
        try:
            # Similar to Step 6, load necessary variables:
            # X_test_dev_scaled, y_test_dev, final_selected_features, scaler, TEST_PATH, TARGET_COLUMN, feature_cols, N_FEATURES_FINAL
            # FIRST_NAME, LAST_NAME, SOLUTION_NAME
            # And reload the model (final_model)
            X_test_dev_scaled = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_X_test_dev_scaled.joblib'))
            y_test_dev = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_y_test_dev.joblib'))
            final_selected_features = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_final_selected_features.joblib'))
            scaler = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_scaler.joblib'))
            TEST_PATH = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_TEST_PATH.joblib'))
            TARGET_COLUMN = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_TARGET_COLUMN.joblib'))
            feature_cols = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_feature_cols.joblib'))
            N_FEATURES_FINAL = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_N_FEATURES_FINAL.joblib'))
            FIRST_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_FIRST_NAME.joblib'))
            LAST_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_LAST_NAME.joblib'))
            SOLUTION_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_SOLUTION_NAME.joblib'))
            
            # Redefine submission paths
            submission_file_base_name = f"Classification_{FIRST_NAME}{LAST_NAME}_{SOLUTION_NAME}"
            PREDICTIONS_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name}.csv")
            VARIABLELIST_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name}_VariableList.csv")
            logger.info(f"Submission paths updated after loading S7 checkpoint: {PREDICTIONS_SUBMISSION_PATH}, {VARIABLELIST_SUBMISSION_PATH}")

            if os.path.exists(FINAL_MODEL_SAVE_PATH_JOBLIB):
                final_model = joblib.load(FINAL_MODEL_SAVE_PATH_JOBLIB)
                logger.info(f"Reloaded final model (joblib) for Step 7 from: {FINAL_MODEL_SAVE_PATH_JOBLIB}")
            elif os.path.exists(FINAL_MODEL_SAVE_PATH_NATIVE):
                # Simplified reload, assuming best_lgbm_params is available for reconstruction
                booster = lgb.Booster(model_file=FINAL_MODEL_SAVE_PATH_NATIVE)
                # Ensure best_lgbm_params is loaded or available
                if 'best_lgbm_params' not in locals(): 
                    best_lgbm_params = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_best_lgbm_params.joblib'))
                final_model = lgb.LGBMClassifier(**best_lgbm_params); final_model._Booster = booster; final_model._fitted = True
                # Attempt to set classes_ attribute if y_train is available or can be loaded
                try:
                    y_ref_for_classes = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_checkpoint_y_train.joblib'))
                    if hasattr(y_ref_for_classes, 'unique'): final_model.classes_=np.sort(y_ref_for_classes.unique()); final_model._n_classes=len(final_model.classes_)
                except: final_model.classes_=np.array([0,1]); final_model._n_classes=2 # Fallback
                logger.info(f"Reloaded final model (native) for Step 7 from: {FINAL_MODEL_SAVE_PATH_NATIVE}")
            else: raise FileNotFoundError("No saved model found for Step 7 loading from checkpoint.")
            logger.info("Variables and model reloaded for Step 7 evaluation.")
        except Exception as e: logger.error(f"Error loading checkpoint or model for Step 7: {e}"); raise

    if 'final_model' not in locals() or final_model is None: 
        logger.error("CRITICAL: 'final_model' is not defined for Step 7."); raise NameError("Missing 'final_model'.")

    # --- Stage 7.1: ROC Plot (Train vs Dev-Test) ---
    # ... (ROC plot logic largely the same, ensure variables are available) ...
    logger.info("\n--- Stage 7.1: Generating ROC Plot (Train vs. Dev Test) for Final Model ---")
    # Ensure X_train_final_subset and y_train are available if not loading S7 from checkpoint
    if 'X_train_final_subset' not in locals() or 'y_train' not in locals():
        # Attempt to reconstruct or load if necessary
        if 'X_train_scaled' in locals() and 'final_selected_features' in locals() and 'y_train' in locals():
             X_train_final_subset = X_train_scaled[final_selected_features]
             # y_train should be available
        else:
            logger.warning("X_train_final_subset or y_train not directly available for ROC plot in Step 7.1. Skipping Train ROC.")
            X_train_final_subset = pd.DataFrame() # Ensure it's an empty df to skip plotting

    if not X_train_final_subset.empty and not y_train.empty :
        sns.set_theme(style="whitegrid"); plt.figure(figsize=(10, 8)); colors = sns.color_palette("deep", 2) 
        train_pred_proba = final_model.predict_proba(X_train_final_subset)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, train_pred_proba); auc_train = roc_auc_score(y_train, train_pred_proba)
        plt.plot(fpr_train, tpr_train, color=colors[0], lw=2, label=f'Train ROC (Top {N_FEATURES_FINAL} Feats, AUC = {auc_train:.4f})')
        logger.info(f"  AUC for ROC on training data: {auc_train:.4f}")

        if not X_test_dev_scaled.empty and not y_test_dev.empty and final_selected_features:
            X_test_dev_subset_for_plot = X_test_dev_scaled[final_selected_features]
            dev_test_pred_proba_for_plot = final_model.predict_proba(X_test_dev_subset_for_plot)[:, 1] 
            fpr_dev, tpr_dev, _ = roc_curve(y_test_dev, dev_test_pred_proba_for_plot); auc_dev_plot = roc_auc_score(y_test_dev, dev_test_pred_proba_for_plot)
            plt.plot(fpr_dev, tpr_dev, color=colors[1], lw=2, label=f'Dev Test ROC (Top {N_FEATURES_FINAL} Feats, AUC = {auc_dev_plot:.4f})')
            logger.info(f"  AUC for ROC on dev-test data: {auc_dev_plot:.4f}")
        
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--'); plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02]) 
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'ROC Curves: Final Model (Top {N_FEATURES_FINAL} Features)'); plt.legend(loc="lower right")
        plt.grid(True); s7_1_roc_train_dev_path = os.path.join(plots_dir, "s7_1_roc_curve_train_vs_dev_test.pdf") 
        plt.savefig(s7_1_roc_train_dev_path, bbox_inches='tight'); plt.clf(); plt.close('all')
        logger.info(f"Train vs. Dev Test ROC curve plot saved to {s7_1_roc_train_dev_path}")
    else:
        logger.warning("Skipping Stage 7.1 ROC plot as training data for plot is unavailable.")


    # --- Stage 7.2: Evaluation on Development Test Set (Metrics) ---
    # ... (Dev Test metrics logic largely the same) ...
    logger.info(f"\n--- Stage 7.2: Evaluating Metrics on Development Test Set ---")
    if not X_test_dev_scaled.empty and not y_test_dev.empty and final_selected_features:
        X_test_dev_subset_eval = X_test_dev_scaled[final_selected_features]
        dev_test_pred_proba_eval = final_model.predict_proba(X_test_dev_subset_eval)[:,1]
        dev_test_pred_class_eval = (dev_test_pred_proba_eval > 0.5).astype(int)
        logger.info("Development Test Performance Metrics:"); logger.info(f"  LogLoss: {log_loss(y_test_dev, dev_test_pred_proba_eval):.4f}"); logger.info(f"  ROC-AUC: {roc_auc_score(y_test_dev, dev_test_pred_proba_eval):.4f}"); logger.info(f"  AUPRC: {average_precision_score(y_test_dev, dev_test_pred_proba_eval):.4f}"); logger.info(f"  F1-Score: {f1_score(y_test_dev, dev_test_pred_class_eval):.4f}")
        logger.info(f"  CM (Dev Test):\n{confusion_matrix(y_test_dev, dev_test_pred_class_eval)}"); logger.info(f"  Report (Dev Test):\n{classification_report(y_test_dev, dev_test_pred_class_eval, target_names=['Not E (0)', 'Electron (1)'])}")
    else: logger.warning("Skipping Stage 7.2 (Dev Test Metrics) as data is not available.")


    # --- Stage 7.3: Evaluation on Final Unseen Test Set & SUBMISSION FILE GENERATION ---
    logger.info(f"\n--- Stage 7.3: Evaluating on Final Unseen Test Set ({TEST_PATH}) & Generating Submission Files ---")
    try: final_test_df_raw = pd.read_hdf(TEST_PATH); logger.info(f"Loaded final test data, shape: {final_test_df_raw.shape}")
    except Exception as e: logger.error(f"Error loading final test data from {TEST_PATH}: {e}"); raise

    if 'feature_cols' not in locals() or not feature_cols: # Should be loaded from checkpoint or available from Step 1
        logger.error("`feature_cols` (original feature list) is not defined. Cannot process final test set.")
        raise NameError("`feature_cols` not defined for final test set processing.")
        
    X_final_test_raw = final_test_df_raw[feature_cols] 
    X_final_test_scaled_full = X_final_test_raw.copy()
    if scaler is not None: 
        # Identify columns that were scaled during training and are present in test data
        # numerical_cols_to_scale should be available from Step 2 or checkpoint
        if 'numerical_cols_to_scale' not in locals(): # Attempt to reload if missing (e.g. checkpoint load for S7 only)
            try: 
                temp_train_df_for_num_cols = pd.read_hdf(TRAIN_PATH, stop=5) # Read a few rows to get cols
                numerical_cols_to_scale = temp_train_df_for_num_cols.select_dtypes(include=np.number).columns.tolist()
                logger.info(f"Re-identified numerical_cols_to_scale for final test set: {len(numerical_cols_to_scale)}")
            except: 
                logger.warning("Could not re-identify numerical_cols_to_scale, using all columns in X_final_test_raw for scaling attempt if scaler exists.")
                numerical_cols_to_scale = X_final_test_raw.columns.tolist()

        cols_to_scale_in_test = [col for col in numerical_cols_to_scale if col in X_final_test_scaled_full.columns]
        if cols_to_scale_in_test:
            X_final_test_scaled_full[cols_to_scale_in_test] = scaler.transform(X_final_test_raw[cols_to_scale_in_test])
            logger.info(f"Applied fitted scaler to {len(cols_to_scale_in_test)} columns in the final test set.")
    elif 'numerical_cols_to_scale' in locals() and numerical_cols_to_scale: 
        logger.error("Scaler is None, but scaling was expected. Cannot scale final test data."); raise ValueError("Scaler not available.")

    X_final_test_subset = X_final_test_scaled_full[final_selected_features]
    logger.info(f"Shape of final test data subset for prediction: {X_final_test_subset.shape}")
    final_test_pred_proba = final_model.predict_proba(X_final_test_subset)[:,1]
    
    # --- Generate Predictions Submission File (NEW FORMATTING) ---
    # ID should be 0-based row number
    ids_for_submission = np.arange(len(final_test_df_raw))
    predictions_submission_df = pd.DataFrame({'id': ids_for_submission, TARGET_COLUMN: final_test_pred_proba})
    predictions_submission_df.to_csv(PREDICTIONS_SUBMISSION_PATH, index=False, header=False) # No header as per example
    logger.info(f"Predictions submission CSV saved to: {PREDICTIONS_SUBMISSION_PATH}")
    logger.info(f"  Format: id, {TARGET_COLUMN} (no header in file)")
    logger.info(f"  First 5 rows of predictions file:\n{predictions_submission_df.head().to_string(index=False, header=False)}")


    # --- Generate Variable List Submission File (NEW) ---
    variable_list_df = pd.DataFrame(final_selected_features, columns=['feature_name']) # Use a header for clarity, though example doesn't show one
    variable_list_df.to_csv(VARIABLELIST_SUBMISSION_PATH, index=False, header=False) # No header as per example
    logger.info(f"Variable list submission CSV saved to: {VARIABLELIST_SUBMISSION_PATH}")
    logger.info(f"  Format: feature_name (one per line, no header in file)")
    logger.info(f"  First 5 variables in list file:\n{variable_list_df.head().to_string(index=False, header=False)}")


    if TARGET_COLUMN in final_test_df_raw.columns: # If true labels are available for the test set
        logger.info(f"Target column '{TARGET_COLUMN}' found in the final test set. Evaluating performance.")
        y_final_test = final_test_df_raw[TARGET_COLUMN]; final_test_pred_class = (final_test_pred_proba > 0.5).astype(int)
        logger.info("Final Unseen Test Performance (target known):"); logger.info(f"  LogLoss: {log_loss(y_final_test, final_test_pred_proba):.4f}"); roc_auc_final_test = roc_auc_score(y_final_test, final_test_pred_proba); logger.info(f"  ROC-AUC: {roc_auc_final_test:.4f}"); logger.info(f"  AUPRC: {average_precision_score(y_final_test, final_test_pred_proba):.4f}"); logger.info(f"  F1-Score: {f1_score(y_final_test, final_test_pred_class):.4f}")
        logger.info(f"  CM (Final Test):\n{confusion_matrix(y_final_test, final_test_pred_class)}"); logger.info(f"  Report (Final Test):\n{classification_report(y_final_test, final_test_pred_class, target_names=['Not E (0)', 'Electron (1)'])}")
        
        sns.set_theme(style="whitegrid"); fpr_final_test,tpr_final_test,_ = roc_curve(y_final_test,final_test_pred_proba)
        plt.figure(figsize=(10,8)); colors_final = sns.color_palette("viridis", 1)
        plt.plot(fpr_final_test,tpr_final_test,color=colors_final[0],lw=2, label=f'Final Unseen Test ROC (Top {N_FEATURES_FINAL} Feats, AUC={roc_auc_final_test:.4f})')
        plt.plot([0,1],[0,1],color='grey',lw=2,linestyle='--'); plt.xlim([-0.02,1.02]); plt.ylim([-0.02,1.02]) 
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve - Final Unseen Test Set Evaluation'); plt.legend(loc="lower right"); plt.grid(True)
        s7_3_roc_final_test_path = os.path.join(plots_dir, "s7_3_roc_curve_final_unseen_test.pdf") 
        plt.savefig(s7_3_roc_final_test_path, bbox_inches='tight'); plt.clf(); plt.close('all'); logger.info(f"ROC curve for final unseen test set evaluation saved to {s7_3_roc_final_test_path}")
    else:
        logger.warning(f"Target column '{TARGET_COLUMN}' not found in the final test set ({TEST_PATH}). Only predictions saved.")

    logger.info("--- Step 7 Complete ---")
    logger.info("\n--- CLASSIFICATION PIPELINE EXECUTION FINISHED SUCCESSFULLY ---")

if __name__ == "__main__":
    cli_args = parse_arguments()
    if cli_args.target_column != 'p_Truth_isElectron':
        logger.warning(f"Target column is '{cli_args.target_column}'. For Dataset 1 (classification), it should be 'p_Truth_isElectron'.")
    main(cli_args)

