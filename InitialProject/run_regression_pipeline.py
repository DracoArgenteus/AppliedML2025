import pandas as pd
import numpy as np
import joblib 
import os
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for Matplotlib BEFORE pyplot import
import matplotlib.pyplot as plt
import seaborn as sns # For styling plots
import argparse 
import logging 
import csv # For writing variable list

from sklearn.model_selection import train_test_split, KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
import shap 
import optuna 

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    datefmt='%H:%M:%S') 
logger = logging.getLogger(__name__)

# --- Custom Metric Function for Regression ---
def mean_absolute_relative_error(y_true, y_pred):
    epsilon = 1e-9 
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    valid_indices = np.abs(y_true_np) > epsilon * 1000 
    if not np.any(valid_indices):
        logger.warning("mean_absolute_relative_error: No valid (non-zero) y_true values after filtering. Returning NaN.")
        return np.nan 
    y_true_filt = y_true_np[valid_indices]
    y_pred_filt = y_pred_np[valid_indices]
    if len(y_true_filt) == 0: 
        logger.warning("mean_absolute_relative_error: y_true_filt is empty after filtering. Returning NaN.")
        return np.nan
    relative_error = (y_pred_filt - y_true_filt) / (y_true_filt + epsilon) 
    return np.mean(np.abs(relative_error))

# --- Argument Parsing ---
def parse_arguments_regression():
    parser = argparse.ArgumentParser(description="Run the LightGBM REGRESSION pipeline for electron energy, adhering to submission guidelines.")
    # I/O
    parser.add_argument('--output_base_dir', type=str, required=True, help='Base directory for all script outputs (plots, models, submission CSVs).')
    parser.add_argument('--train_path', type=str, default="./data/AppML_InitialProject_train.h5", help='Path to the training data HDF5 file.')
    parser.add_argument('--test_path', type=str, default="./data/AppML_InitialProject_test_regression.h5", help='Path to the REGRESSION test data HDF5 file.')
    
    # --- Submission Naming Arguments (NEW) ---
    parser.add_argument('--firstname', type=str, required=True, help='Your first name for submission file naming.')
    parser.add_argument('--lastname', type=str, required=True, help='Your last name for submission file naming.')
    parser.add_argument('--solution_name', type=str, required=True, help='A descriptive name for your regression solution (e.g., LGBM_Optuna_Reg).')

    # Data Config
    parser.add_argument('--electron_flag_column', type=str, default='p_Truth_isElectron', help='Column name for electron flag.')
    parser.add_argument('--target_column_reg', type=str, default='p_Truth_Energy', help='Name of the regression target column (energy).')
    parser.add_argument('--exclude_columns_reg', nargs='*', default=['p_Truth_isElectron', 'p_Truth_Energy'], help='List of columns to exclude from features for regression.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--validation_size_reg', type=float, default=0.20, help='Proportion for validation set (from electron-only data).')
    parser.add_argument('--dev_test_size_reg', type=float, default=0.15, help='Proportion for development test set (from electron-only data).')

    # Feature Analysis (Step R4)
    parser.add_argument('--cv_folds_reg', type=int, default=5, help='Number of CV folds for regression.')
    parser.add_argument('--max_features_analysis_reg', type=int, default=15, help='Max features for iterative performance analysis (regression). Max 12 will be used for model.')
    parser.add_argument('--acceptable_rel_mae', type=float, default=0.1, help='Acceptable Mean Absolute Relative Error threshold for N_FEATURES_FINAL_REG determination.')
    parser.add_argument('--outer_loop_patience_reg', type=int, default=3, help='Patience for early stopping in feature iteration loop.')
    parser.add_argument('--min_rel_mae_improvement', type=float, default=0.001, help='Min relative MAE improvement for feature iteration (lower is better).')
    
    # Optuna (Step R5)
    parser.add_argument('--n_optuna_trials_reg', type=int, default=50, help='Number of Optuna trials for regression (if not using predefined).')
    parser.add_argument('--optuna_n_jobs_reg', type=int, default=1, help='Number of parallel jobs for Optuna study.optimize().')
    parser.add_argument('--use_predefined_params_reg', action='store_true', default=False, help='Skip Optuna and use predefined best hyperparameters for Regression Step R5.')

    # Checkpoints
    parser.add_argument('--save_checkpoint_reg_s5_5', action='store_true', default=False, help='Enable saving regression checkpoint after Step R5.5.')
    parser.add_argument('--load_checkpoint_reg_s6', action='store_true', default=False, help='Enable loading regression checkpoint before Step R6.')
    parser.add_argument('--load_checkpoint_reg_s7', action='store_true', default=False, help='Enable loading regression checkpoint before Step R7.')
    
    # Plotting
    parser.add_argument('--skip_visualization_reg_s2_5', action='store_true', default=False, help="Skip feature/target visualization in Step R2.5")
    parser.add_argument('--skip_shap_summary_plot_reg_s3_5', action='store_true', default=False, help="Skip SHAP summary plot in Step R3.5")

    return parser.parse_args()

def main_regression(args):
    """
    Main function to orchestrate the regression pipeline.
    """
    # --- Initialize variables from parsed arguments ---
    OUTPUT_BASE_DIR = args.output_base_dir
    TRAIN_PATH = args.train_path
    TEST_PATH_REG = args.test_path 

    # --- Submission Naming Variables (NEW) ---
    FIRST_NAME = args.firstname
    LAST_NAME = args.lastname
    SOLUTION_NAME_REG = args.solution_name # Specific solution name for regression

    ELECTRON_FLAG_COLUMN = args.electron_flag_column
    TARGET_COLUMN_REG = args.target_column_reg
    EXCLUDE_COLUMNS_REG = args.exclude_columns_reg
    RANDOM_SEED = args.random_seed
    VALIDATION_SIZE_REG = args.validation_size_reg
    DEV_TEST_SIZE_REG = args.dev_test_size_reg
    CV_FOLDS_REG = args.cv_folds_reg
    MAX_FEATURES_FOR_ANALYSIS_REG = args.max_features_analysis_reg
    ACCEPTABLE_RELATIVE_MAE_THRESHOLD = args.acceptable_rel_mae
    OUTER_LOOP_EARLY_STOPPING_PATIENCE_REG = args.outer_loop_patience_reg
    MIN_REL_MAE_IMPROVEMENT = args.min_rel_mae_improvement 
    N_OPTUNA_TRIALS_REG = args.n_optuna_trials_reg
    OPTUNA_N_JOBS_REG = args.optuna_n_jobs_reg
    
    MAX_FEATURES_ALLOWED_REG = 12 # Project constraint

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    # Create subdirectories for plots and models if they are specific to this script's outputs
    plots_dir_reg = os.path.join(OUTPUT_BASE_DIR, "plots_regression")
    models_dir_reg = os.path.join(OUTPUT_BASE_DIR, "models_regression")
    os.makedirs(plots_dir_reg, exist_ok=True)
    os.makedirs(models_dir_reg, exist_ok=True)

    logger.info(f"REGRESSION Pipeline started. Output base directory: {OUTPUT_BASE_DIR}")
    logger.info(f"Plots will be saved to: {plots_dir_reg}")
    logger.info(f"Models will be saved to: {models_dir_reg}")
    logger.debug(f"REGRESSION: Full arguments received: {args}")

    # --- Define paths for internal outputs (scaler, models, checkpoints) ---
    SCALER_SAVE_PATH_REG = os.path.join(models_dir_reg, "r2_scaler_regression.joblib")
    CHECKPOINT_DIR_R5_5 = os.path.join(OUTPUT_BASE_DIR, "r5_5_checkpoint_regression") 
    FINAL_MODEL_SAVE_PATH_NATIVE_REG = os.path.join(models_dir_reg, f"r6_final_lgbm_model_regression_{SOLUTION_NAME_REG}_native.txt")
    FINAL_MODEL_SAVE_PATH_JOBLIB_REG = os.path.join(models_dir_reg, f"r6_final_lgbm_model_regression_{SOLUTION_NAME_REG}_joblib.pkl")
    
    # --- Define submission file paths (NEW - will be used in Step R7.3) ---
    submission_file_base_name_reg = f"Regression_{FIRST_NAME}{LAST_NAME}_{SOLUTION_NAME_REG}"
    REGRESSION_PREDICTIONS_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name_reg}.csv")
    REGRESSION_VARIABLELIST_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name_reg}_VariableList.csv")
    logger.info(f"Regression predictions submission CSV will be: {REGRESSION_PREDICTIONS_SUBMISSION_PATH}")
    logger.info(f"Regression variable list submission CSV will be: {REGRESSION_VARIABLELIST_SUBMISSION_PATH}")


    # ==============================================================================
    # --- Step R1: Data Loading and Preparation for Regression ---
    # ==============================================================================
    logger.info("--- Step R1: Data Loading and Preparation for Regression ---")
    logger.info(f"Loading original training data from: {TRAIN_PATH}")
    try: original_train_df = pd.read_hdf(TRAIN_PATH) 
    except FileNotFoundError: logger.error(f"Training data file not found at {TRAIN_PATH}"); raise
    except Exception as e: logger.error(f"Error loading HDF5 file '{TRAIN_PATH}': {e}."); raise
    
    logger.info(f"Filtering for true electrons using column: '{ELECTRON_FLAG_COLUMN}' for training...")
    electron_df = original_train_df[original_train_df[ELECTRON_FLAG_COLUMN] == 1].copy()
    if electron_df.empty: 
        logger.error("No true electrons found in the training data. Regression pipeline cannot proceed.")
        raise ValueError("No true electrons for regression training.")
    logger.info(f"Filtered for true electrons for training. Shape of electron_df: {electron_df.shape}")
    
    all_cols_reg = electron_df.columns.tolist()
    # Ensure target and electron flag are in exclude list for feature selection
    current_exclude_reg = list(set(EXCLUDE_COLUMNS_REG + [ELECTRON_FLAG_COLUMN, TARGET_COLUMN_REG]))
    feature_cols_reg = [col for col in all_cols_reg if col not in current_exclude_reg]

    if not feature_cols_reg: 
        logger.error("No feature columns identified for regression after exclusions.")
        raise ValueError("No regression features identified.")
        
    X_reg_original = electron_df[feature_cols_reg]
    y_reg_original = electron_df[TARGET_COLUMN_REG]
    logger.info(f"Number of original features for regression: {len(feature_cols_reg)}")
    logger.debug(f"Regression feature names (first 5): {feature_cols_reg[:5]}")

    logger.info(f"Splitting electron-only data (Validation size: {VALIDATION_SIZE_REG*100:.0f}%, Dev Test size: {DEV_TEST_SIZE_REG*100:.0f}%, Seed: {RANDOM_SEED})...")
    X_dev_temp_reg, X_test_dev_reg, y_dev_temp_reg, y_test_dev_reg = train_test_split(
        X_reg_original, y_reg_original, test_size=DEV_TEST_SIZE_REG, random_state=RANDOM_SEED
    )
    
    val_size_adjusted_reg = 0
    if (1 - DEV_TEST_SIZE_REG) > 0 and VALIDATION_SIZE_REG > 0 :
        val_size_adjusted_reg = VALIDATION_SIZE_REG / (1 - DEV_TEST_SIZE_REG)
    
    if not (0 < val_size_adjusted_reg < 1) and val_size_adjusted_reg != 0: # Can be 0 if VALIDATION_SIZE_REG is 0
        logger.error(f"Invalid adjusted validation size for regression: {val_size_adjusted_reg}. Check DEV_TEST_SIZE_REG and VALIDATION_SIZE_REG.")
        raise ValueError("Invalid data split sizes for regression.")

    if val_size_adjusted_reg == 0 or X_dev_temp_reg.empty:
        X_train_reg, X_val_reg, y_train_reg, y_val_reg = X_dev_temp_reg, pd.DataFrame(), y_dev_temp_reg, pd.Series(dtype='float64')
        logger.info("Validation set for regression is empty or not created from this split.")
    else:
        X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
            X_dev_temp_reg, y_dev_temp_reg, test_size=val_size_adjusted_reg, random_state=RANDOM_SEED
        )

    logger.info(f"  Shape of X_train_reg: {X_train_reg.shape}, y_train_reg: {y_train_reg.shape}")
    logger.info(f"  Shape of X_val_reg: {X_val_reg.shape}, y_val_reg: {y_val_reg.shape}")
    logger.info(f"  Shape of X_test_dev_reg: {X_test_dev_reg.shape}, y_test_dev_reg: {y_test_dev_reg.shape}")
    if not y_train_reg.empty: logger.info(f"  Descriptive stats for y_train_reg (Target: {TARGET_COLUMN_REG}):\n{y_train_reg.describe().to_string()}")

    # ==============================================================================
    # --- Step R2: Pre-processing: Direct Scaling (Fitting on X_train_reg) ---
    # ==============================================================================
    logger.info("\n--- Step R2: Pre-processing for Regression: Direct Scaling (Fitting on X_train_reg) ---")
    X_train_reg_scaled = X_train_reg.copy()
    X_val_reg_scaled = X_val_reg.copy() if not X_val_reg.empty else pd.DataFrame()
    X_test_dev_reg_scaled = X_test_dev_reg.copy() if not X_test_dev_reg.empty else pd.DataFrame()
    
    numerical_cols_to_scale_reg = X_train_reg.select_dtypes(include=np.number).columns.tolist()
    scaler_reg = StandardScaler()

    if numerical_cols_to_scale_reg and not X_train_reg.empty:
        logger.info(f"Fitting StandardScaler on {len(numerical_cols_to_scale_reg)} numerical features from X_train_reg...")
        X_train_reg_scaled[numerical_cols_to_scale_reg] = scaler_reg.fit_transform(X_train_reg[numerical_cols_to_scale_reg])
        logger.info("   StandardScaler_reg fitted on X_train_reg and X_train_reg_scaled created.")
        
        if hasattr(scaler_reg, 'mean_'): 
            if not X_val_reg.empty: 
                X_val_reg_scaled[numerical_cols_to_scale_reg] = scaler_reg.transform(X_val_reg[numerical_cols_to_scale_reg])
                logger.info("   X_val_reg scaled.")
            if not X_test_dev_reg.empty: 
                X_test_dev_reg_scaled[numerical_cols_to_scale_reg] = scaler_reg.transform(X_test_dev_reg[numerical_cols_to_scale_reg])
                logger.info("   X_test_dev_reg scaled.")
            joblib.dump(scaler_reg, SCALER_SAVE_PATH_REG)
            logger.info(f"   Fitted scaler_reg saved to: {SCALER_SAVE_PATH_REG}")
        else: 
            logger.warning("   Scaler_reg was not fitted (e.g., no numeric columns or empty data after selection). Skipping transform and save.")
            scaler_reg = None 
    elif X_train_reg.empty:
        logger.warning("   X_train_reg is empty. Skipping scaling for regression.")
        scaler_reg = None
    else: 
        logger.warning("   No numerical columns identified/selected to scale in X_train_reg.")
        scaler_reg = None 
    logger.info("--- Regression Pre-processing with Direct Scaling complete ---")
    if not X_train_reg_scaled.empty: logger.debug(f"X_train_reg_scaled head (first 3 rows):\n{X_train_reg_scaled.head(3).to_string()}")

    # ==============================================================================
    # --- Step R2.5: Visualizing Data for Regression ---
    # ==============================================================================
    if not args.skip_visualization_reg_s2_5:
        logger.info("\n--- Step R2.5: Visualizing selected features and target for Regression ---")
        if not y_train_reg.empty:
            plt.figure(figsize=(8, 5)); sns.histplot(y_train_reg, kde=True, color='skyblue', bins=50)
            plt.title(f'Distribution of Target: {TARGET_COLUMN_REG} (True Electrons, Training Set)'); plt.xlabel("Energy (GeV)"); plt.ylabel("Frequency")
            target_vis_path = os.path.join(plots_dir_reg, "r2_5_target_energy_distribution.pdf"); plt.savefig(target_vis_path, bbox_inches='tight'); plt.clf(); plt.close(); 
            logger.info(f"   Target distribution plot saved to {target_vis_path}")
        else:
            logger.warning("   y_train_reg is empty. Skipping target distribution plot.")

        if not X_train_reg.empty and len(feature_cols_reg) > 0:
            example_feature_reg_vis = feature_cols_reg[0] # Default to first if specific one not found
            if 'pX_topoetcone20ptCorrection' in feature_cols_reg: # Prefer this specific one if available
                 example_feature_reg_vis = 'pX_topoetcone20ptCorrection'
            
            features_to_visualize_reg_list = []
            if example_feature_reg_vis in X_train_reg.columns : features_to_visualize_reg_list.append(example_feature_reg_vis)
            
            if len(X_train_reg.columns) > 1:
                available_others_reg = [col for col in X_train_reg.columns if col != example_feature_reg_vis]
                if available_others_reg: 
                    num_additional_plots = min(2, len(available_others_reg))
                    other_features_indices = np.random.choice(len(available_others_reg), size=num_additional_plots, replace=False)
                    features_to_visualize_reg_list.extend([available_others_reg[i] for i in other_features_indices])
            
            features_to_visualize_reg_list = list(set(features_to_visualize_reg_list))

            if not features_to_visualize_reg_list: 
                logger.warning("   No features available/selected for distribution visualization in Step R2.5.")
            else:
                logger.info(f"   Visualizing feature distributions (from electron-only training data): {features_to_visualize_reg_list}")
                for feature_name_to_vis in features_to_visualize_reg_list:
                    if feature_name_to_vis in X_train_reg.columns:
                        plt.figure(figsize=(12, 5)); 
                        plt.subplot(1, 2, 1); sns.histplot(X_train_reg[feature_name_to_vis], kde=True, color='blue', bins=50)
                        plt.title(f'Original ({feature_name_to_vis})\nSkew: {X_train_reg[feature_name_to_vis].skew():.2f}'); plt.xlabel("Value"); plt.ylabel("Frequency")
                        plt.subplot(1, 2, 2)
                        if feature_name_to_vis in X_train_reg_scaled.columns: 
                            sns.histplot(X_train_reg_scaled[feature_name_to_vis], kde=True, color='orange', bins=50)
                            plt.title(f'Scaled ({feature_name_to_vis})\nSkew: {X_train_reg_scaled[feature_name_to_vis].skew():.2f}'); plt.xlabel("Value (Std)"); plt.ylabel("Frequency")
                        else: plt.title(f'Scaled ({feature_name_to_vis}) - Not found in scaled data')
                        plt.tight_layout(); 
                        feature_vis_path_reg = os.path.join(plots_dir_reg, f"r2_5_feature_visualization_reg_{feature_name_to_vis}.pdf"); 
                        plt.savefig(feature_vis_path_reg, bbox_inches='tight'); plt.clf(); plt.close(); 
                        logger.info(f"   Feature visualization for '{feature_name_to_vis}' saved to {feature_vis_path_reg}")
                    else:
                        logger.warning(f"   Feature '{feature_name_to_vis}' not found in X_train_reg for visualization.")
            plt.close('all')
        elif X_train_reg.empty:
            logger.warning("   X_train_reg is empty. Skipping feature distribution plots.")
    else: 
        logger.info("Skipping Step R2.5: Feature/Target Visualization for Regression as per CLI argument.")

    # Ensure data is available for subsequent steps
    if X_train_reg_scaled.empty or y_train_reg.empty:
        logger.error("X_train_reg_scaled or y_train_reg is empty. Cannot proceed with regression model training.")
        raise ValueError("Regression training data is empty, cannot proceed.")

    # ==============================================================================
    # --- Step R3: Fitting Preliminary LightGBM Regressor (for SHAP Analysis) ---
    # ==============================================================================
    logger.info("\n--- Step R3: Fitting Preliminary LightGBM Regressor (for SHAP Analysis) ---")
    prelim_regressor_for_shap = lgb.LGBMRegressor(objective='regression_l1', random_state=RANDOM_SEED, num_leaves=31, n_jobs=-1, verbosity=-1, n_estimators=500, learning_rate=0.05)
    logger.info("Fitting preliminary LGBMRegressor for SHAP analysis...")
    try: 
        eval_set_shap_reg = []
        if not X_val_reg_scaled.empty and not y_val_reg.empty:
            eval_set_shap_reg = [(X_val_reg_scaled, y_val_reg)]
        prelim_regressor_for_shap.fit(
            X_train_reg_scaled, y_train_reg, 
            eval_set=eval_set_shap_reg if eval_set_shap_reg else None, 
            eval_metric='mae', 
            callbacks=[lgb.early_stopping(20, verbose=False)] if eval_set_shap_reg else [])
    except Exception as e: logger.error(f"Error during prelim_regressor_for_shap.fit(): {e}"); raise
    logger.info("Preliminary LGBMRegressor fitted successfully.")

    # ==============================================================================
    # --- Step R3.5: SHAP Analysis & Feature Ranking for Regression ---
    # ==============================================================================
    logger.info("\n--- Step R3.5: SHAP Analysis & Feature Ranking for Regression ---")
    if not hasattr(prelim_regressor_for_shap, '_Booster') or prelim_regressor_for_shap._Booster is None: 
        logger.error("Preliminary regressor (prelim_regressor_for_shap) is not fitted."); raise RuntimeError("Prelim regressor not fitted.")
    logger.info("Generating SHAP values for regressor...")
    explainer_reg = shap.TreeExplainer(prelim_regressor_for_shap)
    shap_values_reg = explainer_reg.shap_values(X_train_reg_scaled) 
    mean_abs_shap_reg = np.mean(np.abs(shap_values_reg), axis=0)
    shap_importance_df_reg = pd.DataFrame({'feature': X_train_reg_scaled.columns, 'importance': mean_abs_shap_reg}).sort_values(by='importance', ascending=False)
    all_shap_ranked_features_reg = shap_importance_df_reg['feature'].tolist()
    logger.info(f"Top features for regression (SHAP - showing up to {MAX_FEATURES_ALLOWED_REG+3}):\n{all_shap_ranked_features_reg[:MAX_FEATURES_ALLOWED_REG+3]}")
    
    if not args.skip_shap_summary_plot_reg_s3_5:
        logger.info("Generating SHAP summary plot for regressor...")
        plt.figure() # Ensure new figure context
        shap.summary_plot(shap_values_reg, X_train_reg_scaled, show=False, plot_type="bar", max_display=min(30, len(X_train_reg_scaled.columns)))
        plt.title("SHAP Feature Importance (Regression - Prelim. Model)")
        summary_plot_path_reg = os.path.join(plots_dir_reg, "r3_5_shap_feature_ranking_regression.pdf"); 
        plt.savefig(summary_plot_path_reg, bbox_inches='tight'); plt.clf(); plt.close('all')
        logger.info(f"SHAP summary plot for regression saved to {summary_plot_path_reg}")
    else: 
        logger.info("Skipping Step R3.5: SHAP Summary Plot for Regression as per CLI argument.")

    # ==============================================================================
    # --- Step R4: Iterative Feature Performance Analysis for Regression (using Relative MAE) ---
    # ==============================================================================
    logger.info("\n--- Step R4: Iterative Feature Performance Analysis for Regression ---")
    # ... (Step R4 logic - ensure it uses y_train_reg, X_train_reg_scaled) ...
    # Fallback if SHAP fails or yields no features
    if not all_shap_ranked_features_reg:
        logger.warning("SHAP analysis yielded no ranked features for regression. Falling back to original feature list for iterative analysis.")
        all_shap_ranked_features_reg = feature_cols_reg # Use the initial list of features for regression
        if not all_shap_ranked_features_reg:
            logger.error("No features available for regression iterative analysis even after fallback.")
            raise ValueError("No features for regression iterative analysis.")

    logger.info(f"Using ACCEPTABLE_RELATIVE_MAE_THRESHOLD: {ACCEPTABLE_RELATIVE_MAE_THRESHOLD} (lower is better)")
    rel_mae_scores_vs_n_features = []; n_features_tested_list_reg = [] 
    n_features_upper_bound_reg = min(len(all_shap_ranked_features_reg), MAX_FEATURES_FOR_ANALYSIS_REG)
    if n_features_upper_bound_reg < 1: 
        logger.error("No features for regression analysis in Step R4 after potential fallbacks."); 
        raise ValueError("No features for reg analysis in Step R4.")
    
    n_features_range_potential_reg = range(1, n_features_upper_bound_reg + 1)
    best_rel_mae_so_far = float('inf'); steps_without_improvement_reg = 0

    for k_features_reg in n_features_range_potential_reg:
        current_top_k_features_reg = all_shap_ranked_features_reg[:k_features_reg]
        X_subset_for_cv_reg = X_train_reg_scaled[current_top_k_features_reg]
        logger.info(f"  Step R4: Testing with top {k_features_reg} features for regression...")
        n_features_tested_list_reg.append(k_features_reg)
        cv_splitter_reg = KFold(n_splits=CV_FOLDS_REG, shuffle=True, random_state=RANDOM_SEED)
        fold_rel_mae_scores = [] 
        for fold_num, (train_idx, val_idx_cv) in enumerate(cv_splitter_reg.split(X_subset_for_cv_reg, y_train_reg)):
            X_train_fold, X_val_fold_cv = X_subset_for_cv_reg.iloc[train_idx], X_subset_for_cv_reg.iloc[val_idx_cv]
            y_train_fold, y_val_fold_cv = y_train_reg.iloc[train_idx], y_train_reg.iloc[val_idx_cv]
            model_iterative_reg = lgb.LGBMRegressor(objective='regression_l1', random_state=RANDOM_SEED + fold_num, num_leaves=31, n_jobs=-1, verbosity=-1, n_estimators=300, learning_rate=0.05)
            try: 
                model_iterative_reg.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold_cv,y_val_fold_cv)], eval_metric='mae', callbacks=[lgb.early_stopping(10,verbose=False)])
            except Exception as e: 
                logger.warning(f"Fold {fold_num+1} error ({k_features_reg} features, reg): {e}"); fold_rel_mae_scores.append(np.nan); continue
            preds_val_fold = model_iterative_reg.predict(X_val_fold_cv); rel_mae_fold = mean_absolute_relative_error(y_val_fold_cv, preds_val_fold)
            fold_rel_mae_scores.append(rel_mae_fold)
        mean_cv_rel_mae = np.nanmean(fold_rel_mae_scores) if fold_rel_mae_scores else np.nan
        rel_mae_scores_vs_n_features.append(mean_cv_rel_mae)
        logger.info(f"    Mean CV Relative MAE for {k_features_reg} features: {mean_cv_rel_mae:.4f}")
        if pd.notna(mean_cv_rel_mae) and mean_cv_rel_mae < best_rel_mae_so_far - MIN_REL_MAE_IMPROVEMENT: 
            best_rel_mae_so_far = mean_cv_rel_mae; steps_without_improvement_reg = 0
        else: steps_without_improvement_reg += 1
        if steps_without_improvement_reg >= OUTER_LOOP_EARLY_STOPPING_PATIENCE_REG: 
            logger.info(f"Outer loop early stop for regression at {k_features_reg} features."); break 
            
    plt.figure(figsize=(12,7)); 
    plt.plot(n_features_tested_list_reg, rel_mae_scores_vs_n_features, marker='o', color='darkgreen', label=f'Mean {CV_FOLDS_REG}-Fold Rel. MAE')
    plt.xlabel("N Top SHAP Features (Regression)"); plt.ylabel(f"Mean {CV_FOLDS_REG}-Fold Relative MAE"); plt.title("Regression Performance vs. N Features")
    if n_features_tested_list_reg:
        max_f_reg = max(n_features_tested_list_reg); min_f_reg = min(n_features_tested_list_reg)
        tick_s_reg = 1 if max_f_reg <=10 else 2 if max_f_reg <=20 else int(np.ceil(max_f_reg/10))
        plt.xticks(list(range(min_f_reg, max_f_reg + 1, tick_s_reg)))
    plt.grid(True, alpha=0.6); plt.axhline(y=ACCEPTABLE_RELATIVE_MAE_THRESHOLD, color='red', ls='--', label=f'Acceptable Rel. MAE ({ACCEPTABLE_RELATIVE_MAE_THRESHOLD})')
    if rel_mae_scores_vs_n_features and not all(np.isnan(s) for s in rel_mae_scores_vs_n_features if s is not None):
        valid_scores_for_min = [s for s in rel_mae_scores_vs_n_features if pd.notna(s)]
        if valid_scores_for_min: min_rel_mae = np.nanmin(valid_scores_for_min); plt.axhline(y=min_rel_mae, color='blue', ls=':', label=f'Min Achieved ({min_rel_mae:.4f})')
    plt.legend(loc='upper right'); plt.tight_layout()
    plot_path_sR4 = os.path.join(plots_dir_reg, "r4_relative_mae_vs_n_features_regression.pdf"); 
    plt.savefig(plot_path_sR4); plt.clf(); plt.close('all')
    logger.info(f"Regression performance plot saved: {plot_path_sR4}")
    for k,s in zip(n_features_tested_list_reg, rel_mae_scores_vs_n_features): logger.info(f"  Reg Features: {k:2d}, Relative MAE: {s:.4f}" if pd.notna(s) else f"  Reg Features: {k:2d}, Relative MAE: NaN")
    logger.info("--- Step R4 Complete ---")


    # ==============================================================================
    # --- Determine N_FEATURES_FINAL_REG (max 12) ---
    # ==============================================================================
    logger.info("\n--- Determining N_FEATURES_FINAL_REG ---")
    # ... (N_FEATURES_FINAL_REG determination logic, respecting MAX_FEATURES_ALLOWED_REG) ...
    N_FEATURES_FINAL_REG = 0; achieved_threshold_rel_mae = float('inf')
    if not rel_mae_scores_vs_n_features or not n_features_tested_list_reg: 
        logger.warning("Cannot determine N_FEATURES_FINAL_REG from Step R4 results. Using all SHAP features up to MAX_FEATURES_ALLOWED_REG or fallback.")
        N_FEATURES_FINAL_REG = min(len(all_shap_ranked_features_reg), MAX_FEATURES_ALLOWED_REG) if all_shap_ranked_features_reg else 0
        if N_FEATURES_FINAL_REG == 0 and feature_cols_reg:
            N_FEATURES_FINAL_REG = min(len(feature_cols_reg), MAX_FEATURES_ALLOWED_REG)
            all_shap_ranked_features_reg = feature_cols_reg # Use original as ranked
            logger.warning(f"Using up to {N_FEATURES_FINAL_REG} original features for regression due to no Step R4 results.")
        elif N_FEATURES_FINAL_REG == 0:
            logger.error("No features available for regression. Cannot determine N_FEATURES_FINAL_REG.")
            raise ValueError("No features available for regression model.")
    else:
        for i, score in enumerate(rel_mae_scores_vs_n_features):
            if np.isnan(score): continue
            num_features = n_features_tested_list_reg[i]
            if score <= ACCEPTABLE_RELATIVE_MAE_THRESHOLD and num_features <= MAX_FEATURES_ALLOWED_REG: 
                N_FEATURES_FINAL_REG = num_features; achieved_threshold_rel_mae = score; break 
        if N_FEATURES_FINAL_REG == 0: # Threshold not met or not met within MAX_FEATURES_ALLOWED_REG
            valid_scores_reg_filtered = [(rel_mae_scores_vs_n_features[i], n_features_tested_list_reg[i]) 
                                         for i in range(len(rel_mae_scores_vs_n_features)) 
                                         if not np.isnan(rel_mae_scores_vs_n_features[i]) and n_features_tested_list_reg[i] <= MAX_FEATURES_ALLOWED_REG]
            if valid_scores_reg_filtered:
                valid_scores_reg_filtered.sort(key=lambda x: (x[0], x[1])) # Sort by MAE (asc), then by num_features (asc)
                achieved_threshold_rel_mae, N_FEATURES_FINAL_REG = valid_scores_reg_filtered[0]
                logger.warning(f"Threshold not met or not met within {MAX_FEATURES_ALLOWED_REG} features. Using best within limit: {N_FEATURES_FINAL_REG} features, Rel. MAE: {achieved_threshold_rel_mae:.4f}.")
            else: # No valid scores within limit
                logger.warning(f"No valid scores found within {MAX_FEATURES_ALLOWED_REG} features. Trying best overall up to limit.")
                # Fallback to best overall score, then cap at MAX_FEATURES_ALLOWED_REG
                valid_scores_overall = [(rel_mae_scores_vs_n_features[i], n_features_tested_list_reg[i]) for i in range(len(rel_mae_scores_vs_n_features)) if not np.isnan(rel_mae_scores_vs_n_features[i])]
                if valid_scores_overall:
                    valid_scores_overall.sort(key=lambda x: (x[0], x[1]))
                    temp_mae, temp_n_feat = valid_scores_overall[0]
                    N_FEATURES_FINAL_REG = min(temp_n_feat, MAX_FEATURES_ALLOWED_REG)
                    # Find the score for this N_FEATURES_FINAL_REG if it was tested
                    if N_FEATURES_FINAL_REG in n_features_tested_list_reg:
                        idx_n_final = n_features_tested_list_reg.index(N_FEATURES_FINAL_REG)
                        achieved_threshold_rel_mae = rel_mae_scores_vs_n_features[idx_n_final] if not np.isnan(rel_mae_scores_vs_n_features[idx_n_final]) else float('inf')
                    else: achieved_threshold_rel_mae = float('inf') # Score not directly available for this capped N_FEATURES_FINAL_REG
                    logger.warning(f"Best overall score had {temp_n_feat} features. Capping to {N_FEATURES_FINAL_REG}. Rel. MAE for {N_FEATURES_FINAL_REG} features: {achieved_threshold_rel_mae:.4f}")
                else:
                    logger.error("All Rel MAE scores in Step R4 were NaN."); raise ValueError("All Step R4 scores are NaN.")
    
    if N_FEATURES_FINAL_REG == 0: # Final fallback if still 0
        N_FEATURES_FINAL_REG = min(MAX_FEATURES_ALLOWED_REG, len(all_shap_ranked_features_reg) if all_shap_ranked_features_reg else (len(feature_cols_reg) if feature_cols_reg else 1))
        logger.warning(f"N_FEATURES_FINAL_REG was 0 after analysis, using fallback: {N_FEATURES_FINAL_REG}")

    final_selected_features_reg = all_shap_ranked_features_reg[:N_FEATURES_FINAL_REG]
    if not final_selected_features_reg : 
        logger.error("Final selected features list for regression is empty!"); 
        if feature_cols_reg:
            logger.warning(f"Falling back to using first {min(MAX_FEATURES_ALLOWED_REG, len(feature_cols_reg))} original features for regression.")
            final_selected_features_reg = feature_cols_reg[:min(MAX_FEATURES_ALLOWED_REG, len(feature_cols_reg))]
            N_FEATURES_FINAL_REG = len(final_selected_features_reg)
        else:
            raise ValueError("final_selected_features_reg is empty and no fallback original features.")
    logger.info(f"Determined N_FEATURES_FINAL_REG for Optuna and final model: {N_FEATURES_FINAL_REG}")
    logger.info(f"Final selected features for regression ({len(final_selected_features_reg)}): {final_selected_features_reg}")


    # ==============================================================================
    # --- Step R5: Hyperparameter Optimization with Optuna for Regression ---
    # ==============================================================================
    logger.info(f"\n--- Step R5: Optuna for Top {N_FEATURES_FINAL_REG} Features (Minimizing Relative MAE) ---")
    # ... (Optuna logic for regression, ensuring it uses y_train_reg, X_train_selected_for_optuna_reg) ...
    if not all(f in X_train_reg_scaled.columns for f in final_selected_features_reg):
        missing_fs_reg = [f for f in final_selected_features_reg if f not in X_train_reg_scaled.columns]
        logger.error(f"Critical error: Regression features selected for Optuna ({missing_fs_reg}) are not in X_train_reg_scaled.")
        raise ValueError(f"Selected regression features for Optuna not found in scaled training data: {missing_fs_reg}")

    X_train_selected_for_optuna_reg = X_train_reg_scaled[final_selected_features_reg]
    best_lgbm_params_reg = {} 
    best_rel_mae_optuna = float('inf') 

    if args.use_predefined_params_reg:
        logger.info("Skipping Optuna REGRESSION study and using predefined hyperparameters.")
        best_lgbm_params_reg = {'boosting_type': 'dart', 'n_estimators': 2000, 'learning_rate': 0.022, 'num_leaves': 35, 'max_depth': 9, 'min_child_samples': 12, 'subsample': 0.503, 'colsample_bytree': 0.998, 'reg_alpha': 1.32e-05, 'reg_lambda': 3.30e-07, 'objective': 'regression_l1', 'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbosity': -1}
        logger.info(f"Using predefined REGRESSION hyperparameters: {best_lgbm_params_reg}")
    else:
        logger.info(f"Starting Optuna study for REGRESSION ({N_OPTUNA_TRIALS_REG} trials, minimizing Relative MAE)...")
        def optuna_objective_relative_mae(trial, X_data, y_data_obj):
            lgbm_reg_params = {'objective': 'regression_l1', 'random_state': RANDOM_SEED, 'n_jobs': -1, 'verbosity': -1, 'boosting_type': trial.suggest_categorical('boosting_type_reg', ['gbdt', 'dart']), 'n_estimators': trial.suggest_int('n_estimators_reg', 200, 2000, step=100), 'learning_rate': trial.suggest_float('learning_rate_reg', 1e-4, 0.1, log=True), 'num_leaves': trial.suggest_int('num_leaves_reg', 5, 60), 'max_depth': trial.suggest_int('max_depth_reg', 3, 10), 'min_child_samples': trial.suggest_int('min_child_samples_reg', 5, 50), 'subsample': trial.suggest_float('subsample_reg', 0.5, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree_reg', 0.4, 1.0), 'reg_alpha': trial.suggest_float('reg_alpha_reg', 1e-8, 10.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda_reg', 1e-8, 10.0, log=True)}
            cv_reg = KFold(n_splits=CV_FOLDS_REG, shuffle=True, random_state=RANDOM_SEED); scores_reg = []
            for fold, (train_idx, val_idx_opt) in enumerate(cv_reg.split(X_data, y_data_obj)):
                X_f, X_v = X_data.iloc[train_idx], X_data.iloc[val_idx_opt]; y_f, y_v_opt = y_data_obj.iloc[train_idx], y_data_obj.iloc[val_idx_opt]
                model_reg = lgb.LGBMRegressor(**lgbm_reg_params)
                try: model_reg.fit(X_f, y_f, eval_set=[(X_v,y_v_opt)], eval_metric='mae', callbacks=[lgb.early_stopping(25,verbose=False)])
                except Exception as e: logger.warning(f"Optuna REG Trial {trial.number} Fold {fold+1} Error: {e}"); return float('inf')
                preds_reg = model_reg.predict(X_v); score_reg = mean_absolute_relative_error(y_v_opt, preds_reg)
                if np.isnan(score_reg): return float('inf') 
                scores_reg.append(score_reg); trial.report(score_reg, step=fold)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            return np.mean(scores_reg) if scores_reg else float('inf')
        optuna.logging.set_verbosity(optuna.logging.WARNING); study_reg = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=max(1, CV_FOLDS_REG-2) ,n_min_trials=max(5, N_OPTUNA_TRIALS_REG //10)))
        study_reg.optimize(lambda trial: optuna_objective_relative_mae(trial, X_train_selected_for_optuna_reg, y_train_reg), n_trials=N_OPTUNA_TRIALS_REG, n_jobs=OPTUNA_N_JOBS_REG, show_progress_bar=(N_OPTUNA_TRIALS_REG > 10))
        best_lgbm_params_reg = study_reg.best_params; best_rel_mae_optuna = study_reg.best_value
        logger.info(f"Optuna study for regression finished. Best Relative MAE: {best_rel_mae_optuna:.4f}")
    
    cleaned_best_lgbm_params_reg = {key.replace('_reg', ''): value for key, value in best_lgbm_params_reg.items()}
    best_lgbm_params_reg = cleaned_best_lgbm_params_reg
    best_lgbm_params_reg.setdefault('objective', 'regression_l1'); best_lgbm_params_reg.setdefault('random_state', RANDOM_SEED); best_lgbm_params_reg.setdefault('n_jobs', -1); best_lgbm_params_reg.setdefault('verbosity', -1)

    logger.info("Best hyperparameters for regression (cleaned):"); [logger.info(f"    {k}: {v}") for k,v in best_lgbm_params_reg.items()]
    logger.info("--- Step R5 Complete ---")

    # ==============================================================================
    # --- Step R5.5: Saving Regression Pipeline State for Checkpoint ---
    # ==============================================================================
    logger.info(f"\n--- Step R5.5: Saving Regression Pipeline State for Checkpoint ---")
    if args.save_checkpoint_reg_s5_5:
        os.makedirs(CHECKPOINT_DIR_R5_5, exist_ok=True) 
        vars_to_save_reg_s5_5 = { 'X_train_reg_scaled': X_train_reg_scaled, 'y_train_reg': y_train_reg, 'X_val_reg_scaled': X_val_reg_scaled, 'y_val_reg': y_val_reg, 'X_test_dev_reg_scaled': X_test_dev_reg_scaled, 'y_test_dev_reg': y_test_dev_reg, 'final_selected_features_reg': final_selected_features_reg, 'best_lgbm_params_reg': best_lgbm_params_reg, 'scaler_reg': scaler_reg, 'RANDOM_SEED': RANDOM_SEED, 'TARGET_COLUMN_REG': TARGET_COLUMN_REG, 'EXCLUDE_COLUMNS_REG': EXCLUDE_COLUMNS_REG, 'ELECTRON_FLAG_COLUMN': ELECTRON_FLAG_COLUMN, 'TEST_PATH_REG': TEST_PATH_REG, 'feature_cols_reg': feature_cols_reg, 'N_FEATURES_FINAL_REG': N_FEATURES_FINAL_REG, 'ACCEPTABLE_RELATIVE_MAE_THRESHOLD': ACCEPTABLE_RELATIVE_MAE_THRESHOLD, 'FIRST_NAME': FIRST_NAME, 'LAST_NAME': LAST_NAME, 'SOLUTION_NAME_REG': SOLUTION_NAME_REG } # Added submission names
        for name, var_val in vars_to_save_reg_s5_5.items():
            try: joblib.dump(var_val, os.path.join(CHECKPOINT_DIR_R5_5, f"r5_5_checkpoint_{name}.joblib")); logger.info(f"  Saved '{name}'") 
            except Exception as e: logger.error(f"  Error saving '{name}': {e}")
    else: logger.info("Skipping regression checkpoint saving for Step R5.5.")

    # ==============================================================================
    # --- Step R6: Final Regression Model Training and Saving ---
    # ==============================================================================
    logger.info(f"\n--- Step R6: Training and Saving Final LightGBM Regressor ---")
    # ... (Loading from checkpoint logic for R6) ...
    if args.load_checkpoint_reg_s6:
        logger.info(f"Attempting to load from regression checkpoint: {CHECKPOINT_DIR_R5_5}")
        try:
            # Load all necessary variables from checkpoint_vars_map as in previous version
            # For brevity, assuming this part is robust.
            X_train_reg_scaled = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_X_train_reg_scaled.joblib'))
            y_train_reg = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_y_train_reg.joblib'))
            X_val_reg_scaled = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_X_val_reg_scaled.joblib'))
            y_val_reg = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_y_val_reg.joblib'))
            final_selected_features_reg = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_final_selected_features_reg.joblib'))
            best_lgbm_params_reg = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_best_lgbm_params_reg.joblib'))
            RANDOM_SEED = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_RANDOM_SEED.joblib'))
            scaler_reg = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_scaler_reg.joblib'))
            FIRST_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_FIRST_NAME.joblib'))
            LAST_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_LAST_NAME.joblib'))
            SOLUTION_NAME_REG = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_SOLUTION_NAME_REG.joblib'))
            # Redefine submission paths
            submission_file_base_name_reg = f"Regression_{FIRST_NAME}{LAST_NAME}_{SOLUTION_NAME_REG}"
            REGRESSION_PREDICTIONS_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name_reg}.csv")
            REGRESSION_VARIABLELIST_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name_reg}_VariableList.csv")
            logger.info(f"Submission paths updated after loading R6 checkpoint.")
            logger.info("Variables loaded from regression checkpoint for Step R6.")
        except Exception as e: logger.error(f"Error loading regression checkpoint for Step R6: {e}"); raise

    if X_train_reg_scaled.empty or y_train_reg.empty or not final_selected_features_reg or not best_lgbm_params_reg:
        logger.error("CRITICAL: Missing essential data for final regression model training."); raise ValueError("Essential data missing.")

    X_train_final_subset_reg = X_train_reg_scaled[final_selected_features_reg]
    X_val_final_subset_reg = X_val_reg_scaled[final_selected_features_reg] if not X_val_reg_scaled.empty else None
    
    final_model_reg = lgb.LGBMRegressor(**best_lgbm_params_reg)
    logger.info("Training final LightGBM Regressor...")
    eval_set_final_reg = []
    if X_val_final_subset_reg is not None and not y_val_reg.empty:
        eval_set_final_reg = [(X_val_final_subset_reg, y_val_reg)]
    try:
        final_model_reg.fit(X_train_final_subset_reg, y_train_reg, 
                            eval_set=eval_set_final_reg if eval_set_final_reg else None, 
                            eval_metric='mae', 
                            callbacks=[lgb.early_stopping(30,verbose=True)] if eval_set_final_reg else [])
        logger.info(f"Final regressor trained. Best iteration: {final_model_reg.best_iteration_ if hasattr(final_model_reg, 'best_iteration_') and final_model_reg.best_iteration_ is not None else 'N/A'}")
    except Exception as e: logger.error(f"Error training final regressor: {e}"); raise
        
    try:
        final_model_reg.booster_.save_model(FINAL_MODEL_SAVE_PATH_NATIVE_REG); logger.info(f"Final regressor (native) saved to: {FINAL_MODEL_SAVE_PATH_NATIVE_REG}")
        joblib.dump(final_model_reg, FINAL_MODEL_SAVE_PATH_JOBLIB_REG); logger.info(f"Final regressor (joblib) saved to: {FINAL_MODEL_SAVE_PATH_JOBLIB_REG}")
    except Exception as e: logger.error(f"Error saving final regressor: {e}")
    logger.info("--- Step R6 Complete ---")

    # ==============================================================================
    # --- Step R7: Prediction, Validation for Regression & SUBMISSION FILE GENERATION ---
    # ==============================================================================
    logger.info(f"\n--- Step R7: Prediction, Validation for Regression & Submission File Generation ---")
    # ... (Loading from checkpoint logic for R7) ...
    if args.load_checkpoint_reg_s7:
        logger.info(f"Attempting to load from regression checkpoint for Step R7: {CHECKPOINT_DIR_R5_5}")
        try:
            # Load all necessary variables from checkpoint
            X_test_dev_reg_scaled = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_X_test_dev_reg_scaled.joblib'))
            y_test_dev_reg = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_y_test_dev_reg.joblib'))
            final_selected_features_reg = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_final_selected_features_reg.joblib'))
            scaler_reg = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_scaler_reg.joblib'))
            TEST_PATH_REG = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_TEST_PATH_REG.joblib'))
            TARGET_COLUMN_REG = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_TARGET_COLUMN_REG.joblib'))
            ELECTRON_FLAG_COLUMN = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_ELECTRON_FLAG_COLUMN.joblib'))
            feature_cols_reg = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_feature_cols_reg.joblib'))
            N_FEATURES_FINAL_REG = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_N_FEATURES_FINAL_REG.joblib'))
            FIRST_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_FIRST_NAME.joblib'))
            LAST_NAME = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_LAST_NAME.joblib'))
            SOLUTION_NAME_REG = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_SOLUTION_NAME_REG.joblib'))
            # Redefine submission paths
            submission_file_base_name_reg = f"Regression_{FIRST_NAME}{LAST_NAME}_{SOLUTION_NAME_REG}"
            REGRESSION_PREDICTIONS_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name_reg}.csv")
            REGRESSION_VARIABLELIST_SUBMISSION_PATH = os.path.join(OUTPUT_BASE_DIR, f"{submission_file_base_name_reg}_VariableList.csv")
            logger.info(f"Submission paths updated after loading R7 checkpoint.")

            if os.path.exists(FINAL_MODEL_SAVE_PATH_JOBLIB_REG):
                final_model_reg = joblib.load(FINAL_MODEL_SAVE_PATH_JOBLIB_REG)
                logger.info(f"Loaded regressor (joblib) for Step R7 from: {FINAL_MODEL_SAVE_PATH_JOBLIB_REG}")
            elif os.path.exists(FINAL_MODEL_SAVE_PATH_NATIVE_REG):
                best_lgbm_params_reg_loaded = joblib.load(os.path.join(CHECKPOINT_DIR_R5_5, 'r5_5_checkpoint_best_lgbm_params_reg.joblib'))
                booster_reg = lgb.Booster(model_file=FINAL_MODEL_SAVE_PATH_NATIVE_REG)
                final_model_reg = lgb.LGBMRegressor(**best_lgbm_params_reg_loaded); final_model_reg._Booster=booster_reg; final_model_reg._fitted=True
                logger.info(f"Loaded regressor (native) for Step R7 from: {FINAL_MODEL_SAVE_PATH_NATIVE_REG}")
            else: raise FileNotFoundError("No saved regressor found for Step R7 loading from checkpoint.")
            logger.info("Variables and regressor loaded for Step R7 from checkpoint.")
        except Exception as e: logger.error(f"Error loading regression checkpoint for Step R7: {e}"); raise
    
    if 'final_model_reg' not in locals() or final_model_reg is None: 
        logger.error("CRITICAL: 'final_model_reg' is not defined for Step R7."); raise NameError("Missing 'final_model_reg'.")

    # --- Stage R7.1: Plot (Train vs Dev-Test) for final_model_reg ---
    # ... (Plotting logic for R7.1) ...
    logger.info("\n--- Stage R7.1: Generating True vs. Predicted Plot (Train vs. Dev Test) for Final Regressor ---")
    if 'X_train_final_subset_reg' in locals() and 'y_train_reg' in locals() and not X_train_final_subset_reg.empty and not y_train_reg.empty:
        sns.set_theme(style="whitegrid"); plt.figure(figsize=(10, 8)); colors_reg = sns.color_palette("viridis", 2) 
        train_pred_energy = final_model_reg.predict(X_train_final_subset_reg)
        plt.scatter(y_train_reg, train_pred_energy, color=colors_reg[0], alpha=0.5, label=f'Train Data (Top {N_FEATURES_FINAL_REG} Feats)', s=10)
        
        if not X_test_dev_reg_scaled.empty and not y_test_dev_reg.empty and final_selected_features_reg:
            X_test_dev_subset_for_plot_reg = X_test_dev_reg_scaled[final_selected_features_reg]
            dev_test_pred_energy_for_plot = final_model_reg.predict(X_test_dev_subset_for_plot_reg)
            plt.scatter(y_test_dev_reg, dev_test_pred_energy_for_plot, color=colors_reg[1], alpha=0.5, marker='^', label=f'Dev Test Data (Top {N_FEATURES_FINAL_REG} Feats)', s=10)
        
        all_true_energies = list(y_train_reg.values) + (list(y_test_dev_reg.values) if not y_test_dev_reg.empty else [])
        if all_true_energies: 
            min_val = np.min(all_true_energies); max_val = np.max(all_true_energies)
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
        plt.xlabel('True Energy (GeV)'); plt.ylabel('Predicted Energy (GeV)'); plt.title(f'True vs. Predicted Energy: Final Regressor'); plt.legend(loc="upper left"); plt.grid(True)
        r7_1_pred_vs_actual_path = os.path.join(plots_dir_reg, "r7_1_pred_vs_actual_train_vs_dev_test.pdf"); 
        plt.savefig(r7_1_pred_vs_actual_path, bbox_inches='tight'); plt.clf(); plt.close('all'); 
        logger.info(f"True vs. Predicted Energy plot saved to {r7_1_pred_vs_actual_path}")
    else:
        logger.warning("Skipping Stage R7.1 plot as training data for plot is unavailable.")

    # --- Stage R7.2: Evaluation on Development Test Set ---
    logger.info(f"\n--- Stage R7.2: Evaluating Metrics on Development Test Set (True Electrons Only) ---")
    if not X_test_dev_reg_scaled.empty and not y_test_dev_reg.empty and final_selected_features_reg:
        X_test_dev_subset_eval_reg = X_test_dev_reg_scaled[final_selected_features_reg]
        dev_test_pred_energy_eval = final_model_reg.predict(X_test_dev_subset_eval_reg)
        logger.info("Development Test Set Performance (Regression):")
        rel_mae_dev = mean_absolute_relative_error(y_test_dev_reg, dev_test_pred_energy_eval); mae_dev = mean_absolute_error(y_test_dev_reg, dev_test_pred_energy_eval); rmse_dev = np.sqrt(mean_squared_error(y_test_dev_reg, dev_test_pred_energy_eval)); r2_dev = r2_score(y_test_dev_reg, dev_test_pred_energy_eval)
        logger.info(f"  Mean Absolute Relative Error: {rel_mae_dev:.4f}"); logger.info(f"  Mean Absolute Error (MAE): {mae_dev:.4f} GeV"); logger.info(f"  Root Mean Squared Error (RMSE): {rmse_dev:.4f} GeV"); logger.info(f"  R-squared (R2): {r2_dev:.4f}")
    else: logger.warning("Skipping Dev Test Set metric evaluation due to missing data.")

    # --- Stage R7.3: Evaluation on Final Unseen Test Set & SUBMISSION FILE GENERATION ---
    logger.info(f"\n--- Stage R7.3: Evaluating Regressor on Final Unseen Test Set from {TEST_PATH_REG} & Generating Submission Files ---")
    try: 
        final_test_df_reg_raw = pd.read_hdf(TEST_PATH_REG)
        logger.info(f"Loaded final regression test data, shape: {final_test_df_reg_raw.shape}")
    except Exception as e: 
        logger.error(f"Error loading final regression test data from {TEST_PATH_REG}: {e}"); raise

    final_test_true_electrons_df = pd.DataFrame() # Initialize
    if ELECTRON_FLAG_COLUMN in final_test_df_reg_raw.columns:
        final_test_true_electrons_df = final_test_df_reg_raw[final_test_df_reg_raw[ELECTRON_FLAG_COLUMN] == 1].copy()
        if final_test_true_electrons_df.empty: 
            logger.warning(f"REG: No true electrons in final test file {TEST_PATH_REG} after filtering. Submission files will be empty or not generated.")
        else: 
            logger.info(f"REG: Filtered final test set for true electrons. New shape: {final_test_true_electrons_df.shape}")
    else: 
        logger.warning(f"REG: Electron flag '{ELECTRON_FLAG_COLUMN}' not in final test file. Assuming all samples are for regression or it's fully blind. Using all rows.")
        final_test_true_electrons_df = final_test_df_reg_raw.copy()

    if not final_test_true_electrons_df.empty:
        # Ensure feature_cols_reg is available (it's defined in R1, should be loaded from checkpoint if needed)
        if 'feature_cols_reg' not in locals(): 
            logger.error("`feature_cols_reg` not defined for final test set processing. This should have been loaded from checkpoint if applicable.")
            # Attempt a fallback or raise error
            current_exclude_reg_fallback = list(set(EXCLUDE_COLUMNS_REG + [ELECTRON_FLAG_COLUMN, TARGET_COLUMN_REG]))
            feature_cols_reg = [col for col in final_test_true_electrons_df.columns.tolist() if col not in current_exclude_reg_fallback]
            if not feature_cols_reg: raise ValueError("Cannot determine feature_cols_reg for final test set.")
            logger.warning("Attempted fallback for feature_cols_reg.")


        X_final_test_reg_raw_electrons = final_test_true_electrons_df[feature_cols_reg]
        
        X_final_test_reg_scaled_electrons = X_final_test_reg_raw_electrons.copy()
        if scaler_reg is not None:
            # numerical_cols_to_scale_reg should be available from Step R2 or checkpoint
            if 'numerical_cols_to_scale_reg' not in locals():
                numerical_cols_to_scale_reg = X_final_test_reg_raw_electrons.select_dtypes(include=np.number).columns.tolist()
                logger.warning(f"Re-identified numerical_cols_to_scale_reg for final test scaling: {len(numerical_cols_to_scale_reg)}")

            cols_to_scale_in_test_reg = [col for col in numerical_cols_to_scale_reg if col in X_final_test_reg_scaled_electrons.columns]
            if cols_to_scale_in_test_reg:
                X_final_test_reg_scaled_electrons[cols_to_scale_in_test_reg] = scaler_reg.transform(X_final_test_reg_raw_electrons[cols_to_scale_in_test_reg])
        elif 'numerical_cols_to_scale_reg' in locals() and numerical_cols_to_scale_reg:
             logger.error("Scaler_reg is None, but scaling was expected for regression. Cannot scale final test data."); raise ValueError("Scaler_reg not available.")

        X_final_test_subset_reg_electrons = X_final_test_reg_scaled_electrons[final_selected_features_reg]
        logger.info(f"Shape of final regression test data subset (true electrons) for prediction: {X_final_test_subset_reg_electrons.shape}")
        final_test_pred_energy = final_model_reg.predict(X_final_test_subset_reg_electrons)
        
        # --- Generate Predictions Submission File ---
        ids_for_submission_reg = np.arange(len(final_test_true_electrons_df)) # 0-based index for the submitted (electron) events
        predictions_reg_submission_df = pd.DataFrame({'id': ids_for_submission_reg, 'prediction': final_test_pred_energy})
        predictions_reg_submission_df.to_csv(REGRESSION_PREDICTIONS_SUBMISSION_PATH, index=False, header=False)
        logger.info(f"Regression predictions submission CSV saved to: {REGRESSION_PREDICTIONS_SUBMISSION_PATH}")
        logger.info(f"  Format: id, prediction_energy_GeV (no header in file)")
        with open(REGRESSION_PREDICTIONS_SUBMISSION_PATH, 'r') as f: # Log first few lines
            logger.info(f"  First 3 lines of predictions file:\n  " + "".join([next(f) for _ in range(min(3, len(predictions_reg_submission_df)))]).replace("\n", "\n  "))


        # --- Generate Variable List Submission File ---
        with open(REGRESSION_VARIABLELIST_SUBMISSION_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for feature in final_selected_features_reg:
                writer.writerow([feature])
        logger.info(f"Regression variable list submission CSV saved to: {REGRESSION_VARIABLELIST_SUBMISSION_PATH}")
        logger.info(f"  Format: feature_name (one per line, no header in file)")
        with open(REGRESSION_VARIABLELIST_SUBMISSION_PATH, 'r') as f: # Log first few lines
            logger.info(f"  First 3 lines of variable list file:\n  " + "".join([next(f) for _ in range(min(3, len(final_selected_features_reg)))]).replace("\n", "\n  "))


        if TARGET_COLUMN_REG in final_test_true_electrons_df.columns: # If true labels are available for evaluation
            y_final_test_reg = final_test_true_electrons_df[TARGET_COLUMN_REG]
            logger.info("Final Unseen Regression Test Performance (True Electrons Only):")
            rel_mae_final = mean_absolute_relative_error(y_final_test_reg, final_test_pred_energy); mae_final = mean_absolute_error(y_final_test_reg, final_test_pred_energy); rmse_final = np.sqrt(mean_squared_error(y_final_test_reg, final_test_pred_energy)); r2_final = r2_score(y_final_test_reg, final_test_pred_energy)
            logger.info(f"  Mean Absolute Relative Error: {rel_mae_final:.4f}"); logger.info(f"  Mean Absolute Error (MAE): {mae_final:.4f} GeV"); logger.info(f"  Root Mean Squared Error (RMSE): {rmse_final:.4f} GeV"); logger.info(f"  R-squared (R2): {r2_final:.4f}")
            
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(10,8)); plt.scatter(y_final_test_reg, final_test_pred_energy, alpha=0.5, edgecolors='w', linewidth=0.5, s=15, label="Predictions")
            min_val_plot = min(y_final_test_reg.min(), final_test_pred_energy.min()); max_val_plot = max(y_final_test_reg.max(), final_test_pred_energy.max())
            plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'grey', ls='--', lw=2, label='Perfect Prediction'); 
            plt.xlabel("True Energy (GeV)"); plt.ylabel("Predicted Energy (GeV)"); plt.title("True vs. Predicted Energy - Final Regression Test (Electrons)"); plt.legend(loc="upper left"); plt.grid(True)
            r7_3_pred_vs_actual_path = os.path.join(plots_dir_reg, "r7_3_regression_pred_vs_actual_final_test.pdf"); 
            plt.savefig(r7_3_pred_vs_actual_path, bbox_inches='tight'); plt.clf(); plt.close('all')
            logger.info(f"Prediction vs Actual plot for final regression test saved to {r7_3_pred_vs_actual_path}")
        else:
            logger.warning(f"Target column '{TARGET_COLUMN_REG}' not found in the final regression test set (true electrons). Only predictions saved.")
    else: 
        logger.info("Skipping evaluation and submission file generation for final regression test set as no relevant electron samples found after filtering.")
    
    logger.info("--- Step R7 Complete ---")
    logger.info("\n--- REGRESSION Pipeline Execution Finished ---")

if __name__ == "__main__":
    cli_args_reg = parse_arguments_regression()
    main_regression(cli_args_reg)

