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
    parser = argparse.ArgumentParser(description="Run the LightGBM classification pipeline.")
    
    # --- I/O Arguments ---
    parser.add_argument('--output_base_dir', type=str, required=True,
                        help='Base directory for all script outputs (plots, models, checkpoints).')
    parser.add_argument('--train_path', type=str, default="./data/AppML_InitialProject_train.h5",
                        help='Path to the training data HDF5 file.')
    parser.add_argument('--test_path', type=str, default="./data/AppML_InitialProject_test_classification.h5",
                        help='Path to the classification test data HDF5 file.')
    
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

    # Ensure output base directory exists
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    logger.info(f"Pipeline started. Output base directory: {OUTPUT_BASE_DIR}")
    logger.debug(f"Full arguments received: {args}")

    # --- Define prefixed file paths for outputs ---
    SCALER_SAVE_PATH = os.path.join(OUTPUT_BASE_DIR, "s2_scaler.joblib")
    CHECKPOINT_DIR_S5_5 = os.path.join(OUTPUT_BASE_DIR, "s5_5_checkpoint_classification") 
    FINAL_MODEL_SAVE_PATH_NATIVE = os.path.join(OUTPUT_BASE_DIR, "s6_final_lgbm_model_classification_native.txt")
    FINAL_MODEL_SAVE_PATH_JOBLIB = os.path.join(OUTPUT_BASE_DIR, "s6_final_lgbm_model_classification_joblib.pkl")
    CLASSIFICATION_PREDICTIONS_SAVE_PATH = os.path.join(OUTPUT_BASE_DIR, "s7_classification_test_predictions.csv")

    # ==============================================================================
    # --- Step 1: Data Loading and Initial 3-Way Split ---
    # ==============================================================================
    logger.info("--- Step 1: Data Loading and Initial 3-Way Split ---")
    # ... (Code for Step 1 as in previous version) ...
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
    val_size_adjusted = VALIDATION_SIZE / (1 - DEV_TEST_SIZE) if (1 - DEV_TEST_SIZE) > 0 else 0
    if not (0 < val_size_adjusted < 1):
        logger.error(f"Invalid adjusted validation size: {val_size_adjusted}. Check DEV_TEST_SIZE and VALIDATION_SIZE.")
        raise ValueError("Invalid data split sizes leading to invalid adjusted validation size.")
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev_temp, y_dev_temp, test_size=val_size_adjusted, random_state=RANDOM_SEED, stratify=y_dev_temp
    )
    
    logger.info(f"  Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"  Shape of X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"  Shape of X_test_dev: {X_test_dev.shape}, y_test_dev: {y_test_dev.shape}")
    logger.info(f"  Target distribution in y_train:\n{y_train.value_counts(normalize=True).to_string()}")
    logger.info(f"  Target distribution in y_val:\n{y_val.value_counts(normalize=True).to_string()}")
    logger.info(f"  Target distribution in y_test_dev:\n{y_test_dev.value_counts(normalize=True).to_string()}")


    # ==============================================================================
    # --- Step 2: Pre-processing: Direct Scaling (Fitting on X_train) ---
    # ==============================================================================
    logger.info("\n--- Step 2: Pre-processing: Direct Scaling (Fitting on X_train) ---")
    # ... (Code for Step 2 as in previous version) ...
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_dev_scaled = X_test_dev.copy()
    numerical_cols_to_scale = X_train.columns.tolist() 
    scaler = StandardScaler()
    if numerical_cols_to_scale:
        logger.info(f"Fitting StandardScaler on {len(numerical_cols_to_scale)} features from X_train...")
        X_train_scaled[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
        logger.info("   StandardScaler fitted on X_train and X_train_scaled created.")
        if hasattr(scaler, 'mean_'): 
            if not X_val.empty:
                 X_val_scaled[numerical_cols_to_scale] = scaler.transform(X_val[numerical_cols_to_scale])
                 logger.info("   X_val scaled using fitted scaler.")
            if not X_test_dev.empty:
                X_test_dev_scaled[numerical_cols_to_scale] = scaler.transform(X_test_dev[numerical_cols_to_scale])
                logger.info("   X_test_dev scaled using fitted scaler.")
            joblib.dump(scaler, SCALER_SAVE_PATH)
            logger.info(f"   Fitted scaler saved to: {SCALER_SAVE_PATH}")
        else:
            logger.warning("   Scaler was not fitted (e.g., no numeric columns or empty data). Skipping transform.")
    else:
        logger.warning("   No numerical columns identified/selected to scale.")
        scaler = None 
    logger.info("--- Pre-processing with Direct Scaling complete ---")
    logger.debug(f"X_train_scaled head (first 3 rows):\n{X_train_scaled.head(3).to_string()}")


    # ==============================================================================
    # --- Step 2.5 Visualizing Data After Direct Scaling ---
    # ==============================================================================
    if not args.skip_visualization_s2_5:
        # ... (Code for Step 2.5 as in previous version, saving plots to OUTPUT_BASE_DIR) ...
        logger.info("\n--- Step 2.5: Visualizing selected features: Original vs. Directly Scaled ---")
        example_feature_vis = 'pX_topoetcone20ptCorrection' 
        if example_feature_vis not in feature_cols: 
            if len(feature_cols) > 0: example_feature_vis = feature_cols[0] 
            else: example_feature_vis = None
        features_to_visualize_list = []
        if example_feature_vis and example_feature_vis in X_train.columns:
             features_to_visualize_list.append(example_feature_vis)
        if len(X_train.columns) > 1: 
            available_other_features = [col for col in X_train.columns if col != example_feature_vis]
            if available_other_features:
                other_features_vis = np.random.choice(available_other_features, size=min(2, len(available_other_features)), replace=False)
                features_to_visualize_list.extend(other_features_vis)
        features_to_visualize_list = list(set(features_to_visualize_list))
        if not features_to_visualize_list: logger.warning("   No features available or selected for visualization in Step 2.5.")
        else:
            logger.info(f"   Will visualize distributions for: {features_to_visualize_list}")
            for feature_name_to_vis in features_to_visualize_list:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1); sns.histplot(X_train[feature_name_to_vis], kde=True, color='blue', bins=50)
                plt.title(f'Original ({feature_name_to_vis})\nSkew: {X_train[feature_name_to_vis].skew():.2f}'); plt.xlabel("Value"); plt.ylabel("Frequency")
                plt.subplot(1, 2, 2)
                if feature_name_to_vis in X_train_scaled.columns: sns.histplot(X_train_scaled[feature_name_to_vis], kde=True, color='orange', bins=50); plt.title(f'Scaled ({feature_name_to_vis})\nSkew: {X_train_scaled[feature_name_to_vis].skew():.2f}'); plt.xlabel("Value (Standardized)"); plt.ylabel("Frequency")
                else: plt.title(f'Scaled ({feature_name_to_vis}) - Not found in scaled data')
                plt.tight_layout(); feature_vis_path = os.path.join(OUTPUT_BASE_DIR, f"s2_5_feature_visualization_{feature_name_to_vis}.pdf"); plt.savefig(feature_vis_path, bbox_inches='tight'); plt.clf(); plt.close(); logger.info(f"   Feature visualization for '{feature_name_to_vis}' saved to {feature_vis_path}")
            plt.close('all') 
    else: 
        logger.info("Skipping Step 2.5: Feature Visualization as per CLI argument.")

    # ==============================================================================
    # --- Part 3: Fitting Preliminary LightGBM Model (for SHAP Analysis) ---
    # ==============================================================================
    logger.info("\n--- Step 3: Fitting Preliminary LightGBM Model (for SHAP Analysis) ---")
    # ... (Code for Step 3 as in previous version) ...
    if y_train.value_counts().get(1, 0) > 0 and y_train.value_counts().get(0, 0) > 0: scale_pos_weight_shap = y_train.value_counts().loc[0] / y_train.value_counts().loc[1]
    else: scale_pos_weight_shap = 1; logger.warning("Could not calculate scale_pos_weight accurately for SHAP model. Defaulting to 1.")
    logger.info(f"Scale_pos_weight for SHAP model: {scale_pos_weight_shap:.2f}")
    prelim_model_for_shap = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=RANDOM_SEED, scale_pos_weight=scale_pos_weight_shap, num_leaves=11, n_jobs=-1, verbosity=-1, n_estimators=500, learning_rate=0.5) 
    logger.info("Fitting preliminary LightGBM model for SHAP analysis (this may take a few moments)...")
    try: prelim_model_for_shap.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], eval_metric='roc_auc', callbacks=[lgb.early_stopping(20, verbose=False)]); logger.info("Preliminary LightGBM model fitted successfully.")
    except Exception as e: logger.error(f"An unexpected error occurred during prelim_model_for_shap.fit(): {e}"); raise

    # ==============================================================================
    # --- Part 3.5: SHAP Analysis & Feature Ranking ---
    # ==============================================================================
    logger.info("\n--- Step 3.5: SHAP Analysis & Feature Ranking ---")
    # ... (Code for Step 3.5 as in previous version, saving plot to OUTPUT_BASE_DIR) ...
    if not hasattr(prelim_model_for_shap, '_Booster') or prelim_model_for_shap._Booster is None: logger.error("Preliminary SHAP model (prelim_model_for_shap) is not fitted."); raise RuntimeError("prelim_model_for_shap not fitted.")
    logger.info("Generating SHAP values (this can take time for many samples/features)...")
    explainer = shap.TreeExplainer(prelim_model_for_shap); shap_values_raw = explainer.shap_values(X_train_scaled) 
    if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2: shap_values_positive_class = shap_values_raw[1]; logger.debug("SHAP values: list of two arrays, using index 1.")
    else: shap_values_positive_class = shap_values_raw; logger.debug("SHAP values: single array.")
    mean_abs_shap = np.mean(np.abs(shap_values_positive_class), axis=0)
    shap_importance_df = pd.DataFrame({'feature': X_train_scaled.columns, 'importance': mean_abs_shap}).sort_values(by='importance', ascending=False)
    all_shap_ranked_features = shap_importance_df['feature'].tolist()
    logger.info(f"Top 10 SHAP ranked features:\n{all_shap_ranked_features[:10]}")
    logger.debug(f"Full SHAP importance DataFrame (head):\n{shap_importance_df.head().to_string()}")
    if not args.skip_shap_summary_plot_s3_5:
        logger.info("Generating SHAP summary plot...")
        shap.summary_plot(shap_values_positive_class, X_train_scaled, show=False); plt.title("SHAP Summary Plot (Preliminary Model on Scaled Training Data)")
        summary_plot_path = os.path.join(OUTPUT_BASE_DIR, "s3_5_shap_summary_plot.pdf"); plt.savefig(summary_plot_path, bbox_inches='tight'); plt.clf(); plt.close('all')
        logger.info(f"SHAP summary plot saved to {summary_plot_path}")
    else: logger.info("Skipping Step 3.5: SHAP Summary Plot as per CLI argument.")

    # ==============================================================================
    # --- Step 4: Iterative Feature Performance Analysis ---
    # ==============================================================================
    logger.info("\n--- Step 4: Iterative Feature Performance Analysis ---")
    # ... (Full code for Step 4 as in previous version, prefixed filenames, no live plot) ...
    logger.info(f"Using ACCEPTABLE_ROC_AUC_THRESHOLD: {ACCEPTABLE_ROC_AUC_THRESHOLD}")
    roc_auc_scores_vs_n_features = []; n_features_tested_list = [] 
    n_features_upper_bound = min(len(all_shap_ranked_features), MAX_FEATURES_FOR_ANALYSIS)
    if n_features_upper_bound < 1: logger.error("No features for analysis in Step 4."); raise ValueError("No features for analysis.")
    n_features_range_potential = range(1, n_features_upper_bound + 1)
    if y_train.value_counts().get(1,0)>0 and y_train.value_counts().get(0,0)>0: scale_pos_weight_iterative = y_train.value_counts().loc[0]/y_train.value_counts().loc[1]
    else: scale_pos_weight_iterative=1; logger.warning("Could not calculate scale_pos_weight_iterative accurately for Step 4. Defaulting to 1.")
    logger.info(f"Scale_pos_weight for CV models (Step 4): {scale_pos_weight_iterative:.2f}")
    best_roc_auc_so_far = -np.inf; steps_without_improvement = 0
    for k_features in n_features_range_potential:
        current_top_k_features = all_shap_ranked_features[:k_features]
        X_subset_for_cv = X_train_scaled[current_top_k_features]
        logger.info(f"  Step 4: Testing with top {k_features} features...")
        n_features_tested_list.append(k_features)
        cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        fold_roc_auc_scores = [] 
        for fold_num, (train_idx, val_idx) in enumerate(cv_splitter.split(X_subset_for_cv, y_train)):
            X_train_fold, X_val_fold = X_subset_for_cv.iloc[train_idx], X_subset_for_cv.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model_iterative = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=RANDOM_SEED + fold_num, scale_pos_weight=scale_pos_weight_iterative, num_leaves=11, n_jobs=-1, verbosity=-1, n_estimators=300, learning_rate=0.05)
            try: model_iterative.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold,y_val_fold)], eval_metric='roc_auc', callbacks=[lgb.early_stopping(10,verbose=False)])
            except Exception as e: logger.warning(f"Fold {fold_num+1} error ({k_features} features): {e}"); fold_roc_auc_scores.append(np.nan); continue
            preds_proba = model_iterative.predict_proba(X_val_fold)[:,1]; roc_auc_val = roc_auc_score(y_val_fold, preds_proba)
            fold_roc_auc_scores.append(roc_auc_val)
        mean_cv_roc_auc = np.nanmean(fold_roc_auc_scores)
        roc_auc_scores_vs_n_features.append(mean_cv_roc_auc)
        logger.info(f"    Mean CV ROC-AUC for {k_features} features: {mean_cv_roc_auc:.4f}")
        if mean_cv_roc_auc > best_roc_auc_so_far + MIN_ROC_AUC_IMPROVEMENT: best_roc_auc_so_far = mean_cv_roc_auc; steps_without_improvement = 0
        else: steps_without_improvement += 1
        if steps_without_improvement >= OUTER_LOOP_EARLY_STOPPING_PATIENCE: logger.info(f"Outer loop early stop at {k_features} ft."); break 
    final_fig_s4, final_ax_s4 = plt.subplots(figsize=(12,7))
    final_ax_s4.plot(n_features_tested_list, roc_auc_scores_vs_n_features, marker='o', color='teal', label=f'Mean {CV_FOLDS}-Fold ROC-AUC')
    final_ax_s4.set_xlabel("N Top SHAP Features"); final_ax_s4.set_ylabel(f"Mean {CV_FOLDS}-Fold CV ROC-AUC"); final_ax_s4.set_title("Performance vs. N Features")
    if n_features_tested_list:
        max_f = max(n_features_tested_list); min_f = min(n_features_tested_list)
        tick_s = 1 if max_f <=10 else 2 if max_f <=20 else int(np.ceil(max_f/10))
        final_ax_s4.set_xticks(list(range(min_f, max_f + 1, tick_s)))
    final_ax_s4.grid(True, alpha=0.6); final_ax_s4.axhline(y=ACCEPTABLE_ROC_AUC_THRESHOLD, color='g', ls='--', label=f'Acceptable ({ACCEPTABLE_ROC_AUC_THRESHOLD})')
    if roc_auc_scores_vs_n_features and not all(np.isnan(roc_auc_scores_vs_n_features)): final_ax_s4.axhline(y=np.nanmax(roc_auc_scores_vs_n_features), color='purple', ls=':', label=f'Max Achieved ({np.nanmax(roc_auc_scores_vs_n_features):.4f})')
    final_ax_s4.legend(loc='lower right'); plt.tight_layout()
    performance_plot_path_s4 = os.path.join(OUTPUT_BASE_DIR, "s4_roc_auc_vs_n_features.pdf"); plt.savefig(performance_plot_path_s4); plt.clf(); plt.close('all') 
    logger.info(f"Performance plot saved: {performance_plot_path_s4}")
    for k,s in zip(n_features_tested_list, roc_auc_scores_vs_n_features): logger.info(f"  Features: {k:2d}, ROC-AUC: {s:.4f}")
    logger.info("--- Step 4 Complete ---")

    # ==============================================================================
    # --- Determine N_FEATURES_FINAL ---
    # ==============================================================================
    logger.info("\n--- Determining N_FEATURES_FINAL ---")
    # ... (Full code for N_FEATURES_FINAL determination as in previous version) ...
    N_FEATURES_FINAL = 0; achieved_threshold_score = -1.0
    if not roc_auc_scores_vs_n_features or not n_features_tested_list: logger.error("Cannot determine N_FEATURES_FINAL."); raise ValueError("ROC AUC scores or features list empty.")
    else:
        for i, score in enumerate(roc_auc_scores_vs_n_features):
            if np.isnan(score): continue # Skip NaN scores
            num_features = n_features_tested_list[i]
            if score >= ACCEPTABLE_ROC_AUC_THRESHOLD: N_FEATURES_FINAL = num_features; achieved_threshold_score = score; logger.info(f"Found {N_FEATURES_FINAL} features with ROC-AUC {score:.4f} (>= threshold {ACCEPTABLE_ROC_AUC_THRESHOLD})."); break
        if N_FEATURES_FINAL == 0 and roc_auc_scores_vs_n_features: # If threshold not met, pick best
            valid_scores = [s for s in roc_auc_scores_vs_n_features if not np.isnan(s)]
            if valid_scores:
                best_idx = np.argmax(valid_scores)
                # Find corresponding index in original list to get correct num_features
                original_indices = [i for i, s in enumerate(roc_auc_scores_vs_n_features) if not np.isnan(s)]
                N_FEATURES_FINAL=n_features_tested_list[original_indices[best_idx]]
                achieved_threshold_score = valid_scores[best_idx]
                logger.warning(f"Threshold ({ACCEPTABLE_ROC_AUC_THRESHOLD}) not met. Using features with best score: {N_FEATURES_FINAL} features, ROC-AUC: {achieved_threshold_score:.4f}.")
            else: # All scores were NaN
                logger.error("All ROC AUC scores in Step 4 were NaN. Cannot determine N_FEATURES_FINAL."); raise ValueError("All Step 4 scores are NaN.")
    if N_FEATURES_FINAL > MAX_FEATURES_FOR_ANALYSIS: # Respect project constraint
        logger.warning(f"N_FEATURES_FINAL ({N_FEATURES_FINAL}) > MAX_FEATURES_FOR_ANALYSIS ({MAX_FEATURES_FOR_ANALYSIS}). Adjusting."); N_FEATURES_FINAL = MAX_FEATURES_FOR_ANALYSIS
        if N_FEATURES_FINAL in n_features_tested_list: achieved_threshold_score = roc_auc_scores_vs_n_features[n_features_tested_list.index(N_FEATURES_FINAL)] if not np.isnan(roc_auc_scores_vs_n_features[n_features_tested_list.index(N_FEATURES_FINAL)]) else -1.0
        else: achieved_threshold_score = -1.0 ; logger.warning(f"Score for N_FEATURES_FINAL={N_FEATURES_FINAL} not directly computed.")
    logger.info(f"Determined N_FEATURES_FINAL for Optuna: {N_FEATURES_FINAL}")
    if achieved_threshold_score > 0: logger.info(f"(Achieved ROC-AUC with these features: {achieved_threshold_score:.4f})")
    if N_FEATURES_FINAL == 0 and len(all_shap_ranked_features) > 0: N_FEATURES_FINAL = min(5, len(all_shap_ranked_features)); logger.warning(f"N_FEATURES_FINAL was 0 after analysis, using fallback: {N_FEATURES_FINAL}")
    elif N_FEATURES_FINAL == 0 and not all_shap_ranked_features: logger.error("No features from SHAP for N_FEATURES_FINAL."); raise ValueError("No SHAP features for N_FEATURES_FINAL.")
    final_selected_features = all_shap_ranked_features[:N_FEATURES_FINAL] if len(all_shap_ranked_features) >= N_FEATURES_FINAL else all_shap_ranked_features[:]
    if not final_selected_features: logger.error("Final selected features list is empty!"); raise ValueError("final_selected_features is empty.")

    
    # ==============================================================================
    # --- Step 5: Hyperparameter Optimization with Optuna (Optimizing for LogLoss) ---
    # ==============================================================================
    logger.info(f"\n--- Step 5: Optuna for Top {N_FEATURES_FINAL} Features (Minimizing LogLoss) ---")
    # ... (Full code for Step 5 as in run_classification_pipeline_argparse_v3, using args.use_predefined_params) ...
    X_train_selected_for_optuna = X_train_scaled[final_selected_features]
    if y_train.value_counts().get(1,0)>0 and y_train.value_counts().get(0,0)>0: global_scale_pos_weight = y_train.value_counts().loc[0]/y_train.value_counts().loc[1]
    else: global_scale_pos_weight = 1.0; logger.warning("Could not calculate global_scale_pos_weight accurately for Optuna. Defaulting to 1.")
    best_lgbm_params = {} 
    best_logloss_optuna = float('inf') 
    if args.use_predefined_params:
        logger.info("Skipping Optuna study and using predefined hyperparameters.")
        best_lgbm_params = {'boosting_type': 'gbdt', 'n_estimators': 1000, 'learning_rate': 0.03919901908726119, 'num_leaves': 20, 'max_depth': 5, 'min_child_samples': 25, 'subsample': 0.995035270269982, 'colsample_bytree': 0.5764574765051973, 'reg_alpha': 1.5125713669457504e-07, 'reg_lambda': 4.9377864058981435}
        best_lgbm_params.setdefault('objective', 'binary'); best_lgbm_params.setdefault('random_state', RANDOM_SEED)
        best_logloss_optuna = 0.1215 
        logger.info(f"Using predefined best LogLoss (from log): {best_logloss_optuna:.4f}")
        logger.info(f"Using predefined hyperparameters: {best_lgbm_params}")
    else:
        logger.info(f"Starting Optuna study for {N_OPTUNA_TRIALS} trials (minimizing LogLoss)...")
        min_cpus_needed_optuna = OPTUNA_N_JOBS * 4 if OPTUNA_N_JOBS > 1 else 4 
        logger.warning(f"Optuna study.optimize n_jobs={OPTUNA_N_JOBS}. LGBM n_jobs=4. Ensure SLURM --cpus-per-task >= {min_cpus_needed_optuna}.")
        def optuna_objective_logloss(trial, X_data, y_data):
            lgbm_params = {'objective': 'binary', 'random_state': RANDOM_SEED, 'n_jobs': 4, 'verbosity': -1, 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']), 'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=100), 'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True), 'num_leaves': trial.suggest_int('num_leaves', 5, 50 if N_FEATURES_FINAL <=10 else 100), 'max_depth': trial.suggest_int('max_depth', 3, 8 if N_FEATURES_FINAL <=10 else 10), 'min_child_samples': trial.suggest_int('min_child_samples', 5, 50), 'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True)}
            imbalance = trial.suggest_categorical('imbalance_handling', ['scale_pos_weight', 'is_unbalance', 'none'])
            if imbalance == 'scale_pos_weight': lgbm_params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight_val', global_scale_pos_weight*0.8, global_scale_pos_weight*1.2)
            elif imbalance == 'is_unbalance': lgbm_params['is_unbalance'] = True
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED); scores = []
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_data, y_data)):
                X_f,X_v=X_data.iloc[train_idx],X_data.iloc[val_idx]; y_f,y_v=y_data.iloc[train_idx],y_data.iloc[val_idx]
                model=lgb.LGBMClassifier(**lgbm_params)
                try: model.fit(X_f,y_f,eval_set=[(X_v,y_v)],eval_metric='logloss',callbacks=[lgb.early_stopping(25,verbose=False)])
                except Exception as e: logger.warning(f"Optuna Trial {trial.number} Fold {fold+1} Error: {e}"); return float('inf')
                preds=model.predict_proba(X_v)[:,1];epsilon=1e-15;preds=np.clip(preds,epsilon,1-epsilon);score=log_loss(y_v,preds);scores.append(score)
                trial.report(score,step=fold);
                if trial.should_prune(): logger.info(f"Optuna Trial {trial.number} Fold {fold+1} Pruned."); raise optuna.exceptions.TrialPruned()
            return np.mean(scores)
        optuna.logging.set_verbosity(optuna.logging.WARNING); study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5,n_min_trials=10))
        study.optimize(lambda trial: optuna_objective_logloss(trial, X_train_selected_for_optuna, y_train), n_trials=N_OPTUNA_TRIALS, n_jobs=OPTUNA_N_JOBS)
        best_lgbm_params = study.best_params; best_logloss_optuna = study.best_value; logger.info(f"Optuna study finished. Best LogLoss: {best_logloss_optuna:.4f}")
        imbalance_choice = best_lgbm_params.pop('imbalance_handling', None)
        if imbalance_choice == 'scale_pos_weight':
            if 'scale_pos_weight_val' in best_lgbm_params: best_lgbm_params['scale_pos_weight'] = best_lgbm_params.pop('scale_pos_weight_val')
        elif imbalance_choice == 'is_unbalance': best_lgbm_params['is_unbalance'] = True
        if 'scale_pos_weight_val' in best_lgbm_params: best_lgbm_params.pop('scale_pos_weight_val')
    logger.info("Cleaned best hyperparameters (either predefined or from Optuna):")
    for k,v in best_lgbm_params.items(): logger.info(f"    {k}: {v}")
    logger.info("--- Step 5 Complete ---")

    # ==============================================================================
    # --- Step 5.5: Saving Pipeline State for Checkpoint ---
    # ==============================================================================
    logger.info(f"\n--- Step 5.5: Saving Pipeline State for Checkpoint ---")
    logger.info(f"SAVE_CHECKPOINT_S5_5 (from CLI) set to: {args.save_checkpoint_s5_5}")
    if args.save_checkpoint_s5_5:
        os.makedirs(CHECKPOINT_DIR_S5_5, exist_ok=True) 
        logger.info(f"Checkpoint files will be saved in: {CHECKPOINT_DIR_S5_5}")
        vars_to_save_s5_5 = { 'X_train_scaled': X_train_scaled, 'y_train': y_train, 'X_val_scaled': X_val_scaled, 'y_val': y_val, 'X_test_dev_scaled': X_test_dev_scaled, 'y_test_dev': y_test_dev, 'final_selected_features': final_selected_features, 'best_lgbm_params': best_lgbm_params, 'scaler': scaler, 'RANDOM_SEED': RANDOM_SEED, 'TARGET_COLUMN': TARGET_COLUMN, 'EXCLUDE_COLUMNS': EXCLUDE_COLUMNS, 'TEST_PATH': TEST_PATH, 'feature_cols': feature_cols, 'N_FEATURES_FINAL': N_FEATURES_FINAL, 'ACCEPTABLE_ROC_AUC_THRESHOLD': ACCEPTABLE_ROC_AUC_THRESHOLD }
        for name, var_val in vars_to_save_s5_5.items(): 
            try: joblib.dump(var_val, os.path.join(CHECKPOINT_DIR_S5_5, f"s5_5_{name}.joblib")); logger.info(f"  Saved '{name}'") 
            except Exception as e: logger.error(f"  Error saving '{name}': {e}")
    else: 
        logger.info("Skipping checkpoint saving for Step 5.5 as per CLI argument or default.")
    logger.info("--- Checkpoint Saving Logic Complete (Step 5.5) ---")

    # ==============================================================================
    # --- Step 6: Final Model Training and Saving ---
    # ==============================================================================
    logger.info(f"\n--- Step 6: Training and Saving Final LightGBM Model ---")
    # ... (Full code for Step 6 as in run_classification_pipeline_argparse_v3, using args.load_checkpoint_s6) ...
    logger.info(f"LOAD_FROM_CHECKPOINT_S6 (from CLI) set to: {args.load_checkpoint_s6}")
    if args.load_checkpoint_s6:
        logger.info(f"Attempting to load from checkpoint: {CHECKPOINT_DIR_S5_5}")
        try:
            X_train_scaled = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_X_train_scaled.joblib')); y_train = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_y_train.joblib')); X_val_scaled = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_X_val_scaled.joblib')); y_val = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_y_val.joblib')); final_selected_features = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_final_selected_features.joblib')); best_lgbm_params = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_best_lgbm_params.joblib')); RANDOM_SEED = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_RANDOM_SEED.joblib'))
            if os.path.exists(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_scaler.joblib')): scaler = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_scaler.joblib'))
            if os.path.exists(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_X_test_dev_scaled.joblib')): X_test_dev_scaled = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_X_test_dev_scaled.joblib'))
            if os.path.exists(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_y_test_dev.joblib')): y_test_dev = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_y_test_dev.joblib'))
            if os.path.exists(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_TARGET_COLUMN.joblib')): TARGET_COLUMN = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_TARGET_COLUMN.joblib'))
            if os.path.exists(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_EXCLUDE_COLUMNS.joblib')): EXCLUDE_COLUMNS = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_EXCLUDE_COLUMNS.joblib'))
            if os.path.exists(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_TEST_PATH.joblib')): TEST_PATH = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_TEST_PATH.joblib'))
            if os.path.exists(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_feature_cols.joblib')): feature_cols = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_feature_cols.joblib'))
            if os.path.exists(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_N_FEATURES_FINAL.joblib')): N_FEATURES_FINAL = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_N_FEATURES_FINAL.joblib'))
            if os.path.exists(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_ACCEPTABLE_ROC_AUC_THRESHOLD.joblib')): ACCEPTABLE_ROC_AUC_THRESHOLD = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, 's5_5_ACCEPTABLE_ROC_AUC_THRESHOLD.joblib'))
            logger.info("Core variables for Step 6 loaded from checkpoint.")
        except Exception as e: logger.error(f"Error loading core checkpoint for Step 6: {e}"); raise
    logger.info("Performing sanity check for Step 6 variables...")
    required_vars_for_step6_logic = {'X_train_scaled': pd.DataFrame, 'y_train': pd.Series, 'X_val_scaled': pd.DataFrame, 'y_val': pd.Series, 'final_selected_features': list, 'best_lgbm_params': dict, 'RANDOM_SEED': int }
    missing_critical_vars_s6 = []
    for var_name, expected_type in required_vars_for_step6_logic.items():
        var_value = locals().get(var_name, globals().get(var_name))
        if var_name not in locals() and var_name not in globals(): missing_critical_vars_s6.append(f"{var_name} (not found)")
        elif var_value is None: missing_critical_vars_s6.append(f"{var_name} (is None)")
        elif not isinstance(var_value, expected_type): missing_critical_vars_s6.append(f"{var_name} (type {type(var_value)} != {expected_type})")
        elif var_name in ['final_selected_features', 'best_lgbm_params'] and not var_value : 
             if var_name == 'best_lgbm_params' and args.use_predefined_params and not best_lgbm_params: missing_critical_vars_s6.append(f"{var_name} (empty despite use_predefined_params)")
             elif var_name == 'final_selected_features' and (N_FEATURES_FINAL == 0 and len(all_shap_ranked_features)>0) : pass 
             elif var_name != 'best_lgbm_params' : missing_critical_vars_s6.append(f"{var_name} (empty)")
    if missing_critical_vars_s6: logger.error(f"CRITICAL: Missing/invalid vars for Step 6: {missing_critical_vars_s6}"); raise NameError(f"Missing/invalid vars for Step 6: {missing_critical_vars_s6}")
    else: logger.info("All required variables for Step 6 are present and appear valid.")
    logger.info(f"Using {len(final_selected_features)} features for final model: {final_selected_features}")
    X_train_final_subset = X_train_scaled[final_selected_features]; X_val_final_subset = X_val_scaled[final_selected_features]
    final_model_params = best_lgbm_params.copy(); final_model_params.setdefault('objective','binary'); final_model_params.setdefault('random_state',RANDOM_SEED); final_model_params.setdefault('n_jobs',-1); final_model_params.setdefault('verbosity',-1)
    final_model = lgb.LGBMClassifier(**final_model_params)
    logger.info("Training final model...")
    try:
        final_model.fit(X_train_final_subset, y_train, eval_set=[(X_val_final_subset,y_val)], eval_metric='logloss', callbacks=[lgb.early_stopping(30,verbose=True)]) 
        logger.info(f"Final model trained. Best iteration: {final_model.best_iteration_}")
        if final_model.best_score_:
            for eval_name, metrics_dict in final_model.best_score_.items():
                for metric_n, score_v in metrics_dict.items(): logger.info(f"  Best score ({eval_name} - {metric_n}): {score_v:.4f}")
    except Exception as e: logger.error(f"Error training final model: {e}"); raise
    try:
        final_model.booster_.save_model(FINAL_MODEL_SAVE_PATH_NATIVE); logger.info(f"Final model (native) saved: {FINAL_MODEL_SAVE_PATH_NATIVE}")
        joblib.dump(final_model, FINAL_MODEL_SAVE_PATH_JOBLIB); logger.info(f"Final model (joblib) saved: {FINAL_MODEL_SAVE_PATH_JOBLIB}")
    except Exception as e: logger.error(f"Error saving final model: {e}")
    logger.info("--- Step 6 Complete ---")

    # ==============================================================================
    # --- Step 7: Prediction and Validation on Test Sets (with LogLoss) ---
    # ==============================================================================
    logger.info(f"\n--- Step 7: Prediction and Validation on Test Sets (with LogLoss) ---")
    # ... (Full code for Step 7 as in run_classification_pipeline_argparse_v3, 
    #      using args.load_checkpoint_s7, CHECKPOINT_DIR_S5_5, 
    #      FINAL_MODEL_LOAD_PATH_NATIVE/JOBLIB, and CLASSIFICATION_PREDICTIONS_SAVE_PATH) ...
    logger.info(f"LOAD_FROM_CHECKPOINT_S7 (from CLI) set to: {args.load_checkpoint_s7}")
    if args.load_checkpoint_s7:
        logger.info(f"Attempting to load from checkpoint: {CHECKPOINT_DIR_S5_5}") 
        try:
            expected_s7_vars_from_checkpoint = { 'X_test_dev_scaled': pd.DataFrame, 'y_test_dev': pd.Series, 'final_selected_features': list, 'scaler': StandardScaler, 'TEST_PATH': str, 'TARGET_COLUMN': str, 'EXCLUDE_COLUMNS': list, 'feature_cols': list, 'best_lgbm_params': dict, 'RANDOM_SEED': int, 'N_FEATURES_FINAL': int, 'ACCEPTABLE_ROC_AUC_THRESHOLD': float } 
            for var_name_s7 in expected_s7_vars_from_checkpoint.keys(): globals()[var_name_s7] = joblib.load(os.path.join(CHECKPOINT_DIR_S5_5, f"s5_5_{var_name_s7}.joblib")); logger.info(f"  Step 7 Checkpoint: Loaded '{var_name_s7}'")
            if os.path.exists(FINAL_MODEL_SAVE_PATH_NATIVE):
                booster = lgb.Booster(model_file=FINAL_MODEL_SAVE_PATH_NATIVE); model_params_re = best_lgbm_params.copy(); model_params_re.setdefault('objective','binary'); model_params_re.setdefault('random_state',RANDOM_SEED); model_params_re.setdefault('n_jobs',-1); model_params_re.setdefault('verbosity',-1)
                final_model = lgb.LGBMClassifier(**model_params_re); final_model._Booster=booster; final_model._fitted=True
                y_ref_for_classes = locals().get('y_train', globals().get('y_test_dev')) 
                if y_ref_for_classes is not None and hasattr(y_ref_for_classes, 'unique'): final_model.classes_=np.sort(y_ref_for_classes.unique()); final_model._n_classes=len(final_model.classes_)
                else: final_model.classes_=np.array([0,1]); final_model._n_classes=2
                logger.info(f"Loaded model (native) for Step 7 from: {FINAL_MODEL_SAVE_PATH_NATIVE}")
            elif os.path.exists(FINAL_MODEL_SAVE_PATH_JOBLIB): final_model = joblib.load(FINAL_MODEL_SAVE_PATH_JOBLIB); logger.info(f"Loaded model (joblib) for Step 7 from: {FINAL_MODEL_SAVE_PATH_JOBLIB}")
            else: raise FileNotFoundError("No saved model found for Step 7 loading from checkpoint.")
            logger.info("Variables and model loaded for Step 7 from checkpoint.")
        except Exception as e: logger.error(f"Error loading checkpoint for Step 7: {e}"); raise
    
    if 'final_model' not in locals() or final_model is None: logger.error("CRITICAL: 'final_model' is not defined or is None before Step 7 execution."); raise NameError("Missing 'final_model' for Step 7.")
    required_s7_logic = ['X_test_dev_scaled','y_test_dev','final_selected_features','scaler','TEST_PATH','TARGET_COLUMN','EXCLUDE_COLUMNS','feature_cols']
    missing_s7_vars = [var for var in required_s7_logic if var not in locals() and var not in globals()]
    if missing_s7_vars: logger.error(f"ERROR: Missing essential variables for Step 7 logic: {missing_s7_vars}"); raise NameError(f"Missing required variables for Step 7 logic: {', '.join(missing_s7_vars)}")
    else: logger.info("All required variables for Step 7 logic appear present and valid.")
    
    # --- Stage 7.1: New ROC Plot (Train vs Dev-Test) for the final_model ---
    logger.info("\n--- Stage 7.1: Generating ROC Plot (Train vs. Dev Test) for Final Model ---")
    sns.set_theme(style="whitegrid") # Apply Seaborn theme
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("deep", 2) # Get 2 distinct Seaborn colors

    # 1. Predictions on Training Data (used to fit final_model)
    if 'X_train_final_subset' in locals() and 'y_train' in locals() and not X_train_final_subset.empty:
        train_pred_proba = final_model.predict_proba(X_train_final_subset)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, train_pred_proba)
        auc_train = roc_auc_score(y_train, train_pred_proba)
        plt.plot(fpr_train, tpr_train, color=colors[0], lw=2, 
                 label=f'Train ROC (Top {N_FEATURES_FINAL} Features, AUC = {auc_train:.4f})')
        logger.info(f"  AUC for ROC on training data (final_model with {N_FEATURES_FINAL} features): {auc_train:.4f}")
    else: logger.warning("  Could not generate ROC for training data: X_train_final_subset or y_train not found/empty.")

    # 2. Predictions on Development Test Data
    if 'X_test_dev_scaled' in locals() and 'final_selected_features' in locals() and not X_test_dev_scaled.empty:
        X_test_dev_subset_for_plot = X_test_dev_scaled[final_selected_features]
        if not X_test_dev_subset_for_plot.empty:
            dev_test_pred_proba_for_plot = final_model.predict_proba(X_test_dev_subset_for_plot)[:, 1] 
            fpr_dev, tpr_dev, _ = roc_curve(y_test_dev, dev_test_pred_proba_for_plot)
            auc_dev_plot = roc_auc_score(y_test_dev, dev_test_pred_proba_for_plot)
            plt.plot(fpr_dev, tpr_dev, color=colors[1], lw=2, 
                     label=f'Dev Test ROC (Top {N_FEATURES_FINAL} Features, AUC = {auc_dev_plot:.4f})')
            logger.info(f"  AUC for ROC on dev-test data (final_model with {N_FEATURES_FINAL} features): {auc_dev_plot:.4f}")
        else: logger.warning("  X_test_dev_subset_for_plot is empty. Skipping dev test ROC.")
    else: logger.warning("  Could not generate ROC for dev-test data: X_test_dev_scaled or final_selected_features not found/empty.")

    logger.info("  (Note: 'All features' model comparison not included in this plot; would require separate model training and prediction steps)")

    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--') # Neutral color for diagonal
    plt.xlim([-0.02, 1.02]) # Added padding
    plt.ylim([-0.02, 1.02]) # Added padding
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves: Final Model (Top {N_FEATURES_FINAL} SHAP Features)', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='-', alpha=0.7) # Ensure grid is visible with whitegrid
    s7_1_roc_train_dev_path = os.path.join(OUTPUT_BASE_DIR, "s7_1_roc_curve_train_vs_dev_test.pdf") 
    plt.savefig(s7_1_roc_train_dev_path, bbox_inches='tight'); plt.clf(); plt.close('all')
    logger.info(f"Train vs. Dev Test ROC curve plot saved to {s7_1_roc_train_dev_path}")


    # --- Stage 7.2: Evaluation on Development Test Set (Metrics) ---
    logger.info(f"\n--- Stage 7.2: Evaluating Metrics on Development Test Set ---")
    # ... (rest of Stage 7.2 as before) ...
    X_test_dev_subset_eval = X_test_dev_scaled[final_selected_features]
    dev_test_pred_proba_eval = final_model.predict_proba(X_test_dev_subset_eval)[:,1]
    dev_test_pred_class_eval = (dev_test_pred_proba_eval > 0.5).astype(int)
    logger.info("Dev Test Performance Metrics:"); logger.info(f"  LogLoss: {log_loss(y_test_dev, dev_test_pred_proba_eval):.4f}"); logger.info(f"  ROC-AUC: {roc_auc_score(y_test_dev, dev_test_pred_proba_eval):.4f}"); logger.info(f"  AUPRC: {average_precision_score(y_test_dev, dev_test_pred_proba_eval):.4f}"); logger.info(f"  F1-Score: {f1_score(y_test_dev, dev_test_pred_class_eval):.4f}")
    logger.info(f"  Confusion Matrix (Dev Test):\n{confusion_matrix(y_test_dev, dev_test_pred_class_eval)}"); logger.info(f"  Classification Report (Dev Test):\n{classification_report(y_test_dev, dev_test_pred_class_eval, target_names=['Not e (0)', 'Electron (1)'])}")


    # --- Stage 7.3: Evaluation on Final Unseen Test Set from TEST_PATH ---
    logger.info(f"\n--- Stage 7.3: Evaluating on Final Unseen Test Set from {TEST_PATH} ---")
    # ... (rest of Stage 7.3 as before, saving s7_3_roc_curve_final_unseen_test.pdf) ...
    try: final_test_df_raw = pd.read_hdf(TEST_PATH)
    except Exception as e: logger.error(f"Error loading final test data from {TEST_PATH}: {e}"); raise
    X_final_test_raw = final_test_df_raw[feature_cols] 
    if scaler is None and numerical_cols_to_scale: logger.error("Scaler not available but was needed."); raise ValueError("Scaler not available/fitted.")
    X_final_test_scaled_full = X_final_test_raw.copy()
    if scaler is not None and numerical_cols_to_scale : 
        cols_s = scaler.feature_names_in_ if hasattr(scaler, 'n_features_in_') else numerical_cols_to_scale
        cols_t = [c for c in cols_s if c in X_final_test_scaled_full.columns]
        if cols_t: X_final_test_scaled_full[cols_t] = scaler.transform(X_final_test_raw[cols_t])
    X_final_test_subset = X_final_test_scaled_full[final_selected_features]
    logger.info(f"Shape of final test data subset for prediction (X_final_test_subset): {X_final_test_subset.shape}")
    final_test_pred_proba = final_model.predict_proba(X_final_test_subset)[:,1]
    if TARGET_COLUMN not in final_test_df_raw.columns:
        logger.warning(f"Target column '{TARGET_COLUMN}' not found in the final test set (blind test set).")
        logger.info("Saving predicted probabilities for the blind test set.")
        ids_for_submission = final_test_df_raw.index if hasattr(final_test_df_raw, 'index') else np.arange(len(final_test_pred_proba))
        predictions_df = pd.DataFrame({'id': ids_for_submission, TARGET_COLUMN: final_test_pred_proba})
        predictions_df.to_csv(CLASSIFICATION_PREDICTIONS_SAVE_PATH, index=False) 
        logger.info(f"Predictions for final (blind) classification test set saved to: {CLASSIFICATION_PREDICTIONS_SAVE_PATH}")
    else:
        y_final_test = final_test_df_raw[TARGET_COLUMN]; final_test_pred_class = (final_test_pred_proba > 0.5).astype(int)
        logger.info("Final Unseen Test Performance (target known):"); logger.info(f"  LogLoss: {log_loss(y_final_test, final_test_pred_proba):.4f}"); logger.info(f"  ROC-AUC: {roc_auc_score(y_final_test, final_test_pred_proba):.4f}"); logger.info(f"  AUPRC: {average_precision_score(y_final_test, final_test_pred_proba):.4f}"); logger.info(f"  F1-Score: {f1_score(y_final_test, final_test_pred_class):.4f}")
        logger.info(f"  Confusion Matrix (Final Test):\n{confusion_matrix(y_final_test, final_test_pred_class)}"); logger.info(f"  Classification Report (Final Test):\n{classification_report(y_final_test, final_test_pred_class, target_names=['Not e (0)', 'Electron (1)'])}")
        
        sns.set_theme(style="whitegrid") # Apply Seaborn theme for this plot too
        fpr_final_test,tpr_final_test,_ = roc_curve(y_final_test,final_test_pred_proba)
        plt.figure(figsize=(10,8)) # Consistent size
        colors_final = sns.color_palette("deep", 1)
        plt.plot(fpr_final_test,tpr_final_test,color=colors_final[0],lw=2,label=f'Final Test ROC (Top {N_FEATURES_FINAL} Feats, AUC={roc_auc_score(y_final_test,final_test_pred_proba):.4f})')
        plt.plot([0,1],[0,1],color='grey',lw=2,linestyle='--'); 
        plt.xlim([-0.02,1.02]); plt.ylim([-0.02,1.02]) # Padding
        plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12); plt.title('ROC Curve - Final Unseen Test Set', fontsize=14); plt.legend(loc="lower right", fontsize=10); plt.grid(True, linestyle='-', alpha=0.7)
        s7_3_roc_final_test_path = os.path.join(OUTPUT_BASE_DIR, "s7_3_roc_curve_final_unseen_test.pdf") 
        plt.savefig(s7_3_roc_final_test_path, bbox_inches='tight'); plt.clf(); plt.close('all'); logger.info(f"ROC curve for final unseen test set saved to {s7_3_roc_final_test_path}")
    logger.info("--- Step 7 Complete ---")
    logger.info("\n--- CLASSIFICATION PIPELINE EXECUTION FINISHED ---")

if __name__ == "__main__":
    cli_args = parse_arguments()
    main(cli_args)
