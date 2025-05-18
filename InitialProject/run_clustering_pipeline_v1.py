import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os
import importlib
import importlib.util
import argparse
import random
import logging

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import silhouette_score


# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Attempt to import optional packages ---
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning(
        "UMAP package not found. UMAP-based visualizations will not be available."
    )

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logging.info(
        "Seaborn package not found. Plots will use basic matplotlib histograms."
    )

TENSORFLOW_AVAILABLE = False
try:
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    TENSORFLOW_AVAILABLE = True
    logging.info(f"TensorFlow version: {tf.__version__}")
    physical_devices_gpu = tf.config.list_physical_devices("GPU")
    if physical_devices_gpu:
        logging.info(f"TensorFlow: Detected GPU(s): {physical_devices_gpu}")
        for gpu_device in physical_devices_gpu:
            tf.config.experimental.set_memory_growth(gpu_device, True)
            logging.info(f"  - GPU: {gpu_device.name}, Memory Growth set to True")
    else:
        logging.info("TensorFlow: No GPU detected, will use CPU.")
    physical_devices_cpu = tf.config.list_physical_devices("CPU")
    logging.info(f"TensorFlow: Detected CPU(s): {physical_devices_cpu}")
except ImportError:
    logging.warning(
        "TensorFlow/Keras not found. Autoencoder feature selection and anomaly detection will not be available."
    )

# --- Feature Sub-categories & Log Transformation List ---
PHOTONIC_FEATURES = ["J", "H", "K"]
ABUNDANCE_FEATURES = [
    "C_FE",
    "N_FE",
    "O_FE",
    "NA_FE",
    "MG_FE",
    "AL_FE",
    "SI_FE",
    "CA_FE",
    "S_FE",
    "V_FE",
    "CR_FE",
    "FE_H",
    "CO_FE",
    "NI_FE",
]
KINEMATIC_FEATURES = ["E", "Energy", "Lz"]
FEATURES_TO_LOG_TRANSFORM = [
    "Energy",
    "Lz",
]  # Features to apply log1p to before scaling

# --- Script-Internal Default Parameters ---
DEFAULT_FIRST_NAME = "DefaultFirstName"
DEFAULT_LAST_NAME = "DefaultLastName"
DEFAULT_BASE_SOLUTION_NAME = "Solution1"
DEFAULT_OFFICIAL_VARIABLE_LIST_2 = (
    PHOTONIC_FEATURES + ABUNDANCE_FEATURES + KINEMATIC_FEATURES
)

DEFAULT_GLOBAL_RANDOM_SEED = 42
DEFAULT_FEATURE_SELECTION_METHOD = "autoencoder_l1"
MAX_FEATURES_PROJECT_LIMIT = 7
DEFAULT_DIM_REDUCTION_FOR_VISUALIZATION = "umap"
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1

# Feature Selection AE Defaults
DEFAULT_FS_AE_ENCODER_UNITS = [16, 8]
DEFAULT_FS_AE_L1_REGULARIZER = 0.001
DEFAULT_FS_AE_ACTIVATION = "relu"
DEFAULT_FS_AE_OPTIMIZER = "adam"
DEFAULT_FS_AE_LOSS = "mse"
DEFAULT_FS_AE_EPOCHS = 50
DEFAULT_FS_AE_BATCH_SIZE = 32
DEFAULT_FS_AE_VERBOSITY = 0

# Anomaly Detection AE Defaults
DEFAULT_AD_AE_ENCODER_UNITS = [6, 4]
DEFAULT_AD_AE_ACTIVATION = "relu"
DEFAULT_AD_AE_OPTIMIZER = "adam"
DEFAULT_AD_AE_LOSS = "mse"
DEFAULT_AD_AE_EPOCHS = 30
DEFAULT_AD_AE_BATCH_SIZE = 32
DEFAULT_AD_AE_VERBOSITY = 0
DEFAULT_AD_AE_ERROR_PERCENTILE_THRESHOLD = 95

# --- Load Core Identifiers from my_config.py (if it exists) ---
FIRST_NAME = DEFAULT_FIRST_NAME
LAST_NAME = DEFAULT_LAST_NAME
BASE_SOLUTION_NAME = DEFAULT_BASE_SOLUTION_NAME
CANDIDATE_FEATURES_POOL = DEFAULT_OFFICIAL_VARIABLE_LIST_2

try:
    config_module_spec = importlib.util.find_spec("my_config")
    if config_module_spec:
        config = importlib.util.module_from_spec(config_module_spec)
        config_module_spec.loader.exec_module(config)
        logging.info("Successfully loaded my_config.py")
        FIRST_NAME = getattr(config, "FIRST_NAME", DEFAULT_FIRST_NAME)
        LAST_NAME = getattr(config, "LAST_NAME", DEFAULT_LAST_NAME)
        BASE_SOLUTION_NAME = getattr(
            config, "BASE_SOLUTION_NAME", DEFAULT_BASE_SOLUTION_NAME
        )
        CANDIDATE_FEATURES_POOL = getattr(
            config, "OFFICIAL_VARIABLE_LIST_2", DEFAULT_OFFICIAL_VARIABLE_LIST_2
        )
    else:
        logging.info(
            "my_config.py not found. Using default identifiers and candidate features."
        )
except Exception as e:
    logging.error(
        f"Error loading or processing my_config.py: {e}. Using default identifiers and candidate features."
    )
    FIRST_NAME = DEFAULT_FIRST_NAME
    LAST_NAME = DEFAULT_LAST_NAME
    BASE_SOLUTION_NAME = DEFAULT_BASE_SOLUTION_NAME
    CANDIDATE_FEATURES_POOL = DEFAULT_OFFICIAL_VARIABLE_LIST_2

# --- Assign operational parameters from script defaults ---
GLOBAL_RANDOM_SEED = DEFAULT_GLOBAL_RANDOM_SEED
FEATURE_SELECTION_METHOD = DEFAULT_FEATURE_SELECTION_METHOD
DIM_REDUCTION_FOR_VISUALIZATION = DEFAULT_DIM_REDUCTION_FOR_VISUALIZATION
UMAP_N_NEIGHBORS = DEFAULT_UMAP_N_NEIGHBORS
UMAP_MIN_DIST = DEFAULT_UMAP_MIN_DIST
FS_AE_ENCODER_UNITS = DEFAULT_FS_AE_ENCODER_UNITS
FS_AE_L1_REGULARIZER = DEFAULT_FS_AE_L1_REGULARIZER
FS_AE_ACTIVATION = DEFAULT_FS_AE_ACTIVATION
FS_AE_OPTIMIZER = DEFAULT_FS_AE_OPTIMIZER
FS_AE_LOSS = DEFAULT_FS_AE_LOSS
FS_AE_EPOCHS = DEFAULT_FS_AE_EPOCHS
FS_AE_BATCH_SIZE = DEFAULT_FS_AE_BATCH_SIZE
FS_AE_VERBOSITY = DEFAULT_FS_AE_VERBOSITY

AD_AE_ENCODER_UNITS = DEFAULT_AD_AE_ENCODER_UNITS
AD_AE_ACTIVATION = DEFAULT_AD_AE_ACTIVATION
AD_AE_OPTIMIZER = DEFAULT_AD_AE_OPTIMIZER
AD_AE_LOSS = DEFAULT_AD_AE_LOSS
AD_AE_EPOCHS = DEFAULT_AD_AE_EPOCHS
AD_AE_BATCH_SIZE = DEFAULT_AD_AE_BATCH_SIZE
AD_AE_VERBOSITY = DEFAULT_AD_AE_VERBOSITY
AD_AE_ERROR_PERCENTILE_THRESHOLD = DEFAULT_AD_AE_ERROR_PERCENTILE_THRESHOLD


# --- Apply Global Random Seed ---
logging.info(f"Using GLOBAL_RANDOM_SEED: {GLOBAL_RANDOM_SEED}")
np.random.seed(GLOBAL_RANDOM_SEED)
random.seed(GLOBAL_RANDOM_SEED)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(GLOBAL_RANDOM_SEED)

PRIMARY_CLUSTERING_ALGORITHM = "kmeans"
INPUT_CSV_FILE = "data/SDSS-Gaia_5950stars.csv"
UMAP_METRIC = "euclidean"


def _prepare_candidate_features(df_full_features, candidate_pool_list):
    """
    Helper to select, log-transform specified features, handle NaNs,
    and scale candidate features from the full dataframe.
    """
    if not candidate_pool_list:
        logging.error("Candidate feature pool is empty.")
        return None, None

    valid_candidate_pool = [
        f for f in candidate_pool_list if f in df_full_features.columns
    ]
    missing_in_df = [
        f for f in candidate_pool_list if f not in df_full_features.columns
    ]
    if missing_in_df:
        logging.warning(
            f"Features from candidate pool not found in DataFrame: {missing_in_df}. They will be excluded."
        )
    if not valid_candidate_pool:
        logging.error(
            "No features from the candidate pool are available in the DataFrame."
        )
        return None, None

    df_candidates = df_full_features[valid_candidate_pool].copy()

    # Apply log transformation to specified features before NaN handling and scaling
    for feature_to_log in FEATURES_TO_LOG_TRANSFORM:
        if feature_to_log in df_candidates.columns:
            # Ensure values are positive before log transform; log1p handles zeros.
            # If features can be negative and log is desired, a shift might be needed first.
            # For Energy/Lz which can be negative, this needs careful consideration.
            # For now, applying log1p, assuming non-negativity or that negative values are rare and handled by NaN fill later.
            # A more robust approach for features with negative values might be a different transformation or careful handling.
            if (df_candidates[feature_to_log] <= 0).any():
                logging.warning(
                    f"Feature '{feature_to_log}' contains non-positive values. Applying log1p, but review if this is appropriate or if values need shifting/clipping."
                )
            df_candidates[feature_to_log] = np.log1p(
                df_candidates[feature_to_log] - df_candidates[feature_to_log].min()
                if df_candidates[feature_to_log].min() <= 0
                else df_candidates[feature_to_log]
            )  # Shift to be >0 if min is <=0
            logging.info(f"Applied log1p transformation to feature: {feature_to_log}")

    if df_candidates.isnull().values.any():
        logging.info(
            "Handling NaNs in candidate features (post-log transform if any)..."
        )
        for col in df_candidates.columns:
            if df_candidates[col].isnull().any():
                median_val = df_candidates[col].median()
                df_candidates[col].fillna(median_val, inplace=True)
                logging.debug(f"Filled NaNs in '{col}' with median: {median_val}")

    if df_candidates.empty:
        logging.error("No data in candidate features after NaN handling.")
        return None, None

    scaler = StandardScaler()
    scaled_candidates = scaler.fit_transform(df_candidates)
    return scaled_candidates, df_candidates.columns.tolist()


def select_features_auto_variance(
    df_full_features, current_candidate_pool, num_features_to_select_iter
):
    """Selects features based on highest variance after scaling."""
    logging.info(f"--- Automatic Feature Selection (Variance Method) ---")
    logging.info(
        f"Candidate pool (total {len(current_candidate_pool)}): {current_candidate_pool}"
    )
    logging.info(
        f"Target number of features to select for this iteration: {num_features_to_select_iter}"
    )

    scaled_candidates, actual_candidate_names = _prepare_candidate_features(
        df_full_features, current_candidate_pool
    )
    if scaled_candidates is None:
        return []

    variances = np.var(scaled_candidates, axis=0)
    feature_variances = pd.Series(variances, index=actual_candidate_names).sort_values(
        ascending=False
    )

    logging.info(
        "Full list of candidate features sorted by variance (after potential log-transform and scaling):"
    )
    for feature, variance in feature_variances.items():
        logging.info(f"  - {feature}: {variance:.4f}")

    actual_num_to_select = min(num_features_to_select_iter, len(feature_variances))
    selected = feature_variances.nlargest(actual_num_to_select).index.tolist()

    logging.info(
        f"Top {len(selected)} features selected by variance for this iteration: {selected}"
    )
    if selected:
        logging.info(
            f"Variances of selected features: {feature_variances[selected].to_dict()}"
        )
    return selected


def select_features_autoencoder_l1(
    df_full_features,
    current_candidate_pool,
    num_features_to_select_iter,
    fs_ae_encoder_units_param,
    fs_ae_l1_reg_strength_iter,
    fs_ae_activation_param,
    fs_ae_optimizer_param,
    fs_ae_loss_fn_param,
    fs_ae_epochs_param,
    fs_ae_batch_size_param,
    fs_ae_verbosity_param,
    random_seed_val,
):
    """
    Selects features using an Autoencoder with L1 regularization on the first encoder layer's weights.
    """
    logging.info(f"--- Automatic Feature Selection (Autoencoder L1 Method) ---")
    if not TENSORFLOW_AVAILABLE:
        logging.error(
            "TensorFlow/Keras is not available. Cannot use Autoencoder feature selection."
        )
        return []

    logging.info(
        f"Candidate pool (total {len(current_candidate_pool)}): {current_candidate_pool}"
    )
    logging.info(
        f"Target number of features to select for this iteration: {num_features_to_select_iter}"
    )
    logging.info(
        f"FS AE Params: EncoderUnits={fs_ae_encoder_units_param}, L1Strength={fs_ae_l1_reg_strength_iter}, Activation={fs_ae_activation_param}, Epochs={fs_ae_epochs_param}, Batch={fs_ae_batch_size_param}"
    )
    logging.info(f"FS AE using random_seed_val: {random_seed_val}")

    scaled_data, actual_candidate_names = _prepare_candidate_features(
        df_full_features, current_candidate_pool
    )
    if scaled_data is None:
        return []

    n_features_input = scaled_data.shape[1]

    encoder_input_layer = keras.Input(shape=(n_features_input,), name="encoder_input")
    x = layers.Dense(
        fs_ae_encoder_units_param[0],
        activation=fs_ae_activation_param,
        kernel_regularizer=regularizers.l1(fs_ae_l1_reg_strength_iter),
        name="encoder_l1_dense",
    )(encoder_input_layer)
    for i, units in enumerate(fs_ae_encoder_units_param[1:]):
        x = layers.Dense(
            units, activation=fs_ae_activation_param, name=f"encoder_hidden_{i+1}"
        )(x)
    encoder_output = x
    encoder = keras.Model(encoder_input_layer, encoder_output, name="encoder")

    decoder_input_layer = keras.Input(
        shape=(fs_ae_encoder_units_param[-1],), name="decoder_input_bottleneck"
    )
    decoder_units_reversed = fs_ae_encoder_units_param[:-1][::-1]
    x = decoder_input_layer
    if decoder_units_reversed:
        for i, units in enumerate(decoder_units_reversed):
            x = layers.Dense(
                units, activation=fs_ae_activation_param, name=f"decoder_hidden_{i+1}"
            )(x)

    decoder_output_layer = layers.Dense(
        n_features_input, activation="linear", name="decoder_output"
    )(x)
    decoder = keras.Model(decoder_input_layer, decoder_output_layer, name="decoder")

    autoencoder = keras.Model(
        encoder_input_layer, decoder(encoder_output), name="autoencoder_fs"
    )
    autoencoder.compile(optimizer=fs_ae_optimizer_param, loss=fs_ae_loss_fn_param)

    if fs_ae_verbosity_param > 0:
        logging.info("\nFeature Selection Autoencoder Summary:")
        autoencoder.summary(print_fn=logging.info)

    logging.info("Training Feature Selection Autoencoder...")
    autoencoder.fit(
        scaled_data,
        scaled_data,
        epochs=fs_ae_epochs_param,
        batch_size=fs_ae_batch_size_param,
        shuffle=True,
        verbose=fs_ae_verbosity_param,
    )

    first_layer_weights = encoder.get_layer("encoder_l1_dense").get_weights()[0]

    feature_importance_scores = np.sum(np.abs(first_layer_weights), axis=1)
    feature_importance_series = pd.Series(
        feature_importance_scores, index=actual_candidate_names
    ).sort_values(ascending=False)

    logging.info(
        "Full list of candidate features sorted by FS Autoencoder L1 importance (after potential log-transform and scaling):"
    )
    for feature, score in feature_importance_series.items():
        logging.info(f"  - {feature}: {score:.6f}")

    actual_num_to_select = min(
        num_features_to_select_iter, len(feature_importance_series)
    )
    selected_features = feature_importance_series.nlargest(
        actual_num_to_select
    ).index.tolist()

    logging.info(
        f"Top {len(selected_features)} features selected by FS Autoencoder L1 for this iteration: {selected_features}"
    )
    if selected_features:
        logging.info(
            f"Importance scores of selected features: {feature_importance_series[selected_features].to_dict()}"
        )

    keras.backend.clear_session()
    return selected_features


def detect_anomalies_autoencoder(
    scaled_selected_data,
    random_seed_val,
    ad_ae_encoder_units_param,
    ad_ae_activation_param,
    ad_ae_optimizer_param,
    ad_ae_loss_fn_param,
    ad_ae_epochs_param,
    ad_ae_batch_size_param,
    ad_ae_verbosity_param,
    error_percentile_threshold,
):
    """
    Detects anomalies using an Autoencoder based on reconstruction error.
    Returns a boolean mask (True for inliers, False for anomalies).
    """
    logging.info(f"--- Anomaly Detection (Autoencoder Method) ---")
    if not TENSORFLOW_AVAILABLE:
        logging.error(
            "TensorFlow/Keras is not available. Cannot use Autoencoder for anomaly detection."
        )
        return np.ones(scaled_selected_data.shape[0], dtype=bool)

    n_features_input = scaled_selected_data.shape[1]
    logging.info(
        f"AD AE Params: EncoderUnits={ad_ae_encoder_units_param}, Activation={ad_ae_activation_param}, Epochs={ad_ae_epochs_param}, Threshold={error_percentile_threshold}%"
    )
    logging.info(f"AD AE using random_seed_val: {random_seed_val}")

    encoder_input_layer_ad = keras.Input(
        shape=(n_features_input,), name="ad_encoder_input"
    )
    x_ad = encoder_input_layer_ad
    for i, units in enumerate(ad_ae_encoder_units_param):
        x_ad = layers.Dense(
            units, activation=ad_ae_activation_param, name=f"ad_encoder_hidden_{i}"
        )(x_ad)
    encoder_output_ad = x_ad
    encoder_ad = keras.Model(
        encoder_input_layer_ad, encoder_output_ad, name="ad_encoder"
    )

    decoder_input_layer_ad = keras.Input(
        shape=(ad_ae_encoder_units_param[-1],), name="ad_decoder_input_bottleneck"
    )
    decoder_units_reversed_ad = ad_ae_encoder_units_param[:-1][::-1]
    x_ad = decoder_input_layer_ad
    if decoder_units_reversed_ad:
        for i, units in enumerate(decoder_units_reversed_ad):
            x_ad = layers.Dense(
                units, activation=ad_ae_activation_param, name=f"ad_decoder_hidden_{i}"
            )(x_ad)

    decoder_output_layer_ad = layers.Dense(
        n_features_input, activation="linear", name="ad_decoder_output"
    )(x_ad)
    decoder_ad = keras.Model(
        decoder_input_layer_ad, decoder_output_layer_ad, name="ad_decoder"
    )

    autoencoder_ad = keras.Model(
        encoder_input_layer_ad, decoder_ad(encoder_output_ad), name="autoencoder_ad"
    )
    autoencoder_ad.compile(optimizer=ad_ae_optimizer_param, loss=ad_ae_loss_fn_param)

    if ad_ae_verbosity_param > 0:
        logging.info("\nAnomaly Detection Autoencoder Summary:")
        autoencoder_ad.summary(print_fn=logging.info)

    logging.info("Training Anomaly Detection Autoencoder...")
    autoencoder_ad.fit(
        scaled_selected_data,
        scaled_selected_data,
        epochs=ad_ae_epochs_param,
        batch_size=ad_ae_batch_size_param,
        shuffle=True,
        verbose=ad_ae_verbosity_param,
    )

    logging.info("Calculating reconstruction errors for anomaly detection...")
    reconstructed_data = autoencoder_ad.predict(
        scaled_selected_data, verbose=ad_ae_verbosity_param
    )
    mse = np.mean(np.power(scaled_selected_data - reconstructed_data, 2), axis=1)

    threshold = np.percentile(mse, error_percentile_threshold)
    logging.info(
        f"Reconstruction error threshold for anomalies (top {100-error_percentile_threshold}%): {threshold:.4f}"
    )

    is_inlier = mse < threshold
    n_anomalies = np.sum(~is_inlier)
    logging.info(
        f"Detected {n_anomalies} anomalies ({n_anomalies / len(mse) * 100:.2f}%) using AE reconstruction error."
    )

    keras.backend.clear_session()
    return is_inlier, mse


def _create_2d_embedding_plot(
    data_2d,
    labels,
    title,
    x_label,
    y_label,
    output_path,
    point_alpha=0.7,
    point_size=40,
):
    """Helper function to create and save a 2D scatter plot with consistent outlier/anomaly coloring."""
    try:
        plt.figure(figsize=(12, 9))

        if labels is not None:
            unique_labels = np.unique(labels)

            is_anomaly_plot = (
                len(unique_labels) == 2 and 0 in unique_labels and 1 in unique_labels
            )

            if is_anomaly_plot:
                inlier_mask = labels == 0
                anomaly_mask = labels == 1

                if np.any(inlier_mask):
                    plt.scatter(
                        data_2d[inlier_mask, 0],
                        data_2d[inlier_mask, 1],
                        c="blue",
                        label="Inliers (0)",
                        alpha=point_alpha,
                        edgecolors="k",
                        s=point_size,
                        linewidths=0.5,
                    )
                if np.any(anomaly_mask):
                    plt.scatter(
                        data_2d[anomaly_mask, 0],
                        data_2d[anomaly_mask, 1],
                        c="yellow",
                        label="Anomalies (1)",
                        alpha=point_alpha,
                        edgecolors="k",
                        s=point_size,
                        linewidths=0.5,
                    )
                plt.legend(title="Groups", bbox_to_anchor=(1.03, 1), loc="upper left")

            else:
                plt.scatter(
                    data_2d[:, 0],
                    data_2d[:, 1],
                    c=labels,
                    cmap="viridis",
                    alpha=point_alpha,
                    edgecolors="k",
                    s=point_size,
                    linewidths=0.5,
                )
                handles = []
                non_outlier_unique_labels = sorted(
                    [l for l in unique_labels if l != -1]
                )
                if non_outlier_unique_labels:
                    vmin_val = min(non_outlier_unique_labels)
                    vmax_val = max(non_outlier_unique_labels)
                    if vmin_val == vmax_val:
                        norm = Normalize(vmin=vmin_val - 0.5, vmax=vmax_val + 0.5)
                    else:
                        norm = Normalize(vmin=vmin_val, vmax=vmax_val)
                    scalar_mappable = ScalarMappable(norm=norm, cmap="viridis")
                else:
                    scalar_mappable = None

                for label_val in sorted(unique_labels):
                    if scalar_mappable:
                        marker_face_color = scalar_mappable.to_rgba(label_val)
                    else:
                        marker_face_color = "blue"
                    cluster_legend_label = f"Cluster {label_val}"
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            label=cluster_legend_label,
                            markerfacecolor=marker_face_color,
                            markersize=8,
                        )
                    )
                if handles:
                    plt.legend(
                        handles=handles,
                        title="Clusters",
                        bbox_to_anchor=(1.03, 1),
                        loc="upper left",
                    )

        else:
            plt.scatter(
                data_2d[:, 0],
                data_2d[:, 1],
                cmap="viridis",
                alpha=point_alpha,
                edgecolors="k",
                s=point_size,
                linewidths=0.5,
            )

        plt.title(title, fontsize=11)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(output_path, bbox_inches="tight")
        logging.info(f"Plot saved: {output_path}")
        plt.close()
    except Exception as e:
        logging.error(f"ERROR generating 2D plot '{title}': {e}")


def plot_initial_feature_distributions(df, plot_output_dir, base_filename_prefix):
    """Plots initial distributions of features by their sub-categories."""
    logging.info("--- Generating Initial Feature Distribution Plots by Category ---")

    feature_categories = {
        "Photonic": PHOTONIC_FEATURES,
        "Abundances": ABUNDANCE_FEATURES,
        "Kinematic": KINEMATIC_FEATURES,
    }

    if base_filename_prefix.endswith("_Plot00"):
        base_filename_prefix = base_filename_prefix[: -len("_Plot00")]

    for i, (category_name, feature_list) in enumerate(feature_categories.items()):
        # Use a copy of the feature list from the global scope
        current_category_features = list(feature_list)

        # Apply log transformation for specified features within this category
        # This transformation is for visualization purposes only here.
        # The main data preprocessing for clustering happens in _prepare_candidate_features
        df_category_plot = df.copy()  # Work on a copy for plotting
        for feature_to_log in FEATURES_TO_LOG_TRANSFORM:
            if (
                feature_to_log in current_category_features
                and feature_to_log in df_category_plot.columns
            ):
                original_min = df_category_plot[feature_to_log].min()
                if (df_category_plot[feature_to_log] <= 0).any():
                    logging.info(
                        f"Initial plot for '{feature_to_log}': Shifting by {-original_min + 1e-9 if original_min <=0 else 0} before log1p due to non-positive values."
                    )
                    df_category_plot[feature_to_log] = np.log1p(
                        df_category_plot[feature_to_log] - original_min + 1e-9
                        if original_min <= 0
                        else df_category_plot[feature_to_log]
                    )
                else:
                    df_category_plot[feature_to_log] = np.log1p(
                        df_category_plot[feature_to_log]
                    )
                logging.info(
                    f"Applied log1p transformation to '{feature_to_log}' for initial plotting."
                )

        available_features = [
            f for f in current_category_features if f in df_category_plot.columns
        ]
        if not available_features:
            logging.warning(
                f"No features from category '{category_name}' found in DataFrame. Skipping this category plot."
            )
            continue

        df_category_processed = df_category_plot[available_features].copy()

        if df_category_processed.isnull().values.any():
            logging.info(
                f"Handling NaNs in '{category_name}' features for initial plotting (post-log)..."
            )
            for col in df_category_processed.columns:
                if df_category_processed[col].isnull().any():
                    df_category_processed[col].fillna(
                        df_category_processed[col].median(), inplace=True
                    )

        if df_category_processed.empty:
            logging.warning(
                f"No data for category '{category_name}' after NaN handling. Skipping plot."
            )
            continue

        plot_filename = f"{base_filename_prefix}_Plot00{chr(ord('a') + i)}.pdf"
        plot_path = os.path.join(plot_output_dir, plot_filename)
        current_fig = None

        if category_name == "Abundances":
            current_fig = plt.figure(
                figsize=(max(10, len(available_features) * 0.7), 6)
            )
            if SEABORN_AVAILABLE:
                sns.boxplot(data=df_category_processed)
            else:
                plt.boxplot(
                    [df_category_processed[col].dropna() for col in available_features],
                    labels=available_features,
                )
            plt.title(
                f"Initial Distributions: {category_name} ({len(available_features)} Features)",
                fontsize=14,
            )
            plt.ylabel("Value (Note: Energy/Lz may be log-transformed)")
            plt.xticks(rotation=45, ha="right", fontsize=8)
        else:
            num_plots = len(available_features)
            if num_plots == 0:
                continue
            cols_subplot = min(3, num_plots)
            rows_subplot = (num_plots + cols_subplot - 1) // cols_subplot
            current_fig, axes = plt.subplots(
                rows_subplot,
                cols_subplot,
                figsize=(cols_subplot * 5, rows_subplot * 4),
                squeeze=False,
            )
            axes = axes.flatten()
            last_ax_idx = -1
            for idx, feature_name in enumerate(available_features):
                ax = axes[idx]
                plot_data = df_category_processed[feature_name].dropna()
                title_suffix = (
                    " (log1p transformed)"
                    if feature_name in FEATURES_TO_LOG_TRANSFORM
                    else ""
                )
                if SEABORN_AVAILABLE:
                    sns.histplot(plot_data, kde=True, ax=ax)
                else:
                    ax.hist(plot_data, bins=30, alpha=0.7, density=True)
                ax.set_title(f"{feature_name}{title_suffix}", fontsize=10)
                ax.set_xlabel("Value", fontsize=8)
                ax.set_ylabel("Density/Frequency", fontsize=8)
                last_ax_idx = idx
            for j_ax in range(last_ax_idx + 1, len(axes)):
                current_fig.delaxes(axes[j_ax])
            current_fig.suptitle(
                f"Initial Distributions: {category_name} ({len(available_features)} Features)",
                fontsize=16,
                y=1.02 if rows_subplot > 1 else 1.05,
            )

        plt.tight_layout(rect=[0, 0, 1, 0.96 if category_name != "Abundances" else 0.9])
        plt.savefig(plot_path, bbox_inches="tight")
        logging.info(f"Initial feature distribution plot saved: {plot_path}")
        if current_fig:
            plt.close(current_fig)
        else:
            plt.close()


def run_clustering_pipeline(
    use_ae_anomaly_detection: bool,
    solution_name_suffix: str,
    initial_candidate_features_list: list,
    actual_selected_features: list,
    python_script_output_dir: str,
    current_global_random_seed: int,
    k_means_num_clusters_iter: int,
):
    """
    Main function to perform clustering, save results, and generate visualization.
    """
    logging.info(f"--- Starting Clustering Pipeline: {solution_name_suffix} ---")
    logging.info(f"Output directory for this run: {python_script_output_dir}")
    logging.info(f"Using AE Anomaly Detection: {use_ae_anomaly_detection}")
    logging.info(
        f"Primary clustering algorithm: {PRIMARY_CLUSTERING_ALGORITHM.upper()}"
    )
    logging.info(
        f"Actual features selected for clustering ({len(actual_selected_features)}): {actual_selected_features}"
    )
    logging.info(
        f"Using random seed: {current_global_random_seed} for scikit-learn components."
    )
    logging.info(
        f"K-Means number of clusters for this run: {k_means_num_clusters_iter}"
    )

    csv_output_dir = os.path.join(python_script_output_dir, "csv_files")
    plot_output_dir = os.path.join(python_script_output_dir, "plot_files")
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True)

    current_silhouette_score = None

    if not actual_selected_features:
        logging.error("No features selected for the run. Skipping pipeline.")
        return current_silhouette_score

    if not (5 <= k_means_num_clusters_iter <= 50):
        logging.error(
            f"K-Means number of clusters ({k_means_num_clusters_iter}) is outside the allowed range [5, 50]. Skipping run."
        )
        return current_silhouette_score

    filename_base_for_run = f"Clustering_{FIRST_NAME}{LAST_NAME}_{solution_name_suffix}"

    output_solution_file = os.path.join(csv_output_dir, f"{filename_base_for_run}.csv")
    output_varlist_file = os.path.join(
        csv_output_dir, f"{filename_base_for_run}_VariableList.csv"
    )

    umap_visualization_enabled = DIM_REDUCTION_FOR_VISUALIZATION.lower() == "umap"

    try:
        df_full_data = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        logging.error(f"Input file '{INPUT_CSV_FILE}' not found.")
        return current_silhouette_score
    except Exception as e:
        logging.error(f"Could not load data. {e}")
        return current_silhouette_score

    original_indices_all = df_full_data.index.copy()

    # Plot 01: Initial UMAP Landscape of Candidate Features (Optional)
    # Note: This uses features *after* potential log-transform in _prepare_candidate_features
    if (
        umap_visualization_enabled
        and len(initial_candidate_features_list) > 0
        and UMAP_AVAILABLE
    ):
        logging.info("Generating UMAP landscape plot of all candidate features...")
        scaled_all_candidates_umap, _ = _prepare_candidate_features(
            df_full_data, initial_candidate_features_list
        )
        if (
            scaled_all_candidates_umap is not None
            and scaled_all_candidates_umap.shape[0] > 1
            and scaled_all_candidates_umap.shape[1] >= 2
        ):
            try:
                reducer_initial_umap = umap.UMAP(
                    n_neighbors=UMAP_N_NEIGHBORS,
                    min_dist=UMAP_MIN_DIST,
                    n_components=2,
                    random_state=current_global_random_seed,
                )
                embedding_initial_umap = reducer_initial_umap.fit_transform(
                    scaled_all_candidates_umap
                )
                plot_path_initial_umap = os.path.join(
                    plot_output_dir, f"{filename_base_for_run}_Plot01.pdf"
                )
                _create_2d_embedding_plot(
                    embedding_initial_umap,
                    None,
                    f"Candidate Feature Landscape (UMAP)\nSolution: {solution_name_suffix} ({len(initial_candidate_features_list)} Features, scaled & log-transformed where applicable)",
                    "UMAP Component 1",
                    "UMAP Component 2",
                    plot_path_initial_umap,
                )
            except Exception as e:
                logging.error(f"Failed to generate UMAP landscape plot: {e}")
        elif not UMAP_AVAILABLE:
            logging.warning("UMAP not available, skipping UMAP landscape plot.")

    try:
        # Prepare features for the current run (this will apply log-transform and scaling)
        scaled_features_for_run, actual_selected_feature_names_for_run = (
            _prepare_candidate_features(df_full_data, actual_selected_features)
        )
        if scaled_features_for_run is None:
            logging.error("Feature preparation failed for the current run.")
            return current_silhouette_score
        # Ensure actual_selected_features reflects what was actually prepared (e.g., if some were missing from df)
        actual_selected_features = actual_selected_feature_names_for_run

    except (
        KeyError
    ) as e:  # Should be caught by _prepare_candidate_features, but as a safeguard
        logging.error(
            f"One or more selected features not found in CSV during run preparation: {e}."
        )
        return current_silhouette_score

    logging.info(
        f"Selected features for processing scaled. Shape: {scaled_features_for_run.shape}"
    )

    scaled_features_for_clustering = scaled_features_for_run
    original_indices_for_clustering = (
        original_indices_all  # Initially all, filtered if anomalies removed
    )

    n_anomalies_detected = 0
    if use_ae_anomaly_detection:
        if (
            TENSORFLOW_AVAILABLE
            and scaled_features_for_run.shape[0] > 1
            and scaled_features_for_run.shape[1] > 0
        ):
            inlier_mask, reconstruction_errors = detect_anomalies_autoencoder(
                scaled_features_for_run,
                current_global_random_seed,
                AD_AE_ENCODER_UNITS,
                AD_AE_ACTIVATION,
                AD_AE_OPTIMIZER,
                AD_AE_LOSS,
                AD_AE_EPOCHS,
                AD_AE_BATCH_SIZE,
                AD_AE_VERBOSITY,
                AD_AE_ERROR_PERCENTILE_THRESHOLD,
            )

            n_anomalies_detected = np.sum(~inlier_mask)
            # --- Plot 02: Anomaly Visualization (Conditional on AE Anomaly Detection & UMAP) ---
            if (
                umap_visualization_enabled
                and n_anomalies_detected > 0
                and UMAP_AVAILABLE
            ):
                logging.info("Generating anomaly visualization plot using UMAP...")
                try:
                    # UMAP on the data *before* anomaly removal, but labels show inliers/anomalies
                    reducer_anomaly = umap.UMAP(
                        n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST,
                        n_components=2,
                        random_state=current_global_random_seed,
                    )
                    embedding_anomaly = reducer_anomaly.fit_transform(
                        scaled_features_for_run
                    )
                    anomaly_labels_for_plot = (~inlier_mask).astype(
                        int
                    )  # 0 for inlier, 1 for anomaly

                    plot_path_anomaly = os.path.join(
                        plot_output_dir, f"{filename_base_for_run}_Plot02.pdf"
                    )
                    _create_2d_embedding_plot(
                        embedding_anomaly,
                        anomaly_labels_for_plot,
                        f"AE Anomaly Detection (UMAP of Selected Features)\n{n_anomalies_detected} Anomalies (Group 1) | Solution: {solution_name_suffix}",
                        "UMAP Component 1",
                        "UMAP Component 2",
                        plot_path_anomaly,
                    )
                except Exception as e:
                    logging.error(f"Failed to generate UMAP anomaly plot: {e}")
            elif not UMAP_AVAILABLE and n_anomalies_detected > 0:
                logging.warning("UMAP not available, skipping anomaly plot.")

            if np.sum(inlier_mask) == 0:
                logging.error(
                    "AE Anomaly Detection classified all points as anomalies. Skipping clustering for this run."
                )
                return current_silhouette_score

            logging.info(
                f"Proceeding with {np.sum(inlier_mask)} inlier points for K-Means after AE anomaly detection."
            )
            scaled_features_for_clustering = scaled_features_for_run[inlier_mask]
            # Filter original_indices_all based on inlier_mask
            # Need to ensure original_indices_all corresponds to df_full_data before any filtering for selected features
            # This assumes df_features_for_run (and thus scaled_features_for_run) used all original rows.
            original_indices_for_clustering = df_full_data.index[inlier_mask].copy()

        else:
            logging.warning(
                "Skipping AE Anomaly Detection: TensorFlow not available or insufficient data."
            )
    else:
        logging.info("Skipping AE Anomaly Detection as per configuration.")

    if scaled_features_for_clustering.shape[0] == 0:
        logging.error(
            "No data points remaining after potential anomaly detection. Cannot proceed."
        )
        return current_silhouette_score

    cluster_labels_final = None
    if PRIMARY_CLUSTERING_ALGORITHM == "kmeans":
        num_samples_for_kmeans = scaled_features_for_clustering.shape[0]
        if num_samples_for_kmeans < k_means_num_clusters_iter:
            logging.warning(
                f"Samples ({num_samples_for_kmeans}) < K-Means clusters ({k_means_num_clusters_iter}). Adjusting k to {num_samples_for_kmeans}."
            )
            current_kmeans_k_iter = max(1, num_samples_for_kmeans)
        else:
            current_kmeans_k_iter = k_means_num_clusters_iter

        logging.info(
            f"K-Means clustering with k={current_kmeans_k_iter} on {num_samples_for_kmeans} points..."
        )
        n_init_val = "auto"
        kmeans = KMeans(
            n_clusters=current_kmeans_k_iter,
            init="k-means++",
            n_init=n_init_val,
            max_iter=300,
            random_state=current_global_random_seed,
        )
        try:
            cluster_labels_final = kmeans.fit_predict(scaled_features_for_clustering)
            logging.info(f"K-Means complete. Inertia: {kmeans.inertia_:.2f}")

            if (
                len(np.unique(cluster_labels_final)) > 1
                and len(np.unique(cluster_labels_final))
                < scaled_features_for_clustering.shape[0]
            ):
                current_silhouette_score = silhouette_score(
                    scaled_features_for_clustering,
                    cluster_labels_final,
                    random_state=current_global_random_seed,
                )
                logging.info(
                    f"Silhouette Score for k={current_kmeans_k_iter}: {current_silhouette_score:.4f}"
                )
            else:
                logging.warning(
                    f"Cannot calculate Silhouette Score for k={current_kmeans_k_iter} (requires 1 < n_labels < n_samples)."
                )

        except ValueError as e:
            logging.error(f"ERROR K-Means: {e}")
            return current_silhouette_score
    else:
        logging.error(f"Algorithm '{PRIMARY_CLUSTERING_ALGORITHM}' not implemented.")
        return current_silhouette_score

    if cluster_labels_final is None:
        logging.error("Cluster labels not generated.")
        return current_silhouette_score

    solution_df = pd.DataFrame(
        {
            "event_index": original_indices_for_clustering,
            "category_number": cluster_labels_final,
        }
    )
    try:
        solution_df.to_csv(output_solution_file, index=False, header=False)
        logging.info(f"Solution saved: {output_solution_file}")
    except Exception as e:
        logging.error(f"ERROR saving solution: {e}")

    varlist_df = pd.DataFrame(actual_selected_features, columns=["variable_name"])
    try:
        varlist_df.to_csv(output_varlist_file, index=False, header=False)
        logging.info(f"Varlist saved: {output_varlist_file}")
    except Exception as e:
        logging.error(f"ERROR saving varlist: {e}")

    # --- Final Clustering Visualization ---
    plot_number_final_clustering = (
        "03"
        if use_ae_anomaly_detection
        and TENSORFLOW_AVAILABLE
        and n_anomalies_detected > 0
        and UMAP_AVAILABLE
        else "02"
    )
    final_plot_path = os.path.join(
        plot_output_dir,
        f"{filename_base_for_run}_Plot{plot_number_final_clustering}.pdf",
    )

    num_viz_features = scaled_features_for_clustering.shape[1]
    visualization_data = None
    x_label = ""
    y_label = ""
    plot_title_detail_method = "UMAP"

    if (
        umap_visualization_enabled
        and num_viz_features >= 2
        and scaled_features_for_clustering.shape[0] > 1
        and UMAP_AVAILABLE
    ):
        data_for_dim_reduction = scaled_features_for_clustering
        n_samples_dim_reduction = data_for_dim_reduction.shape[0]

        logging.info(
            f"UMAP viz for final clusters (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST})..."
        )
        try:
            effective_umap_neighbors = min(
                UMAP_N_NEIGHBORS, n_samples_dim_reduction - 1
            )
            if effective_umap_neighbors < 2:
                logging.warning(
                    f"Skipping UMAP for final plot: Samples ({n_samples_dim_reduction}) vs neighbors ({effective_umap_neighbors})."
                )
                visualization_data = None
            else:
                reducer = umap.UMAP(
                    n_neighbors=effective_umap_neighbors,
                    min_dist=UMAP_MIN_DIST,
                    n_components=2,
                    metric=UMAP_METRIC,
                    random_state=current_global_random_seed,
                )
                visualization_data = reducer.fit_transform(data_for_dim_reduction)
                x_label, y_label = "UMAP Component 1", "UMAP Component 2"
        except Exception as e:
            logging.error(f"ERROR during UMAP for final plot: {e}.")
            visualization_data = None

        if visualization_data is not None:
            plot_main_title = f"Final K-Means Clusters (UMAP of Processed Features)\nSolution: {solution_name_suffix} | {current_kmeans_k_iter} Clusters on {visualization_data.shape[0]} pts"
            _create_2d_embedding_plot(
                visualization_data,
                cluster_labels_final,
                plot_main_title,
                x_label,
                y_label,
                final_plot_path,
            )
        else:
            logging.warning(
                "Skipping final 2D UMAP cluster plot due to issues or insufficient data."
            )

    elif num_viz_features == 1 and scaled_features_for_clustering.shape[0] > 0:
        logging.info("1D cluster viz for final clusters (histograms/density plots)...")
        try:
            plt.figure(figsize=(12, 7))
            feature_name = actual_selected_features[0]
            df_plot = pd.DataFrame(
                {
                    "feature_value": scaled_features_for_clustering[:, 0],
                    "cluster": cluster_labels_final,
                }
            )
            unique_labels_sorted = sorted(np.unique(cluster_labels_final))
            palette = plt.cm.viridis(np.linspace(0, 1, len(unique_labels_sorted)))
            for idx, cl_num in enumerate(unique_labels_sorted):
                subset = df_plot[df_plot["cluster"] == cl_num]
                if not subset.empty:
                    if SEABORN_AVAILABLE:
                        sns.histplot(
                            subset["feature_value"],
                            label=f"Cluster {cl_num}",
                            kde=True,
                            stat="density",
                            common_norm=False,
                            color=palette[idx],
                            element="step",
                        )
                    else:
                        plt.hist(
                            subset["feature_value"],
                            label=f"Cluster {cl_num}",
                            density=True,
                            alpha=0.6,
                            bins=30,
                            color=palette[idx],
                            histtype="step",
                            linewidth=1.5,
                        )
            plt.title(
                f"Feature Distribution by Final Cluster - {solution_name_suffix}\nFeature: {feature_name} (Scaled)",
                fontsize=14,
            )
            plt.xlabel(f"Scaled {feature_name}", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend(title="Clusters", bbox_to_anchor=(1.03, 1), loc="upper left")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(final_plot_path, bbox_inches="tight")
            logging.info(f"1D Viz saved: {final_plot_path}")
            plt.close()
        except Exception as e:
            logging.error(f"ERROR 1D plot: {e}")
    else:
        logging.warning(
            "Skipping final visualization: Not enough features for 2D plot or no samples after processing."
        )

    logging.info(f"--- Pipeline {solution_name_suffix} Finished ---")
    return current_silhouette_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run unsupervised clustering pipeline."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory for all Python script outputs (CSVs, PDFs).",
    )
    args = parser.parse_args()

    # Create main output directory and subdirectories for CSVs and Plots
    # This needs to be done *before* calling plot_initial_feature_distributions
    csv_output_dir_main = os.path.join(args.output_dir, "csv_files")
    plot_output_dir_main = os.path.join(args.output_dir, "plot_files")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(csv_output_dir_main, exist_ok=True)
        os.makedirs(plot_output_dir_main, exist_ok=True)
        logging.info(
            f"Ensured output directories exist: {args.output_dir}, {csv_output_dir_main}, {plot_output_dir_main}"
        )
    except OSError as e:
        logging.error(f"ERROR: Could not create output directories: {e}")
        exit(1)

    input_data_dir = os.path.dirname(INPUT_CSV_FILE)
    if input_data_dir and not os.path.exists(input_data_dir):
        try:
            os.makedirs(input_data_dir)
            logging.info(f"Created input data directory: '{input_data_dir}'")
        except OSError as e:
            logging.error(
                f"Error creating input data directory '{input_data_dir}': {e}."
            )

    if not os.path.exists(INPUT_CSV_FILE):
        logging.critical(f"Input data file '{INPUT_CSV_FILE}' not found.")
        exit(1)
    elif FIRST_NAME == DEFAULT_FIRST_NAME or LAST_NAME == DEFAULT_LAST_NAME:
        logging.warning(
            f"Using default names: {FIRST_NAME} {LAST_NAME}. Update my_config.py."
        )

    # --- Define Iteration Parameters for Experiments ---
    NUM_FEATURES_OPTIONS = [
        2,
        3,
    ]  # Focus on 3-5 features, also check project max if different
    if (
        MAX_FEATURES_PROJECT_LIMIT not in NUM_FEATURES_OPTIONS
        and MAX_FEATURES_PROJECT_LIMIT > max(NUM_FEATURES_OPTIONS, default=0)
    ):
        NUM_FEATURES_OPTIONS.insert(0, MAX_FEATURES_PROJECT_LIMIT)
    NUM_FEATURES_OPTIONS = sorted(list(set(NUM_FEATURES_OPTIONS)), reverse=True)

    K_MEANS_OPTIONS = [5, 7, 10]
    AE_L1_OPTIONS = (
        [0.0001, 0.001]
        if FEATURE_SELECTION_METHOD.lower() == "autoencoder_l1"
        else [FS_AE_L1_REGULARIZER]
    )

    all_run_metrics = []
    total_pipeline_runs = 0
    num_fs_ae_l1_options = (
        len(AE_L1_OPTIONS)
        if FEATURE_SELECTION_METHOD.lower() == "autoencoder_l1"
        else 1
    )
    total_pipeline_runs = (
        len(NUM_FEATURES_OPTIONS) * num_fs_ae_l1_options * len(K_MEANS_OPTIONS) * 2
    )
    if not TENSORFLOW_AVAILABLE:  # If TF not available, AE_AD branch is skipped
        total_pipeline_runs = (
            len(NUM_FEATURES_OPTIONS) * num_fs_ae_l1_options * len(K_MEANS_OPTIONS) * 1
        )

    logging.info(f"Starting experimental runs...")
    logging.info(f"  Iterating over Num Features: {NUM_FEATURES_OPTIONS}")
    logging.info(f"  Iterating over K-Means K: {K_MEANS_OPTIONS}")
    if FEATURE_SELECTION_METHOD.lower() == "autoencoder_l1":
        logging.info(f"  Iterating over FS AE L1 Regularizer: {AE_L1_OPTIONS}")

    try:
        df_for_feature_selection_and_initial_plots = pd.read_csv(INPUT_CSV_FILE)
        if not df_for_feature_selection_and_initial_plots.empty:
            initial_plots_base_filename = (
                f"Clustering_{FIRST_NAME}{LAST_NAME}_InitialDataExploration"
            )
            plot_initial_feature_distributions(
                df_for_feature_selection_and_initial_plots,
                plot_output_dir_main,
                initial_plots_base_filename,
            )
            logging.info("\n--- Descriptive Statistics for All Candidate Features ---")
            valid_candidates = [
                col
                for col in CANDIDATE_FEATURES_POOL
                if col in df_for_feature_selection_and_initial_plots.columns
            ]
            if valid_candidates:
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    1000,
                ):
                    desc_stats_str = (
                        df_for_feature_selection_and_initial_plots[valid_candidates]
                        .describe()
                        .to_string()
                    )
                logging.info(f"\n{desc_stats_str}")
            else:
                logging.warning(
                    "No valid candidate features found in the DataFrame for describe()."
                )
        else:
            logging.warning(
                "DataFrame for initial plots is empty. Skipping initial plots and describe()."
            )

    except Exception as e:
        logging.critical(
            f"Failed to load data for initial plots/describe from {INPUT_CSV_FILE}: {e}"
        )
        exit(1)

    current_pipeline_run_count = 0
    for num_feat_to_select_iter in NUM_FEATURES_OPTIONS:
        for fs_ae_l1_iter_strength in AE_L1_OPTIONS:

            if (
                FEATURE_SELECTION_METHOD.lower() != "autoencoder_l1"
                and fs_ae_l1_iter_strength != AE_L1_OPTIONS[0]
            ):
                continue

            logging.info(
                f"\n{'='*20} Starting runs for MAX_FEATURES_TO_SELECT = {num_feat_to_select_iter} {'='*20}"
            )
            if FEATURE_SELECTION_METHOD.lower() == "autoencoder_l1":
                logging.info(
                    f"{'='*20} Using FS AE L1 Strength = {fs_ae_l1_iter_strength} {'='*20}"
                )

            current_max_features_to_select = min(
                num_feat_to_select_iter, MAX_FEATURES_PROJECT_LIMIT
            )
            if current_max_features_to_select <= 0:
                logging.warning(
                    f"Skipping iteration with num_feat_to_select_iter = {num_feat_to_select_iter} as it's not positive."
                )
                continue

            fs_method_short = ""
            current_fs_method_for_iter = FEATURE_SELECTION_METHOD.lower()

            if current_fs_method_for_iter == "auto_variance":
                fs_method_short = "Var"
            elif current_fs_method_for_iter == "autoencoder_l1":
                fs_method_short = "AE"
            else:
                logging.warning(
                    f"Unknown FEATURE_SELECTION_METHOD '{FEATURE_SELECTION_METHOD}'. Defaulting to 'autoencoder_l1'."
                )
                current_fs_method_for_iter = "autoencoder_l1"
                fs_method_short = "AE"

            features_for_run_iter = []
            df_fs_data_source = df_for_feature_selection_and_initial_plots

            if current_fs_method_for_iter == "auto_variance":
                features_for_run_iter = select_features_auto_variance(
                    df_fs_data_source,
                    CANDIDATE_FEATURES_POOL,
                    current_max_features_to_select,
                )
            elif current_fs_method_for_iter == "autoencoder_l1":
                if not TENSORFLOW_AVAILABLE:
                    logging.error(
                        "TensorFlow not available for 'autoencoder_l1'. Falling back to 'auto_variance' for this feature selection iteration."
                    )
                    fs_method_short = "Var"
                    features_for_run_iter = select_features_auto_variance(
                        df_fs_data_source,
                        CANDIDATE_FEATURES_POOL,
                        current_max_features_to_select,
                    )
                else:
                    features_for_run_iter = select_features_autoencoder_l1(
                        df_fs_data_source,
                        CANDIDATE_FEATURES_POOL,
                        current_max_features_to_select,
                        FS_AE_ENCODER_UNITS,
                        fs_ae_l1_iter_strength,
                        FS_AE_ACTIVATION,
                        FS_AE_OPTIMIZER,
                        FS_AE_LOSS,
                        FS_AE_EPOCHS,
                        FS_AE_BATCH_SIZE,
                        FS_AE_VERBOSITY,
                        GLOBAL_RANDOM_SEED,
                    )

            if not features_for_run_iter:
                logging.error(
                    f"No features selected for MAX_FEATURES_TO_SELECT = {current_max_features_to_select} using {current_fs_method_for_iter}. Skipping K-Means iterations for this feature set."
                )
                continue

            for k_clusters_iter in K_MEANS_OPTIONS:
                l1_suffix = (
                    f"_L1-{fs_ae_l1_iter_strength:.0e}"
                    if current_fs_method_for_iter == "autoencoder_l1"
                    else ""
                )
                current_solution_name_base_iter = f"{BASE_SOLUTION_NAME}_{fs_method_short}{l1_suffix}_N{len(features_for_run_iter)}_k{k_clusters_iter}"

                # Run 1: Without AE Anomaly Detection
                current_pipeline_run_count += 1
                logging.info(
                    f"\n>>> Starting Main Pipeline Run {current_pipeline_run_count} / {total_pipeline_runs} <<<"
                )
                solution_name_no_ad = f"{current_solution_name_base_iter}_NoAD"
                silhouette_no_ad = run_clustering_pipeline(
                    use_ae_anomaly_detection=False,
                    solution_name_suffix=solution_name_no_ad,
                    initial_candidate_features_list=CANDIDATE_FEATURES_POOL,
                    actual_selected_features=features_for_run_iter,
                    python_script_output_dir=args.output_dir,
                    current_global_random_seed=GLOBAL_RANDOM_SEED,
                    k_means_num_clusters_iter=k_clusters_iter,
                )
                all_run_metrics.append(
                    {
                        "solution_name": solution_name_no_ad,
                        "num_features": len(features_for_run_iter),
                        "k_clusters": k_clusters_iter,
                        "fs_ae_l1_strength": (
                            fs_ae_l1_iter_strength
                            if current_fs_method_for_iter == "autoencoder_l1"
                            else "N/A"
                        ),
                        "feature_selection_method": fs_method_short,
                        "anomaly_detection": "NoAD",
                        "silhouette_score": (
                            silhouette_no_ad if silhouette_no_ad is not None else np.nan
                        ),
                    }
                )
                logging.info("-" * 70)

                # Run 2: With AE Anomaly Detection
                if TENSORFLOW_AVAILABLE:
                    current_pipeline_run_count += 1
                    logging.info(
                        f"\n>>> Starting Main Pipeline Run {current_pipeline_run_count} / {total_pipeline_runs} <<<"
                    )
                    solution_name_with_ad = f"{current_solution_name_base_iter}_AeAD"
                    silhouette_with_ad = run_clustering_pipeline(
                        use_ae_anomaly_detection=True,
                        solution_name_suffix=solution_name_with_ad,
                        initial_candidate_features_list=CANDIDATE_FEATURES_POOL,
                        actual_selected_features=features_for_run_iter,
                        python_script_output_dir=args.output_dir,
                        current_global_random_seed=GLOBAL_RANDOM_SEED,
                        k_means_num_clusters_iter=k_clusters_iter,
                    )
                    all_run_metrics.append(
                        {
                            "solution_name": solution_name_with_ad,
                            "num_features": len(features_for_run_iter),
                            "k_clusters": k_clusters_iter,
                            "fs_ae_l1_strength": (
                                fs_ae_l1_iter_strength
                                if current_fs_method_for_iter == "autoencoder_l1"
                                else "N/A"
                            ),
                            "feature_selection_method": fs_method_short,
                            "anomaly_detection": "AeAD",
                            "silhouette_score": (
                                silhouette_with_ad
                                if silhouette_with_ad is not None
                                else np.nan
                            ),
                        }
                    )
                else:
                    logging.warning(
                        f"Skipping AE Anomaly Detection run for {current_solution_name_base_iter} as TensorFlow is not available."
                    )

                logging.info("=" * 70)

    logging.info("\n--- All Experimental Pipelines Finished ---")

    if all_run_metrics:
        results_df = pd.DataFrame(all_run_metrics)
        results_df.sort_values(by="silhouette_score", ascending=False, inplace=True)

        logging.info(
            "\n--- Summary of All Runs (Sorted by Silhouette Score Descending) ---"
        )
        summary_cols = [
            "solution_name",
            "silhouette_score",
            "num_features",
            "k_clusters",
            "anomaly_detection",
        ]
        if FEATURE_SELECTION_METHOD.lower() == "autoencoder_l1":
            summary_cols.insert(4, "fs_ae_l1_strength")

        display_df = results_df[
            [col for col in summary_cols if col in results_df.columns]
        ].copy()

        try:
            pd.options.display.float_format = "{:.4f}".format
            results_summary_str = display_df.to_string(index=False)
        except Exception:
            results_summary_str = display_df.to_string(index=False)

        logging.info(f"\n{results_summary_str}")

        if (
            not results_df.empty
            and "silhouette_score" in results_df.columns
            and results_df["silhouette_score"].notna().any()
        ):
            best_run = results_df.iloc[0]
            logging.info(
                "\n--- Best Performing Run (based on highest Silhouette Score) ---"
            )
            logging.info(f"Solution Name: {best_run['solution_name']}")
            logging.info(f"Silhouette Score: {best_run['silhouette_score']:.4f}")
            logging.info(f"Number of Features: {best_run['num_features']}")
            logging.info(f"Number of K-Means Clusters: {best_run['k_clusters']}")
            if (
                "fs_ae_l1_strength" in best_run
                and best_run["fs_ae_l1_strength"] != "N/A"
            ):
                logging.info(f"FS AE L1 Strength: {best_run['fs_ae_l1_strength']:.0e}")
            logging.info(
                f"Feature Selection Method: {best_run['feature_selection_method']}"
            )
            logging.info(f"Anomaly Detection: {best_run['anomaly_detection']}")
            logging.info(
                f"Corresponding CSV solution file: Clustering_{FIRST_NAME}{LAST_NAME}_{best_run['solution_name']}.csv"
            )
        else:
            logging.warning(
                "Could not determine the best run as no valid Silhouette scores were recorded."
            )
    else:
        logging.info("No experimental runs were completed to summarize.")
