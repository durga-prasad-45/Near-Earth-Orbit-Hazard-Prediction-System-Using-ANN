"""
train_model.py
----------------
Reproduces the pipeline from ANN_Project.ipynb and saves the artifacts
(model, scaler, threshold, feature list) that app.py needs.

Run this ONCE, locally, in a folder that contains `sprint1.csv`
(the same dataset used in the notebook):

    pip install -r requirements.txt
    python train_model.py

It will create an `artifacts/` folder containing:
    - ann_model.keras     (trained Keras model)
    - scaler.pkl           (fitted StandardScaler)
    - config.json          (best threshold + feature order + test metrics)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

tf.random.set_seed(42)
np.random.seed(42)

DATA_PATH = "sprint1.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as the notebook."""
    df = df.copy()
    df["Avg_diameter"] = (df["est_diameter_min"] + df["est_diameter_max"]) / 2
    df["Diameter_range"] = df["est_diameter_max"] - df["est_diameter_min"]
    df["Velocity_distance_ratio"] = df["relative_velocity"] / df["miss_distance"]
    df["log_velocity"] = np.log1p(df["relative_velocity"])
    df["log_miss_distance"] = np.log1p(df["miss_distance"])
    return df


def build_ann(input_dim, unit_1=64, unit_2=32, unit_3=16,
              dropout_rate=0.2, learning_rate=0.001, l2_reg=0.001):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(unit_1, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg), name="dense_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(dropout_rate, name="dropout_1"),

        layers.Dense(unit_2, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg), name="dense_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(dropout_rate, name="dropout_2"),

        layers.Dense(unit_3, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg), name="dense_3"),
        layers.Dropout(dropout_rate / 2, name="dropout_3"),

        layers.Dense(1, activation="sigmoid", name="output"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = engineer_features(df)

    X = df.drop("hazardous", axis=1)
    y = df["hazardous"]
    feature_order = list(X.columns)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42,
        stratify=y_train_full
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: cw[0], 1: cw[1]}

    # Best hyperparameters found in the notebook's manual grid search
    best_config = {"units_1": 64, "units_2": 32, "units_3": 16,
                    "dropout_rate": 0.2, "lr": 0.001, "l2": 0.001}

    print("Training final ANN with best config:", best_config)
    model = build_ann(
        input_dim=X_train_s.shape[1],
        unit_1=best_config["units_1"],
        unit_2=best_config["units_2"],
        unit_3=best_config["units_3"],
        dropout_rate=best_config["dropout_rate"],
        learning_rate=best_config["lr"],
        l2_reg=best_config["l2"],
    )

    cb_list = [
        callbacks.EarlyStopping(monitor="val_auc", patience=10,
                                 restore_best_weights=True, mode="max", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=5, min_lr=1e-6, verbose=1),
    ]

    model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=100,
        batch_size=512,
        class_weight=class_weight_dict,
        callbacks=cb_list,
        verbose=1,
    )

    # Threshold tuning on validation set (maximize F1)
    val_probs = model.predict(X_val_s, verbose=0).flatten()
    thresholds = np.arange(0.10, 0.90, 0.01)
    f1_scores = [f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
                 for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1_scores))])

    # Final test evaluation
    y_prob_test = model.predict(X_test_s, verbose=0).flatten()
    y_pred_test = (y_prob_test >= best_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "precision": float(precision_score(y_test, y_pred_test)),
        "recall": float(recall_score(y_test, y_pred_test)),
        "f1": float(f1_score(y_test, y_pred_test)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_test)),
    }
    print("Test metrics:", metrics)

    # Save artifacts
    model.save(os.path.join(ARTIFACT_DIR, "ann_model.keras"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

    config = {
        "feature_order": feature_order,
        "best_threshold": best_threshold,
        "test_metrics": metrics,
        "best_config": best_config,
    }
    with open(os.path.join(ARTIFACT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved artifacts to ./{ARTIFACT_DIR}/")
    print("You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()