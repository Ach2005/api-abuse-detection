import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/api_behavioral_features.csv"

OUTPUT_DIR = "results"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "model_validation_results.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "session_request_count",
    "session_duration_seconds",
    "requests_per_minute",
    "failed_auth_ratio",
    "error_ratio",
    "unique_endpoint_count",
    "endpoint_diversity",
    "mean_request_interval",
    "request_interval_std",
    "night_activity_ratio",
    "total_geo_distance_km",
    "max_geo_distance_km",
    "mean_geo_distance_km",
    "burst_score"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading behavioral dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(df):,} sessions"
    )

    X = df[FEATURES]

    y = df["label"]

    return X, y


# ============================================================
# SUPERVISED MODEL VALIDATION
# ============================================================

def validate_supervised_models(X, y):

    print("\nRunning 5-fold cross-validation...")

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    for name, model in models.items():

        print(
            f"\nValidating {name}..."
        )

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({

            "model": name,

            "precision_mean":
                scores[
                    "test_precision"
                ].mean(),

            "precision_std":
                scores[
                    "test_precision"
                ].std(),

            "recall_mean":
                scores[
                    "test_recall"
                ].mean(),

            "recall_std":
                scores[
                    "test_recall"
                ].std(),

            "f1_mean":
                scores[
                    "test_f1"
                ].mean(),

            "f1_std":
                scores[
                    "test_f1"
                ].std(),

            "roc_auc_mean":
                scores[
                    "test_roc_auc"
                ].mean(),

            "roc_auc_std":
                scores[
                    "test_roc_auc"
                ].std()
        })

    return pd.DataFrame(results)


# ============================================================
# FALSE POSITIVE RATE VALIDATION
# ============================================================

def calculate_fpr_cross_validation(X, y):

    print(
        "\nCalculating false-positive rates..."
    )

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fpr_results = []

    for name, model in models.items():

        fold_fprs = []

        for train_idx, test_idx in cv.split(
            X,
            y
        ):

            X_train = X.iloc[
                train_idx
            ]

            X_test = X.iloc[
                test_idx
            ]

            y_train = y.iloc[
                train_idx
            ]

            y_test = y.iloc[
                test_idx
            ]

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            cm = confusion_matrix(
                y_test,
                predictions
            )

            tn, fp, fn, tp = cm.ravel()

            fpr = (
                fp / (fp + tn)
                if (fp + tn) > 0
                else 0
            )

            fold_fprs.append(
                fpr
            )

        fpr_results.append({

            "model": name,

            "false_positive_rate_mean":
                np.mean(fold_fprs),

            "false_positive_rate_std":
                np.std(fold_fprs)
        })

    return pd.DataFrame(
        fpr_results
    )


# ============================================================
# ISOLATION FOREST VALIDATION
# ============================================================

def validate_isolation_forest(X, y):

    print(
        "\nValidating Isolation Forest..."
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    precisions = []
    recalls = []
    f1_scores = []
    fprs = []

    for train_idx, test_idx in cv.split(
        X,
        y
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=200,
                    contamination="auto",
                    random_state=42
                )
            )
        ])

        model.fit(
            X_train
        )

        predictions = model.predict(
            X_test
        )

        predictions = (
            predictions == -1
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        tn, fp, fn, tp = cm.ravel()

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1_scores.append(
            f1
        )

        fprs.append(
            fpr
        )

    return pd.DataFrame([
        {

            "model": "Isolation Forest",

            "precision_mean":
                np.mean(precisions),

            "precision_std":
                np.std(precisions),

            "recall_mean":
                np.mean(recalls),

            "recall_std":
                np.std(recalls),

            "f1_mean":
                np.mean(f1_scores),

            "f1_std":
                np.std(f1_scores),

            "roc_auc_mean": np.nan,

            "roc_auc_std": np.nan,

            "false_positive_rate_mean":
                np.mean(fprs),

            "false_positive_rate_std":
                np.std(fprs)
        }
    ])


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(results):

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL VALIDATION RESULTS"
    )

    print(
        "=" * 60
    )

    display = results.copy()

    percentage_columns = [
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "false_positive_rate_mean"
    ]

    for column in percentage_columns:

        if column in display.columns:

            display[column] = (
                display[column] * 100
            ).round(2)

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "AI-Based API Abuse Detection"
    )

    print(
        "Model Validation & Comparison"
    )

    print("=" * 60)

    # Load
    X, y = load_data()

    # Supervised validation
    supervised_results = (
        validate_supervised_models(
            X,
            y
        )
    )

    # FPR
    fpr_results = (
        calculate_fpr_cross_validation(
            X,
            y
        )
    )

    # Isolation Forest
    isolation_results = (
        validate_isolation_forest(
            X,
            y
        )
    )

    # Merge results
    results = (
        supervised_results
        .merge(
            fpr_results,
            on="model",
            how="left"
        )
    )

    # Add Isolation Forest
    results = pd.concat(
        [
            results,
            isolation_results
        ],
        ignore_index=True
    )

    # Display
    display_results(
        results
    )

    # Save
    save_results(
        results
    )

    print(
        "\nModel validation completed."
    )


if __name__ == "__main__":

    main()