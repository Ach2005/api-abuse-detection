import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    "ml_model_results.csv"
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

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\nPreparing ML data...")

    X = df[FEATURES]

    y = df["label"]

    # Same test split will be used for supervised models
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Testing samples: {len(X_test):,}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# EVALUATE SUPERVISED MODEL
# ============================================================

def evaluate_supervised_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):

    print(
        f"\nTraining {model_name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    # Probability for ROC-AUC
    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

    else:

        probabilities = None

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

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    if probabilities is not None:

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

    else:

        roc_auc = 0

    print(
        f"{model_name} completed."
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1-score: {f1:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate":
            false_positive_rate,
        "roc_auc": roc_auc
    }


# ============================================================
# ISOLATION FOREST
# ============================================================

def evaluate_isolation_forest(
    X_train,
    X_test,
    y_test
):

    print(
        "\nTraining Isolation Forest..."
    )

    model = Pipeline(
        [
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
        ]
    )

    model.fit(
        X_train
    )

    # Isolation Forest:
    # 1  = normal
    # -1 = anomaly

    predictions = model.predict(
        X_test
    )

    # Convert to our labels:
    # 0 = normal
    # 1 = abuse

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

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    print(
        "Isolation Forest completed."
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1-score: {f1:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

    print(
        "ROC-AUC: Not used for this initial "
        "unsupervised evaluation."
    )

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )

    return {
        "model": "Isolation Forest",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate":
            false_positive_rate,
        "roc_auc": None
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "AI-Based API Abuse Detection"
    )

    print(
        "Machine Learning Model Training"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_data(df)

    results = []

    # ========================================================
    # 1. LOGISTIC REGRESSION
    # ========================================================

    logistic_model = Pipeline(
        [
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
        ]
    )

    results.append(
        evaluate_supervised_model(
            logistic_model,
            "Logistic Regression",
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ========================================================
    # 2. DECISION TREE
    # ========================================================

    decision_tree = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    )

    results.append(
        evaluate_supervised_model(
            decision_tree,
            "Decision Tree",
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ========================================================
    # 3. RANDOM FOREST
    # ========================================================

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    results.append(
        evaluate_supervised_model(
            random_forest,
            "Random Forest",
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # ========================================================
    # 4. ISOLATION FOREST
    # ========================================================

    results.append(
        evaluate_isolation_forest(
            X_train,
            X_test,
            y_test
        )
    )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)

    print(
        "MACHINE LEARNING RESULTS"
    )

    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        "\nML model training completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()