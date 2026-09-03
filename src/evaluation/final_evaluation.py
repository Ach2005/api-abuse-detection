import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed/api_behavioral_features.csv"

RESULTS_DIR = "results"

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "final_system_evaluation.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    RESULTS_DIR,
    "random_forest_confusion_matrix.png"
)

MODEL_COMPARISON_FILE = os.path.join(
    RESULTS_DIR,
    "model_comparison.png"
)

SCENARIO_SCORE_FILE = os.path.join(
    RESULTS_DIR,
    "scenario_abuse_scores.png"
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

    print("=" * 60)

    print(
        "AI-Based API Abuse Detection"
    )

    print(
        "Final System Evaluation"
    )

    print("=" * 60)

    print(
        "\nLoading behavioral dataset..."
    )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"Loaded {len(df):,} sessions"
    )

    return df


# ============================================================
# RANDOM FOREST CROSS-VALIDATION
# ============================================================

def evaluate_random_forest(df):

    print(
        "\nRunning final Random Forest evaluation..."
    )

    X = df[FEATURES]

    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    # Five-fold cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Out-of-fold predictions
    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=-1
    )

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL RANDOM FOREST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy:           {accuracy * 100:.2f}%"
    )

    print(
        f"Precision:          {precision * 100:.2f}%"
    )

    print(
        f"Recall:             {recall * 100:.2f}%"
    )

    print(
        f"F1-score:           {f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC:            {roc_auc * 100:.2f}%"
    )

    print(
        f"False Positive Rate:{false_positive_rate * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "false_positive_rate":
            false_positive_rate,
        "confusion_matrix": cm
    }


# ============================================================
# MITIGATION POLICY EVALUATION
# ============================================================

def evaluate_mitigation(df):

    print(
        "\nEvaluating automated mitigation policy..."
    )

    actions = []

    for _, row in df.iterrows():

        rpm = row[
            "requests_per_minute"
        ]

        # Calculate a simple behavioral score
        score = 0

        if rpm >= 60:
            score += 25

        elif rpm >= 30:
            score += 15

        if row[
            "failed_auth_ratio"
        ] >= 0.50:

            score += 20

        elif row[
            "failed_auth_ratio"
        ] >= 0.20:

            score += 10

        if row[
            "error_ratio"
        ] >= 0.50:

            score += 15

        elif row[
            "error_ratio"
        ] >= 0.20:

            score += 8

        if row[
            "unique_endpoint_count"
        ] >= 12:

            score += 10

        if row[
            "mean_request_interval"
        ] <= 2:

            score += 15

        elif row[
            "mean_request_interval"
        ] <= 5:

            score += 8

        if row[
            "max_geo_distance_km"
        ] >= 3000:

            score += 10

        if row[
            "burst_score"
        ] >= 0.70:

            score += 15

        score = min(
            score,
            100
        )

        # Final mitigation decision
        if (
            rpm >= 60
            or (
                rpm >= 30
                and score >= 70
            )
        ):

            action = "BLOCK"

        elif (
            rpm >= 30
            or score >= 40
        ):

            action = "RATE LIMIT"

        else:

            action = "ALLOW"

        actions.append(
            action
        )

    df = df.copy()

    df["mitigation_action"] = actions

    print(
        "\nMitigation actions:"
    )

    print(
        df[
            "mitigation_action"
        ].value_counts()
    )

    return df


# ============================================================
# MODEL COMPARISON GRAPH
# ============================================================

def create_model_comparison_graph():

    file = os.path.join(
        RESULTS_DIR,
        "model_validation_results.csv"
    )

    if not os.path.exists(file):

        print(
            "\nModel validation file not found."
        )

        return

    df = pd.read_csv(
        file
    )

    models = df["model"]

    f1 = (
        df["f1_mean"] * 100
    )

    precision = (
        df["precision_mean"] * 100
    )

    recall = (
        df["recall_mean"] * 100
    )

    x = np.arange(
        len(models)
    )

    width = 0.25

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width,
        precision,
        width,
        label="Precision"
    )

    plt.bar(
        x,
        recall,
        width,
        label="Recall"
    )

    plt.bar(
        x + width,
        f1,
        width,
        label="F1-score"
    )

    plt.xticks(
        x,
        models,
        rotation=20
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.title(
        "Machine Learning Model Comparison"
    )

    plt.ylim(
        0,
        105
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        MODEL_COMPARISON_FILE,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved: {MODEL_COMPARISON_FILE}"
    )


# ============================================================
# SCENARIO SCORE GRAPH
# ============================================================

def create_scenario_graph(df):

    scenario_scores = {}

    for scenario, group in df.groupby(
        "scenario"
    ):

        score = 0

        score += min(
            group[
                "requests_per_minute"
            ].mean() / 60 * 25,
            25
        )

        score += min(
            group[
                "failed_auth_ratio"
            ].mean() * 20,
            20
        )

        score += min(
            group[
                "error_ratio"
            ].mean() * 15,
            15
        )

        scenario_scores[
            scenario
        ] = score

    scores = pd.Series(
        scenario_scores
    ).sort_values(
        ascending=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        scores.index,
        scores.values
    )

    plt.xlabel(
        "Behavioral Risk Score"
    )

    plt.title(
        "Behavioral Risk by Abuse Scenario"
    )

    plt.tight_layout()

    plt.savefig(
        SCENARIO_SCORE_FILE,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {SCENARIO_SCORE_FILE}"
    )


# ============================================================
# CONFUSION MATRIX GRAPH
# ============================================================

def create_confusion_matrix_graph(cm):

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm
    )

    plt.title(
        "Random Forest Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        [0, 1],
        ["Normal", "Abuse"]
    )

    plt.yticks(
        [0, 1],
        ["Normal", "Abuse"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {CONFUSION_MATRIX_FILE}"
    )


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

def save_summary(
    rf_results,
    mitigation_df
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    summary = pd.DataFrame([

        {
            "system": "Random Forest",

            "accuracy":
                rf_results["accuracy"],

            "precision":
                rf_results["precision"],

            "recall":
                rf_results["recall"],

            "f1_score":
                rf_results["f1_score"],

            "roc_auc":
                rf_results["roc_auc"],

            "false_positive_rate":
                rf_results[
                    "false_positive_rate"
                ]
        }

    ])

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        f"\nFinal summary saved to: "
        f"{SUMMARY_FILE}"
    )

    print(
        "\nFinal mitigation distribution:"
    )

    print(
        mitigation_df[
            "mitigation_action"
        ].value_counts()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    # Final ML evaluation
    rf_results = evaluate_random_forest(
        df
    )

    # Automated mitigation evaluation
    mitigation_df = evaluate_mitigation(
        df
    )

    # Graphs
    create_confusion_matrix_graph(
        rf_results["confusion_matrix"]
    )

    create_model_comparison_graph()

    create_scenario_graph(
        df
    )

    # Save final results
    save_summary(
        rf_results,
        mitigation_df
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL SYSTEM EVALUATION COMPLETED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()