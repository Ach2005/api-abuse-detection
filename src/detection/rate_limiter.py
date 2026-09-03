import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/raw/api_traffic_raw.csv"

OUTPUT_DIR = "results"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "rate_limit_baseline_results.csv"
)

# Fixed-window rate-limit thresholds
THRESHOLDS = [
    30,
    60,
    100,
    150
]


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_data():

    print(
        "Loading raw API traffic..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(
        f"Loaded {len(df):,} API requests"
    )

    return df


# ============================================================
# APPLY FIXED-WINDOW RATE LIMIT
# ============================================================

def apply_rate_limit(
    df,
    threshold
):

    """
    Traditional fixed-window rate limiting.

    For each IP address:
        1. Divide traffic into 1-minute windows.
        2. Count requests in each window.
        3. If requests exceed threshold,
           mark those requests as blocked.

    Prediction:
        0 = allowed
        1 = blocked
    """

    data = df.copy()

    # --------------------------------------------------------
    # Create one-minute windows
    # --------------------------------------------------------

    data["window"] = (
        data["timestamp"]
        .dt.floor("min")
    )

    # --------------------------------------------------------
    # Count requests per IP per minute
    # --------------------------------------------------------

    request_counts = (
        data.groupby(
            [
                "ip_address",
                "window"
            ]
        )
        .size()
        .rename(
            "requests_in_window"
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Attach count to each request
    # --------------------------------------------------------

    data = data.merge(
        request_counts,
        on=[
            "ip_address",
            "window"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # Apply rate limit
    # --------------------------------------------------------

    data[
        "baseline_prediction"
    ] = (
        data[
            "requests_in_window"
        ]
        > threshold
    ).astype(int)

    return data


# ============================================================
# MAP REQUEST RESULTS TO SESSION
# ============================================================

def create_session_predictions(
    data
):

    """
    A session is considered detected if at least
    one request from that session is blocked.
    """

    session_predictions = (
        data.groupby(
            "session_id"
        )
        .agg(
            baseline_prediction=(
                "baseline_prediction",
                "max"
            ),
            label=(
                "label",
                "first"
            )
        )
        .reset_index()
    )

    return session_predictions


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    # True Positive
    tp = (
        (y_true == 1)
        &
        (y_pred == 1)
    ).sum()

    # True Negative
    tn = (
        (y_true == 0)
        &
        (y_pred == 0)
    ).sum()

    # False Positive
    fp = (
        (y_true == 0)
        &
        (y_pred == 1)
    ).sum()

    # False Negative
    fn = (
        (y_true == 1)
        &
        (y_pred == 0)
    ).sum()

    # Precision
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    # Recall / Detection Rate
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    # F1 score
    f1 = (
        2
        * precision
        * recall
        /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # False Positive Rate
    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate":
            false_positive_rate
    }


# ============================================================
# EVALUATE ONE THRESHOLD
# ============================================================

def evaluate_threshold(
    df,
    threshold
):

    print(
        f"\nTesting threshold: "
        f"{threshold} requests/minute"
    )

    # Apply rate limiter
    request_results = (
        apply_rate_limit(
            df,
            threshold
        )
    )

    # Convert request-level detection
    # into session-level detection
    session_results = (
        create_session_predictions(
            request_results
        )
    )

    y_true = (
        session_results["label"]
    )

    y_pred = (
        session_results[
            "baseline_prediction"
        ]
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    metrics[
        "threshold"
    ] = threshold

    return (
        metrics,
        request_results
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    df
):

    results = []

    request_results_by_threshold = {}

    for threshold in THRESHOLDS:

        metrics, request_results = (
            evaluate_threshold(
                df,
                threshold
            )
        )

        results.append(
            metrics
        )

        request_results_by_threshold[
            threshold
        ] = request_results

    return (
        pd.DataFrame(results),
        request_results_by_threshold
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results
):

    print("\n" + "=" * 60)
    print(
        "FIXED RATE-LIMIT BASELINE RESULTS"
    )
    print("=" * 60)

    columns = [
        "threshold",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate"
    ]

    print(
        results[
            columns
        ].to_string(
            index=False
        )
    )

    print("\nPercentages:")

    percentage_results = (
        results.copy()
    )

    percentage_columns = [
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate"
    ]

    for column in percentage_columns:

        percentage_results[
            column
        ] = (
            percentage_results[
                column
            ]
            * 100
        ).round(2)

    print(
        percentage_results[
            [
                "threshold",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results
):

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
        "Traditional Fixed Rate-Limit Baseline"
    )
    print("=" * 60)

    # Load raw traffic
    df = load_data()

    # Run experiment
    results, _ = (
        run_experiment(df)
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
        "\nBaseline evaluation completed."
    )


if __name__ == "__main__":
    main()