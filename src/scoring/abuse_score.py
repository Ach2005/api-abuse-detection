import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/api_behavioral_features.csv"

OUTPUT_DIR = "results"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "api_abuse_scores.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading behavioral feature dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(df):,} sessions"
    )

    return df


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(series, low, high):

    """
    Convert a feature into a 0-100 risk score.

    Values below low  -> 0
    Values above high -> 100
    """

    if high <= low:

        return pd.Series(
            0,
            index=series.index
        )

    score = (
        (series - low)
        /
        (high - low)
        * 100
    )

    return score.clip(
        0,
        100
    )


# ============================================================
# BEHAVIORAL RISK COMPONENTS
# ============================================================

def calculate_risk_components(df):

    print(
        "\nCalculating behavioral risk components..."
    )

    # --------------------------------------------------------
    # 1. REQUEST FREQUENCY
    # --------------------------------------------------------
    #
    # Low traffic is normal.
    # Above ~30 requests/min becomes increasingly suspicious.
    #
    # --------------------------------------------------------

    df["frequency_risk"] = normalize(
        df["requests_per_minute"],
        low=10,
        high=60
    )

    # --------------------------------------------------------
    # 2. FAILED AUTHENTICATION
    # --------------------------------------------------------

    df["authentication_risk"] = (
        df["failed_auth_ratio"]
        * 100
    ).clip(
        0,
        100
    )

    # --------------------------------------------------------
    # 3. ERROR BEHAVIOR
    # --------------------------------------------------------

    df["error_risk"] = (
        df["error_ratio"]
        * 100
    ).clip(
        0,
        100
    )

    # --------------------------------------------------------
    # 4. ENDPOINT BEHAVIOR
    # --------------------------------------------------------
    #
    # We use unique endpoint count instead of only endpoint
    # diversity because a session touching many endpoints
    # can indicate enumeration/scanning behavior.
    #
    # --------------------------------------------------------

    df["endpoint_risk"] = normalize(
        df["unique_endpoint_count"],
        low=3,
        high=12
    )

    # --------------------------------------------------------
    # 5. REQUEST TIMING
    # --------------------------------------------------------
    #
    # Very short intervals can indicate automation.
    #
    # < 2 sec  -> high risk
    # 2-5 sec  -> moderate risk
    # > 10 sec -> low risk
    #
    # --------------------------------------------------------

    timing_score = np.where(
        df["mean_request_interval"] <= 2,
        100,
        np.where(
            df["mean_request_interval"] <= 5,
            70,
            np.where(
                df["mean_request_interval"] <= 10,
                30,
                0
            )
        )
    )

    df["timing_risk"] = timing_score

    # --------------------------------------------------------
    # 6. GEOGRAPHIC BEHAVIOR
    # --------------------------------------------------------
    #
    # max_geo_distance_km represents the largest geographic
    # movement observed for the user.
    #
    # 0-100 km     -> normal
    # 100-1000 km  -> suspicious
    # >1000 km     -> increasingly suspicious
    #
    # --------------------------------------------------------

    df["geographic_risk"] = normalize(
        df["max_geo_distance_km"],
        low=100,
        high=3000
    )

    # --------------------------------------------------------
    # 7. BURST BEHAVIOR
    # --------------------------------------------------------

    df["burst_risk"] = (
        df["burst_score"]
        * 100
    ).clip(
        0,
        100
    )

    return df


# ============================================================
# ABUSE SCORE
# ============================================================

def calculate_abuse_score(df):

    print(
        "\nCalculating API Abuse Score..."
    )

    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------
    #
    # Total = 1.00
    #
    # Frequency + authentication receive more weight because
    # they are strong API abuse indicators.
    #
    # --------------------------------------------------------

    weights = {

        "frequency_risk": 0.20,

        "authentication_risk": 0.20,

        "error_risk": 0.10,

        "endpoint_risk": 0.10,

        "timing_risk": 0.15,

        "geographic_risk": 0.10,

        "burst_risk": 0.15
    }

    # --------------------------------------------------------
    # Weighted score
    # --------------------------------------------------------

    df["abuse_score"] = (

        df["frequency_risk"]
        * weights["frequency_risk"]

        +

        df["authentication_risk"]
        * weights["authentication_risk"]

        +

        df["error_risk"]
        * weights["error_risk"]

        +

        df["endpoint_risk"]
        * weights["endpoint_risk"]

        +

        df["timing_risk"]
        * weights["timing_risk"]

        +

        df["geographic_risk"]
        * weights["geographic_risk"]

        +

        df["burst_risk"]
        * weights["burst_risk"]
    )

    df["abuse_score"] = (
        df["abuse_score"]
        .clip(
            0,
            100
        )
        .round(2)
    )

    return df


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(score):

    if score < 40:

        return "NORMAL"

    elif score < 70:

        return "SUSPICIOUS"

    else:

        return "HIGH_RISK"


def create_risk_classification(df):

    print(
        "\nClassifying sessions..."
    )

    df["risk_classification"] = (
        df["abuse_score"]
        .apply(
            classify_risk
        )
    )

    return df


# ============================================================
# EXPLANATIONS
# ============================================================

def generate_reasons(row):

    reasons = []

    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    if row["frequency_risk"] >= 70:

        reasons.append(
            "High request frequency"
        )

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    if row["authentication_risk"] >= 50:

        reasons.append(
            "High failed authentication ratio"
        )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    if row["error_risk"] >= 50:

        reasons.append(
            "High API error ratio"
        )

    # --------------------------------------------------------
    # Endpoint behavior
    # --------------------------------------------------------

    if row["endpoint_risk"] >= 70:

        reasons.append(
            "Unusual endpoint exploration"
        )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    if row["timing_risk"] >= 70:

        reasons.append(
            "Very short request intervals"
        )

    # --------------------------------------------------------
    # Geography
    # --------------------------------------------------------

    if row["geographic_risk"] >= 70:

        reasons.append(
            "Large geographic movement"
        )

    # --------------------------------------------------------
    # Burst
    # --------------------------------------------------------

    if row["burst_risk"] >= 70:

        reasons.append(
            "High burst activity"
        )

    # --------------------------------------------------------
    # No strong anomaly
    # --------------------------------------------------------

    if len(reasons) == 0:

        reasons.append(
            "No major behavioral anomaly detected"
        )

    return "; ".join(
        reasons
    )


def create_explanations(df):

    print(
        "\nGenerating explanations..."
    )

    df["detection_reasons"] = (
        df.apply(
            generate_reasons,
            axis=1
        )
    )

    return df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(df):

    print(
        "\n" + "=" * 60
    )

    print(
        "API ABUSE SCORE RESULTS"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Classification distribution
    # --------------------------------------------------------

    print(
        "\nRisk classification:"
    )

    print(
        df[
            "risk_classification"
        ]
        .value_counts()
    )

    # --------------------------------------------------------
    # Average score by scenario
    # --------------------------------------------------------

    print(
        "\nAverage abuse score by scenario:"
    )

    scenario_scores = (
        df.groupby(
            "scenario"
        )["abuse_score"]
        .mean()
        .sort_values(
            ascending=False
        )
        .round(2)
    )

    print(
        scenario_scores
    )

    # --------------------------------------------------------
    # Normal vs abuse
    # --------------------------------------------------------

    print(
        "\nAverage abuse score by label:"
    )

    print(
        df.groupby(
            "label"
        )["abuse_score"]
        .mean()
        .round(2)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\nScore statistics:"
    )

    print(
        df[
            "abuse_score"
        ]
        .describe()
        .round(2)
    )

    # --------------------------------------------------------
    # Top detections
    # --------------------------------------------------------

    print(
        "\nSample detections:"
    )

    print(
        df[
            [
                "session_id",
                "scenario",
                "label",
                "abuse_score",
                "risk_classification",
                "detection_reasons"
            ]
        ]
        .sort_values(
            "abuse_score",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
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
        "Behavioral API Abuse Scoring"
    )

    print(
        "=" * 60
    )

    # Load
    df = load_data()

    # Risk components
    df = calculate_risk_components(
        df
    )

    # Abuse score
    df = calculate_abuse_score(
        df
    )

    # Classification
    df = create_risk_classification(
        df
    )

    # Explanations
    df = create_explanations(
        df
    )

    # Save
    save_results(
        df
    )

    # Display
    display_results(
        df
    )

    print(
        "\nAbuse scoring completed."
    )


if __name__ == "__main__":

    main()