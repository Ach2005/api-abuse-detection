import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed/api_behavioral_features.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "random_forest_model.pkl"
)


# ============================================================
# FEATURES USED BY THE MODEL
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
# TRAIN MODEL
# ============================================================

def train_model():

    print("Training Random Forest model...")

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]

    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"Model saved to: {MODEL_FILE}"
    )

    return model


# ============================================================
# LOAD OR TRAIN MODEL
# ============================================================

def get_model():

    if os.path.exists(
        MODEL_FILE
    ):

        print(
            "Loading trained Random Forest model..."
        )

        return joblib.load(
            MODEL_FILE
        )

    return train_model()


# ============================================================
# CALCULATE BEHAVIORAL SCORE
# ============================================================

def calculate_abuse_score(data):

    score = 0

    reasons = []

    # --------------------------------------------------------
    # Request frequency
    # --------------------------------------------------------

    if data["requests_per_minute"] >= 60:

        score += 25

        reasons.append(
            "Very high request frequency"
        )

    elif data["requests_per_minute"] >= 30:

        score += 15

        reasons.append(
            "High request frequency"
        )

    # --------------------------------------------------------
    # Failed authentication
    # --------------------------------------------------------

    if data["failed_auth_ratio"] >= 0.50:

        score += 20

        reasons.append(
            "High failed authentication ratio"
        )

    elif data["failed_auth_ratio"] >= 0.20:

        score += 10

        reasons.append(
            "Elevated failed authentication"
        )

    # --------------------------------------------------------
    # Error behavior
    # --------------------------------------------------------

    if data["error_ratio"] >= 0.50:

        score += 15

        reasons.append(
            "High API error ratio"
        )

    elif data["error_ratio"] >= 0.20:

        score += 8

        reasons.append(
            "Elevated API errors"
        )

    # --------------------------------------------------------
    # Endpoint behavior
    # --------------------------------------------------------

    if data["unique_endpoint_count"] >= 12:

        score += 10

        reasons.append(
            "Unusual endpoint exploration"
        )

    # --------------------------------------------------------
    # Request timing
    # --------------------------------------------------------

    if data["mean_request_interval"] <= 2:

        score += 15

        reasons.append(
            "Very short request intervals"
        )

    elif data["mean_request_interval"] <= 5:

        score += 8

        reasons.append(
            "Short request intervals"
        )

    # --------------------------------------------------------
    # Geographic behavior
    # --------------------------------------------------------

    if data["max_geo_distance_km"] >= 3000:

        score += 10

        reasons.append(
            "Large geographic movement"
        )

    # --------------------------------------------------------
    # Burst behavior
    # --------------------------------------------------------

    if data["burst_score"] >= 0.70:

        score += 15

        reasons.append(
            "High burst activity"
        )

    # Keep score between 0 and 100
    score = min(
        score,
        100
    )

    return score, reasons


# ============================================================
# CLASSIFY RISK
# ============================================================

def classify_risk(
    abuse_score,
    ml_probability
):

    # --------------------------------------------------------
    # Combine behavioral score and ML confidence
    # --------------------------------------------------------

    if (
        abuse_score >= 70
        or ml_probability >= 0.80
    ):

        return "HIGH RISK"

    elif (
        abuse_score >= 40
        or ml_probability >= 0.50
    ):

        return "SUSPICIOUS"

    else:

        return "NORMAL"


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def determine_action(
    risk
):

    if risk == "HIGH RISK":

        return "BLOCK"

    elif risk == "SUSPICIOUS":

        return "RATE LIMIT / MONITOR"

    else:

        return "ALLOW"


# ============================================================
# DETECT API ABUSE
# ============================================================

def detect_abuse(
    model,
    input_data
):

    # Convert input into DataFrame
    data = pd.DataFrame(
        [input_data]
    )

    # Ensure correct feature order
    X = data[FEATURES]

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X
    )[0]

    probability = model.predict_proba(
        X
    )[0][1]

    # --------------------------------------------------------
    # Behavioral Abuse Score
    # --------------------------------------------------------

    abuse_score, reasons = (
        calculate_abuse_score(
            input_data
        )
    )

    # --------------------------------------------------------
    # Final risk classification
    # --------------------------------------------------------

    risk = classify_risk(
        abuse_score,
        probability
    )

    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    action = determine_action(
        risk
    )

    return {
        "ml_prediction": int(
            prediction
        ),

        "ml_abuse_probability": round(
            probability * 100,
            2
        ),

        "abuse_score": abuse_score,

        "risk": risk,

        "action": action,

        "reasons": reasons
    }


# ============================================================
# DEMO
# ============================================================

def main():

    print("=" * 60)

    print(
        "AI-Based API Abuse Detection"
    )

    print(
        "Real-Time Detection Demo"
    )

    print("=" * 60)

    # Load model
    model = get_model()

    # --------------------------------------------------------
    # Example suspicious API session
    # --------------------------------------------------------

    sample_request = {

        "session_request_count": 70,

        "session_duration_seconds": 60,

        "requests_per_minute": 70,

        "failed_auth_ratio": 0.65,

        "error_ratio": 0.40,

        "unique_endpoint_count": 15,

        "endpoint_diversity": 0.30,

        "mean_request_interval": 1.2,

        "request_interval_std": 0.5,

        "night_activity_ratio": 0.20,

        "total_geo_distance_km": 3500,

        "max_geo_distance_km": 3500,

        "mean_geo_distance_km": 1750,

        "burst_score": 0.80
    }

    # Detect
    result = detect_abuse(
        model,
        sample_request
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "DETECTION RESULT"
    )

    print(
        "=" * 60
    )

    print(
        f"\nML Prediction: "
        f"{'ABUSE' if result['ml_prediction'] == 1 else 'NORMAL'}"
    )

    print(
        f"ML Abuse Probability: "
        f"{result['ml_abuse_probability']}%"
    )

    print(
        f"Behavioral Abuse Score: "
        f"{result['abuse_score']}/100"
    )

    print(
        f"Risk Level: "
        f"{result['risk']}"
    )

    print(
        f"Recommended Action: "
        f"{result['action']}"
    )

    print(
        "\nReasons:"
    )

    if result["reasons"]:

        for reason in result["reasons"]:

            print(
                f" - {reason}"
            )

    else:

        print(
            " - No major behavioral anomaly detected"
        )

    print(
        "\nDetection completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()