import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/raw/api_traffic_raw.csv"

OUTPUT_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "api_behavioral_features.csv"
)


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_data():

    print("Loading raw API traffic...")

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(
        f"Loaded {len(df):,} requests"
    )

    return df


# ============================================================
# BASIC REQUEST FEATURES
# ============================================================

def create_basic_features(df):

    print("\nCreating basic request features...")

    # HTTP error indicator
    df["is_error"] = (
        df["status_code"] >= 400
    ).astype(int)

    # Authentication failure
    df["auth_failure"] = (
        df["auth_success"] == False
    ).astype(int)

    # Hour of day
    df["hour"] = df["timestamp"].dt.hour

    # Night activity
    df["is_night"] = (
        df["hour"] < 6
    ).astype(int)

    return df


# ============================================================
# TIME-BASED FEATURES
# ============================================================

def create_time_features(df):

    print(
        "Creating time-based behavioral features..."
    )

    # Sort requests chronologically within sessions
    df = df.sort_values(
        [
            "user_id",
            "session_id",
            "timestamp"
        ]
    ).reset_index(drop=True)

    # Time between consecutive requests
    df["request_interval"] = (
        df.groupby("session_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # First request has no previous request
    df["request_interval"] = (
        df["request_interval"]
        .fillna(0)
    )

    # Prevent negative values
    df["request_interval"] = (
        df["request_interval"]
        .clip(lower=0)
    )

    return df


# ============================================================
# SESSION-LEVEL FEATURES
# ============================================================

def create_session_features(df):

    print(
        "Creating session-level behavioral features..."
    )

    grouped = df.groupby("session_id")

    # --------------------------------------------------------
    # Total requests
    # --------------------------------------------------------

    session_request_count = (
        grouped.size()
        .rename("session_request_count")
    )

    # --------------------------------------------------------
    # Session duration
    # --------------------------------------------------------

    session_duration = (
        grouped["timestamp"].max()
        -
        grouped["timestamp"].min()
    ).dt.total_seconds()

    session_duration = (
        session_duration
        .rename("session_duration_seconds")
    )

    # --------------------------------------------------------
    # Requests per minute
    # --------------------------------------------------------

    requests_per_minute = (
        session_request_count
        /
        (
            session_duration / 60
        ).replace(0, np.nan)
    )

    requests_per_minute = (
        requests_per_minute
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(
            session_request_count
        )
        .rename("requests_per_minute")
    )

    # --------------------------------------------------------
    # Failed authentication ratio
    # --------------------------------------------------------

    failed_auth_ratio = (
        grouped["auth_failure"]
        .mean()
        .rename("failed_auth_ratio")
    )

    # --------------------------------------------------------
    # Error ratio
    # --------------------------------------------------------

    error_ratio = (
        grouped["is_error"]
        .mean()
        .rename("error_ratio")
    )

    # --------------------------------------------------------
    # Unique endpoints
    # --------------------------------------------------------

    unique_endpoint_count = (
        grouped["endpoint"]
        .nunique()
        .rename("unique_endpoint_count")
    )

    # --------------------------------------------------------
    # Endpoint diversity
    # --------------------------------------------------------

    endpoint_diversity = (
        unique_endpoint_count
        /
        session_request_count
    ).rename("endpoint_diversity")

    # --------------------------------------------------------
    # Mean request interval
    # --------------------------------------------------------

    mean_request_interval = (
        grouped["request_interval"]
        .mean()
        .rename("mean_request_interval")
    )

    # --------------------------------------------------------
    # Request interval standard deviation
    # --------------------------------------------------------

    request_interval_std = (
        grouped["request_interval"]
        .std()
        .fillna(0)
        .rename("request_interval_std")
    )

    # --------------------------------------------------------
    # Night activity ratio
    # --------------------------------------------------------

    night_activity_ratio = (
        grouped["is_night"]
        .mean()
        .rename("night_activity_ratio")
    )

    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    session_features = pd.concat(
        [
            session_request_count,
            session_duration,
            requests_per_minute,
            failed_auth_ratio,
            error_ratio,
            unique_endpoint_count,
            endpoint_diversity,
            mean_request_interval,
            request_interval_std,
            night_activity_ratio
        ],
        axis=1
    )

    return session_features


# ============================================================
# GEOGRAPHIC FEATURES
# ============================================================

def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    """
    Calculate Haversine distance in kilometers.
    """

    earth_radius = 6371

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    delta_lat = np.radians(
        lat2 - lat1
    )

    delta_lon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(delta_lat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return earth_radius * c


def create_geographic_features(df):

    print(
        "Creating geographic behavioral features..."
    )

    df = df.sort_values(
        [
            "user_id",
            "timestamp"
        ]
    ).reset_index(drop=True)

    # Previous location
    df["previous_latitude"] = (
        df.groupby("user_id")["latitude"]
        .shift(1)
    )

    df["previous_longitude"] = (
        df.groupby("user_id")["longitude"]
        .shift(1)
    )

    # Distance from previous location
    df["geo_distance_km"] = calculate_distance(
        df["previous_latitude"].fillna(
            df["latitude"]
        ),
        df["previous_longitude"].fillna(
            df["longitude"]
        ),
        df["latitude"],
        df["longitude"]
    )

    df["geo_distance_km"] = (
        df["geo_distance_km"]
        .fillna(0)
    )

    return df


# ============================================================
# BURST FEATURES
# ============================================================

def create_burst_features(df):

    print(
        "Creating burst behavior features..."
    )

    # Requests less than 2 seconds apart
    # are considered burst requests.

    df["burst_request"] = (
        (df["request_interval"] > 0)
        &
        (df["request_interval"] < 2)
    ).astype(int)

    burst_ratio = (
        df.groupby("session_id")[
            "burst_request"
        ]
        .mean()
        .rename("burst_score")
    )

    return burst_ratio


# ============================================================
# BUILD FINAL DATASET
# ============================================================

def build_feature_dataset(df):

    print(
        "\nBuilding final behavioral dataset..."
    )

    # Basic features
    df = create_basic_features(df)

    # Time features
    df = create_time_features(df)

    # Session features
    session_features = (
        create_session_features(df)
    )

    # Geographic features
    df = create_geographic_features(df)

    # Geographic session statistics
    geo_features = (
        df.groupby("session_id")[
            "geo_distance_km"
        ]
        .agg(
            total_geo_distance_km="sum",
            max_geo_distance_km="max",
            mean_geo_distance_km="mean"
        )
    )

    # Burst features
    burst_features = (
        create_burst_features(df)
    )

    # Combine session features
    feature_df = session_features.join(
        geo_features
    )

    feature_df = feature_df.join(
        burst_features
    )

    # Labels
    labels = (
        df.groupby("session_id")
        .agg(
            label=("label", "first"),
            scenario=("scenario", "first")
        )
    )

    feature_df = feature_df.join(
        labels
    )

    # User information
    user_info = (
        df.groupby("session_id")
        .agg(
            user_id=("user_id", "first"),
            ip_address=("ip_address", "first")
        )
    )

    feature_df = feature_df.join(
        user_info
    )

    # Reset index
    feature_df = (
        feature_df
        .reset_index()
    )

    # Replace invalid values
    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_df = (
        feature_df
        .fillna(0)
    )

    return feature_df


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nFeature dataset saved to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI-Based API Abuse Detection")
    print("Behavioral Feature Engineering")
    print("=" * 60)

    # Load raw dataset
    df = load_data()

    # Build behavioral features
    feature_df = build_feature_dataset(df)

    # Save
    save_dataset(feature_df)

    # Results
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)

    print(
        f"\nRows: {len(feature_df):,}"
    )

    print(
        f"Columns: {len(feature_df.columns)}"
    )

    print("\nFeature columns:")

    for column in feature_df.columns:
        print(f" - {column}")

    print("\nLabel distribution:")

    print(
        feature_df["label"]
        .value_counts()
    )

    print("\nScenario distribution:")

    print(
        feature_df["scenario"]
        .value_counts()
    )

    print("\nMissing values:")

    print(
        feature_df.isnull().sum()
    )

    print("\nFeature preview:")

    print(
        feature_df.head()
    )

    print("\n[Done]")


if __name__ == "__main__":
    main()