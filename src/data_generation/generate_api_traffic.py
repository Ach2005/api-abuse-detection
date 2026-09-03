import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

NUM_REQUESTS = 100_000
NUM_USERS = 2_000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "api_traffic_raw.csv"
)


# ============================================================
# API CONFIGURATION
# ============================================================

ENDPOINTS = [
    "/api/login",
    "/api/logout",
    "/api/profile",
    "/api/users",
    "/api/products",
    "/api/products/search",
    "/api/orders",
    "/api/orders/history",
    "/api/cart",
    "/api/payment",
    "/api/notifications",
    "/api/settings",
    "/api/reviews",
    "/api/recommendations",
]

USER_AGENTS = [
    "Chrome",
    "Firefox",
    "Safari",
    "Edge",
    "MobileApp",
]

NORMAL_LOCATIONS = [
    (28.6139, 77.2090, "Delhi"),
    (19.0760, 72.8777, "Mumbai"),
    (18.5204, 73.8567, "Pune"),
    (12.9716, 77.5946, "Bangalore"),
    (17.3850, 78.4867, "Hyderabad"),
    (22.5726, 88.3639, "Kolkata"),
    (13.0827, 80.2707, "Chennai"),
    (26.9124, 75.7873, "Jaipur"),
]

ABNORMAL_LOCATIONS = [
    (40.7128, -74.0060, "New York"),
    (51.5074, -0.1278, "London"),
    (35.6762, 139.6503, "Tokyo"),
    (37.7749, -122.4194, "San Francisco"),
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_ip():
    return f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"


def generate_session_id():
    return f"session_{uuid.uuid4().hex[:12]}"


def choose_normal_location():
    return random.choice(NORMAL_LOCATIONS)


def choose_normal_endpoint():
    weights = [
        0.08, 0.03, 0.12, 0.08,
        0.15, 0.12, 0.10, 0.06,
        0.08, 0.03, 0.03, 0.03,
        0.04, 0.05
    ]

    return random.choices(
        ENDPOINTS,
        weights=weights,
        k=1
    )[0]


def random_endpoint():
    return random.choice(ENDPOINTS)


# ============================================================
# RESPONSE TIME
# ============================================================

def response_time(scenario):

    distributions = {
        "normal": (180, 60),
        "brute_force": (190, 70),
        "flooding": (140, 50),
        "endpoint_spam": (170, 60),
        "scraping": (210, 80),
        "geo_anomaly": (190, 70),
        "unusual_timing": (180, 70),
        "stealth_abuse": (190, 70),
    }

    mean, std = distributions[scenario]

    return max(
        20,
        np.random.normal(mean, std)
    )


# ============================================================
# STATUS CODE
# ============================================================

def generate_status_code(scenario):

    if scenario == "normal":
        return random.choices(
            [200, 201, 400, 401, 404, 500],
            weights=[0.84, 0.05, 0.04, 0.02, 0.04, 0.01]
        )[0]

    if scenario == "brute_force":
        return random.choices(
            [200, 401, 403, 429],
            weights=[0.10, 0.65, 0.15, 0.10]
        )[0]

    if scenario == "flooding":
        return random.choices(
            [200, 429, 500, 503],
            weights=[0.55, 0.25, 0.10, 0.10]
        )[0]

    if scenario == "endpoint_spam":
        return random.choices(
            [200, 404, 429, 500],
            weights=[0.60, 0.15, 0.15, 0.10]
        )[0]

    if scenario == "scraping":
        return random.choices(
            [200, 404, 429],
            weights=[0.78, 0.12, 0.10]
        )[0]

    if scenario == "geo_anomaly":
        return random.choices(
            [200, 401, 403, 429],
            weights=[0.65, 0.15, 0.10, 0.10]
        )[0]

    if scenario == "unusual_timing":
        return random.choices(
            [200, 401, 404, 429],
            weights=[0.78, 0.08, 0.09, 0.05]
        )[0]

    if scenario == "stealth_abuse":
        return random.choices(
            [200, 401, 403, 404, 429],
            weights=[0.70, 0.10, 0.08, 0.07, 0.05]
        )[0]

    return 200


# ============================================================
# AUTHENTICATION
# ============================================================

def generate_auth_success(scenario):

    if scenario == "normal":
        return random.random() < 0.97

    if scenario == "brute_force":
        return random.random() < 0.25

    if scenario == "flooding":
        return random.random() < 0.90

    if scenario == "endpoint_spam":
        return random.random() < 0.90

    if scenario == "scraping":
        return random.random() < 0.95

    if scenario == "geo_anomaly":
        return random.random() < 0.80

    if scenario == "unusual_timing":
        return random.random() < 0.92

    if scenario == "stealth_abuse":
        return random.random() < 0.75

    return True


# ============================================================
# REQUEST INTERVALS
# ============================================================

def generate_interval(scenario, normal_profile=None):

    # --------------------------------------------------------
    # NORMAL TRAFFIC
    # --------------------------------------------------------

    if scenario == "normal":

        # Ordinary legitimate user
        if normal_profile == "ordinary":
            return np.random.exponential(
                scale=30
            )

        # Legitimate high-traffic user
        #
        # Roughly 35-60 requests/minute.
        # This should create false positives for
        # aggressive traditional rate limits.
        elif normal_profile == "high_traffic":
            return np.random.uniform(
                1.0,
                1.7
            )

        # Legitimate bursty user
        #
        # Most requests are normal, but occasionally
        # the user makes a short burst.
        elif normal_profile == "bursty":

            if random.random() < 0.35:
                return np.random.uniform(
                    0.4,
                    1.5
                )

            return np.random.uniform(
                15,
                45
            )

        return 30

    # --------------------------------------------------------
    # ABUSIVE TRAFFIC
    # --------------------------------------------------------

    if scenario == "brute_force":
        return np.random.uniform(
            0.5,
            3
        )

    if scenario == "flooding":
        return np.random.uniform(
            0.05,
            1
        )

    if scenario == "endpoint_spam":
        return np.random.uniform(
            0.5,
            5
        )

    if scenario == "scraping":
        return np.random.uniform(
            1,
            8
        )

    if scenario == "geo_anomaly":
        return np.random.uniform(
            5,
            30
        )

    if scenario == "unusual_timing":

        if random.random() < 0.5:
            return np.random.uniform(
                1,
                5
            )

        return np.random.uniform(
            300,
            900
        )

    if scenario == "stealth_abuse":
        return np.random.uniform(
            5,
            25
        )

    return 30


# ============================================================
# ENDPOINT BEHAVIOR
# ============================================================

def choose_endpoint(scenario):

    if scenario == "normal":
        return choose_normal_endpoint()

    if scenario == "brute_force":
        return "/api/login"

    if scenario == "flooding":
        return random_endpoint()

    if scenario == "endpoint_spam":
        return random.choice([
            "/api/products",
            "/api/products/search",
            "/api/users",
            "/api/orders"
        ])

    if scenario == "scraping":
        return random.choice([
            "/api/products",
            "/api/products/search",
            "/api/reviews",
            "/api/recommendations"
        ])

    if scenario == "geo_anomaly":
        return choose_normal_endpoint()

    if scenario == "unusual_timing":
        return choose_normal_endpoint()

    if scenario == "stealth_abuse":
        return random.choice([
            "/api/users",
            "/api/profile",
            "/api/products",
            "/api/orders",
            "/api/products/search",
            "/api/reviews"
        ])

    return choose_normal_endpoint()


# ============================================================
# CREATE USERS
# ============================================================

def create_users():

    users = []

    for i in range(1, NUM_USERS + 1):

        users.append({
            "user_id": f"user_{i:04d}",
            "ip_address": generate_ip(),
            "home_location": choose_normal_location(),
            "user_agent": random.choice(USER_AGENTS),
        })

    return users


# ============================================================
# ASSIGN SESSIONS
# ============================================================

def assign_sessions(users):

    scenarios = [
        "normal",
        "brute_force",
        "flooding",
        "endpoint_spam",
        "scraping",
        "geo_anomaly",
        "unusual_timing",
        "stealth_abuse",
    ]

    scenario_weights = [
        0.65,
        0.07,
        0.06,
        0.05,
        0.05,
        0.04,
        0.04,
        0.04,
    ]

    sessions = []

    for user in users:

        number_of_sessions = random.randint(
            1,
            3
        )

        for _ in range(number_of_sessions):

            scenario = random.choices(
                scenarios,
                weights=scenario_weights,
                k=1
            )[0]

            # ------------------------------------------------
            # Only normal users receive normal traffic
            # profiles.
            # ------------------------------------------------

            normal_profile = None

            if scenario == "normal":

                normal_profile = random.choices(
                    [
                        "ordinary",
                        "high_traffic",
                        "bursty"
                    ],
                    weights=[
                        0.70,
                        0.20,
                        0.10
                    ],
                    k=1
                )[0]

            sessions.append({
                "session_id": generate_session_id(),
                "user_id": user["user_id"],
                "ip_address": user["ip_address"],
                "home_location": user["home_location"],
                "user_agent": user["user_agent"],
                "scenario": scenario,
                "normal_profile": normal_profile,
            })

    return sessions


# ============================================================
# GENERATE ONE REQUEST
# ============================================================

def generate_request(
    session,
    timestamp,
    request_number
):

    scenario = session["scenario"]

    home_location = session["home_location"]

    # --------------------------------------------------------
    # Geographic behavior
    # --------------------------------------------------------

    if scenario == "geo_anomaly":

        if request_number < 5:
            location = home_location
        else:
            location = random.choice(
                ABNORMAL_LOCATIONS
            )

    else:
        location = home_location

    # --------------------------------------------------------
    # Scraping user agent
    # --------------------------------------------------------

    if scenario == "scraping":

        user_agent = random.choice([
            "Chrome",
            "Firefox",
            "MobileApp",
            "Python-Requests",
            "ScraperBot"
        ])

    else:
        user_agent = session["user_agent"]

    # --------------------------------------------------------
    # HTTP method
    # --------------------------------------------------------

    if scenario == "brute_force":
        method = "POST"

    elif scenario == "scraping":
        method = "GET"

    else:
        method = random.choice([
            "GET",
            "GET",
            "GET",
            "POST",
            "PUT"
        ])

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    return {
        "timestamp": timestamp,
        "user_id": session["user_id"],
        "ip_address": session["ip_address"],
        "session_id": session["session_id"],
        "endpoint": choose_endpoint(scenario),
        "http_method": method,
        "status_code": generate_status_code(scenario),
        "response_time_ms": round(
            response_time(scenario),
            2
        ),
        "auth_success": generate_auth_success(
            scenario
        ),
        "latitude": location[0],
        "longitude": location[1],
        "location": location[2],
        "user_agent": user_agent,
        "request_size": random.randint(
            200,
            5000
        ),
        "response_size": random.randint(
            500,
            50000
        ),
        "scenario": scenario,
        "label": (
            0 if scenario == "normal"
            else 1
        ),
    }


# ============================================================
# DATASET GENERATOR
# ============================================================

def generate_dataset(num_requests):

    start_time = datetime(
        2026,
        1,
        1,
        0,
        0,
        0
    )

    users = create_users()

    sessions = assign_sessions(
        users
    )

    print(
        f"Created {len(users):,} users"
    )

    print(
        f"Created {len(sessions):,} sessions"
    )

    # --------------------------------------------------------
    # Display normal profile distribution
    # --------------------------------------------------------

    normal_sessions = [
        s for s in sessions
        if s["scenario"] == "normal"
    ]

    profile_counts = pd.Series(
        [
            s["normal_profile"]
            for s in normal_sessions
        ]
    ).value_counts()

    print("\nNormal traffic profiles:")

    print(profile_counts)

    # --------------------------------------------------------
    # Determine requests per session
    # --------------------------------------------------------

    session_requests = []

    for session in sessions:

        scenario = session["scenario"]
        profile = session["normal_profile"]

        if scenario == "normal":

            if profile == "ordinary":
                count = random.randint(
                    15,
                    80
                )

            elif profile == "high_traffic":
                count = random.randint(
                    35,
                    100
                )

            elif profile == "bursty":
                count = random.randint(
                    25,
                    90
                )

            else:
                count = 30

        elif scenario == "brute_force":
            count = random.randint(
                25,
                100
            )

        elif scenario == "flooding":
            count = random.randint(
                40,
                120
            )

        elif scenario == "endpoint_spam":
            count = random.randint(
                25,
                90
            )

        elif scenario == "scraping":
            count = random.randint(
                20,
                100
            )

        elif scenario == "geo_anomaly":
            count = random.randint(
                20,
                80
            )

        elif scenario == "unusual_timing":
            count = random.randint(
                15,
                60
            )

        elif scenario == "stealth_abuse":
            count = random.randint(
                25,
                90
            )

        else:
            count = 30

        session_requests.append(
            (session, count)
        )

    # --------------------------------------------------------
    # Scale to target number of requests
    # --------------------------------------------------------

    total_generated = sum(
        count
        for _, count
        in session_requests
    )

    scale = (
        num_requests /
        total_generated
    )

    adjusted_sessions = []

    for session, count in session_requests:

        adjusted_count = max(
            5,
            int(count * scale)
        )

        adjusted_sessions.append(
            (
                session,
                adjusted_count
            )
        )

    # --------------------------------------------------------
    # Generate request sequences
    # --------------------------------------------------------

    rows = []

    print(
        "\nGenerating request sequences..."
    )

    for index, (
        session,
        count
    ) in enumerate(
        adjusted_sessions
    ):

        current_time = (
            start_time
            + timedelta(
                seconds=random.randint(
                    0,
                    86400
                )
            )
        )

        for request_number in range(
            count
        ):

            interval = generate_interval(
                session["scenario"],
                session["normal_profile"]
            )

            current_time += timedelta(
                seconds=float(interval)
            )

            timestamp = current_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            row = generate_request(
                session,
                timestamp,
                request_number
            )

            rows.append(row)

        if (index + 1) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1:,} / "
                f"{len(adjusted_sessions):,} "
                f"sessions"
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    # Ensure exactly requested number
    if len(df) > num_requests:

        df = df.sample(
            n=num_requests,
            random_state=RANDOM_SEED
        )

    elif len(df) < num_requests:

        extra = df.sample(
            n=(
                num_requests
                - len(df)
            ),
            replace=True,
            random_state=RANDOM_SEED
        )

        df = pd.concat(
            [
                df,
                extra
            ],
            ignore_index=True
        )

    # Sort chronologically
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI-Based API Abuse Detection")
    print("Synthetic API Traffic Generator - Version 3")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(
        f"\nGenerating "
        f"{NUM_REQUESTS:,} realistic API requests..."
    )

    df = generate_dataset(
        NUM_REQUESTS
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET GENERATED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nFile: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print("\nScenario distribution:")

    print(
        df["scenario"].value_counts()
    )

    print("\nLabel distribution:")

    print(
        df["label"].value_counts()
    )

    print("\nUnique users:")

    print(
        df["user_id"].nunique()
    )

    print("\nUnique sessions:")

    print(
        df["session_id"].nunique()
    )

    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    print("\nDataset saved successfully.")


if __name__ == "__main__":
    main()