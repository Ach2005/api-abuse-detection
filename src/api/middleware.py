import time
from collections import defaultdict, deque


# ============================================================
# REAL-TIME API ABUSE TRACKER
# ============================================================

class AbuseTracker:

    def __init__(self):

        # Recent request timestamps for each client
        self.requests = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Recently accessed endpoints
        self.endpoints = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Recent response status codes
        self.status_codes = defaultdict(
            lambda: deque(maxlen=100)
        )

    # --------------------------------------------------------
    # Record request
    # --------------------------------------------------------

    def record_request(
        self,
        client_id,
        endpoint
    ):

        now = time.time()

        self.requests[client_id].append(now)

        self.endpoints[client_id].append(endpoint)

        # Placeholder until response is available
        self.status_codes[client_id].append(0)

        self.cleanup(client_id, now)

    # --------------------------------------------------------
    # Record response
    # --------------------------------------------------------

    def record_response(
        self,
        client_id,
        status_code
    ):

        # Update the latest request's status code
        if self.status_codes[client_id]:

            self.status_codes[client_id][-1] = status_code

    # --------------------------------------------------------
    # Remove old requests
    # --------------------------------------------------------

    def cleanup(
        self,
        client_id,
        now
    ):

        while (
            self.requests[client_id]
            and now - self.requests[client_id][0] > 60
        ):

            self.requests[client_id].popleft()

            if self.endpoints[client_id]:
                self.endpoints[client_id].popleft()

            if self.status_codes[client_id]:
                self.status_codes[client_id].popleft()

    # --------------------------------------------------------
    # Calculate live behavioral features
    # --------------------------------------------------------

    def get_features(
        self,
        client_id
    ):

        timestamps = list(
            self.requests[client_id]
        )

        endpoints = list(
            self.endpoints[client_id]
        )

        status_codes = list(
            self.status_codes[client_id]
        )

        request_count = len(timestamps)

        # ----------------------------------------------------
        # Requests per minute
        # ----------------------------------------------------

        requests_per_minute = request_count

        # ----------------------------------------------------
        # Request intervals
        # ----------------------------------------------------

        intervals = []

        for i in range(
            1,
            len(timestamps)
        ):

            intervals.append(
                timestamps[i] - timestamps[i - 1]
            )

        if intervals:

            mean_interval = (
                sum(intervals) / len(intervals)
            )

            if len(intervals) > 1:

                mean = mean_interval

                variance = sum(
                    (x - mean) ** 2
                    for x in intervals
                ) / len(intervals)

                interval_std = variance ** 0.5

            else:

                interval_std = 0.0

        else:

            mean_interval = 60.0

            interval_std = 0.0

        # ----------------------------------------------------
        # Unique endpoints
        # ----------------------------------------------------

        unique_endpoints = len(
            set(endpoints)
        )

        endpoint_diversity = (
            unique_endpoints / request_count
            if request_count > 0
            else 0.0
        )

        # ----------------------------------------------------
        # HTTP error ratio
        # ----------------------------------------------------

        valid_status_codes = [
            code
            for code in status_codes
            if code != 0
        ]

        if valid_status_codes:

            error_count = sum(
                1
                for code in valid_status_codes
                if code >= 400
            )

            error_ratio = (
                error_count / len(valid_status_codes)
            )

        else:

            error_ratio = 0.0

        # ----------------------------------------------------
        # Failed authentication ratio
        #
        # 401 = Unauthorized
        # 403 = Forbidden
        # ----------------------------------------------------

        if valid_status_codes:

            failed_auth_count = sum(
                1
                for code in valid_status_codes
                if code in (401, 403)
            )

            failed_auth_ratio = (
                failed_auth_count /
                len(valid_status_codes)
            )

        else:

            failed_auth_ratio = 0.0

        # ----------------------------------------------------
        # Burst score
        # ----------------------------------------------------

        if mean_interval <= 1:

            burst_score = 1.0

        elif mean_interval <= 2:

            burst_score = 0.8

        elif mean_interval <= 5:

            burst_score = 0.5

        else:

            burst_score = 0.1

        # ----------------------------------------------------
        # Session duration
        # ----------------------------------------------------

        if len(timestamps) >= 2:

            duration = (
                timestamps[-1] -
                timestamps[0]
            )

        else:

            duration = 0.0

        # ----------------------------------------------------
        # Return all behavioral features
        # ----------------------------------------------------

        return {

            "session_request_count":
                request_count,

            "session_duration_seconds":
                duration,

            "requests_per_minute":
                requests_per_minute,

            "failed_auth_ratio":
                failed_auth_ratio,

            "error_ratio":
                error_ratio,

            "unique_endpoint_count":
                unique_endpoints,

            "endpoint_diversity":
                endpoint_diversity,

            "mean_request_interval":
                mean_interval,

            "request_interval_std":
                interval_std,

            "night_activity_ratio":
                0.0,

            "total_geo_distance_km":
                0.0,

            "max_geo_distance_km":
                0.0,

            "mean_geo_distance_km":
                0.0,

            "burst_score":
                burst_score
        }


# ============================================================
# GLOBAL TRACKER
# ============================================================

tracker = AbuseTracker()