from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
import sys
import time

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from detection.detect_abuse import get_model, detect_abuse
from api.middleware import tracker


app = FastAPI(
    title="AI-Based API Abuse Detection",
    description="Real-time API abuse detection and automated mitigation",
    version="1.0"
)


model = get_model()


# Temporary rate-limit information
rate_limited_until = {}

# Number of times each client has triggered rate limiting
rate_limit_violations = {}


class BehavioralData(BaseModel):

    session_request_count: int
    session_duration_seconds: float
    requests_per_minute: float
    failed_auth_ratio: float
    error_ratio: float
    unique_endpoint_count: int
    endpoint_diversity: float
    mean_request_interval: float
    request_interval_std: float
    night_activity_ratio: float
    total_geo_distance_km: float
    max_geo_distance_km: float
    mean_geo_distance_km: float
    burst_score: float


@app.get("/")
async def home():

    return {
        "message": "AI-Based API Abuse Detection System",
        "status": "running"
    }


@app.get("/api/data")
async def protected_api():

    return {
        "message": "API request allowed",
        "data": "Sample protected API data"
    }


@app.post("/detect")
async def detect(data: BehavioralData):

    input_data = data.model_dump()

    result = detect_abuse(
        model,
        input_data
    )

    return {
        "status": "success",
        "detection": result
    }


@app.middleware("http")
async def abuse_detection_middleware(
    request: Request,
    call_next
):

    ignored_paths = [
        "/",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/detect"
    ]

    if request.url.path in ignored_paths:
        return await call_next(request)


    # Identify the client.
    # X-Test-Client is used only by the local test script.

    test_client = request.headers.get(
        "X-Test-Client"
    )

    if test_client:
        client_id = test_client
    else:
        client_id = (
            request.client.host
            if request.client
            else "unknown"
        )


    now = time.time()


    # Check whether this client is already blocked.

    if rate_limit_violations.get(
        client_id,
        0
    ) >= 3:

        return JSONResponse(

            status_code=429,

            headers={
                "Retry-After": "60",
                "X-Abuse-Risk": "HIGH RISK"
            },

            content={

                "status": "blocked",

                "message":
                    "Client blocked after repeated suspicious activity",

                "client":
                    client_id,

                "risk":
                    "HIGH RISK",

                "mitigation":
                    "BLOCK",

                "reason":
                    "Repeated rate-limit violations"
            }
        )


    # Check temporary rate limit.

    if client_id in rate_limited_until:

        if now < rate_limited_until[client_id]:

            remaining = int(
                rate_limited_until[client_id] - now
            ) + 1

            return JSONResponse(

                status_code=429,

                headers={
                    "Retry-After": str(remaining),
                    "X-Abuse-Risk": "SUSPICIOUS"
                },

                content={

                    "status":
                        "rate_limited",

                    "message":
                        "Too many suspicious API requests. Please slow down.",

                    "client":
                        client_id,

                    "risk":
                        "SUSPICIOUS",

                    "mitigation":
                        "RATE LIMIT",

                    "retry_after_seconds":
                        remaining
                }
            )

        else:

            del rate_limited_until[client_id]


    # Record request.

    tracker.record_request(
        client_id,
        request.url.path
    )


    # Calculate behavioral features.

    features = tracker.get_features(
        client_id
    )


    # Run ML detection.

    result = detect_abuse(
        model,
        features
    )


    requests_per_minute = (
        features["requests_per_minute"]
    )

    ml_probability = (
        result["ml_abuse_probability"]
    )

    abuse_score = (
        result["abuse_score"]
    )


    # Decide risk level.

    if (

        requests_per_minute >= 60

        or (

            requests_per_minute >= 30

            and (

                ml_probability >= 80

                or abuse_score >= 70

            )

        )

    ):

        risk = "HIGH RISK"
        action = "BLOCK"


    elif (

        requests_per_minute >= 20

        and (

            ml_probability >= 50
            or abuse_score >= 30

        )

    ):

        risk = "SUSPICIOUS"
        action = "RATE LIMIT"


    else:

        risk = "NORMAL"
        action = "ALLOW"


    # High-risk traffic is blocked immediately.

    if action == "BLOCK":

        return JSONResponse(

            status_code=429,

            headers={
                "Retry-After": "60",
                "X-Abuse-Risk": "HIGH RISK"
            },

            content={

                "status":
                    "blocked",

                "message":
                    "Request blocked due to suspected API abuse",

                "client":
                    client_id,

                "requests_per_minute":
                    requests_per_minute,

                "risk":
                    risk,

                "mitigation":
                    "BLOCK",

                "abuse_score":
                    abuse_score,

                "ml_abuse_probability":
                    ml_probability,

                "reasons":
                    result["reasons"]
            }
        )


    # Suspicious traffic is temporarily rate limited.

    if action == "RATE LIMIT":

        rate_limit_violations[client_id] = (
            rate_limit_violations.get(
                client_id,
                0
            ) + 1
        )

        rate_limited_until[client_id] = (
            time.time() + 3
        )

        return JSONResponse(

            status_code=429,

            headers={
                "Retry-After": "3",
                "X-Abuse-Risk": "SUSPICIOUS",
                "X-Rate-Limit-Warning":
                    "Suspicious API activity detected"
            },

            content={

                "status":
                    "rate_limited",

                "message":
                    "Request rate temporarily limited due to suspicious activity",

                "client":
                    client_id,

                "requests_per_minute":
                    requests_per_minute,

                "risk":
                    risk,

                "mitigation":
                    "RATE LIMIT",

                "abuse_score":
                    abuse_score,

                "ml_abuse_probability":
                    ml_probability,

                "rate_limit_violations":
                    rate_limit_violations[client_id],

                "reasons":
                    result["reasons"],

                "retry_after_seconds":
                    3
            }
        )


    # Normal request.

    response = await call_next(
        request
    )


    # Record response status.

    tracker.record_response(
        client_id,
        response.status_code
    )


    response.headers[
        "X-Abuse-Risk"
    ] = "NORMAL"

    return response