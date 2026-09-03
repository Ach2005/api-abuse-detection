import requests
import time


BASE_URL = "http://127.0.0.1:8000"


def send_request(client_id):

    return requests.get(
        f"{BASE_URL}/api/data",
        headers={
            "X-Test-Client": client_id
        }
    )


# NORMAL TRAFFIC

print("\n==============================")
print("NORMAL TRAFFIC TEST")
print("==============================")

normal_client = "normal-client"

for i in range(5):

    response = send_request(
        normal_client
    )

    print(
        f"Request {i + 1}: "
        f"{response.status_code} "
        f"{response.headers.get('X-Abuse-Risk')}"
    )

    time.sleep(2)


print("\nNormal traffic test completed.")


# SUSPICIOUS TRAFFIC

print("\n==============================")
print("SUSPICIOUS TRAFFIC TEST")
print("==============================")

suspicious_client = "suspicious-client"

for i in range(25):

    response = send_request(
        suspicious_client
    )

    data = response.json()

    print(
        f"Request {i + 1}: "
        f"{response.status_code} "
        f"{response.headers.get('X-Abuse-Risk')}"
    )

    if response.status_code == 429:

        print(
            "Mitigation:",
            data.get("mitigation")
        )

        if data.get("mitigation") == "RATE LIMIT":

            break

    time.sleep(0.03)


# HIGH-SPEED ATTACK

print("\n==============================")
print("HIGH-SPEED ABUSE TEST")
print("==============================")

attack_client = "attack-client"

for i in range(100):

    response = send_request(
        attack_client
    )

    data = response.json()

    print(
        f"Request {i + 1}: "
        f"{response.status_code} "
        f"{response.headers.get('X-Abuse-Risk')}"
    )

    if response.status_code == 429:

        mitigation = data.get(
            "mitigation"
        )

        print(
            "Mitigation:",
            mitigation
        )

        if mitigation == "BLOCK":

            print(
                "\n================================"
            )

            print(
                "HIGH-RISK ATTACK BLOCKED!"
            )

            print(
                "================================"
            )

            print(
                "Reason:",
                data.get("reason")
            )

            break

        # Wait for temporary rate limit
        # before continuing the attack.

        time.sleep(3.2)

    else:

        time.sleep(0.01)