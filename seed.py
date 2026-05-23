import requests
import uuid
import random
import os

BASE_URL = os.environ.get("BASE_URL", "http://138.16.162.15:8000")
SEED_COUNT = int(os.environ.get("LOCUST_SEED_COUNT", "1000"))


def create_user_and_account(idx: int) -> int:
    username = f"user_{idx}"

    user = requests.post(f"{BASE_URL}/users/", json={
        "username": username,
        "password": "password123"
    }).json()

    account = requests.post(f"{BASE_URL}/accounts/", json={
        "user_id": user["id"],
        "account_number": f"acc_{idx}",
        "balance": random.randint(1000, 10000),
        "currency": "RUB"
    }).json()

    return account["id"]


def main():
    account_ids = []
    for idx in range(SEED_COUNT):
        try:
            account_ids.append(create_user_and_account(idx + 1))
        except Exception as e:
            print("seed error:", e)

    print(f"seeded accounts: {len(account_ids)}")


if __name__ == "__main__":
    main()