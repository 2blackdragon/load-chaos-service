import random
import uuid
import time
import os
import threading

from locust import HttpUser, task, between


# Configurable knobs (via env vars)
SEED_ACCOUNT_MIN = int(os.environ.get("LOCUST_SEED_ACCOUNTS", "10"))
ACCOUNT_REFRESH_TTL = int(os.environ.get("LOCUST_ACCOUNT_REFRESH_TTL", "60"))  # seconds
# Default creation probabilities lowered to avoid excessive account churn during runs
NORMAL_CREATE_PROB = float(os.environ.get("LOCUST_CREATE_PROB_NORMAL", "0.01"))
MIXED_CREATE_PROB = float(os.environ.get("LOCUST_CREATE_PROB_MIXED", "0.005"))
# Probability that an InvalidUser will actually perform invalid actions (keeps dataset cleaner)
INVALID_USER_PROB = float(os.environ.get("LOCUST_INVALID_PROB", "0.05"))

# Global cap for creations during a run (0 = unlimited)
LOCUST_MAX_CREATIONS = int(os.environ.get("LOCUST_MAX_CREATIONS", "200"))
_created_count = 0
_created_lock = threading.Lock()


class BaseBankingUser(HttpUser):

    wait_time = between(0.1, 1.5)

    def on_start(self):
        self.cached_account_ids = []
        self._accounts_loaded_at = 0
        self.load_accounts()

        # Ensure there is a small pool of existing accounts to operate on.
        if len(self.cached_account_ids) < SEED_ACCOUNT_MIN:
            self.ensure_seed_accounts(SEED_ACCOUNT_MIN)
        # debug: report loaded account cache size
        try:
            print(f"[locust] on_start loaded {len(self.cached_account_ids)} account ids; sample={self.cached_account_ids[:5]}")
        except Exception:
            pass


    def load_accounts(self, limit: int = 1000):

        response = self.client.get(f"/accounts/?limit={limit}")

        if response.status_code != 200:
            return

        try:
            accounts = response.json()

            # robustly extract ids from returned list-like structure
            ids = []
            if isinstance(accounts, dict) and "items" in accounts:
                iterable = accounts["items"]
            else:
                iterable = accounts

            for account in iterable:
                if isinstance(account, dict) and "id" in account:
                    ids.append(account["id"])

            self.cached_account_ids = ids
            self._accounts_loaded_at = time.time()
            # debug: report how many ids were loaded
            try:
                print(f"[locust] load_accounts fetched {len(self.cached_account_ids)} ids; sample={self.cached_account_ids[:5]}")
            except Exception:
                pass

        except Exception:
            self.cached_account_ids = []

    def get_random_account(self):
        # refresh cache periodically to avoid stale IDs
        if (not self.cached_account_ids) or (time.time() - getattr(self, "_accounts_loaded_at", 0) > ACCOUNT_REFRESH_TTL):
            print("[locust] cache empty or stale, reloading accounts")
            self.load_accounts()

        if not self.cached_account_ids:
            print("[locust] no accounts available in cache")
            return None

        return random.choice(self.cached_account_ids)
    

    def create_random_user(self):

        username = f"user_{uuid.uuid4().hex[:8]}"

        response = self.client.post(
            "/users/",
            json={
                "username": username,
                "password": "password123"
            }
        )

        if response.status_code not in (200, 201):
            return None

        return response.json()

    def _increment_creation_count(self):
        global _created_count
        if LOCUST_MAX_CREATIONS <= 0:
            return True

        with _created_lock:
            if _created_count >= LOCUST_MAX_CREATIONS:
                return False
            _created_count += 1
            return True

    def create_user_and_account_limited(self):
        """Create a user and account atomically and increment global creation counter.

        Returns created account object or None.
        """
        # check + create user
        user = self.create_random_user()
        if not user:
            return None

        account = self.create_account(user["id"])
        if not account:
            return None

        # only commit creation if under cap (prevents over-creation)
        if not self._increment_creation_count():
            return None

        return account

    def create_account(self, user_id):

        response = self.client.post(
            "/accounts/",
            json={
                "user_id": user_id,
                "account_number": str(uuid.uuid4())[:16],
                "balance": random.randint(1000, 10000),
                "currency": "RUB"
            }
        )

        if response.status_code not in (200, 201):
            return None

        return response.json()


    def get_account(self, account_id):
        response = self.client.get(f"/accounts/{account_id}")

        # If account not found, remove from cache to reduce repeated 404s
        if response.status_code == 404:
            try:
                self.cached_account_ids.remove(account_id)
            except ValueError:
                pass

            # attempt a quick reload if cache is now too small
            if len(self.cached_account_ids) < 3:
                self.load_accounts()

        return response

    def ensure_seed_accounts(self, min_count: int = 10):
        # create a few accounts if there are not enough existing ones
        attempts = 0
        while len(self.cached_account_ids) < min_count and attempts < min_count * 3:
            user = self.create_random_user()
            if not user:
                attempts += 1
                continue

            account = self.create_account(user["id"])
            if account and isinstance(account, dict) and "id" in account:
                self.cached_account_ids.append(account["id"])

            attempts += 1

    def list_accounts(self):

        return self.client.get("/accounts/?limit=100")



class NormalUser(BaseBankingUser):

    weight = 5

    @task(5)
    def read_account(self):

        account_id = self.get_random_account()

        if not account_id:
            return

        self.get_account(account_id)

    @task(3)
    def make_transfer(self):

        from_acc = self.get_random_account()
        to_acc = self.get_random_account()

        if not from_acc or not to_acc:
            return

        if from_acc == to_acc:
            return

        self.client.post(
            "/transfers/",
            json={
                "from_account_id": from_acc,
                "to_account_id": to_acc,
                "amount": random.randint(1, 500),
                "currency": "RUB"
            }
        )

    @task(1)
    def create_new_user_and_account(self):
        # create new accounts only rarely to avoid overwhelming the service
        if random.random() > NORMAL_CREATE_PROB:
            return

        account = self.create_user_and_account_limited()

        if not account:
            return

        self.cached_account_ids.append(account["id"])

    @task(2)
    def browse_accounts(self):

        self.list_accounts()


class AggressiveUser(BaseBankingUser):

    weight = 2

    wait_time = between(0.01, 0.2)

    @task(10)
    def aggressive_reads(self):

        account_id = self.get_random_account()

        if not account_id:
            return

        self.get_account(account_id)

    @task(8)
    def aggressive_transfers(self):

        from_acc = self.get_random_account()
        to_acc = self.get_random_account()

        if not from_acc or not to_acc:
            return

        if from_acc == to_acc:
            return

        self.client.post(
            "/transfers/",
            json={
                "from_account_id": from_acc,
                "to_account_id": to_acc,
                "amount": random.randint(1, 2000),
                "currency": "RUB"
            }
        )

    @task(5)
    def account_listing_spam(self):

        self.list_accounts()


class MixedUser(BaseBankingUser):

    weight = 3

    wait_time = between(0.05, 1.0)

    @task(6)
    def mixed_reads(self):
        # mostly normal reads but sometimes more aggressive
        account_id = self.get_random_account()

        if not account_id:
            return

        self.get_account(account_id)

    @task(3)
    def mixed_transfers(self):
        from_acc = self.get_random_account()
        to_acc = self.get_random_account()

        if not from_acc or not to_acc:
            return

        if from_acc == to_acc:
            return

        amount = random.randint(1, 1500)

        self.client.post(
            "/transfers/",
            json={
                "from_account_id": from_acc,
                "to_account_id": to_acc,
                "amount": amount,
                "currency": "RUB",
            },
        )

    @task(1)
    def occasional_create(self):
        if random.random() < MIXED_CREATE_PROB:
            account = self.create_user_and_account_limited()
            if account:
                self.cached_account_ids.append(account["id"])


class InvalidUser(BaseBankingUser):

    weight = 1

    wait_time = between(0.5, 3.0)

    @task(5)
    def invalid_transfer(self):
        # run invalid actions only occasionally to avoid polluting dataset
        if random.random() > INVALID_USER_PROB:
            return

        self.client.post(
            "/transfers/",
            json={
                "from_account_id": 999999,
                "to_account_id": 888888,
                "amount": -100,
                "currency": "INVALID"
            }
        )

    @task(3)
    def invalid_account_lookup(self):
        if random.random() > INVALID_USER_PROB:
            return

        self.client.get("/accounts/99999999")

    @task(2)
    def malformed_request(self):
        if random.random() > INVALID_USER_PROB:
            return

        self.client.post(
            "/transfers/",
            json={
                "broken": "payload"
            }
        )
