import random
import uuid
import time
import os

from locust import HttpUser, task, between


ACCOUNT_REFRESH_TTL = int(os.environ.get("LOCUST_ACCOUNT_REFRESH_TTL", "60"))
NORMAL_CREATE_PROB = float(os.environ.get("LOCUST_CREATE_PROB_NORMAL", "0.05"))
MIXED_CREATE_PROB = float(os.environ.get("LOCUST_CREATE_PROB_MIXED", "0.02"))


class BaseBankingUser(HttpUser):

    wait_time = between(0.1, 1.5)

    def on_start(self):
        self.cached_account_ids = []
        self._accounts_loaded_at = 0
        self.load_accounts()


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

        except Exception:
            self.cached_account_ids = []

    def get_random_account(self):
        # refresh cache periodically to avoid stale IDs
        if (not self.cached_account_ids) or (time.time() - getattr(self, "_accounts_loaded_at", 0) > ACCOUNT_REFRESH_TTL):
            self.load_accounts()

        if not self.cached_account_ids:
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

        user = self.create_random_user()

        if not user:
            return

        account = self.create_account(user["id"])

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
            user = self.create_random_user()
            if not user:
                return

            account = self.create_account(user["id"])
            if account:
                self.cached_account_ids.append(account["id"])


class InvalidUser(BaseBankingUser):

    weight = 1

    wait_time = between(0.5, 3.0)

    @task(5)
    def invalid_transfer(self):

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

        self.client.get("/accounts/99999999")

    @task(2)
    def malformed_request(self):

        self.client.post(
            "/transfers/",
            json={
                "broken": "payload"
            }
        )


class Chaos500User(HttpUser):
    """
    Пользователь, который намеренно вызывает серверные ошибки (500)
    через header-инъекцию.
    """

    weight = 1
    wait_time = between(0.1, 0.5)

    def injected_headers(self):
        return {
            "X-Inject-Failure": "true"
        }

    @task(5)
    def break_accounts_read(self):
        account_id = random.randint(1, 100000)

        self.client.get(
            f"/accounts/{account_id}",
            headers=self.injected_headers()
        )

    @task(5)
    def break_accounts_list(self):
        self.client.get(
            "/accounts/?limit=50",
            headers=self.injected_headers()
        )

    @task(3)
    def break_transfers(self):
        self.client.post(
            "/transfers/",
            json={
                "from_account_id": random.randint(1, 1000),
                "to_account_id": random.randint(1, 1000),
                "amount": random.randint(1, 10000),
                "currency": "RUB"
            },
            headers=self.injected_headers()
        )

    @task(2)
    def break_account_creation(self):
        username = f"chaos_{uuid.uuid4().hex[:6]}"

        self.client.post(
            "/users/",
            json={
                "username": username,
                "password": "password123"
            },
            headers=self.injected_headers()
        )

    @task(1)
    def random_endpoint_stress(self):
        endpoints = [
            "/accounts/",
            "/accounts/?limit=10",
            "/transfers/",
        ]

        self.client.get(
            random.choice(endpoints),
            headers=self.injected_headers()
        )
