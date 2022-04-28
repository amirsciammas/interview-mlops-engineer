import random
from typing import List

from locust import HttpUser, constant, task


class APIUser(HttpUser):

    """
    Basic user looking for predictions
    """

    wait_time = constant(1)

    @task()
    def get_prediction(self) -> None:

        """
        Prediction request
        """

        data = {"data": self._generate_random_data()}
        self.client.post(url="/predict", json=data, stream=True)

    def _generate_random_data(self) -> List:
        """
        Generate 4 float numbers for iris predictions.

        Returns:
            A list of 4 integers between 1 and 20
        """
        return [random.randrange(1, 20) for _ in range(4)]
