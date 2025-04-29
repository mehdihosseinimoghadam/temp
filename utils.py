import random
import requests
from typing import List
from pydantic import BaseModel


def get_data(url):
    """Fetch data from an external URL"""
    response = requests.get(url)
    return response.json()


def generate_random_data(count: int):
    """Generate random sample data"""
    result = []
    for i in range(count):
        result.append({
            "id": i + 1,
            "name": f"Item-{random.randint(1000, 9999)}",
            "value": round(random.uniform(0, 100), 2),
            "tags": random.sample(["red", "green", "blue", "alpha", "beta", "gamma"], k=random.randint(1, 3))
        })
    return result

# Define response models


class GeneratedData(BaseModel):
    id: int
    name: str
    value: float
    tags: List[str]


class DataAnalysis(BaseModel):
    count: int
    average_value: float
    most_common_tag: str
    min_value: float
    max_value: float
