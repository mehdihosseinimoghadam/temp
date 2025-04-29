import random
import requests
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field


def get_data(url: str, timeout: int = 10, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Fetch data from an external URL with improved error handling and options

    Args:
        url: The URL to fetch data from
        timeout: Request timeout in seconds
        headers: Optional request headers

    Returns:
        Parsed JSON response
    """
    if headers is None:
        headers = {"User-Agent": "DataGeneratorAPI/1.0"}

    try:
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"error": "Invalid JSON response"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def generate_random_data(count: int, seed: int = None) -> List[Dict[str, Any]]:
    """
    Generate random sample data with improved randomization

    Args:
        count: Number of items to generate
        seed: Optional random seed for reproducible results

    Returns:
        List of generated data items
    """
    if seed is not None:
        random.seed(seed)

    categories = ["electronics", "clothing", "food", "books", "toys"]
    adjectives = ["amazing", "innovative", "quality", "affordable", "premium"]

    result = []
    for i in range(count):
        category = random.choice(categories)
        adjective = random.choice(adjectives)
        result.append({
            "id": i + 1,
            "name": f"{adjective.title()} {category.title()} Item",
            "value": round(random.uniform(10, 500), 2),
            "category": category,
            "rating": round(random.uniform(1, 5), 1),
            "tags": random.sample(["new", "sale", "limited", "popular", "exclusive"],
                                  k=random.randint(1, 3)),
            "in_stock": random.choice([True, False])
        })

    return result

# Define response models


class GeneratedData(BaseModel):
    id: int
    name: str
    value: float
    category: str = Field(..., description="Product category")
    rating: float = Field(..., ge=1, le=5, description="Product rating (1-5)")
    tags: List[str]
    in_stock: bool = Field(..., description="Availability status")


class DataAnalysis(BaseModel):
    total_items: int = Field(..., description="Total number of items analyzed")
    price_metrics: Dict[str,
                        float] = Field(..., description="Price statistics")
    category_distribution: Dict[str,
                                int] = Field(..., description="Items per category")
    availability: Dict[str, int] = Field(...,
                                         description="In-stock vs out-of-stock counts")
    top_tags: List[str] = Field(..., description="Most frequently used tags")
