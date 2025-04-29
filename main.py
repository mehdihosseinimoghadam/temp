from fastapi import FastAPI, HTTPException
import requests
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

app = FastAPI(title="Data Generator API")


def get_data(url):
    response = requests.get(url)
    return response.json()

# Define a response model


class GeneratedData(BaseModel):
    id: int
    name: str
    value: float
    tags: List[str]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Generator API"}


@app.get("/generate-data", response_model=List[GeneratedData])
def generate_data(count: Optional[int] = 10):
    """
    Generate random sample data.

    Parameters:
    - count: Number of data items to generate (default: 10)

    Returns:
    - List of generated data items
    """
    if count <= 0 or count > 100:
        raise HTTPException(
            status_code=400, detail="Count must be between 1 and 100")

    result = []
    for i in range(count):
        result.append({
            "id": i + 1,
            "name": f"Item-{random.randint(1000, 9999)}",
            "value": round(random.uniform(0, 100), 2),
            "tags": random.sample(["red", "green", "blue", "alpha", "beta", "gamma"], k=random.randint(1, 3))
        })

    return result


@app.get("/fetch-external-data")
def fetch_external_data(url: str):
    """
    Fetch data from an external URL using the existing get_data function.

    Parameters:
    - url: The URL to fetch data from

    Returns:
    - The JSON response from the URL
    """
    try:
        return get_data(url)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
