from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional, Dict
import statistics
from collections import Counter

from utils import get_data, generate_random_data, GeneratedData, DataAnalysis

app = FastAPI(title="Data Generator API")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Generator API"}


@app.get("/generate-data", response_model=List[GeneratedData])
def generate_data(count: Optional[int] = Query(10, ge=1, le=100)):
    """
    Generate random sample data.

    Parameters:
    - count: Number of data items to generate (default: 10, min: 1, max: 100)

    Returns:
    - List of generated data items
    """
    return generate_random_data(count)


@app.get("/fetch-external-data")
def fetch_external_data(url: str):
    """
    Fetch data from an external URL.

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


@app.get("/analyze-data", response_model=DataAnalysis)
def analyze_data(count: Optional[int] = Query(20, ge=1, le=100)):
    """
    Generate and analyze random data.

    Parameters:
    - count: Number of data items to generate and analyze (default: 20)

    Returns:
    - Statistical analysis of the generated data
    """
    data = generate_random_data(count)

    # Extract values for analysis
    values = [item["value"] for item in data]
    all_tags = [tag for item in data for tag in item["tags"]]

    # Find most common tag
    tag_counter = Counter(all_tags)
    most_common = tag_counter.most_common(1)[0][0] if all_tags else "none"

    analysis = {
        "count": len(data),
        "average_value": round(statistics.mean(values), 2),
        "most_common_tag": most_common,
        "min_value": min(values),
        "max_value": max(values)
    }

    return analysis


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
