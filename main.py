from fastapi import FastAPI, HTTPException, Query, Depends, Path
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import statistics
from collections import Counter
import logging

from utils import generate_random_data, GeneratedData, DataAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Data Generator API",
    description="An API for generating and analyzing random data",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def read_root():
    """Return a welcome message for the API"""
    logger.info("Root endpoint accessed")
    return {
        "message": "Welcome to the Enhanced Data Generator API",
        "version": "2.0.0",
        "documentation": "/docs"
    }


@app.get("/generate-data", response_model=List[GeneratedData], tags=["Data Generation"])
def generate_data(
    count: int = Query(
        10, ge=1, le=100, description="Number of items to generate"),
    seed: Optional[int] = Query(
        None, description="Random seed for reproducible results")
):
    """
    Generate random sample data with customizable parameters.

    The data includes product information with categories, ratings, and availability.
    """
    logger.info(f"Generating {count} data items with seed={seed}")
    return generate_random_data(count, seed)


@app.get("/fetch-external-data", tags=["External Data"])
def fetch_external_data(
    url: str = Query(..., description="URL to fetch data from"),
    timeout: int = Query(
        10, ge=1, le=60, description="Request timeout in seconds")
):
    """
    Fetch data from an external URL with configurable timeout.

    This endpoint acts as a proxy to external data sources.
    """
    logger.info(f"Fetching external data from {url}")
    result = get_data(url, timeout)

    if "error" in result:
        logger.error(f"Error fetching data: {result['error']}")
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.get("/analyze-data", response_model=DataAnalysis, tags=["Data Analysis"])
def analyze_data(
    count: int = Query(
        20, ge=1, le=100, description="Number of items to analyze"),
    seed: Optional[int] = Query(
        None, description="Random seed for reproducible results")
):
    """
    Generate and perform comprehensive analysis on random data.

    The analysis includes price statistics, category distribution, and tag frequency.
    """
    logger.info(f"Analyzing {count} data items with seed={seed}")
    data = generate_random_data(count, seed)

    # Extract values for analysis
    values = [item["value"] for item in data]
    categories = [item["category"] for item in data]
    all_tags = [tag for item in data for tag in item["tags"]]
    in_stock_count = sum(1 for item in data if item["in_stock"])

    # Calculate category distribution
    category_counts = Counter(categories)

    # Find most common tags
    tag_counter = Counter(all_tags)
    top_tags = [tag for tag, _ in tag_counter.most_common(3)]

    # Create analysis result
    analysis = {
        "total_items": len(data),
        "price_metrics": {
            "average": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "min": min(values),
            "max": max(values),
            "standard_deviation": round(statistics.stdev(values), 2) if len(values) > 1 else 0
        },
        "category_distribution": dict(category_counts),
        "availability": {
            "in_stock": in_stock_count,
            "out_of_stock": len(data) - in_stock_count
        },
        "top_tags": top_tags
    }

    return analysis


@app.get("/data/{item_id}", response_model=GeneratedData, tags=["Data Retrieval"])
def get_item(
    item_id: int = Path(..., ge=1,
                        description="The ID of the item to retrieve"),
    count: int = Query(100, ge=1, le=1000,
                       description="Pool size to generate"),
    seed: int = Query(42, description="Random seed for reproducible results")
):
    """
    Retrieve a specific item by ID from a generated data pool.

    Uses a fixed seed to ensure consistent results for the same ID.
    """
    logger.info(f"Retrieving item with ID {item_id}")
    data = generate_random_data(count, seed)

    if item_id > len(data):
        raise HTTPException(
            status_code=404, detail=f"Item with ID {item_id} not found")

    return data[item_id - 1]


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
