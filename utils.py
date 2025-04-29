import random
import requests
import json
import time
import hashlib
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Set
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum, auto
from functools import lru_cache
import numpy as np
from dataclasses import dataclass

# Constants
REQUEST_TIMEOUT = 15  # Changed from DEFAULT_TIMEOUT
DATA_CACHE_TTL = 3nged from USER_AGENTS
   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
]

    # Cache for API responses
    api_cache= {}  # Changed from response_cache


    class ProductCategory(str, Enum):  # Changed from DataCategory
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD = "food"
    BOOKS = "books"
    TOYS = "toys"
    FURNITURE = "furniture"
    SPORTS = "sports"
    BEAUTY = "beauty"


    class PriceBounds(BaseModel):  # Changed from PriceRange
    min_value: float = Field(0.0, ge=0)  # Changed from min_price
    max_value: float = Field(1000.0, gt=0)  # Changed from max_price

    @ validator('max_value')  # Changed from max_price
    def max_must_exceed_min(cls, v, values):  # Changed method name
        if 'min_value' in values and v <= values['min_value']:
            raise ValueError('max_value must be greater than min_value')
    return v


    class GeneratorConfig(BaseModel):  # Changed from DataGenerationOptions
    item_count: int = Field(10, ge=1, le=1000)  # Changed from count
    random_seed: Optional[int] = None  # Changed from seed
    product_types: Optional[List[ProductCategory]
                           ]= None  # Changed from categories
                            price_bounds: Optional[PriceBounds]= None  # Changed from price_range
                                with_metadata: bool= False  # Changed from include_metadata
                                min_rating: Optional[float]= Field(
        None, ge = 1, le = 5)  # Changed from rating_threshold
        tag_filters: Optional[Set[str]] = None  # Changed from tags_filter and type

    @ root_validator
        def validate_filters(cls, values):  # Changed method name
        if values.get('min_rating') is not None and values.get('min_rating') > 4.5:
        if not values.get('tag_filters'):
        values['tag_filters'] = set(['premium'])  # Changed to set
            elif 'premium' not in values.get('tag_filters'):
        values['tag_filters'].add('premium')  # Using add() for set
            return values


@ dataclass
        class CacheItem:  # Changed from CacheEntry
        content: Any  # Changed from data
        created: float  # Changed from timestamp
        lifetime: int  # Changed from ttl


        # Changed function name and parameters
        def store_in_cache(key: str, content: Any, lifetime: int = DATA_CACHE_TTL):
        """Cache API response data with expiration"""
        api_cache[key] = CacheItem(
        content =content, created=time.time(), lifetime=lifetime)


        def retrieve_from_cache(key: str) -> Optional[Any]:  # Changed function name
        """Retrieve cached response if valid"""
        if key in api_cache:
        item= api_cache[key]
        if time.time() - item.created < item.lifetime:
        return item.content
        else:
        del api_cache[key]  # Expired
        return None


        # Changed function name
        def create_cache_key(url: str, params: Dict[str, Any] = None) -> str:
        """Generate a unique cache key for a request"""
        key_parts = [url]
        if params:
        key_parts.append(json.dumps(params, sort_keys=True))
        return hashlib.md5(''.join(key_parts).encode()).hexdigest()


        async def fetch_data_async(url: str, timeout: int = REQUEST_TIMEOUT,  # Changed function name and parameter
                          headers: Dict[str, str] = None,
                           # Changed parameter name
                           use_cache: bool = True) -> Dict[str, Any]:
                           """
    Asynchronously fetch data from an external URL with caching

    Args:
        url: The URL to fetch data from
        timeout: Request timeout in seconds
        headers: Optional request headers
        use_cache: Whether to use caching

    Returns:
        Parsed JSON response
    """
                           if headers is None:
                           headers = {"User-Agent": random.choice(BROWSER_AGENTS)}

                               cache_key = create_cache_key(url, {"timeout": timeout, "headers": headers})

                               if use_cache:
        cached_content= retrieve_from_cache(cache_key)
        if cached_content:
            return cached_content

                               try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout, headers=headers) as response:
                if response.status >= 400:
                    # Changed key
                    return {"error": f"HTTP Error: {response.status}", "code": response.status}

                content= await response.json()  # Changed variable name

                if use_cache:
                    store_in_cache(cache_key, content)

                return content
                               except aiohttp.ClientError as e:
        return {"error": f"Request failed: {str(e)}"}
                               except asyncio.TimeoutError:
        return {"error": f"Request timed out after {timeout} seconds"}
                               except json.JSONDecodeError:
        return {"error": "Invalid JSON response"}
                               except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


                               def fetch_data(url: str, timeout: int = REQUEST_TIMEOUT,  # Changed function name and parameter
               headers: Dict[str, str] = None,
               # Changed parameter name
               use_cache: bool = True) -> Dict[str, Any]:
               """
    Synchronous wrapper for fetch_data_async
    """
               return asyncio.run(fetch_data_async(url, timeout, headers, use_cache))


               # Changed function name
               def fetch_multiple(urls: List[str], timeout: int = REQUEST_TIMEOUT) -> List[Dict[str, Any]]:
               """
    Fetch data from multiple URLs in parallel
    """
               async def _fetch_all():
               tasks = [fetch_data_async(url, timeout) for url in urls]
               return await asyncio.gather(*tasks)

               return asyncio.run(_fetch_all())


@ lru_cache(maxsize=128)
                   # Changed function name
                   def load_product_templates() -> Dict[str, Dict[str, Any]]:
    """
    Get product templates for more realistic data generation
    """
    templates= {
        ProductCategory.ELECTRONICS: {  # Changed from DataCategory
            # Changed from name_format
            "name_pattern": "{brand} {product} {model}",
            # Changed from brands
            "manufacturers": ["Sony", "Samsung", "Apple", "LG", "Dell"],
            # Changed from products
            "items": ["TV", "Laptop", "Smartphone", "Tablet", "Headphones"],
            "price_bounds": (199.99, 1999.99),  # Changed from price_range
            # Changed from common_tags
            "typical_tags": ["tech", "gadget", "wireless", "smart", "portable"]
        },
        ProductCategory.CLOTHING: {  # Changed from DataCategory
            # Changed from name_format
            "name_pattern": "{brand} {type} {style}",
            # Changed from brands
            "manufacturers": ["Nike", "Adidas", "Zara", "H&M", "Levi's"],
            # Changed from products
            "items": ["Shirt", "Pants", "Jacket", "Dress", "Shoes"],
            "price_bounds": (19.99, 299.99),  # Changed from price_range
            # Changed from common_tags
            "typical_tags": ["casual", "formal", "summer", "winter", "sport"]
        },
        # More categories...
    }

        # Add default template for any missing categories
        for category in ProductCategory:  # Changed from DataCategory
    if category not in templates:
    templates[category] = {
               # Changed from name_format
               "name_pattern": "Generic {category} Item",
                # Changed from brands
                "manufacturers": ["Brand A", "Brand B", "Brand C"],
                # Changed from products
                "items": ["Basic", "Premium", "Standard"],
                "price_bounds": (9.99, 99.99),  # Changed from price_range
                # Changed from common_tags
                "typical_tags": ["basic", "standard", "quality"]
            }

            return templates


            def create_sample_data(config: Union[GeneratorConfig, Dict[str, Any], int] = 10,  # Changed function name and parameter
                      # Changed parameter name
                      random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
                       """
    Generate random sample data with extensive customization

    Args:
        config: Generation options or simple count
        random_seed: Optional random seed for reproducible results

    Returns:
        List of generated data items
    """
                       # Handle different input types
                       if isinstance(config, int):
        # Changed parameter names
        config= GeneratorConfig(item_count=config, random_seed=random_seed)
                           elif isinstance(config, dict):
        config= GeneratorConfig(**config)

                           # Set random seed if provided
                           if config.random_seed is not None:  # Changed from seed
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

                           # Get available categories
                           available_types = list(ProductCategory)  # Changed variable and class names
                           if config.product_types:  # Changed from categories
        available_types= [
            cat for cat in available_types if cat in config.product_types]

            # Get product templates
            templates = load_product_templates()  # Changed function name

            # Generate data
            result = []
            for i in range(config.item_count):  # Changed from count
            # Select category
            product_type = random.choice(available_types)  # Changed variable name
            template = templates[product_type]

            # Generate price within range
            price_bounds = config.price_bounds if config.price_bounds else PriceBounds(  # Changed from price_range
            # Changed from min_price and price_range
            min_value = template["price_bounds"][0],
            # Changed from max_price and price_range
            max_value = template["price_bounds"][1]
        )
            # Changed from min_price/max_price
            price = round(random.uniform(
            price_bounds.min_value, price_bounds.max_value), 2)

            # Generate rating
            rating = round(max(1.0, min(5.0, np.random.normal(3.5, 0.8))), 1)

            # Skip items below rating threshold if specified
            if config.min_rating and rating < config.min_rating:  # Changed from rating_threshold
            continue

            # Generate tags
            all_possible_tags = template["typical_tags"] + ["new", "sale",
                                                       "limited", "popular", "exclusive"]  # Changed from common_tags
                                                        num_tags = random.randint(1, 4)
                                                            tags = random.sample(all_possible_tags, k=min(
            num_tags, len(all_possible_tags)))

                                                            # Apply tags filter if specified
                                                            # Changed from tags_filter
                                                            if config.tag_filters and not any(tag in config.tag_filters for tag in tags):
            # Convert set to list for random.choice
            tags.append(random.choice(list(config.tag_filters)))

                                                            # Generate name
                                                            manufacturer = random.choice(
            template["manufacturers"])  # Changed from brands
            product = random.choice(template["items"])  # Changed from products
            name = template["name_pattern"].format(  # Changed from name_format
            brand = manufacturer,  # Changed from brand
            product = product,
            model = f"Model-{random.randint(100, 999)}",
            type = product,
            style = random.choice(["Classic", "Modern", "Vintage", "Sport"]),
            category = product_type.value.title()  # Changed from category
        )

            # Create item
            item = {
           "id": i + 1,
            "name": name,
            "price": price,  # Changed from value
            "category": product_type.value,  # Changed from category
            "rating": rating,
            "tags": tags,
            "available": random.random() > 0.2,  # Changed from in_stock
            # Changed from created_at
            "created_date": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
            # Changed from discount_percent
            "discount": random.choice([0, 0, 0, 10, 15, 20, 25, 30]),
        }

            # Add metadata if requested
            if config.with_metadata:  # Changed from include_metadata
        item["details"] = {  # Changed from metadata
               "manufacturer": manufacturer,  # Changed from brand
                "product_type": product,
                "weight_kg": round(random.uniform(0.1, 10.0), 2),
                "dimensions": {
                    "width_cm": round(random.uniform(5, 100), 1),
                    "height_cm": round(random.uniform(5, 100), 1),
                    "depth_cm": round(random.uniform(5, 100), 1),
                },
                # Changed from manufacturer_country
                "origin_country": random.choice(["USA", "China", "Japan", "Germany", "Vietnam"]),
                "warranty_months": random.choice([0, 12, 24, 36])
            }

            result.append(item)

            return result


            class ProductData(BaseModel):  # Changed from GeneratedData
            id: int
            name: str
            price: float  # Changed from value
            category: str = Field(..., description="Product category")
            rating: float = Field(..., ge=1, le=5, description="Product rating (1-5)")
            tags: List[str]
            # Changed from in_stock
            available: bool = Field(..., description="Availability status")
            # Changed from created_at
            created_date: str = Field(..., description="ISO format creation date")
            # Changed from discount_percent
            discount: int = Field(0, ge=0, le=100, description="Discount percentage")
            details: Optional[Dict[str, Any]] = Field(
        None, description ="Additional product metadata")  # Changed from metadata

    @ property
        def final_price(self) -> float:  # Changed from discounted_price
        """Calculate the discounted price"""
            return round(self.price * (1 - self.discount / 100), 2)  # Changed from value and discount_percent

    @ property
        def is_new_arrival(self) -> bool:  # Changed from is_new
        """Check if product is new (created within last 30 days)"""
            created = datetime.fromisoformat(
           self.created_date)  # Changed from created_at
            return (datetime.now() - created).days <= 30

    @ property
            def has_discount(self) -> bool:  # Changed from is_on_sale
            """Check if product is on sale"""
            return self.discount > 0  # Changed from discount_percent


            class PriceStats(BaseModel):  # Changed from PriceMetrics
            mean: float  # Changed from average
            median: float
            minimum: float  # Changed from min
            maximum: float  # Changed from max
            std_dev: float  # Changed from standard_deviation
            percentile_values: Dict[str, float]  # Changed from percentiles


            class AnalysisResult(BaseModel):  # Changed from DataAnalysis
            # Changed from total_items
            item_count: int = Field(..., description="Total number of items analyzed")
            # Changed from price_metrics
            price_stats: PriceStats = Field(..., description="Price statistics")
            # Changed from category_distribution
            category_counts: Dict[str,
                         int] = Field(..., description="Items per category")
                          # Changed from availability
                          stock_status: Dict[str, int] = Field(...,
                                         description ="Available vs unavailable counts")
                                         # Changed from top_tags
                                         popular_tags: List[str] = Field(...,
                                    description ="Most frequently used tags")
                                    # Changed from rating_distribution
                                    rating_counts: Dict[str,
                        int]= Field(..., description="Rating distribution")
                        # Changed from discount_analysis
                        discount_stats: Dict[str,
                        Any] = Field(..., description="Discount statistics")

                         class Config:
                         schema_extra = {
            "example": {
                "item_count": 50,  # Changed from total_items
                "price_stats": {  # Changed from price_metrics
                    "mean": 199.99,  # Changed from average
                    "median": 149.99,
                    "minimum": 9.99,  # Changed from min
                    "maximum": 999.99,  # Changed from max
                    "std_dev": 150.5,  # Changed from standard_deviation
                    # Changed from percentiles
                    "percentile_values": {"25": 99.99, "75": 299.99, "90": 499.99}
                },
                # Changed from category_distribution
                "category_counts": {"electronics": 15, "clothing": 20, "books": 15},
                # Changed from availability and keys
                "stock_status": {"available": 40, "unavailable": 10},
                # Changed from top_tags
                "popular_tags": ["sale", "new", "popular"],
                # Changed from rating_distribution
                "rating_counts": {"1-2": 5, "2-3": 10, "3-4": 20, "4-5": 15},
                "discount_stats": {  # Changed from discount_analysis
                    "discounted_items": 15,  # Changed from items_on_sale
                    "mean_discount": 18.5,  # Changed from average_discount
                    "total_savings": 550.75
                }
            }
        }


        # Changed function and return type names
        def analyze_product_data(data: List[Dict[str, Any]]) -> AnalysisResult:
        """
    Perform comprehensive analysis on data

    Args:
        data: List of data items to analyze

    Returns:
        Detailed statistical analysis
    """
        if not data:
        raise ValueError("Cannot analyze empty dataset")

        # Extract values for analysis
        prices = [item["price"] for item in data]  # Changed from values
        categories = [item["category"] for item in data]
        all_tags = [tag for item in data for tag in item["tags"]]
        ratings = [item["rating"] for item in data]
        discounts = [item["discount"]
                for item in data]  # Changed from discount_percent

                 # Calculate category distribution
                 category_counts = {cat: categories.count(
        cat) for cat in set(categories)}  # Changed variable name

                     # Calculate in-stock counts
                     # Changed from in_stock
                     available_count = sum(1 for item in data if item["available"])

                     # Find most common tags
                     tag_counter = {}
    for tag in all_tags:
                tag_counter[tag] = tag_counter.get(tag, 0) + 1
                     top_tags = sorted(tag_counter.items(),
                      key =lambda x: x[1], reverse=True)[:5]
                      # Changed from top_tag_names
                      popular_tag_names = [tag for tag, _ in top_tags]

                      # Calculate rating distribution
                      rating_counts = {  # Changed from rating_dist
        "1-2": sum(1 for r in ratings if 1 <= r < 2),
        "2-3": sum(1 for r in ratings if 2 <= r < 3),
        "3-4": sum(1 for r in ratings if 3 <= r < 4),
        "4-5": sum(1 for r in ratings if 4 <= r <= 5)
    }

        # Calculate discount analysis
        # Changed from items_on_sale
        discounted_items = sum(1 for d in discounts if d > 0)
        # Changed from avg_discount
        mean_discount = sum(discounts) / len(discounts) if discounts else 0
        # Changed from value and discount_percent
        total_savings = sum(
       item["price"] * item["discount"] / 100 for item in data)

        # Calculate percentiles
        percentiles = {
        "25": np.percentile(prices, 25),  # Changed from values
        "75": np.percentile(prices, 75),  # Changed from values
        "90": np.percentile(prices, 90)  # Changed from values
    }

        # Create price metrics
        price_stats = PriceStats(  # Changed from price_metrics
       # Changed from average and values
        mean = round(sum(prices) / len(prices), 2),
        # Changed from values
        median = round(sorted(prices)[len(prices) // 2], 2),
        minimum = min(prices),  # Changed from min and values
        maximum = max(prices),  # Changed from max and values
        # Changed from standard_deviation and values
        std_dev = round(np.std(prices), 2),
        # Changed from percentiles
        percentile_values = {k: round(v, 2) for k, v in percentiles.items()}
    )

        # Create analysis result
        analysis = AnalysisResult(  # Changed from DataAnalysis
        item_count = len(data),  # Changed from total_items
        price_stats = price_stats,  # Changed from price_metrics
        category_counts = category_counts,  # Changed from category_distribution
        stock_status ={  # Changed from availability
            "available": available_count,  # Changed from in_stock
            # Changed from out_of_stock
            "unavailable": len(data) - available_count
        },
        popular_tags = popular_tag_names,  # Changed from top_tags
        rating_counts = rating_counts,  # Changed from rating_distribution
        discount_stats ={  # Changed from discount_analysis
            "discounted_items": discounted_items,  # Changed from items_on_sale
            # Changed from average_discount
            "mean_discount": round(mean_discount, 1),
            "total_savings": round(total_savings, 2)
        }
    )

        return analysis
