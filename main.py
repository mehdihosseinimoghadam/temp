from fastapi import FastAPI, HTTPException, Query, Depends, Path, Body, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Optional, Dict, Any, Union, Callable
import statistics
from collections import Counter
import logging
import json
import csv
import io
import time
import asyncio
import uuid
from datetime import datetime, timedelta
import jwt
from pydantic import BaseModel, Field, validator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from contextlib import asynccontextmanager

from utils import (
    get_data, batch_get_data, generate_random_data, analyze_data,
    GeneratedData, DataAnalysis, DataGenerationOptions, DataCategory, PriceRange
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Background tasks queue
background_tasks = {}

# JWT Configuration
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Fake users database
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
        "role": "admin"
    },
    "user": {
        "username": "user",
        "full_name": "Normal User",
        "email": "user@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
        "role": "user"
    }
}

# Models for authentication


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: str = "user"


class UserInDB(User):
    hashed_password: str

# Models for API


class ErrorResponse(BaseModel):
    detail: str


class SuccessResponse(BaseModel):
    message: str
    data: Optional[Any] = None


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Any] = None
    created_at: str
    completed_at: Optional[str] = None


class DataExportFormat(str):
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


class DataExportRequest(BaseModel):
    count: int = Field(100, ge=1, le=10000)
    format: str = DataExportFormat.JSON
    include_metadata: bool = False
    categories: Optional[List[DataCategory]] = None

    @validator('format')
    def validate_format(cls, v):
        if v not in [DataExportFormat.JSON, DataExportFormat.CSV, DataExportFormat.EXCEL]:
            raise ValueError(
                f"Format must be one of: {DataExportFormat.JSON}, {DataExportFormat.CSV}, {DataExportFormat.EXCEL}")
        return v


class ChartType(str):
    BAR = "bar"
    PIE = "pie"
    LINE = "line"
    SCATTER = "scatter"


class ChartRequest(BaseModel):
    data_count: int = Field(50, ge=10, le=1000)
    chart_type: str = ChartType.BAR
    category: Optional[str] = None

    @validator('chart_type')
    def validate_chart_type(cls, v):
        if v not in [ChartType.BAR, ChartType.PIE, ChartType.LINE, ChartType.SCATTER]:
            raise ValueError(
                f"Chart type must be one of: {ChartType.BAR}, {ChartType.PIE}, {ChartType.LINE}, {ChartType.SCATTER}")
        return v

# Startup and shutdown events


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("API server starting up")
    yield
    # Shutdown
    logger.info("API server shutting down")

app = FastAPI(
    title="Advanced Data Generator API",
    description="A comprehensive API for generating, analyzing, and visualizing random data",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication functions


def fake_hash_password(password: str):
    return "fakehashed" + password


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not fake_hash_password(password) == user.hashed_password:
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def admin_required(current_user: User = Depends(get_current_active_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Not enough permissions"
        )
    return current_user

# Background task functions


async def process_data_export(task_id: str, request: DataExportRequest):
    try:
        # Update task status
        background_tasks[task_id]["status"] = "processing"
        background_tasks[task_id]["progress"] = 0.1

        # Generate data
        logger.info(
            f"Generating {request.count} items for export in {request.format} format")
        options = DataGenerationOptions(
            count=request.count,
            include_metadata=request.include_metadata,
            categories=request.categories
        )

        # Simulate long processing
        for i in range(1, 10):
            await asyncio.sleep(0.5)  # Simulate work
            background_tasks[task_id]["progress"] = i / 10

        data = generate_random_data(options)

        # Process based on format
        if request.format == DataExportFormat.JSON:
            result = json.dumps(data, indent=2)
            content_type = "application/json"
            filename = f"data_export_{task_id}.json"

        elif request.format == DataExportFormat.CSV:
            # Flatten data for CSV
            flattened_data = []
            for item in data:
                flat_item = item.copy()
                if "metadata" in flat_item:
                    for k, v in flat_item["metadata"].items():
                        if isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                flat_item[f"metadata_{k}_{sub_k}"] = sub_v
                        else:
                            flat_item[f"metadata_{k}"] = v
                    del flat_item["metadata"]
                flat_item["tags"] = ",".join(flat_item["tags"])
                flattened_data.append(flat_item)

            # Create CSV
            output = io.StringIO()
            writer = csv.DictWriter(
                output, fieldnames=flattened_data[0].keys())
            writer.writeheader()
            writer.writerows(flattened_data)
            result = output.getvalue()
            content_type = "text/csv"
            filename = f"data_export_{task_id}.csv"

        elif request.format == DataExportFormat.EXCEL:
            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Handle nested data
            if request.include_metadata:
                metadata_df = pd.json_normalize(
                    [item.get("metadata", {}) for item in data])
                metadata_df.columns = [
                    f"metadata_{col}" for col in metadata_df.columns]
                df = pd.concat(
                    [df.drop("metadata", axis=1, errors="ignore"), metadata_df], axis=1)

            # Convert tags list to string
            df["tags"] = df["tags"].apply(lambda x: ",".join(x))

            # Create Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(writer, sheet_name="Data", index=False)

            result = output.getvalue()
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"data_export_{task_id}.xlsx"

        # Update task with result
        background_tasks[task_id]["status"] = "completed"
        background_tasks[task_id]["progress"] = 1.0
        background_tasks[task_id]["result"] = {
            "content": result,
            "content_type": content_type,
            "filename": filename
        }
        background_tasks[task_id]["completed_at"] = datetime.now().isoformat()

        logger.info(f"Export task {task_id} completed")

    except Exception as e:
        logger.error(f"Error in export task {task_id}: {str(e)}")
        background_tasks[task_id]["status"] = "failed"
        background_tasks[task_id]["error"] = str(e)
        background_tasks[task_id]["completed_at"] = datetime.now().isoformat()


async def generate_chart(task_id: str, request: ChartRequest):
    try:
        # Update task status
        background_tasks[task_id]["status"] = "processing"
        background_tasks[task_id]["progress"] = 0.1

        # Generate data
        logger.info(
            f"Generating chart of type {request.chart_type} with {request.data_count} items")
        data = generate_random_data(request.data_count)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Create figure
        plt.figure(figsize=(10, 6))

        # Generate chart based on type
        if request.chart_type == ChartType.BAR:
            if request.category:
                # Filter data by category
                filtered_data = df[df["category"] == request.category]
                plt.bar(filtered_data["name"], filtered_data["value"])
            else:
                plt.bar(df["name"], df["value"])
        elif request.chart_type == ChartType.PIE:
            plt.pie(df["value"], labels=df["name"], autopct="%1.1f%%")

        # Save figure to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        # Update task status
        background_tasks[task_id]["status"] = "completed"
        background_tasks[task_id]["progress"] = 1.0
        background_tasks[task_id]["result"] = {
            "content": buffer.getvalue(),
            "content_type": "image/png",
            "filename": f"chart_{task_id}.png"
        }
        background_tasks[task_id]["completed_at"] = datetime.now().isoformat()

        logger.info(f"Chart task {task_id} completed")

    except Exception as e:
        logger.error(f"Error in chart task {task_id}: {str(e)}")
        background_tasks[task_id]["status"] = "failed"
        background_tasks[task_id]["error"] = str(e)
        background_tasks[task_id]["completed_at"] = datetime.now().isoformat()


@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to the Data Generator API"}

            @app.get("/generate-data", response_model=List[GeneratedData], tags=["Data Generation"])
def generate_data(
    count: Optional[int] = Query(10, ge=1, le=100),
    seed: Optional[int] = Query(
        None, description="Random seed for reproducible results")
):
    """
    Generate random sample data.
    
    Parameters:
    - count: Number of data items to generate (default: 10, min: 1, max: 100)
    - seed: Random seed for reproducible results
    
    Returns:
