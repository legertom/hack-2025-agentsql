"""
Synthetic data generator for privacy-safe SQL agent execution.
Generates fake data based on schema analysis to avoid sending real data to LLMs.
"""

import random
import string
import json
import pandas as pd
import duckdb
from typing import Any, List, Dict, Optional

def analyze_column(df: pd.DataFrame, col_name: str) -> Dict[str, Any]:
    """
    Analyze a column to determine its type and characteristics.
    """
    series = df[col_name]
    dtype = str(series.dtype)
    
    # Basic type detection
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_string = pd.api.types.is_string_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)
    
    stats = {
        "name": col_name,
        "dtype": dtype,
        "is_numeric": is_numeric,
        "is_string": is_string,
        "is_datetime": is_datetime,
        "has_nulls": series.isnull().any(),
    }
    
    if is_numeric:
        stats["min"] = float(series.min()) if not series.empty else 0
        stats["max"] = float(series.max()) if not series.empty else 100
        stats["is_int"] = pd.api.types.is_integer_dtype(series)
    
    if is_string:
        # Check for common patterns
        non_null = series.dropna().astype(str)
        if not non_null.empty:
            avg_len = non_null.str.len().mean()
            stats["avg_len"] = avg_len
            
            # Check if it looks like an email
            if non_null.str.contains("@").mean() > 0.5:
                stats["semantic_type"] = "email"
            # Check if it looks like a phone number (digits and dashes)
            elif non_null.str.contains(r"[\d-]").mean() > 0.8 and avg_len < 15:
                stats["semantic_type"] = "phone"
            # Check low cardinality (enum)
            elif series.nunique() < 10 and len(series) > 20:
                stats["semantic_type"] = "enum"
                stats["values"] = series.unique().tolist()
            else:
                stats["semantic_type"] = "text"
    
    return stats

def generate_value(stats: Dict[str, Any]) -> Any:
    """
    Generate a single fake value based on column stats.
    """
    # Handle nulls randomly (10% chance if column has nulls)
    if stats["has_nulls"] and random.random() < 0.1:
        return None
        
    if stats.get("semantic_type") == "enum":
        return random.choice(stats["values"])
        
    if stats["is_numeric"]:
        min_val = stats.get("min", 0)
        max_val = stats.get("max", 100)
        if stats.get("is_int"):
            return random.randint(int(min_val), int(max_val))
        else:
            return random.uniform(min_val, max_val)
            
    if stats.get("semantic_type") == "email":
        domains = ["example.com", "test.org", "fake.net"]
        name = "".join(random.choices(string.ascii_lowercase, k=8))
        return f"{name}@{random.choice(domains)}"
        
    if stats.get("semantic_type") == "phone":
        return f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        
    if stats["is_string"]:
        length = int(stats.get("avg_len", 10))
        return "".join(random.choices(string.ascii_letters + " ", k=length)).strip()
        
    # Fallback
    return "mock_value"

def generate_synthetic_df(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """
    Generate a synthetic DataFrame based on the patterns in the input DataFrame.
    """
    if df.empty:
        return pd.DataFrame(columns=df.columns)
        
    synthetic_data = {}
    
    for col in df.columns:
        stats = analyze_column(df, col)
        synthetic_data[col] = [generate_value(stats) for _ in range(n_rows)]
        
    return pd.DataFrame(synthetic_data)

def extract_schema_with_synthetic_data(file_path: str, file_format: str, table_name: str) -> dict:
    """
    Extract schema and generate synthetic sample rows.
    Replaces the old extract_schema that used real data.
    """
    con = duckdb.connect(":memory:")
    
    if file_format == "csv":
        # Read full file to analyze patterns (local only)
        query = f"SELECT * FROM read_csv('{file_path}', header=true, all_varchar=true)"
    elif file_format in ("json", "jsonl"):
        query = f"SELECT * FROM read_json_auto('{file_path}', records=true, sample_size=-1)"
    else:
        raise ValueError(f"Unsupported format: {file_format}")
    
    # Load real data into pandas for analysis
    real_df = con.execute(query).fetchdf()
    con.close()
    
    # Generate synthetic samples
    synthetic_df = generate_synthetic_df(real_df, n_rows=5)
    
    # Get column info (we use the real DF for types as it's the ground truth)
    columns = []
    for col in real_df.columns:
        col_type = str(real_df[col].dtype)
        if col_type == "object":
            col_type = "VARCHAR"
        columns.append({"name": col, "type": col_type})
    
    # Convert synthetic data to list of dicts
    sample_rows = synthetic_df.to_dict(orient='records')
    
    return {
        "table_name": table_name,
        "columns": columns,
        "sample_rows": sample_rows,
        "row_count": len(real_df),
        "is_synthetic": True
    }
