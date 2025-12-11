"""
Core execution logic: DuckDB runner, diff comparison, schema extraction.

This module contains the foundational components for:
- Scenario configuration (ScenarioConfig, InputFile, ExpectedOutput)
- DuckDB execution (loading files, running SQL)
- Output comparison (diff logic)
- Schema extraction (for prompts)
"""

import duckdb
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any


# ============================================================================
# Scenario Configuration
# ============================================================================

@dataclass
class InputFile:
    """Represents an input file to be loaded into DuckDB."""
    path: str           # Path to the input file (CSV or JSONL)
    table_name: str     # Name of the table to create in DuckDB
    format: str = "csv" # "csv" or "jsonl"

@dataclass
class ExpectedOutput:
    """Represents the expected output file for comparison."""
    path: str           # Path to expected output CSV
    table_name: str     # Name used in final SELECT (for reference)

@dataclass
class ScenarioConfig:
    """Configuration for a single SQL transform scenario."""
    name: str
    inputs: list[InputFile]
    expected_output: ExpectedOutput
    max_iters: int = 5
    docs: list[str] = field(default_factory=list)  # Optional mapping docs


# ============================================================================
# DuckDB Runner
# ============================================================================

@dataclass
class DuckDBRunResult:
    """Result of running SQL in DuckDB."""
    success: bool
    result_df: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    tables_created: list[str] = field(default_factory=list)


def load_input_file(con: duckdb.DuckDBPyConnection, input_file: InputFile) -> str:
    """
    Load an input file into DuckDB as a table.
    Returns the SQL used for loading (for debugging).
    """
    if input_file.format == "csv":
        # Load CSV with all columns as VARCHAR (like the Go code does with raw columns)
        load_sql = f"""
            CREATE TABLE {input_file.table_name} AS 
            SELECT * FROM read_csv(
                '{input_file.path}',
                header=true,
                all_varchar=true
            )
        """
    elif input_file.format in ("json", "jsonl"):
        # Load JSONL with auto-detection
        load_sql = f"""
            CREATE TABLE {input_file.table_name} AS 
            SELECT * FROM read_json_auto(
                '{input_file.path}',
                records=true,
                sample_size=-1
            )
        """
    else:
        raise ValueError(f"Unsupported format: {input_file.format}")
    
    con.execute(load_sql)
    return load_sql


def run_duckdb(scenario: ScenarioConfig, sql: str) -> DuckDBRunResult:
    """
    Execute SQL against input files and return the result.
    
    The SQL should produce a result table/view that we can compare against expected output.
    The final statement should be a SELECT that produces the output rows.
    """
    tables_created = []
    
    try:
        # Create in-memory DuckDB connection
        con = duckdb.connect(":memory:")
        
        # Enable auto-install and auto-load of extensions (matches Go sis-config-transforms)
        con.execute("SET autoinstall_known_extensions=1;")
        con.execute("SET autoload_known_extensions=1;")
        
        # Load all input files as tables
        for input_file in scenario.inputs:
            load_input_file(con, input_file)
            tables_created.append(input_file.table_name)
        
        # Execute the candidate SQL
        result = con.execute(sql)
        result_df = result.fetchdf()
        
        con.close()
        
        return DuckDBRunResult(
            success=True,
            result_df=result_df,
            tables_created=tables_created
        )
        
    except Exception as e:
        return DuckDBRunResult(
            success=False,
            error=str(e),
            tables_created=tables_created
        )


# ============================================================================
# Diff / Comparison Logic
# ============================================================================

@dataclass
class DiffResult:
    """Result of comparing actual output to expected output."""
    ok: bool  # True if outputs match exactly
    error: Optional[str] = None  # Error message if DuckDB execution failed
    
    # Column differences
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    
    # Row count differences
    row_count_expected: int = 0
    row_count_actual: int = 0
    
    # Sample mismatched rows (capped for context)
    # Each mismatch now includes: row_idx, expected, actual, and input_row
    sample_mismatches: list[dict] = field(default_factory=list)
    max_sample_mismatches: int = 5
    
    # Full tables for small datasets (for detailed comparison)
    expected_table: Optional[list[dict]] = None
    actual_table: Optional[list[dict]] = None
    input_table: Optional[list[dict]] = None
    
    def to_summary(self) -> str:
        """Return a human-readable summary of the diff."""
        if self.ok:
            return "Output matches expected exactly!"
        
        if self.error:
            return f"Execution error: {self.error}"
        
        parts = ["Output does not match expected:"]
        
        if self.missing_columns:
            parts.append(f"  Missing columns: {self.missing_columns}")
        if self.extra_columns:
            parts.append(f"  Extra columns: {self.extra_columns}")
        if self.row_count_expected != self.row_count_actual:
            parts.append(f"  Row count: expected {self.row_count_expected}, got {self.row_count_actual}")
        if self.sample_mismatches:
            parts.append(f"  Sample mismatches ({len(self.sample_mismatches)} shown):")
            for m in self.sample_mismatches:
                parts.append(f"    Row {m.get('row_idx', '?')}: {m}")
        
        return "\n".join(parts)


def normalize_row(row: pd.Series) -> dict[str, str]:
    """Normalize a DataFrame row to a dict of strings for comparison."""
    normalized: dict[str, str] = {}
    for col in row.index:
        value = row[col]
        if isinstance(value, bool):
            normalized[col] = str(value).lower()
        elif pd.isna(value):
            normalized[col] = ""
        else:
            normalized[col] = str(value)
    return normalized


def compare_outputs(
    actual_df: pd.DataFrame, 
    expected_path: str, 
    input_df: Optional[pd.DataFrame] = None,
    max_samples: int = 5
) -> DiffResult:
    """
    Compare actual DataFrame output to expected CSV file.
    
    Args:
        actual_df: The output from running the SQL
        expected_path: Path to the expected output CSV
        input_df: Optional input DataFrame to include in mismatch context
        max_samples: Max number of mismatched rows to include
    
    Returns:
        DiffResult with comparison details and optional input context
    """
    # Load expected output
    expected_df = pd.read_csv(expected_path, dtype=str)
    
    # Normalize: strip whitespace from string columns
    for col in actual_df.columns:
        if actual_df[col].dtype == object:
            actual_df[col] = actual_df[col].astype(str).str.strip()
    for col in expected_df.columns:
        if expected_df[col].dtype == object:
            expected_df[col] = expected_df[col].astype(str).str.strip()
    
    result = DiffResult(
        ok=False,
        row_count_expected=len(expected_df),
        row_count_actual=len(actual_df),
        max_sample_mismatches=max_samples
    )
    
    # For small datasets, store the full tables for detailed comparison
    if len(expected_df) <= 10:
        result.expected_table = expected_df.to_dict(orient='records')
        result.actual_table = actual_df.to_dict(orient='records')
        if input_df is not None:
            result.input_table = input_df.to_dict(orient='records')
    
    # Check columns
    expected_cols = set(expected_df.columns)
    actual_cols = set(actual_df.columns)
    
    result.missing_columns = list(expected_cols - actual_cols)
    result.extra_columns = list(actual_cols - expected_cols)
    
    # If columns don't match, we can't do row-by-row comparison
    if result.missing_columns or result.extra_columns:
        return result
    
    # Sort both DataFrames by all columns for deterministic comparison
    actual_sorted = actual_df.sort_values(by=list(actual_df.columns)).reset_index(drop=True)
    expected_sorted = expected_df.sort_values(by=list(expected_df.columns)).reset_index(drop=True)
    
    # Compare row by row
    min_rows = min(len(actual_sorted), len(expected_sorted))
    mismatches = []
    
    for i in range(min_rows):
        actual_row = actual_sorted.iloc[i]
        expected_row = expected_sorted.iloc[i]
        actual_norm = normalize_row(actual_row)
        expected_norm = normalize_row(expected_row)
        
        if actual_norm != expected_norm:
            if len(mismatches) < max_samples:
                mismatch = {
                    "row_idx": i,
                    "expected": expected_norm,
                    "actual": actual_norm,
                }
                # Include input row if available (match by a key column or index)
                if input_df is not None and i < len(input_df):
                    mismatch["input"] = normalize_row(input_df.iloc[i])
                mismatches.append(mismatch)
    
    result.sample_mismatches = mismatches
    
    # Check if everything matches
    if (not result.missing_columns and 
        not result.extra_columns and 
        result.row_count_expected == result.row_count_actual and 
        not result.sample_mismatches):
        result.ok = True
    
    return result


def get_output_path_for_expected(expected_path: str) -> Path:
    """Get the output path for writing results based on expected path."""
    expected = Path(expected_path)
    output_dir = expected.parent.parent / "output_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_name = expected.name
    suffix = expected_name[len("expected_"):] if expected_name.startswith("expected_") else expected_name
    return output_dir / f"output_{suffix}"


def load_input_as_df(scenario: ScenarioConfig) -> Optional[pd.DataFrame]:
    """
    Load the first input file as a DataFrame for context in diffs.
    """
    if not scenario.inputs:
        return None
    
    inp = scenario.inputs[0]
    try:
        if inp.format == "csv":
            return pd.read_csv(inp.path, dtype=str)
        elif inp.format in ("json", "jsonl"):
            # For JSONL, load via DuckDB to handle nested structures
            con = duckdb.connect(":memory:")
            df = con.execute(f"SELECT * FROM read_json_auto('{inp.path}', records=true)").fetchdf()
            con.close()
            # Convert all columns to string for consistent comparison
            return df.astype(str)
    except Exception:
        return None
    return None


def run_and_compare(scenario: ScenarioConfig, sql: str) -> DiffResult:
    """
    Execute SQL against the scenario's input files and compare to expected output.
    
    This is the main "tool" function that the LangGraph agent will use.
    
    Args:
        scenario: The scenario configuration with input/expected file paths
        sql: The candidate SQL to execute (should be a SELECT that produces output rows)
    
    Returns:
        DiffResult with comparison details including input context
    """
    # Run DuckDB
    run_result = run_duckdb(scenario, sql)
    
    # If execution failed, return error result
    if not run_result.success:
        return DiffResult(
            ok=False,
            error=run_result.error
        )

    output_path = get_output_path_for_expected(scenario.expected_output.path)
    run_result.result_df.to_csv(output_path, index=False)

    actual_df = pd.read_csv(output_path, dtype=str)
    
    # Load input data for context
    input_df = load_input_as_df(scenario)
    
    # Compare outputs with input context
    return compare_outputs(
        actual_df=actual_df,
        expected_path=scenario.expected_output.path,
        input_df=input_df
    )


# ============================================================================
# Schema Extraction (for prompts)
# ============================================================================

def extract_schema(file_path: str, file_format: str, table_name: str) -> dict:
    """
    Extract schema (column names + inferred types) from an input file.
    USES SYNTHETIC DATA GENERATION to avoid reading real values.
    Returns a dict with table_name, columns, and sample rows (FAKE).
    """
    from agent_sql.synthetic import extract_schema_with_synthetic_data
    return extract_schema_with_synthetic_data(file_path, file_format, table_name)


def extract_expected_schema(file_path: str) -> dict:
    """
    Extract schema from expected output CSV file.
    USES SYNTHETIC DATA GENERATION.
    """
    from agent_sql.synthetic import extract_schema_with_synthetic_data
    # Re-use the logic, just don't need table name in the return as much
    return extract_schema_with_synthetic_data(file_path, "csv", "expected_output")


def format_schema_for_prompt(schema: dict, is_input: bool = True) -> str:
    """
    Format a schema dict into a concise string for the LLM prompt.
    """
    prefix = "Input" if is_input else "Expected Output"
    lines = [f"### {prefix} Table: `{schema.get('table_name', 'result')}`"]
    
    # Column info
    cols = ", ".join([f"`{c['name']}` ({c['type']})" for c in schema["columns"]])
    lines.append(f"Columns: {cols}")
    
    # Sample rows (limit to 5 for prompt brevity)
    lines.append("Sample rows (SYNTHETIC/FAKE DATA matching the schema patterns):")
    for i, row in enumerate(schema["sample_rows"][:5]):
        # Truncate long values but keep structure
        truncated = {}
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                truncated[k] = v
            else:
                str_v = str(v)
                truncated[k] = str_v[:50] + "..." if len(str_v) > 50 else v
        lines.append(f"  {i+1}. {json.dumps(truncated)}")
    
    return "\n".join(lines)

