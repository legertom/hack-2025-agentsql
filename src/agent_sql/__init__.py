"""
AgentSQL - Agentic SQL Transform with LangGraph

A LangGraph-based agent that generates and refines DuckDB SQL 
to transform raw input data into target schemas.
"""

__version__ = "0.1.0"

# Core exports
from agent_sql.core import (
    ScenarioConfig,
    InputFile,
    ExpectedOutput,
    DuckDBRunResult,
    DiffResult,
    run_duckdb,
    run_and_compare,
    extract_schema,
    extract_expected_schema,
)

# Prompt builder exports
from agent_sql.prompts import (
    SQLGeneratorPromptBuilder,
    EvaluatorPromptBuilder,
    format_diff_for_prompt,
    format_input_for_evaluation,
)

# Agent exports
from agent_sql.agent import (
    AgentState,
    HistoryEntry,
    build_sql_agent_graph,
    run_sql_agent,
)

# Extension exports (for future use)
from agent_sql.extensions import (
    SQLExampleRetriever,
    FieldMappingGuide,
    build_multi_table_context,
    build_multi_transform_sql,
)

__all__ = [
    # Core
    "ScenarioConfig",
    "InputFile",
    "ExpectedOutput",
    "DuckDBRunResult",
    "DiffResult",
    "run_duckdb",
    "run_and_compare",
    "extract_schema",
    "extract_expected_schema",
    # Prompts
    "SQLGeneratorPromptBuilder",
    "EvaluatorPromptBuilder",
    "format_diff_for_prompt",
    "format_input_for_evaluation",
    # Agent
    "AgentState",
    "HistoryEntry",
    "build_sql_agent_graph",
    "run_sql_agent",
    # Extensions
    "SQLExampleRetriever",
    "FieldMappingGuide",
    "build_multi_table_context",
    "build_multi_transform_sql",
]

