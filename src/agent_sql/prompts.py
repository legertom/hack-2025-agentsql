"""
Prompt builders for SQL generation and evaluation.

This module contains:
- SQLGeneratorPromptBuilder: Base prompt builder for SQL generation
- EvaluatorPromptBuilder: Base prompt builder for failure analysis
- Helper functions for formatting diffs and context
"""

from typing import Optional
from agent_sql.core import (
    ScenarioConfig,
    DiffResult,
    extract_schema,
    extract_expected_schema,
    format_schema_for_prompt,
    load_input_as_df,
)
import json


# ============================================================================
# SQL Generator Prompt Builder
# ============================================================================

SQL_SYSTEM_PROMPT = """You are a DuckDB SQL expert. Your task is to write a SQL query that transforms input data into the expected output format.

## RULES:
1. Output ONLY valid DuckDB SQL - no explanations, no markdown code blocks, just the SQL.
2. The SQL should be a single SELECT statement (CTEs are allowed).
3. Use the exact column names from the expected output.
4. Look at the values for each column in the input data compared to those in the expected output data to understand the type and format of the data we need to enforce.
5. Handle data type conversions and validation carefully.
7. Do NOT use DDL statements (CREATE TABLE, etc.) - just SELECT.

## DUCKDB TIPS:
- Use TRIM() to remove whitespace
- Use CASE WHEN for conditional logic
- Use CAST() for type conversions
- For struct fields: `column_name.field_name`
- Use `regexp_full_match(string, pattern)` for regex matching.
- Pay attention to integer vs float values when it comes to numeric values.
"""


class SQLGeneratorPromptBuilder:
    """
    Base prompt builder for SQL generation.
    
    This is the current implementation. Extend this class to add:
    - Few-shot examples (FewShotSQLGeneratorPromptBuilder)
    - Field mapping guides (MappingGuideSQLGeneratorPromptBuilder)
    """
    
    def build_prompt(
        self,
        scenario: ScenarioConfig,
        previous_sql: Optional[str] = None,
        last_diff: Optional[DiffResult] = None,
        evaluation: Optional[str] = None,
        iteration: int = 0,
        **kwargs  # For extensions like few_shot_examples, mapping_guide
    ) -> tuple[str, str]:
        """
        Build the system and user prompts for SQL generation.
        
        Returns:
            (system_prompt, user_prompt) tuple
        """
        system_prompt = SQL_SYSTEM_PROMPT
        
        # Build user prompt
        user_prompt = self._build_user_prompt(
            scenario=scenario,
            previous_sql=previous_sql,
            last_diff=last_diff,
            evaluation=evaluation,
            iteration=iteration,
            **kwargs
        )
        
        return system_prompt, user_prompt
    
    def _build_user_prompt(
        self,
        scenario: ScenarioConfig,
        previous_sql: Optional[str] = None,
        last_diff: Optional[DiffResult] = None,
        evaluation: Optional[str] = None,
        iteration: int = 0,
        **kwargs
    ) -> str:
        """Build the user prompt content."""
        lines = []
        
        # Add scenario context (schemas and samples)
        lines.append(self._build_scenario_context(scenario))
        lines.append("\n---\n")
        
        if iteration == 0:
            lines.append("## TASK")
            lines.append("Write a DuckDB SQL query that transforms the input table(s) to produce the expected output.")
            lines.append("Analyze the sample data carefully to understand the required transformations.")
            lines.append("NOTE: The sample data provided is SYNTHETIC/FAKE but matches the schema patterns of the real data.")
        else:
            lines.append(f"## ITERATION {iteration} - REFINEMENT NEEDED")
            lines.append("⚠️ YOUR PREVIOUS SQL FAILED. You MUST fix it based on the analysis below.")
            lines.append("")
            
            if previous_sql:
                lines.append("### Your Previous (INCORRECT) SQL:")
                lines.append(f"```sql\n{previous_sql}\n```")
                lines.append("")
            
            if evaluation:
                lines.append("### 🔍 Expert Analysis of Why It Failed:")
                lines.append(evaluation)
                lines.append("")
            
            if last_diff:
                lines.append("### Detailed Failure Feedback:")
                lines.append(format_diff_for_prompt(last_diff))
                lines.append("")
            
            lines.append("### TASK")
            lines.append("Based on the expert analysis above, write a DIFFERENT SQL query that fixes the root cause.")
            lines.append("Pay special attention to the FIX section in the expert analysis.")
            lines.append("Output only the corrected SQL - do NOT repeat the same query.")
        
        return "\n".join(lines)
    
    def _build_scenario_context(self, scenario: ScenarioConfig) -> str:
        """
        Build the full context for a scenario: input schemas, expected output schema.
        This is the "grounding" information that doesn't change between iterations.
        """
        lines = [f"## Scenario: {scenario.name}\n"]
        
        # Input tables
        lines.append("## INPUT TABLES\n")
        for inp in scenario.inputs:
            schema = extract_schema(inp.path, inp.format, inp.table_name)
            lines.append(format_schema_for_prompt(schema, is_input=True))
            lines.append("")
        
        # Expected output
        lines.append("## EXPECTED OUTPUT\n")
        expected_schema = extract_expected_schema(scenario.expected_output.path)
        expected_schema["table_name"] = scenario.expected_output.table_name
        lines.append(format_schema_for_prompt(expected_schema, is_input=False))
        
        return "\n".join(lines)


# ============================================================================
# Evaluator Prompt Builder
# ============================================================================

class EvaluatorPromptBuilder:
    """
    Base prompt builder for SQL failure evaluation.
    
    Extend this class to add:
    - More examples (EnhancedEvaluatorPromptBuilder)
    - Debugging guides (GuidedEvaluatorPromptBuilder)
    """
    
    def build_prompt(
        self,
        scenario: ScenarioConfig,
        sql: str,
        last_diff: DiffResult,
        **kwargs  # For examples, guides, etc.
    ) -> tuple[str, str]:
        """
        Build the system and user prompts for evaluation.
        
        Returns:
            (system_prompt, user_prompt) tuple
        """
        system_prompt = "You are a precise SQL debugging expert. You analyze failures and provide actionable, specific fixes."
        
        user_prompt = self._build_user_prompt(
            scenario=scenario,
            sql=sql,
            last_diff=last_diff,
            **kwargs
        )
        
        return system_prompt, user_prompt
    
    def _build_user_prompt(
        self,
        scenario: ScenarioConfig,
        sql: str,
        last_diff: DiffResult,
        **kwargs
    ) -> str:
        """Build the user prompt content for evaluation."""
        input_context = format_input_for_evaluation(scenario)
        
        eval_prompt = f"""You are a SQL debugging expert. Analyze why this SQL query failed.

SQL QUERY:
```sql
{sql}
```

FAILURE DETAILS:
{format_diff_for_prompt(last_diff)}

{input_context}

TASK:
Analyze the failure and provide specific guidance. Focus on:

1. ROOT CAUSE: What is the fundamental reason the validation/SQL query is failing? It's important to consider the values for other rows in the input data and the expected output data to establish a pattern adn single-out the root cause, especially if other rows are working correctly with the same SQL query. This will help you understand why a specific row is failing as opposed to other rows.
2. INPUT ANALYSIS: What does the input value ACTUALLY contain compared to the expected output? Show the exact characters.
3. WHY IT FAILS: Explain why the current SQL query logic incorrectly matches or processes the input.
4. SPECIFIC FIX: Provide concrete SQL code or pattern changes needed.

Output your analysis in this format:
ROOT CAUSE: [one sentence explanation]
INPUT VALUE: [what the actual input string contains, with repr() if helpful]
WHY IT FAILS: [why the regex/validation incorrectly processes it]
FIX: [specific SQL pattern or approach - be concrete, show code if helpful]
"""
        return eval_prompt


# ============================================================================
# Helper Functions
# ============================================================================

def format_diff_for_prompt(diff: DiffResult) -> str:
    """
    Convert a DiffResult into a detailed summary for the LLM prompt.
    Includes input data context to help the LLM understand WHY values should fail validation.
    """
    if diff.ok:
        return "✓ SUCCESS: Output matches expected exactly!"
    
    if diff.error:
        # For execution errors, provide the error message
        return f"✗ EXECUTION ERROR:\n{diff.error}"
    
    # For diff mismatches, provide structured feedback WITHOUT leaking real values
    lines = ["Your previous SQL produced WRONG results. Below is the summary of the mismatch:"]
    lines.append("")
    
    if diff.missing_columns:
        lines.append(f"  - Missing columns (need to add): {diff.missing_columns}")
    
    if diff.extra_columns:
        lines.append(f"  - Extra columns (should remove): {diff.extra_columns}")
    
    if diff.row_count_expected != diff.row_count_actual:
        lines.append(f"  - Row count: expected {diff.row_count_expected}, got {diff.row_count_actual}")
    
    if diff.sample_mismatches:
        lines.append(f"\n### VALUE MISMATCHES ({len(diff.sample_mismatches)} samples):")
        lines.append("Note: Actual values are hidden for privacy. Focus on the column logic.")
        
        # Group mismatches by column to give high-level feedback
        mismatched_cols = {}
        for m in diff.sample_mismatches:
            expected = m.get('expected', {})
            actual = m.get('actual', {})
            
            for col in expected:
                if expected.get(col) != actual.get(col):
                    if col not in mismatched_cols:
                        mismatched_cols[col] = 0
                    mismatched_cols[col] += 1
        
        for col, count in mismatched_cols.items():
             lines.append(f"  - Column `{col}`: {count} rows had incorrect values.")
             lines.append(f"    Check your logic for `{col}`. Ensure data types and transformations match the expected format.")

    return "\n".join(lines)


def format_input_for_evaluation(scenario: ScenarioConfig) -> str:
    """
    Format input data in a way that helps the evaluator understand the exact values.
    Uses SYNTHETIC data to avoid leaking real values.
    """
    # We need to generate synthetic data here too, we can't load real input
    # But wait, load_input_as_df in core.py loads real data.
    # We should update load_input_as_df or just generate fresh synthetic data here.
    # Let's generate fresh synthetic data to be safe.
    
    if not scenario.inputs:
        return "No input data available"
        
    inp = scenario.inputs[0]
    from agent_sql.synthetic import generate_synthetic_df
    import duckdb
    
    # Load real data just to analyze patterns (local only)
    con = duckdb.connect(":memory:")
    if inp.format == "csv":
        query = f"SELECT * FROM read_csv('{inp.path}', header=true, all_varchar=true)"
    else:
        query = f"SELECT * FROM read_json_auto('{inp.path}', records=true, sample_size=-1)"
        
    real_df = con.execute(query).fetchdf()
    con.close()
    
    # Generate synthetic
    synthetic_df = generate_synthetic_df(real_df, n_rows=5)
    
    lines = []
    lines.append("INPUT DATA (SYNTHETIC SAMPLES matching real patterns):")
    lines.append("")
    
    for idx, row in synthetic_df.iterrows():
        lines.append(f"Row {idx}:")
        for col in synthetic_df.columns:
            val = str(row[col])
            lines.append(f"  {col}: {repr(val)}")
        lines.append("")
    
    return "\n".join(lines)

