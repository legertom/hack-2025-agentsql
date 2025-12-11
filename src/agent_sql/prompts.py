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
    
    # For diff mismatches, provide structured feedback
    lines = ["Your previous SQL produced WRONG results - the output from your SQL does not match the expected output. Below is the full comparison of the input, expected, and actual outputs:"]
    lines.append("")
    lines.append("")
    
    if diff.missing_columns:
        lines.append(f"  - Missing columns (need to add): {diff.missing_columns}")
    
    if diff.extra_columns:
        lines.append(f"  - Extra columns (should remove): {diff.extra_columns}")
    
    if diff.row_count_expected != diff.row_count_actual:
        lines.append(f"  - Row count: expected {diff.row_count_expected}, got {diff.row_count_actual}")
    
    # Show full tables for small datasets to give complete context
    if diff.input_table and diff.expected_table and diff.actual_table:
        lines.append("\n### FULL COMPARISON (input → expected vs actual):")
        for i, (inp, exp, act) in enumerate(zip(diff.input_table, diff.expected_table, diff.actual_table)):
            exp_norm = {k: str(v).strip() for k, v in exp.items()}
            act_norm = {k: str(v).strip() for k, v in act.items()}
            
            if exp_norm != act_norm:
                lines.append(f"\n  Row {i} MISMATCH:")
                lines.append(f"    INPUT:    {inp}")
                lines.append(f"    EXPECTED: {exp}")
                lines.append(f"    ACTUAL:   {act}")
                # Highlight specific differences
                for col in exp:
                    if exp_norm.get(col) != act_norm.get(col):
                        lines.append(f"    → Column `{col}`: expected '{exp_norm.get(col)}' but got '{act_norm.get(col)}'")
            else:
                lines.append(f"\n  Row {i} OK: {exp}")
    
    elif diff.sample_mismatches:
        lines.append(f"\n### VALUE MISMATCHES ({len(diff.sample_mismatches)} samples):")
        for m in diff.sample_mismatches[:5]:
            row_idx = m.get('row_idx', '?')
            expected = m.get('expected', {})
            actual = m.get('actual', {})
            input_row = m.get('input', {})
            
            lines.append(f"\n  Row {row_idx}:")
            if input_row:
                lines.append(f"    INPUT:    {input_row}")
            lines.append(f"    EXPECTED: {expected}")
            lines.append(f"    ACTUAL:   {actual}")
            
            # Show only differing columns with clear explanation
            diff_cols = [k for k in expected if expected.get(k) != actual.get(k)]
            for col in diff_cols:
                # Try to find corresponding input column
                input_col = col.replace('_clever', '').replace('_raw', '')
                input_val = input_row.get(input_col) or input_row.get(col) or input_row.get(input_col + '_raw')
                if input_val:
                    lines.append(f"    → `{col}`: input was '{input_val}' → expected '{expected.get(col)}' but got '{actual.get(col)}'")
                else:
                    lines.append(f"    → `{col}`: expected '{expected.get(col)}' but got '{actual.get(col)}'")
    
    # Add strong guidance at the end
    lines.append("")
    # #lines.append("=" * 60)
    # lines.append("Your previous SQL produced WRONG results. You MUST write a DIFFERENT SQL query that fixes the issue. Make sure you look at all the values for the failing columns in the input and expected output data to understand the pattern for the expected data type and format. DO NOT output the same SQL again - it does not work!")
    #lines.append("=" * 60)
    
    return "\n".join(lines)


def format_input_for_evaluation(scenario: ScenarioConfig) -> str:
    """
    Format input data in a way that helps the evaluator understand the exact values.
    Shows the raw input with character-level detail for problematic rows.
    """
    input_df = load_input_as_df(scenario)
    if input_df is None or len(input_df) == 0:
        return "No input data available"
    
    lines = []
    lines.append("INPUT DATA (showing exact character content):")
    lines.append("")
    
    for idx, row in input_df.iterrows():
        lines.append(f"Row {idx}:")
        for col in input_df.columns:
            val = str(row[col])
            # Show the exact characters
            lines.append(f"  {col}: {repr(val)}")
            # # Show what TRIM would produce
            # trimmed = val.strip()
            # if trimmed != val:
            #     lines.append(f"    → After TRIM: {repr(trimmed)}")
            # # Highlight if it contains quotes
            # if '"' in trimmed or "'" in trimmed:
            #     lines.append(f"    → ⚠️ Contains quotes!")
        lines.append("")
    
    return "\n".join(lines)

