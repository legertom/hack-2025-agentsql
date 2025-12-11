# AgentSQL Architecture

## High-Level Overview

AgentSQL is a LangGraph-based agent that iteratively generates and refines DuckDB SQL to transform raw input data into target schemas.

## Core Components

### 1. Core Execution (`core.py`)

**ScenarioConfig:**
- Defines input files, expected output, and metadata
- Used throughout the agent lifecycle

**DuckDB Runner:**
- `run_duckdb()`: Loads input files and executes SQL
- `run_and_compare()`: Executes SQL and compares to expected output

**Diff Logic:**
- `compare_outputs()`: Compares actual vs expected output
- `DiffResult`: Structured comparison results

**Schema Extraction:**
- `extract_schema()`: Extracts column info and sample rows
- Used for building prompts

### 2. Prompts (`prompts.py`)

**SQLGeneratorPromptBuilder:**
- Builds prompts for SQL generation
- Includes scenario context, previous attempts, feedback
- Extensible for few-shot examples, mapping guides

**EvaluatorPromptBuilder:**
- Builds prompts for failure analysis
- Analyzes why SQL failed and suggests fixes
- Extensible for more examples, guides

### 3. Agent (`agent.py`)

**AgentState:**
- TypedDict holding scenario, SQL, diff, iteration, history
- Passed between nodes

**Nodes:**
- `sql_generator`: Generates/refines SQL using LLM
- `duckdb_node`: Executes SQL and compares output
- `evaluator`: Analyzes failures (optional)
- `controller`: Routes to END or next iteration

**Graph:**
```
START → sql_generator → duckdb_node → controller
                              ↓
                         (if not done)
                              ↓
                         evaluator → sql_generator
                              ↓
                         (if done)
                              ↓
                            END
```

### 4. Extensions (`extensions.py`)

Placeholder for complex features:
- Multi-table support
- Multi-step transforms
- Vector search for examples
- Field mapping guides

## Data Flow

1. **Initialization:**
   - Create `ScenarioConfig` with input/expected files
   - Initialize `AgentState` with scenario

2. **SQL Generation:**
   - `sql_generator` node builds prompt
   - Calls LLM to generate SQL
   - Updates state with new SQL

3. **Execution:**
   - `duckdb_node` runs SQL via `run_and_compare()`
   - Gets `DiffResult` with comparison
   - Updates state with diff

4. **Evaluation (if failed):**
   - `evaluator` node analyzes failure
   - Provides specific guidance
   - Updates state with evaluation

5. **Control:**
   - `controller` checks success or max iterations
   - Routes to END or back to evaluator/sql_generator

6. **Termination:**
   - Success: Returns final SQL
   - Max iterations: Returns last attempt
