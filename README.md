# AgentSQL - Agentic SQL Transform with LangGraph

A LangGraph-based agent that generates and refines DuckDB SQL to transform raw input data into target schemas.

## Quick Start

1. **Setup environment:**
   ```bash
   uv sync
   ```

2. **Configure keys:**
   Copy the example environment file and add your `OPENAI_API_KEY`:
   ```bash
   cp .env.example .env
   # Edit .env and paste your key
   ```

3. **Run the Local Workbench:**
   Launch the Streamlit web interface to run scenarios and view logs:
   ```bash
   uv run streamlit run src/agent_sql/web_app.py
   ```
   Access the app at `http://localhost:8501`.

4. **(Optional) Run CLI example:**
   ```bash
   uv run example.py
   ```


## Project Structure

```
AgentSQL/
├── src/agent_sql/          # Main package
│   ├── core.py             # DuckDB runner, diff, schema extraction
│   ├── prompts.py          # Prompt builders (SQL generator + evaluator)
│   ├── agent.py            # Agent state, nodes, graph
│   └── extensions.py       # Extension features (multi-table, vector search)
├── notebooks/              # Prototype & experiments
├── testdata/               # Test scenarios
└── docs/                   # Documentation
```

## Workstreams

### 1. Prompt Engineering

**Location:** `src/agent_sql/prompts.py`

**Tasks:**
- SQL Generator: Add few-shot examples, field mapping guides
- Evaluator: Enhance with more examples, debugging guides

**How to extend:**
```python
# In prompts.py
class FewShotSQLGeneratorPromptBuilder(SQLGeneratorPromptBuilder):
    def build_prompt(self, scenario, ..., few_shot_examples=None, **kwargs):
        # Add few-shot examples to the prompt
        ...
```

### 2. Complex Features

**Location:** `src/agent_sql/extensions.py`

**Tasks:**
- Multi-table input support
- Multi-step transforms
- Vector search for SQL examples

**How to implement:**
```python
# In extensions.py
def build_multi_table_context(scenario: ScenarioConfig) -> str:
    # Implement multi-table logic
    ...
```

### 3. CLI/Web UI

**Tasks:**
- CLI: Command-line interface for running scenarios
- Web UI: Streamlit/Gradio interface

