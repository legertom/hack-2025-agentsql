"""
Streamlit-based local web UI for AgentSQL.

This module provides a browser-based workbench for:
- Browsing scenarios under `testdata/`
- Inspecting configuration (inputs, expected outputs)
- Running the AgentSQL LangGraph agent
- Viewing logs, generated SQL, and diffs

Implementation is incremental and follows the steps outlined in
implementation_plan.md. This file currently contains the skeleton
layout and entrypoint; scenario loading, logging, and result rendering
will be wired up in subsequent tasks.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Dict, Optional

import logging
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure `src` is on sys.path so `agent_sql` can be imported when running
# this file as a script (e.g., `streamlit run src/agent_sql/web_app.py`).
SRC_DIR = Path(__file__).parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Load environment variables from .env if present so OPENAI_API_KEY is available
load_dotenv(override=True)

# Import core types directly (these do not require an OpenAI key)
from agent_sql.core import ScenarioConfig, InputFile, ExpectedOutput  # type: ignore

# Try to import the agent runner only if an API key is configured
AGENT_IMPORT_ERROR: Optional[str] = None
run_sql_agent = None

if not os.getenv("OPENAI_API_KEY"):
    AGENT_IMPORT_ERROR = (
        "OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=... "
        "or export it in your shell before running the app."
    )
else:
    try:
        from agent_sql.agent import run_sql_agent as _run_sql_agent  # type: ignore

        run_sql_agent = _run_sql_agent
    except Exception as exc:  # pragma: no cover - defensive path
        AGENT_IMPORT_ERROR = f"Failed to initialize AgentSQL LLM client: {exc}"


# ---------------------------------------------------------------------------
# Filesystem configuration and scenario loading
# ---------------------------------------------------------------------------

SCRIPT_DIR = SRC_DIR.parent  # Repository root
TESTDATA_DIR = SCRIPT_DIR / "testdata"


def discover_scenarios(testdata_dir: Path = TESTDATA_DIR) -> Dict[str, Path]:
    """Discover valid scenarios under `testdata/`.

    A scenario is any directory directly under `testdata/` that contains an
    `input_files/` subdirectory, mirroring the behavior of the Textual TUI.
    """

    scenarios: Dict[str, Path] = {}
    if not testdata_dir.exists():
        return scenarios

    for scenario_dir in testdata_dir.iterdir():
        if not scenario_dir.is_dir():
            continue
        input_dir = scenario_dir / "input_files"
        if input_dir.exists():
            scenarios[scenario_dir.name] = scenario_dir

    return scenarios


def build_scenario_config(name: str, path: Path) -> ScenarioConfig:
    """Build a ScenarioConfig for the given scenario directory.

    Logic closely follows AgentSQLApp.load_scenario_config from the TUI:
    - Search for `expected_*.csv` under `input_files/`, then scenario root.
    - Treat the first match as the expected output file.
    - All other `*.csv` files under `input_files/` become InputFile entries.
    """

    input_dir = path / "input_files"

    # Find expected output first (it's often in input_files)
    expected_files = list(input_dir.glob("expected_*.csv"))
    if not expected_files:
        expected_files = list(path.glob("expected_*.csv"))

    if not expected_files:
        raise FileNotFoundError(f"No expected_*.csv found for scenario '{name}' in {path}")

    expected_path = expected_files[0]

    # Find input files (exclude expected output). Support both CSV and
    # JSON/JSONL inputs, mirroring the TUI and core engine behavior.
    inputs: list[InputFile] = []
    for f in input_dir.iterdir():
        if not f.is_file():
            continue

        if f.name == expected_path.name:
            continue

        ext = f.suffix.lower()
        if ext == ".csv":
            fmt = "csv"
        elif ext in (".json", ".jsonl"):
            fmt = "jsonl"  # handled alongside "json" in core.run_duckdb
        else:
            # Ignore unsupported formats (e.g., .sql helpers)
            continue

        inputs.append(
            InputFile(
                path=str(f.absolute()),
                table_name=f.stem.replace("input_", ""),  # same heuristic as TUI
                format=fmt,
            )
        )

    return ScenarioConfig(
        name=name,
        inputs=inputs,
        expected_output=ExpectedOutput(
            path=str(expected_path.absolute()),
            table_name="output",
        ),
        max_iters=5,
    )


def get_current_scenario() -> Optional[ScenarioConfig]:
    """Return the currently selected ScenarioConfig from session_state, if any."""

    return st.session_state.get("current_scenario_config")  # type: ignore[return-value]


def set_current_scenario(config: ScenarioConfig, path: Path) -> None:
    """Store the active ScenarioConfig and path in session_state for later use."""

    st.session_state["current_scenario_config"] = config
    st.session_state["current_scenario_path"] = str(path)


class StreamlitLogHandler(logging.Handler):
    """Logging handler that buffers log messages into Streamlit session state.

    This allows the Logs tab to render messages emitted during `run_sql_agent`.
    """

    def __init__(self, buffer_key: str = "agent_logs", placeholder=None) -> None:
        super().__init__()
        self.buffer_key = buffer_key
        self.placeholder = placeholder

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI plumbing
        msg = self.format(record)
        logs = st.session_state.setdefault(self.buffer_key, [])
        logs.append(msg)
        if self.placeholder:
            self.placeholder.code("\n".join(logs))


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def execute_agent_run(config: ScenarioConfig, log_placeholder) -> None:
    """Execute the agent run, streaming logs to the given placeholder."""
    
    if config is None:
        st.sidebar.error("No scenario configuration is available to run.")
        return

    # Clear previous logs for a fresh run
    st.session_state["agent_logs"] = []
    
    # Clear the placeholder initially
    log_placeholder.empty()

    # Attach a StreamlitLogHandler to the root logger
    root_logger = logging.getLogger()
    handler = StreamlitLogHandler(placeholder=log_placeholder)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    try:
        with st.spinner(f"Running AgentSQL for scenario '{config.name}'..."):
            try:
                result = run_sql_agent(config, verbose=True)
            except Exception as exc:  # pragma: no cover - UI-only error path
                # Store a minimal failure dict so downstream UI can react
                result = {"success": False, "error": str(exc)}
    finally:
        # Always detach handler to avoid duplicate logs across runs
        root_logger.removeHandler(handler)

    # Persist latest result in session_state for later tabs to use
    st.session_state["agent_result"] = result

    # Provide a brief status summary in the sidebar
    if result.get("success"):
        st.sidebar.success("Agent run completed successfully.")
    else:
        reason = result.get("reason") or result.get("error") or "unknown error"
        st.sidebar.error(f"Agent run failed or did not converge (reason: {reason}).")


def render_sidebar() -> tuple[Optional[ScenarioConfig], bool]:
    """Render the sidebar container with scenario selection.

    Returns:
        Tuple of (selected_config, run_button_clicked)
    """

    st.sidebar.header("Scenarios")

    # If the agent could not be initialized (e.g., missing OPENAI_API_KEY),
    # show a clear error and disable the Run button. Users can still browse
    # scenarios and configuration.
    if AGENT_IMPORT_ERROR is not None:
        st.sidebar.error(AGENT_IMPORT_ERROR)

    scenarios = discover_scenarios()
    if not scenarios:
        st.sidebar.error(
            f"No valid scenarios found under {TESTDATA_DIR} "
            "(expected directories with an 'input_files' subdirectory)."
        )
        st.sidebar.selectbox("Select scenario", options=["(no scenarios available)"])
        st.sidebar.button("Run Agent", disabled=True)
        return None, False

    scenario_names = sorted(scenarios.keys())

    default_index = 0
    if "current_scenario_name" in st.session_state:
        try:
            default_index = scenario_names.index(st.session_state["current_scenario_name"])
        except ValueError:
            default_index = 0

    selected_name = st.sidebar.selectbox(
        "Select scenario",
        options=scenario_names,
        index=default_index,
    )

    selected_path = scenarios[selected_name]
    config = None
    try:
        config = build_scenario_config(selected_name, selected_path)
        set_current_scenario(config, selected_path)
        st.session_state["current_scenario_name"] = selected_name
    except FileNotFoundError as exc:
        st.sidebar.error(str(exc))
        st.sidebar.button("Run Agent", disabled=True)
        return None, False

    # Run Agent button: invoke the LangGraph agent and store result
    # If the agent is unavailable (e.g., missing OPENAI_API_KEY), keep
    # the button disabled but still show configuration and logs.
    if run_sql_agent is None or AGENT_IMPORT_ERROR is not None:
        st.sidebar.button("Run Agent", disabled=True)
        return config, False

    run_clicked = st.sidebar.button("Run Agent")
    return config, run_clicked


def _render_config_tab(config: Optional[ScenarioConfig]) -> None:
    """Render configuration and data preview for the selected scenario."""

    st.markdown("### Scenario Configuration")

    if config is None:
        st.info("Select a scenario in the sidebar to see its configuration and data preview.")
        return

    st.markdown(f"#### Scenario: `{config.name}`")

    # Inputs section
    st.markdown("##### Inputs")
    if not config.inputs:
        st.warning("No input files configured for this scenario.")
    else:
        for inp in config.inputs:
            st.markdown(f"**{inp.table_name}** — `{inp.path}` ({inp.format})")
            try:
                if inp.format == "csv":
                    df_preview = pd.read_csv(inp.path, nrows=5, dtype=str)
                elif inp.format in ("json", "jsonl"):
                    # JSONL preview – treat each line as a record
                    df_preview = pd.read_json(inp.path, lines=True, nrows=5)
                else:
                    raise ValueError(f"Unsupported format for preview: {inp.format}")

                st.dataframe(df_preview, use_container_width=True)
            except Exception as exc:  # pragma: no cover - UI-only error path
                st.error(f"Could not preview `{inp.path}`: {exc}")

    # Expected output schema section
    st.markdown("##### Expected Output Schema")
    expected_path = config.expected_output.path
    try:
        schema_sample = pd.read_csv(expected_path, nrows=5)
        schema_rows = [
            {"column": col, "dtype": str(dtype)} for col, dtype in schema_sample.dtypes.items()
        ]
        if schema_rows:
            st.table(schema_rows)
        else:
            st.info("Expected output file is empty; no columns to display.")
    except Exception as exc:  # pragma: no cover - UI-only error path
        st.error(f"Could not infer schema from `{expected_path}`: {exc}")


def _render_result_tab(result: Optional[dict], config: Optional[ScenarioConfig]) -> None:
    """Render final SQL, reasoning, and diff summary for the last run."""

    st.markdown("### Resulting SQL")

    if result is None:
        st.info("Run the agent to see SQL and diffs here.")
        return

    success = bool(result.get("success"))
    sql = result.get("sql")
    reason = result.get("reason")
    error = result.get("error")
    iteration = result.get("iteration")
    evaluation = result.get("evaluation")
    last_diff = result.get("last_diff")

    if success:
        st.success("Agent reported success for the last run.")
    else:
        st.error("Agent did not converge or reported failure for the last run.")

    # SQL section
    st.markdown("#### Final SQL")
    if sql:
        st.code(sql, language="sql")
    else:
        st.info("No SQL was returned by the agent.")

    # Reason / evaluation section
    if reason or evaluation:
        st.markdown("#### Reason / Evaluation")
        if reason:
            st.write(f"Reason: {reason}")
        if evaluation:
            st.write(evaluation)

    if iteration is not None:
        st.markdown("#### Iteration Summary")
        st.write(f"Iterations: {iteration}")

    # Diff summary (if available)
    if last_diff is not None:
        st.markdown("#### Diff Summary")
        try:
            summary = last_diff.to_summary()  # type: ignore[attr-defined]
        except Exception:
            summary = str(last_diff)
        st.code(summary)
    elif not success:
        # Failed but no diff – fall back to error text only
        if error:
            st.markdown("#### Error Detail")
            st.code(error)


def render_main_panel(config: Optional[ScenarioConfig], run_clicked: bool) -> None:
    """Render the main content area with tabs for config, logs, and SQL."""

    st.title("AgentSQL Local Workbench")

    result = st.session_state.get("agent_result")

    # Use tabs to mirror the TUI structure conceptually
    config_tab, logs_tab, sql_tab = st.tabs(["Config", "Logs & Output", "Result SQL"])

    with config_tab:
        _render_config_tab(config)

    with logs_tab:
        st.markdown("### Agent Logs")
        log_placeholder = st.empty()
        
        # Always show the latest logs (either from the just-finished run or previous)
        logs = st.session_state.get("agent_logs", [])
        if logs:
            log_placeholder.code("\n".join(logs))
        elif not run_clicked:
            st.info("Run the agent to see logs here.")

    with sql_tab:
        _render_result_tab(result, config)

    # If running, execute the agent. We do this in the sidebar context so the
    # spinner is visible regardless of which tab is active, but we pass the
    # log_placeholder so logs stream into the Logs tab.
    if run_clicked and config:
        with st.sidebar:
            execute_agent_run(config, log_placeholder)


def main() -> None:
    """Main entrypoint for the Streamlit app.

    This function sets up global page configuration and delegates to
    sidebar and main panel renderers. Business logic is added in later
    tasks to keep changes reviewable and aligned with the implementation plan.
    """

    st.set_page_config(page_title="AgentSQL", layout="wide")

    config, run_clicked = render_sidebar()
    render_main_panel(config, run_clicked)


if __name__ == "__main__":
    # When run directly (e.g., `python src/agent_sql/web_app.py`), this will
    # render a static version of the layout. In normal usage, Streamlit runs
    # this module via `streamlit run src/agent_sql/web_app.py`.
    main()
