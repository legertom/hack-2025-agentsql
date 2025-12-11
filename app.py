import streamlit as st
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import io

# Add src to path
script_dir = Path(__file__).parent.absolute()
src_path = script_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load env vars
load_dotenv(override=True)

from agent_sql import (
    ScenarioConfig,
    InputFile,
    ExpectedOutput,
    run_sql_agent,
)

# Constants
TESTDATA_DIR = script_dir / "testdata"

st.set_page_config(
    page_title="AgentSQL Prototype",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AgentSQL Prototype")

# Sidebar: Scenario Selection
st.sidebar.header("Scenario Selection")

if not TESTDATA_DIR.exists():
    st.error(f"Testdata directory not found at {TESTDATA_DIR}")
    st.stop()

scenarios = [d.name for d in TESTDATA_DIR.iterdir() if d.is_dir() and (d / "input_files").exists()]
selected_scenario_name = st.sidebar.selectbox("Choose a scenario", scenarios)

if selected_scenario_name:
    scenario_path = TESTDATA_DIR / selected_scenario_name
    input_dir = scenario_path / "input_files"
    
    # Logic to load scenario config (mirrored from TUI)
    expected_file = list(input_dir.glob("expected_*.csv"))
    if not expected_file:
        expected_file = list(scenario_path.glob("expected_*.csv"))
    
    if not expected_file:
        st.error(f"No expected output found for {selected_scenario_name}")
        st.stop()
        
    expected_path = expected_file[0]
    
    inputs = []
    for f in input_dir.glob("*.csv"):
        if f.name == expected_path.name:
            continue
        inputs.append(InputFile(
            path=str(f.absolute()),
            table_name=f.stem.replace("input_", ""),
            format="csv"
        ))
        
    scenario_config = ScenarioConfig(
        name=selected_scenario_name,
        inputs=inputs,
        expected_output=ExpectedOutput(
            path=str(expected_path.absolute()),
            table_name="output"
        ),
        max_iters=5
    )
    
    # Display Config
    st.sidebar.subheader("Configuration")
    st.sidebar.markdown(f"**Expected Output:** `{expected_path.name}`")
    st.sidebar.markdown("**Input Files:**")
    for inp in inputs:
        st.sidebar.markdown(f"- `{Path(inp.path).name}` as `{inp.table_name}`")

    # Main Area
    st.subheader(f"Scenario: {selected_scenario_name}")
    
    # Initialize session state for logs if not exists
    if "logs" not in st.session_state:
        st.session_state.logs = []

    # Custom Handler for Streamlit
    class StreamlitLogHandler(logging.Handler):
        def __init__(self, container):
            super().__init__()
            self.container = container
            self.formatter = logging.Formatter('%(message)s')

        def emit(self, record):
            msg = self.format(record)
            st.session_state.logs.append(msg)
            # Update the container with all logs
            self.container.code("\n".join(st.session_state.logs))

    # Log display area
    st.subheader("Execution Logs")
    log_container = st.empty()
    
    # If we have previous logs, show them
    if st.session_state.logs:
        log_container.code("\n".join(st.session_state.logs))

    if st.button("Run Agent", type="primary"):
        # Clear previous logs on new run
        st.session_state.logs = []
        log_container.empty()
        
        with st.spinner("Agent is thinking..."):
            # Setup logging
            logger = logging.getLogger()
            handler = StreamlitLogHandler(log_container)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            
            try:
                # Run agent
                result = run_sql_agent(scenario_config, verbose=True)
                
                if result.get("success"):
                    st.success("Agent successfully generated SQL!")
                    st.markdown("### Generated SQL")
                    st.code(result.get("sql"), language="sql")
                    
                    st.markdown("### Reasoning")
                    st.write(result.get("reason"))
                else:
                    st.error("Agent failed to find a solution.")
                    if result.get("last_diff"):
                        st.error(f"Last Error: {result['last_diff'].error or 'Output mismatch'}")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Exception: {e}")
            finally:
                logger.removeHandler(handler)

else:
    st.info("Please select a scenario from the sidebar.")
