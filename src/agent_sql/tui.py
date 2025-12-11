from pathlib import Path
import sys
import threading
import asyncio
from typing import List, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Button, Static, ListView, ListItem, Label, RichLog, TabbedContent, TabPane
from textual.reactive import reactive
from textual.worker import Worker, WorkerState

from agent_sql.core import ScenarioConfig, InputFile, ExpectedOutput
from agent_sql.agent import run_sql_agent

# Constants
SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()
TESTDATA_DIR = SCRIPT_DIR / "testdata"

class ScenarioItem(ListItem):
    """A list item representing a scenario."""
    def __init__(self, name: str, path: Path) -> None:
        self.scenario_name = name
        self.scenario_path = path
        super().__init__(Label(name))

class AgentSQLApp(App):
    """A Textual app for AgentSQL."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 30;
        dock: left;
        height: 100%;
        background: $panel;
        border-right: vkey $accent;
    }

    #main-content {
        height: 100%;
        width: 1fr;
        padding: 1;
    }

    #scenario-list {
        height: 1fr;
    }

    .box {
        height: auto;
        border: solid $accent;
        padding: 1;
        margin-bottom: 1;
    }

    #log-view {
        height: 1fr;
        border: solid $secondary;
        background: $surface;
    }

    Button {
        width: 100%;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run_agent", "Run Agent"),
    ]

    current_scenario: reactive[Optional[ScenarioConfig]] = reactive(None)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        with Container(id="sidebar"):
            yield Label("Scenarios", classes="box")
            yield ListView(id="scenario-list")
            yield Button("Run Agent", id="run-btn", variant="primary", disabled=True)

        with Container(id="main-content"):
            with TabbedContent():
                with TabPane("Config", id="config-tab"):
                    yield Static(id="scenario-details", markup=True)
                with TabPane("Logs & Output", id="logs-tab"):
                    yield RichLog(id="log-view", highlight=True, markup=True)
                with TabPane("Result SQL", id="sql-tab"):
                    yield Static(id="sql-view", markup=True)

        yield Footer()

    def on_mount(self) -> None:
        """Load scenarios on startup."""
        self.load_scenarios()

    def load_scenarios(self) -> None:
        """Scan testdata for scenarios."""
        scenario_list = self.query_one("#scenario-list", ListView)
        
        if not TESTDATA_DIR.exists():
            self.query_one("#log-view", RichLog).write(f"[red]Error: testdata not found at {TESTDATA_DIR}[/]")
            return

        for scenario_dir in TESTDATA_DIR.iterdir():
            if scenario_dir.is_dir():
                # Check for input/expected files to confirm it's a valid scenario
                input_dir = scenario_dir / "input_files"
                if input_dir.exists():
                    scenario_list.append(ScenarioItem(scenario_dir.name, scenario_dir))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle scenario selection."""
        item = event.item
        if isinstance(item, ScenarioItem):
            self.load_scenario_config(item.scenario_name, item.scenario_path)
            self.query_one("#run-btn", Button).disabled = False

    def load_scenario_config(self, name: str, path: Path) -> None:
        """Load the configuration for a selected scenario."""
        input_dir = path / "input_files"
        
        # Find expected output first (it's often in input_files)
        expected_file = list(input_dir.glob("expected_*.csv"))
        if not expected_file:
            # Try looking in the root of scenario dir just in case
            expected_file = list(path.glob("expected_*.csv"))
            
        if not expected_file:
            self.query_one("#log-view", RichLog).write(f"[red]No expected output found for {name}[/]")
            return
            
        expected_path = expected_file[0]

        # Find input files (exclude expected output)
        inputs = []
        for f in input_dir.glob("*.csv"):
            if f.name == expected_path.name:
                continue
                
            inputs.append(InputFile(
                path=str(f.absolute()),
                table_name=f.stem.replace("input_", ""), # simple heuristic
                format="csv"
            ))
        
        self.current_scenario = ScenarioConfig(
            name=name,
            inputs=inputs,
            expected_output=ExpectedOutput(
                path=str(expected_path.absolute()),
                table_name="output"
            ),
            max_iters=5
        )
        
        # Update details view
        details = f"[b]Scenario:[/b] {name}\n\n"
        details += "[b]Inputs:[/b]\n"
        for inp in inputs:
            details += f"- {inp.path} (Table: {inp.table_name})\n"
        details += f"\n[b]Expected Output:[/b]\n- {expected_path}\n"
        
        self.query_one("#scenario-details", Static).update(details)
        self.query_one("#sql-view", Static).update("") # Clear previous SQL

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle run button press."""
        if event.button.id == "run-btn":
            self.action_run_agent()

    def action_run_agent(self) -> None:
        """Run the agent in a worker."""
        if not self.current_scenario:
            return
            
        self.query_one("#run-btn", Button).disabled = True
        self.query_one("#log-view", RichLog).clear()
        self.query_one("#log-view", RichLog).write(f"[bold green]Starting agent for {self.current_scenario.name}...[/]")
        
        # Switch to logs tab
        self.query_one(TabbedContent).active = "logs-tab"
        
        self.run_worker(self.run_agent_thread, thread=True)

    def run_agent_thread(self) -> None:
        """Run the agent (blocking) and capture output via logging."""
        import logging
        import traceback
        
        log_view = self.query_one("#log-view", RichLog)
        
        # Custom handler to write to RichLog
        class TextualHandler(logging.Handler):
            def __init__(self, app, log_view):
                super().__init__()
                self.app = app
                self.log_view = log_view
                
            def emit(self, record):
                msg = self.format(record)
                self.app.call_from_thread(self.log_view.write, msg)
        
        # Setup logger
        root_logger = logging.getLogger()
        handler = TextualHandler(self.app, log_view)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        
        result = {}
        
        try:
            # Check if we have inputs
            if not self.current_scenario.inputs:
                self.app.call_from_thread(log_view.write, "[red]❌ Error: No input files found for this scenario![/]")
                result = {"success": False}
            else:
                result = run_sql_agent(self.current_scenario, verbose=True)
        except Exception as e:
            # Capture the exception
            self.app.call_from_thread(log_view.write, f"[red]❌ Error: {str(e)}[/]")
            self.app.call_from_thread(log_view.write, traceback.format_exc())
            result = {"success": False}
        finally:
            # Cleanup handler
            root_logger.removeHandler(handler)
            
            # Update SQL view
            if result.get("success"):
                sql = result.get("sql", "-- No SQL generated")
                self.app.call_from_thread(self.update_sql_view, sql)
            else:
                self.app.call_from_thread(self.update_sql_view, "-- Agent failed or encountered an error. Check Logs tab.")
                
            self.app.call_from_thread(self.enable_button)

    def update_sql_view(self, sql: str) -> None:
        """Update the SQL view tab."""
        self.query_one("#sql-view", Static).update(f"```sql\n{sql}\n```")
        self.query_one(TabbedContent).active = "sql-tab"

    def enable_button(self) -> None:
        """Re-enable the run button."""
        self.query_one("#run-btn", Button).disabled = False

if __name__ == "__main__":
    app = AgentSQLApp()
    app.run()
