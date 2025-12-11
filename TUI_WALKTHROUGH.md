# AgentSQL TUI Walkthrough

I have implemented a Text User Interface (TUI) for AgentSQL using the `Textual` library. This allows you to browse scenarios, run the agent, and view results directly from your terminal.

## How to Run

1.  **Ensure dependencies are installed:**
    ```bash
    uv sync
    ```

2.  **Run the TUI app:**
    ```bash
    uv run tui_app.py
    ```

## Features

-   **Scenario Browser:** The sidebar lists all valid scenarios found in the `testdata` directory.
-   **Config View:** Select a scenario to see its input files and expected output path.
-   **Live Logs:** When you click "Run Agent", the logs (including the agent's reasoning and SQL generation) are streamed to the "Logs & Output" tab.
-   **SQL Result:** If the agent succeeds, the generated SQL is displayed in the "Result SQL" tab with syntax highlighting.

## Screenshots

*(Since I cannot take screenshots of the TUI running in this environment, please run the app to see it in action!)*

## Troubleshooting

-   **"testdata not found":** Ensure you are running the script from the project root.
-   **Agent fails:** Check the "Logs" tab for error messages. Ensure your `OPENAI_API_KEY` is set in `.env`.
