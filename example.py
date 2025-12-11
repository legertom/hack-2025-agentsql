"""
Example script to run the SQL agent.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path so we can import agent_sql
script_dir = Path(__file__).parent.absolute()
src_path = script_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import from the package
from agent_sql import (
    ScenarioConfig,
    InputFile,
    ExpectedOutput,
    run_sql_agent,
)

# Base path for testdata - use absolute path based on script location
SCRIPT_DIR = Path(__file__).parent.absolute()
TESTDATA_DIR = SCRIPT_DIR / "testdata"


def main():
    """Run the agent on a test scenario."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Set it in your .env file or export it:")
        print("   export OPENAI_API_KEY=your_key_here")
        return
    
    # Verify testdata directory exists
    if not TESTDATA_DIR.exists():
        print(f"❌ Error: testdata directory not found at {TESTDATA_DIR}")
        print(f"   Current working directory: {Path.cwd()}")
        print(f"   Script location: {SCRIPT_DIR}")
        return
    
    # Create CSV test scenario
    input_path = TESTDATA_DIR / "csv_test" / "input_files" / "input_users.csv"
    expected_path = TESTDATA_DIR / "csv_test" / "input_files" / "expected_users.csv"
    
    # Verify files exist
    if not input_path.exists():
        print(f"❌ Error: Input file not found at {input_path}")
        return
    if not expected_path.exists():
        print(f"❌ Error: Expected output file not found at {expected_path}")
        return
    
    csv_scenario = ScenarioConfig(
        name="csv_users_transform",
        inputs=[
            InputFile(
                path=str(input_path.absolute()),
                table_name="users_raw",
                format="csv"
            )
        ],
        expected_output=ExpectedOutput(
            path=str(expected_path.absolute()),
            table_name="users_output"
        ),
        max_iters=5
    )
    
    print("=" * 70)
    print("Running SQL Agent on CSV Scenario")
    print("=" * 70)
    
    # Run the agent
    result = run_sql_agent(csv_scenario, verbose=True)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Success: {result.get('success', False)}")
    print(f"Iterations: {result.get('iteration', 0)}")
    print(f"Reason: {result.get('reason', 'unknown')}")
    
    if result.get("success"):
        print("\n✅ Agent successfully generated SQL!")
        print(f"\nFinal SQL:\n{result.get('sql', 'N/A')}")
    else:
        print("\n❌ Agent did not find a solution")
        if result.get("last_diff"):
            print(f"\nLast error: {result['last_diff'].error or 'Output mismatch'}")


if __name__ == "__main__":
    main()

