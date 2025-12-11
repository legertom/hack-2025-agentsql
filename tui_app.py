#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent.absolute()
src_path = script_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

# Load environment variables first
load_dotenv(override=True)

from agent_sql.tui import AgentSQLApp

def main():
    app = AgentSQLApp()
    app.run()

if __name__ == "__main__":
    main()
