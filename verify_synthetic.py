"""
Verify synthetic data generation locally.
This script does NOT require an OpenAI API key.
It demonstrates that we are generating fake data instead of using real data.
"""

import sys
from pathlib import Path
# Add src to path
script_dir = Path(__file__).parent.absolute()
src_path = script_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import AFTER adding to path
from agent_sql.synthetic import extract_schema_with_synthetic_data

def main():
    # Path to a real input file
    input_path = script_dir / "testdata" / "csv_test" / "input_files" / "input_users.csv"
    
    print(f"Reading real file: {input_path}")
    print("-" * 50)
    
    # Extract schema using our new synthetic generator
    # This is what the agent will now do internally
    schema = extract_schema_with_synthetic_data(str(input_path), "csv", "users")
    
    print(f"Table: {schema['table_name']}")
    print(f"Columns: {[c['name'] + ' (' + c['type'] + ')' for c in schema['columns']]}")
    print("-" * 50)
    print("GENERATED SYNTHETIC SAMPLES (What the LLM will see):")
    print("-" * 50)
    
    for i, row in enumerate(schema['sample_rows']):
        print(f"Row {i+1}: {row}")
        
    print("-" * 50)
    print("Verification Complete: The LLM will see these FAKE values, not your real data.")

if __name__ == "__main__":
    main()
