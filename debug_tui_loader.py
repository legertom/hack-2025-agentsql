from pathlib import Path

# Replicate the logic from TUI
TESTDATA_DIR = Path("testdata").absolute()

def debug_load_scenario(name, path):
    print(f"--- Debugging {name} ---")
    print(f"Path: {path}")
    
    input_dir = path / "input_files"
    print(f"Input Dir: {input_dir} (Exists: {input_dir.exists()})")
    
    # Logic from TUI
    expected_file = list(input_dir.glob("expected_*.csv"))
    if not expected_file:
        expected_file = list(path.glob("expected_*.csv"))
    
    if not expected_file:
        print("❌ No expected output found")
    else:
        print(f"✅ Expected output found: {expected_file[0].name}")
        expected_path = expected_file[0]

    inputs = []
    for f in input_dir.glob("*.csv"):
        if expected_file and f.name == expected_file[0].name:
            print(f"   Skipping expected file: {f.name}")
            continue
            
        print(f"   Found input: {f.name}")
        inputs.append(f)
        
    print(f"Total Inputs: {len(inputs)}")

if __name__ == "__main__":
    if not TESTDATA_DIR.exists():
        print("Testdata dir not found!")
    else:
        for scenario_dir in TESTDATA_DIR.iterdir():
            if scenario_dir.is_dir():
                debug_load_scenario(scenario_dir.name, scenario_dir)
