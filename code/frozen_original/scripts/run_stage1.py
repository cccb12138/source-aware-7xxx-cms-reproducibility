from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "00_environment_check.py",
    "01_data_audit.py",
    "02_build_source_folds.py",
]


for script in SCRIPTS:
    path = ROOT / script
    print("\n" + "=" * 80)
    print(f"RUNNING: {path.name}")
    print("=" * 80)
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)

print("\nSTAGE 1 COMPLETE: audit and source-group folds passed.")

