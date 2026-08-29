from __future__ import annotations

import importlib
import platform
import sys

import config
from src.data_utils import ensure_output_dirs, write_json


REQUIRED_STAGE1 = ["pandas", "numpy", "sklearn", "openpyxl"]
REQUIRED_SINGLE_TARGET = ["xgboost", "optuna", "shap", "matplotlib", "seaborn"]
OPTIONAL_MTL = ["torch"]


def package_status(names):
    result = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            result[name] = {"available": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:
            result[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def main():
    ensure_output_dirs()
    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "workbook_exists": config.WORKBOOK_PATH.exists(),
        "workbook_path": config.WORKBOOK_PATH,
        "stage1": package_status(REQUIRED_STAGE1),
        "single_target": package_status(REQUIRED_SINGLE_TARGET),
        "mtl": package_status(OPTIONAL_MTL),
    }
    write_json(config.OUTPUT_DIRS["reports"] / "environment.json", payload)

    print(f"Python: {sys.executable}")
    print(f"Workbook: {config.WORKBOOK_PATH} | exists={config.WORKBOOK_PATH.exists()}")
    for section in ("stage1", "single_target", "mtl"):
        print(f"\n[{section}]")
        for name, status in payload[section].items():
            print(f"  {name:12s} {status}")

    missing_stage1 = [k for k, v in payload["stage1"].items() if not v["available"]]
    if missing_stage1:
        raise SystemExit(f"Stage 1 cannot run; missing packages: {missing_stage1}")
    if not config.WORKBOOK_PATH.exists():
        raise SystemExit("Input workbook does not exist.")
    if not payload["mtl"]["torch"]["available"]:
        print("\n[WARN] PyTorch is not installed. This does not block Stage 1 or single-target models;")
        print("       it must be resolved before the masked-loss MTL stage.")


if __name__ == "__main__":
    main()

