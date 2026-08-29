from pathlib import Path

from PIL import Image
import pandas as pd


ROOT = Path(r"F:\CC\outputs\paper_results_v2")
FIG = ROOT / "figures"
DATA = ROOT / "figure_data"
checks = []


def add(name, passed, detail):
    checks.append({"Check": name, "Passed": bool(passed), "Detail": str(detail)})


pngs = sorted(FIG.glob("Fig*.png"))
pdfs = sorted(FIG.glob("Fig*.pdf"))
add("Eight_PNG_figures", len(pngs) == 8, len(pngs))
add("Eight_PDF_figures", len(pdfs) == 8, len(pdfs))
for image_path in pngs:
    with Image.open(image_path) as image:
        width, height = image.size
    add(image_path.stem, image_path.stat().st_size > 100_000 and width > 2000 and height > 1200,
        f"{width}x{height}; {image_path.stat().st_size} bytes")

scope = pd.read_csv(DATA / "Fig1a_scope_audit_counts.csv")
combos = pd.read_csv(DATA / "Fig1b_partial_label_combinations.csv")
add("Fig1_scope_counts", scope.set_index("Target").loc["UTS", "Scope_clean"] == 675 and scope.set_index("Target").loc["EL", "Scope_clean"] == 537, scope.to_dict("records"))
add("Fig1_partial_labels_sum_689", combos["size"].sum() == 689, combos["size"].sum())
add("Fig1_triple_complete_266", combos.loc[combos["Combination"].eq("YS+UTS+EL"), "size"].iloc[0] == 266, combos.to_dict("records"))

models = pd.read_csv(DATA / "Fig7a_model_comparison.csv")
features = pd.read_csv(DATA / "Fig7b_sparse_feature_decision.csv")
add("Fig7_ensemble_R2", abs(models.loc[models["Model"].eq("RF+XGB ensemble"), "R2"].iloc[0] - 0.5268211291102216) < 1e-10, models.to_dict("records"))
add("Fig7_refined5_R2", abs(features.loc[features["Label"].eq("Refined 5"), "R2"].iloc[0] - 0.5268211291102216) < 1e-10, features.to_dict("records"))

datasets = pd.read_csv(DATA / "Fig8a_dataset_counts.csv")
holdouts = pd.read_csv(DATA / "Fig8b_dataset_holdout_metrics.csv")
add("Fig8_dataset_rows_sum_675", datasets["Rows"].sum() == 675, datasets["Rows"].sum())
add("Fig8_six_dataset_holdouts", len(holdouts) == 6, len(holdouts))

result = pd.DataFrame(checks)
result.to_csv(ROOT / "paper_figures_v2_verification.csv", index=False, encoding="utf-8-sig")
if not result["Passed"].all():
    raise AssertionError(result.loc[~result["Passed"]].to_string(index=False))
print(f"All {len(result)} V2 figure checks passed.")
