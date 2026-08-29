from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import config
from src.data_utils import ensure_output_dirs


SETS = {
    "composition_core": config.PRIMARY_FIXED_FEATURES,
    "plus_solution_temp": config.PRIMARY_FIXED_FEATURES + ["Sol_Temp_C"],
    "plus_all_process": config.PRIMARY_FIXED_FEATURES + config.PROCESS_FEATURES,
}


def main():
    ensure_output_dirs(); out = config.OUTPUT_DIRS["single"] / "process_sensitivity_strict"
    out.mkdir(parents=True, exist_ok=True); preds = []
    for task in ("YS", "UTS", "EL"):
        df = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / f"{task}_with_outer_folds.csv")
        target = config.TARGET_COLUMNS[task]
        for set_name, features in SETS.items():
            df[features + [target]] = df[features + [target]].apply(pd.to_numeric, errors="coerce")
            for fold in sorted(df.Outer_Fold.unique()):
                tr = df[df.Outer_Fold != fold]; te = df[df.Outer_Fold == fold]
                model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("rf", RandomForestRegressor(n_estimators=300, max_features=0.8,
                                                               min_samples_leaf=2, random_state=config.RANDOM_SEED+int(fold),
                                                               n_jobs=4))])
                model.fit(tr[features], tr[target]); p = model.predict(te[features])
                preds.append(pd.DataFrame({"Task": task, "Feature_Set": set_name, "Outer_Fold": int(fold),
                                           "Model_Row_ID": te.Model_Row_ID, "Source_Group": te.Source_Group,
                                           "y_true": te[target], "y_pred": p}))
    pred = pd.concat(preds, ignore_index=True); rows = []
    for (task, fs), part in pred.groupby(["Task", "Feature_Set"]):
        rows.append({"Task": task, "Feature_Set": fs, "Rows": len(part),
                     "R2": r2_score(part.y_true, part.y_pred),
                     "RMSE": np.sqrt(mean_squared_error(part.y_true, part.y_pred)),
                     "MAE": mean_absolute_error(part.y_true, part.y_pred)})
    summary = pd.DataFrame(rows).sort_values(["Task", "R2"], ascending=[True, False])
    pred.to_csv(out / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
