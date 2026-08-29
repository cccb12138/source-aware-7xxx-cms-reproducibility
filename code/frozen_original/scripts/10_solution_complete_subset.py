from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import config
from src.data_utils import ensure_output_dirs


def main():
    ensure_output_dirs(); out = config.OUTPUT_DIRS["single"] / "solution_temp_complete_subset"
    out.mkdir(parents=True, exist_ok=True); parts=[]
    sets={"composition_only":config.PRIMARY_FIXED_FEATURES,
          "composition_plus_solution_temp":config.PRIMARY_FIXED_FEATURES+["Sol_Temp_C"]}
    for task in ("YS","UTS","EL"):
        d=pd.read_csv(config.OUTPUT_DIRS["processed"]/"strict"/f"{task}_with_outer_folds.csv")
        d=d.dropna(subset=["Sol_Temp_C"]).copy(); target=config.TARGET_COLUMNS[task]
        for name,features in sets.items():
            d[features+[target]]=d[features+[target]].apply(pd.to_numeric,errors="coerce")
            for fold in sorted(d.Outer_Fold.unique()):
                tr=d[d.Outer_Fold!=fold]; te=d[d.Outer_Fold==fold]
                m=Pipeline([("imp",SimpleImputer(strategy="median")),
                            ("rf",RandomForestRegressor(n_estimators=300,max_features=.8,min_samples_leaf=2,
                                                        random_state=config.RANDOM_SEED+int(fold),n_jobs=4))])
                m.fit(tr[features],tr[target]); p=m.predict(te[features])
                parts.append(pd.DataFrame({"Task":task,"Feature_Set":name,"Outer_Fold":int(fold),
                                           "Model_Row_ID":te.Model_Row_ID,"Source_Group":te.Source_Group,
                                           "y_true":te[target],"y_pred":p}))
    pred=pd.concat(parts,ignore_index=True); rows=[]
    for (task,fs),x in pred.groupby(["Task","Feature_Set"]):
        rows.append({"Task":task,"Feature_Set":fs,"Rows":len(x),"Sources":x.Source_Group.nunique(),
                     "R2":r2_score(x.y_true,x.y_pred),"RMSE":np.sqrt(mean_squared_error(x.y_true,x.y_pred)),
                     "MAE":mean_absolute_error(x.y_true,x.y_pred)})
    s=pd.DataFrame(rows).sort_values(["Task","R2"],ascending=[True,False])
    pred.to_csv(out/"oof_predictions.csv",index=False,encoding="utf-8-sig")
    s.to_csv(out/"metrics_oof_summary.csv",index=False,encoding="utf-8-sig")
    print(s.to_string(index=False))


if __name__=="__main__":main()
