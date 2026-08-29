from __future__ import annotations
import numpy as np,pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import config


def main():
    b=config.OUTPUT_DIRS["single"];out=b/"rf_xgb_ensemble_strict";out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(b/"baseline_strict"/"oof_predictions.csv");base=base[(base.Model=="RandomForest")&(base.Feature_Set=="composition_core")]
    tuned=pd.read_csv(b/"nested_optuna_strict"/"oof_predictions.csv");xgb=pd.read_csv(b/"nested_optuna_xgb_strict"/"oof_predictions.csv")
    parts=[];rows=[]
    for task in ("YS","UTS","EL"):
        rf=tuned[tuned.Task==task] if task=="UTS" else base[base.Task==task]
        rf=rf[["Task","Outer_Fold","Model_Row_ID","Source_Group","y_true","y_pred"]].rename(columns={"y_pred":"pred_rf"})
        x=xgb[xgb.Task==task][["Model_Row_ID","y_pred"]].rename(columns={"y_pred":"pred_xgb"})
        z=rf.merge(x,on="Model_Row_ID",validate="one_to_one");z["y_pred"]=(z.pred_rf+z.pred_xgb)/2;parts.append(z)
        rows.append({"Task":task,"Rows":len(z),"R2":r2_score(z.y_true,z.y_pred),"RMSE":np.sqrt(mean_squared_error(z.y_true,z.y_pred)),"MAE":mean_absolute_error(z.y_true,z.y_pred)})
    p=pd.concat(parts,ignore_index=True);s=pd.DataFrame(rows);p.to_csv(out/"oof_predictions.csv",index=False,encoding="utf-8-sig");s.to_csv(out/"metrics_oof_summary.csv",index=False,encoding="utf-8-sig");print(s.to_string(index=False))
if __name__=="__main__":main()
