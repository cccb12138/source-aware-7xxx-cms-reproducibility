from __future__ import annotations

import copy
import random
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

import config
from src.data_utils import ensure_output_dirs, write_json


warnings.filterwarnings("ignore")
torch.set_num_threads(4)
TASKS = ["YS", "UTS", "EL"]
TARGETS = [config.TARGET_COLUMNS[t] for t in TASKS]
MASKS = [config.MASK_COLUMNS[t] for t in TASKS]
SEEDS = [20260721, 20260722, 20260723]


class MaskedMTL(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.10),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.10))
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in TASKS])

    def forward(self, x):
        h = self.shared(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def masked_loss(pred, y, mask):
    losses = []
    for j in range(len(TASKS)):
        valid = mask[:, j] > 0.5
        if valid.any():
            losses.append(torch.mean((pred[valid, j] - y[valid, j]) ** 2))
    return torch.stack(losses).mean()


def train_with_early_stop(xtr, ytr, mtr, xva, yva, mva, seed):
    set_seed(seed); model = MaskedMTL(xtr.shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    best, best_epoch, wait = None, 1, 0
    for epoch in range(1, 501):
        model.train(); opt.zero_grad(); loss = masked_loss(model(xtr), ytr, mtr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val = float(masked_loss(model(xva), yva, mva))
        if best is None or val < best - 1e-5:
            best, best_epoch, wait = val, epoch, 0; best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= 40: break
    return best_epoch, best


def train_full(x, y, mask, epochs, seed):
    set_seed(seed); model = MaskedMTL(x.shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    for _ in range(epochs):
        model.train(); opt.zero_grad(); loss = masked_loss(model(x), y, mask); loss.backward(); opt.step()
    return model


def metric(y, p):
    return {"R2": r2_score(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p)}


def main():
    ensure_output_dirs(); out = config.OUTPUT_DIRS["mtl"] / "masked_mlp_strict"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.OUTPUT_DIRS["processed"] / "strict" / "MTL_with_outer_folds.csv")
    features = config.PRIMARY_FIXED_FEATURES
    df[features + TARGETS + MASKS] = df[features + TARGETS + MASKS].apply(pd.to_numeric, errors="coerce")
    pred_rows, training_rows = [], []

    for outer in sorted(df.Outer_Fold.unique()):
        train = df[df.Outer_Fold != outer].copy(); test = df[df.Outer_Fold == outer].copy()
        val_fold = sorted(train.Outer_Fold.unique())[int(outer) % len(train.Outer_Fold.unique())]
        fit = train[train.Outer_Fold != val_fold].copy(); val = train[train.Outer_Fold == val_fold].copy()
        if set(train.Source_Group) & set(test.Source_Group): raise AssertionError("outer source leakage")

        imputer = SimpleImputer(strategy="median"); scaler = StandardScaler()
        x_fit = scaler.fit_transform(imputer.fit_transform(fit[features]))
        x_val = scaler.transform(imputer.transform(val[features]))
        # Final preprocessing is refit on all outer-training rows after epoch selection.
        y_mean = np.array([fit.loc[fit[MASKS[j]] == 1, TARGETS[j]].mean() for j in range(3)])
        y_std = np.array([fit.loc[fit[MASKS[j]] == 1, TARGETS[j]].std() for j in range(3)])
        y_std = np.where(y_std > 0, y_std, 1.0)
        def tensors(part, x):
            y = part[TARGETS].fillna(pd.Series(y_mean, index=TARGETS)).to_numpy()
            return (torch.tensor(x, dtype=torch.float32),
                    torch.tensor((y-y_mean)/y_std, dtype=torch.float32),
                    torch.tensor(part[MASKS].to_numpy(), dtype=torch.float32))
        tf = tensors(fit, x_fit); tv = tensors(val, x_val)
        epochs = []
        for seed in SEEDS:
            ep, vl = train_with_early_stop(*tf, *tv, seed)
            epochs.append(ep); training_rows.append({"Outer_Fold": int(outer), "Seed": seed, "Best_Epoch": ep, "Validation_Loss": vl})

        imputer = SimpleImputer(strategy="median"); scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(train[features])); x_test = scaler.transform(imputer.transform(test[features]))
        y_mean = np.array([train.loc[train[MASKS[j]] == 1, TARGETS[j]].mean() for j in range(3)])
        y_std = np.array([train.loc[train[MASKS[j]] == 1, TARGETS[j]].std() for j in range(3)]); y_std = np.where(y_std > 0, y_std, 1.0)
        tt = tensors(train, x_train); xt = torch.tensor(x_test, dtype=torch.float32)
        ensemble = []
        for seed, ep in zip(SEEDS, epochs):
            model = train_full(*tt, ep, seed); model.eval()
            with torch.no_grad(): ensemble.append(model(xt).numpy()*y_std+y_mean)
        mtl_pred = np.mean(ensemble, axis=0)

        for j, task in enumerate(TASKS):
            valid_test = test[MASKS[j]].eq(1).to_numpy()
            for idx in np.where(valid_test)[0]:
                pred_rows.append({"Model": "Masked_MTL", "Task": task, "Outer_Fold": int(outer),
                                  "Model_Row_ID": test.iloc[idx].Model_Row_ID, "Source_Group": test.iloc[idx].Source_Group,
                                  "y_true": test.iloc[idx][TARGETS[j]], "y_pred": mtl_pred[idx, j]})
            observed_train = train[MASKS[j]].eq(1)
            imp = SimpleImputer(strategy="median")
            xr = imp.fit_transform(train.loc[observed_train, features]); xe = imp.transform(test.loc[valid_test, features])
            rf = RandomForestRegressor(n_estimators=300, max_features=0.8, min_samples_leaf=2,
                                       random_state=config.RANDOM_SEED+int(outer), n_jobs=4)
            rf.fit(xr, train.loc[observed_train, TARGETS[j]]); rp = rf.predict(xe)
            valid_indices = np.where(valid_test)[0]
            for k, idx in enumerate(valid_indices):
                pred_rows.append({"Model": "Independent_RF", "Task": task, "Outer_Fold": int(outer),
                                  "Model_Row_ID": test.iloc[idx].Model_Row_ID, "Source_Group": test.iloc[idx].Source_Group,
                                  "y_true": test.iloc[idx][TARGETS[j]], "y_pred": rp[k]})
        print(f"outer fold {outer}: selected epochs {epochs}")

    pred = pd.DataFrame(pred_rows); summary = []
    for (model, task), part in pred.groupby(["Model", "Task"]):
        summary.append({"Model": model, "Task": task, "Rows": len(part), "Sources": part.Source_Group.nunique(),
                        **metric(part.y_true, part.y_pred)})
    summary = pd.DataFrame(summary).sort_values(["Task", "R2"], ascending=[True, False])
    pred.to_csv(out / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(training_rows).to_csv(out / "training_diagnostics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "metrics_oof_summary.csv", index=False, encoding="utf-8-sig")
    write_json(out / "run_config.json", {"features": features, "seeds": SEEDS, "shared_layers": [64,32],
                                          "loss": "equal-task masked MSE", "augmentation": False})
    print("\nPARTIAL-LABEL MTL SUMMARY\n", summary.to_string(index=False))


if __name__ == "__main__": main()
