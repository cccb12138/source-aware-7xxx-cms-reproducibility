from __future__ import annotations

import copy, random
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

import config
from src.data_utils import ensure_output_dirs,write_json


torch.set_num_threads(4)
TASKS=["YS","UTS"]; TARGETS=[config.TARGET_COLUMNS[t] for t in TASKS]; MASKS=[config.MASK_COLUMNS[t] for t in TASKS]
SEEDS=[20260721,20260722,20260723]


class SelectiveNet(nn.Module):
    def __init__(self,n):
        super().__init__(); self.shared=nn.Sequential(nn.Linear(n,48),nn.ReLU(),nn.Dropout(.1))
        self.towers=nn.ModuleList([nn.Sequential(nn.Linear(48,24),nn.ReLU(),nn.Linear(24,1)) for _ in TASKS])
    def forward(self,x):
        h=self.shared(x); return torch.cat([tower(h) for tower in self.towers],1)


def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def loss_fn(p,y,m):
    z=[]
    for j in range(2):
        ok=m[:,j]>.5
        if ok.any():z.append(((p[ok,j]-y[ok,j])**2).mean())
    return torch.stack(z).mean()
def tensors(part,x,mean,std):
    y=part[TARGETS].fillna(pd.Series(mean,index=TARGETS)).to_numpy()
    return torch.tensor(x,dtype=torch.float32),torch.tensor((y-mean)/std,dtype=torch.float32),torch.tensor(part[MASKS].to_numpy(),dtype=torch.float32)


def choose_epochs(fit,val,xf,xv,mean,std,s):
    seed_all(s);m=SelectiveNet(xf.shape[1]);o=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4)
    a=tensors(fit,xf,mean,std);b=tensors(val,xv,mean,std);best=1e9;ep=1;wait=0
    for e in range(1,501):
        m.train();o.zero_grad();l=loss_fn(m(a[0]),a[1],a[2]);l.backward();o.step();m.eval()
        with torch.no_grad():v=float(loss_fn(m(b[0]),b[1],b[2]))
        if v<best-1e-5:best=v;ep=e;wait=0
        else:wait+=1
        if wait>=40:break
    return ep,best
def fit_full(data,x,mean,std,epochs,s):
    seed_all(s);m=SelectiveNet(x.shape[1]);o=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4);a=tensors(data,x,mean,std)
    for _ in range(epochs):m.train();o.zero_grad();l=loss_fn(m(a[0]),a[1],a[2]);l.backward();o.step()
    return m


def main():
    ensure_output_dirs();out=config.OUTPUT_DIRS["mtl"]/"selective_ys_uts_strict";out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(config.OUTPUT_DIRS["processed"]/"strict"/"MTL_with_outer_folds.csv");f=config.PRIMARY_FIXED_FEATURES
    d[f+TARGETS+MASKS]=d[f+TARGETS+MASKS].apply(pd.to_numeric,errors="coerce");parts=[];diag=[]
    for outer in sorted(d.Outer_Fold.unique()):
        tr=d[d.Outer_Fold!=outer].copy();te=d[d.Outer_Fold==outer].copy();vf=sorted(tr.Outer_Fold.unique())[int(outer)%4]
        fit=tr[tr.Outer_Fold!=vf];val=tr[tr.Outer_Fold==vf]
        imp=SimpleImputer(strategy="median");sc=StandardScaler();xf=sc.fit_transform(imp.fit_transform(fit[f]));xv=sc.transform(imp.transform(val[f]))
        mean=np.array([fit.loc[fit[MASKS[j]]==1,TARGETS[j]].mean() for j in range(2)]);std=np.array([fit.loc[fit[MASKS[j]]==1,TARGETS[j]].std() for j in range(2)])
        epochs=[]
        for s in SEEDS:
            e,v=choose_epochs(fit,val,xf,xv,mean,std,s);epochs.append(e);diag.append({"Outer_Fold":outer,"Seed":s,"Epoch":e,"Val_Loss":v})
        imp=SimpleImputer(strategy="median");sc=StandardScaler();xtr=sc.fit_transform(imp.fit_transform(tr[f]));xte=sc.transform(imp.transform(te[f]))
        mean=np.array([tr.loc[tr[MASKS[j]]==1,TARGETS[j]].mean() for j in range(2)]);std=np.array([tr.loc[tr[MASKS[j]]==1,TARGETS[j]].std() for j in range(2)])
        xt=torch.tensor(xte,dtype=torch.float32);ens=[]
        for s,e in zip(SEEDS,epochs):
            m=fit_full(tr,xtr,mean,std,e,s);m.eval()
            with torch.no_grad():ens.append(m(xt).numpy()*std+mean)
        p=np.mean(ens,0)
        for j,t in enumerate(TASKS):
            ok=te[MASKS[j]].eq(1).to_numpy()
            parts.append(pd.DataFrame({"Task":t,"Outer_Fold":outer,"Model_Row_ID":te.loc[ok,"Model_Row_ID"],"Source_Group":te.loc[ok,"Source_Group"],"y_true":te.loc[ok,TARGETS[j]],"y_pred":p[ok,j]}))
        print("fold",outer,"epochs",epochs)
    pred=pd.concat(parts,ignore_index=True);rows=[]
    for t,x in pred.groupby("Task"):rows.append({"Task":t,"Rows":len(x),"Sources":x.Source_Group.nunique(),"R2":r2_score(x.y_true,x.y_pred),"RMSE":np.sqrt(mean_squared_error(x.y_true,x.y_pred)),"MAE":mean_absolute_error(x.y_true,x.y_pred)})
    s=pd.DataFrame(rows);pred.to_csv(out/"oof_predictions.csv",index=False,encoding="utf-8-sig");pd.DataFrame(diag).to_csv(out/"training_diagnostics.csv",index=False,encoding="utf-8-sig");s.to_csv(out/"metrics_oof_summary.csv",index=False,encoding="utf-8-sig")
    write_json(out/"run_config.json",{"tasks":TASKS,"seeds":SEEDS,"architecture":"shared48 + task towers24","augmentation":False});print(s.to_string(index=False))


if __name__=="__main__":main()
