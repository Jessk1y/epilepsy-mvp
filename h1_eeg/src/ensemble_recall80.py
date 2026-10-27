import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
r=pd.read_csv('reve_v2_per_person.csv'); d=pd.read_csv('eegdino_finetune_per_person.csv')
d['dino_prob']=d[['eegdino_seed0','eegdino_seed1','eegdino_seed2']].mean(1)
m=r.merge(d[['dataset','id','dino_prob']],on=['dataset','id'])
m['ens']=(m.reve2_ens_prob+m.dino_prob)/2
print({k:len(v) for k,v in m.groupby('dataset')})
def mets(y,p):
    tp=((p==1)&(y==1)).sum();fp=((p==1)&(y==0)).sum();fn=((p==0)&(y==1)).sum();tn=((p==0)&(y==0)).sum()
    rec=tp/max(tp+fn,1);spec=tn/max(tn+fp,1);prec=tp/max(tp+fp,1) if tp+fp else np.nan
    return dict(precision=prec,recall=rec,spec=spec,bal=(rec+spec)/2)
def thr_youden(y,s):
    f,t,th=roc_curve(y,s);return th[np.argmax(t-f)]
def thr_sens(y,s,target=0.8):
    f,t,th=roc_curve(y,s);i=np.where(t>=target)[0][0];return th[i]
def boot(y,s,thr,n=2000):
    rng=np.random.default_rng(0);out=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y));out.append(mets(y[i],(s[i]>=thr).astype(int)))
    return pd.DataFrame(out).quantile([.025,.975])
rows=[]
for col,name in [('reve2_ens_prob','REVE v2'),('dino_prob','EEG-DINO'),('ens','REVE v2 + EEG-DINO')]:
    v=m[m.dataset=='nmt_val'];e=m[m.dataset=='nmt_eval']
    for rule,f in [('Юден',thr_youden),('recall 80 %',thr_sens)]:
        t=f(v.y.values,v[col].values);ye,se=e.y.values,e[col].values
        mm=mets(ye,(se>=t).astype(int));ci=boot(ye,se,t)
        rows.append(dict(модель=name,порог=rule,значение=round(t,3),набор='NMT eval',ROC_AUC=round(roc_auc_score(ye,se),3),**{k:f"{mm[k]:.2f} ({ci[k].iloc[0]:.2f}–{ci[k].iloc[1]:.2f})" for k in mm}))
nmt=pd.DataFrame(rows);print(nmt.to_string(index=False))
# локальная калибровка на Турции: 500 выборок по 20, правила Юден и recall>=80 %
t=m[m.dataset=='turkey'].reset_index(drop=True);y=t.y.values
rng=np.random.default_rng(0);res=[]
for rep in range(500):
    idx=rng.permutation(len(t));cal,test=idx[:20],idx[20:]
    if len(set(y[cal]))<2: continue
    for col,name in [('reve2_ens_prob','REVE v2'),('dino_prob','EEG-DINO'),('ens','REVE v2 + EEG-DINO')]:
        s=t[col].values
        for rule,f in [('Юден',thr_youden),('recall 80 %',thr_sens)]:
            th=f(y[cal],s[cal]);res.append(dict(модель=name,порог=rule,**mets(y[test],(s[test]>=th).astype(int))))
R=pd.DataFrame(res)
g=R.groupby(['модель','порог'])
out=g.mean().round(2).astype(str)+' ('+g.quantile(.025).round(2).astype(str)+'–'+g.quantile(.975).round(2).astype(str)+')'
print(out.to_string())
nmt.to_csv('ensemble_nmt_eval_threshold_metrics.csv',index=False)
out.reset_index().to_csv('turkey_calibration_youden_vs_recall80.csv',index=False)
