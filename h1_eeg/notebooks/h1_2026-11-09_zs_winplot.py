# Г1: самые подозрительные и самые спокойные окна REVE v4 у больных и здоровых + матрицы корреляции каналов (1–30 Гц)
import os, glob, numpy as np, pandas as pd, scipy.signal as ss, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = os.path.dirname(glob.glob("/kaggle/input/**/h1_shared_meta.csv", recursive=True)[0])
meta = pd.read_csv(f"{root}/h1_shared_meta.csv")
EL = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","Fz","Cz","Pz"]
L = 2000; SOS = ss.butter(4, (1, 30), btype="bandpass", fs=200, output="sos"); IU = np.triu_indices(19, 1)
PICK = [  # набор, запись, метка, окно max p, p, окно min p, p
 ("turkey", "1_epilepsy (47).mat", 1, 8, .72, 30, .13), ("turkey", "1_epilepsy (39).mat", 1, 93, .54, 10, .11),
 ("turkey", "2_Normal (4).mat", 0, 0, .34, 86, .10), ("turkey", "2_Normal (6).mat", 0, 11, .21, 82, .03),
 ("nmt_eval", "0001653.edf", 1, 58, .88, 19, .50), ("nmt_eval", "0002143.edf", 1, 84, .99, 35, .57),
 ("nmt_eval", "0002047.edf", 0, 17, .59, 91, .19), ("nmt_eval", "0000584.edf", 0, 43, .83, 11, .44)]
SPL = {"turkey": "turkey", "nmt_eval": "eval"}
def rec(d, rid):
    r = meta[(meta.split == SPL[d]) & (meta.id == rid)].iloc[0]; n = int(r.n200)
    return np.asarray(np.load(f"{root}/fs200/{r.shard200}", mmap_mode="r")[int(r.off200):int(r.off200) + 19 * n], np.float32).reshape(19, n)
def traces(ax, seg, step, title):
    seg = seg - seg.mean(1, keepdims=True); t = np.arange(L) / 200
    for i in range(19): ax.plot(t, seg[i] - i * step, lw=0.45, color="k")
    ax.set_yticks(-np.arange(19) * step); ax.set_yticklabels(EL, fontsize=6); ax.set_xlim(0, 10); ax.set_ylim(-19.5 * step, step)
    ax.set_title(title, fontsize=9); ax.tick_params(axis="x", labelsize=6)
def cmat(ax, segf):
    C = np.corrcoef(segf); im = ax.imshow(np.abs(C), vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(19)); ax.set_xticklabels(EL, fontsize=5, rotation=90); ax.set_yticks(range(19)); ax.set_yticklabels(EL, fontsize=5)
    ax.set_title(f"|корр.| 1–30 Гц, среднее {np.abs(C[IU]).mean():.2f}", fontsize=9); return im
for d, nm in [("turkey", "Турция"), ("nmt_eval", "NMT eval")]:
    rows = [p for p in PICK if p[0] == d]
    fig, axs = plt.subplots(len(rows), 4, figsize=(20, 5.2 * len(rows)), gridspec_kw=dict(width_ratios=[2.2, 1, 2.2, 1]))
    for ax, (_, rid, y, wt, pt, wl, pl) in zip(axs, rows):
        X = rec(d, rid); Xf = ss.sosfiltfilt(SOS, X, axis=-1)
        st = X[:, wt * L:(wt + 1) * L]; sl = X[:, wl * L:(wl + 1) * L]
        step = float(np.percentile(np.abs(np.hstack([st - st.mean(1, keepdims=True), sl - sl.mean(1, keepdims=True)])), 99.5) * 1.5)
        lab = "БОЛЬНОЙ" if y else "ЗДОРОВЫЙ"
        traces(ax[0], st, step, f"{lab} {rid}: самое подозрительное окно #{wt}, p={pt}   (шаг {step:.0f} мкВ)")
        cmat(ax[1], Xf[:, wt * L:(wt + 1) * L])
        traces(ax[2], sl, step, f"{lab} {rid}: самое спокойное окно #{wl}, p={pl}")
        cmat(ax[3], Xf[:, wl * L:(wl + 1) * L])
    fig.suptitle(f"{nm}: окна REVE v4 (сигнал к среднему референту, шкала общая в строке)", fontsize=13, y=1.0)
    plt.tight_layout(); plt.savefig(f"/kaggle/working/winplot_{d}.png", dpi=110, bbox_inches="tight"); plt.close()
print("ok")
