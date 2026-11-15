"""Шаблон команды Г2 для MS-DSA-NET по контракту приложения (epiapp/g2/contract.py).

    python g2_msdsa_cli.py --t1 T1.nii.gz --flair FLAIR.nii.gz --out ПАПКА

НЕ ПРОВЕРЕН НА НАСТОЯЩЕЙ МОДЕЛИ: библиотеки msdsa_lib.py в репозитории нет. Файл собран по ячейкам блокнота
h2_mri/notebooks/ms_dsa_net_stage2b_cv.ipynb (разделы 2, 7 и 10: build_model, load_weights, make_ensemble_infer, predict_new).
Ярославе: проверить и поправить места с пометкой «ПРОВЕРИТЬ», запустить на одном пациенте.

Что должно лежать на машине (пути — переменными окружения):
  MSDSA_PROJ    папка проекта: msdsa_lib.py, deploy/{config.json, fold0..4.pt, nyul_ref_uhb.npy}
  MSDSA_REPO    клон кода MS-DSA-NET (в нём networks2.py)
  MSDSA_MNI     шаблон MNI, который использовался в блокноте (переменная MNI)
  FSL в PATH (FSLDIR), окружение с torch, monai, nibabel.
Запуск приложения:  EPI_G2_CMD="~/venvs/mri/bin/python ~/epilepsy-mvp/app/tools/g2_msdsa_cli.py"
"""
import argparse, json, os, shutil, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True); ap.add_argument("--flair", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--id", default="sub-new")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    PROJ, REPO, MNI = Path(os.environ["MSDSA_PROJ"]), Path(os.environ["MSDSA_REPO"]), os.environ["MSDSA_MNI"]
    DEPLOY = PROJ / "deploy"
    sys.path[:0] = [str(PROJ), str(REPO)]
    import numpy as np, nibabel as nib, torch
    from monai.inferers import sliding_window_inference
    import msdsa_lib as ml
    from networks2 import MS_DSA_NET

    cfg = json.load(open(DEPLOY / "config.json", encoding="utf-8"))
    PATCH, OVERLAP, device = (128, 128, 128), 0.25, torch.device("cuda")

    def build_model():                                    # как в блокноте, раздел 2
        return MS_DSA_NET(spatial_dims=3, in_channels=2, out_channels=2, img_size=list(PATCH), feature_size=16, pos_embed=True,
                          project_size=64, sa_type="parallel", norm_name="instance",
                          act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}), res_block=True, bias=False, dropout_rate=0.1)

    def load_weights(m, path):
        sd = torch.load(path, map_location="cpu"); sd = sd.get("model", sd.get("state_dict", sd))
        m.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=True); return m

    def make_run_sw(m):
        def run_sw(x, sw):
            with torch.no_grad(), torch.autocast("cuda", enabled=True):       # ПРОВЕРИТЬ: в блокноте enabled=AMP
                return sliding_window_inference(x, PATCH, sw, m, overlap=OVERLAP)
        return run_sw

    (out / "progress.txt").write_text("0 0.05")
    models = [load_weights(build_model(), DEPLOY / f).to(device).eval() for f in cfg["models"]]
    infer = ml.make_ensemble_infer([make_run_sw(m) for m in models], device, 4, tta=cfg.get("tta", True))
    NYUL = ml.NyulHarmonizer(np.load(DEPLOY / cfg["nyul_ref"]))               # ПРОВЕРИТЬ: как в блокноте создаётся NYUL из NYUL_REF
    NEW = ml.Dataset("NEW", "-", out / "data", out / "work", out / "pred")
    NEW.files = {a.id: dict(t1=Path(a.t1), flair=Path(a.flair), mask=None)}
    (out / "progress.txt").write_text("1 0.3")
    r = ml.run_pipeline(NEW, [a.id], infer=infer, harmonizer=NYUL, variant="ensemble", mni=MNI, n_fsl=1, progress=False)
    if r["failures"]:
        sys.exit("ошибка расчёта: " + str(r["failures"]))
    (out / "progress.txt").write_text("3 0.9")
    prob = NEW.pred("ensemble") / f"{a.id}_prob.nii.gz"
    shutil.copy(prob, out / "prob.nii.gz")
    brain_path = NEW.ref / f"{a.id}_brain.nii.gz"
    shutil.copy(brain_path, out / "brain_mask.nii.gz")
    # снимки в сетке карты — чтобы наложение в просмотрщике было точным. ПРОВЕРИТЬ имена файлов после FSL.
    for src, dst in ((NEW.fsl(a.id) / "t1_reg.nii.gz", "t1.nii.gz"), (NEW.fsl(a.id) / "flair_reg.nii.gz", "flair.nii.gz")):
        if Path(src).exists():
            shutil.copy(src, out / dst)
    mni_arr = nib.load(MNI).get_fdata(dtype=np.float32); brain_ref = mni_arr > np.percentile(mni_arr[mni_arr > 0], 10)
    qc = ml.input_qc(a.t1, a.flair, NEW.fsl(a.id) / "t1_reg.nii.gz", mni_arr, brain_ref, cfg["qc_rules"])
    json.dump(qc, open(out / "qc.json", "w", encoding="utf-8"), ensure_ascii=False, default=float)
    img = nib.load(prob); brain = nib.load(brain_path).get_fdata() > 0
    cands = ml.cluster_table(img.get_fdata(dtype=np.float32), brain, ml.Post(**cfg["post"]), img.affine, top=cfg.get("show_top"))
    cands[["peak_prob", "volume_ml", "x", "y", "z"]].to_csv(out / "candidates.csv", index=False)   # x, y, z — мм в мире карты (ПРОВЕРИТЬ)


if __name__ == "__main__":
    main()
