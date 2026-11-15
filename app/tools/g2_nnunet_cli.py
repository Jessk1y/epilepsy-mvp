"""Шаблон команды Г2 для nnU-Net по контракту приложения (epiapp/g2/contract.py).

    python g2_nnunet_cli.py --t1 T1.nii.gz --flair FLAIR.nii.gz --out ПАПКА

НЕ ПРОВЕРЕН НА НАСТОЯЩЕЙ МОДЕЛИ. Собран по блокноту h2_mri/notebooks/improve.ipynb (набор 504, план nnUNetResEncUNetMPlans, 3d_fullres).
Дарье: проверить места с пометкой «ПРОВЕРИТЬ» — порядок каналов, тренер, фолды, нужна ли подготовка снимков (совмещение FLAIR с T1,
удаление черепа), как при обучении.
Переменные окружения: nnUNet_results (как при обучении); NNUNET_BIN — папка с nnUNetv2_predict (bin окружения);
NNUNET_DATASET (504), NNUNET_TRAINER (nnUNetTrainer_FCD), NNUNET_FOLDS («0» или «0 1 2 3 4»).
Запуск приложения:  EPI_G2_CMD="~/projects/mrt/.venv/bin/python ~/epilepsy-mvp/app/tools/g2_nnunet_cli.py"  EPI_G2_TITLE="nnU-Net"
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True); ap.add_argument("--flair", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); inp, pred = out / "in", out / "pred"
    inp.mkdir(parents=True, exist_ok=True); pred.mkdir(exist_ok=True)
    import numpy as np, nibabel as nib
    # ПРОВЕРИТЬ: канал 0000 — T1, 0001 — FLAIR (как в dataset.json набора 504); FLAIR должен быть в сетке T1
    t1, fl = nib.load(a.t1), nib.load(a.flair)
    if fl.shape != t1.shape or not np.allclose(fl.affine, t1.affine, atol=1e-3):
        from nibabel.processing import resample_from_to
        fl = resample_from_to(fl, t1, order=1)                       # только пересчёт сетки; настоящее совмещение — если делалось при обучении
    nib.save(t1, inp / "case_0000.nii.gz"); nib.save(fl, inp / "case_0001.nii.gz")
    (out / "progress.txt").write_text("2 0.3")
    exe = str(Path(os.environ.get("NNUNET_BIN", "")) / "nnUNetv2_predict")
    cmd = [exe, "-i", str(inp), "-o", str(pred), "-d", os.environ.get("NNUNET_DATASET", "504"), "-c", "3d_fullres",
           "-p", "nnUNetResEncUNetMPlans", "-tr", os.environ.get("NNUNET_TRAINER", "nnUNetTrainer_FCD"),
           "-f", *os.environ.get("NNUNET_FOLDS", "0").split(), "--save_probabilities"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("nnUNetv2_predict: " + r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "nnUNetv2_predict завершился с ошибкой")
    p = np.load(pred / "case.npz")["probabilities"][1]              # класс 1 — ФКД; оси nnU-Net: (z, y, x)
    prob = np.transpose(p, (2, 1, 0)).astype(np.float32)             # ПРОВЕРИТЬ порядок осей на одном пациенте
    assert prob.shape == t1.shape[:3], (prob.shape, t1.shape)
    nib.save(nib.Nifti1Image(prob, t1.affine), out / "prob.nii.gz")
    shutil.copy(inp / "case_0000.nii.gz", out / "t1.nii.gz"); shutil.copy(inp / "case_0001.nii.gz", out / "flair.nii.gz")
    z = lambda im: float(max(im.header.get_zooms()[:3]))
    json.dump(dict(t1_mm=z(nib.load(a.t1)), flair_mm=z(nib.load(a.flair)), flags=[], ok=True), open(out / "qc.json", "w"))
    # candidates.csv не пишется: кандидатов посчитает приложение (связные области выше рабочей точки, не меньше 0,1 см³)
    (out / "progress.txt").write_text("3 0.9")


if __name__ == "__main__":
    main()
