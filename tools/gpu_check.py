"""Проверка, можно ли нагрузить все видеокарты машины одновременно.

Запуск на Linux-машине (в любом окружении с PyTorch и CUDA, например meld_graph):
    python gpu_check.py            # ~3 минуты
    python gpu_check.py --sec 30   # дольше и точнее

Что делает:
  1. печатает топологию и параметры шины PCIe каждой карты (nvidia-smi);
  2. на каждой карте ОТДЕЛЬНО меряет скорость счёта (TFLOPS) и скорость передачи
     данных процессор -> карта (ГБ/с);
  3. то же самое на ВСЕХ картах ОДНОВРЕМЕННО и во время нагрузки снимает
     текущие поколение и ширину шины;
  4. пишет вывод: во сколько раз проседает каждая карта при совместной работе.
Ничего не устанавливает и не меняет в системе.
"""
import argparse
import multiprocessing as mp
import subprocess
import time

import torch


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # nvidia-smi может отсутствовать в PATH
        return f"(не удалось: {e})"


def bench(dev, sec, q):
    torch.cuda.set_device(dev)
    # счёт: перемножение матриц в fp16 (как обучение нейросети)
    a = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    n, t0 = 0, time.time()
    while time.time() - t0 < sec:
        a @ b
        n += 1
        if n % 10 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    tflops = 2 * 8192 ** 3 * n / (time.time() - t0) / 1e12
    # шина: копирование 512 МБ из закреплённой памяти процессора на карту
    host = torch.empty(256 * 1024 * 1024, dtype=torch.float16).pin_memory()
    gpu = torch.empty_like(host, device=dev)
    torch.cuda.synchronize()
    n, t0 = 0, time.time()
    while time.time() - t0 < sec / 2:
        gpu.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        n += 1
    gbs = host.numel() * 2 * n / (time.time() - t0) / 1e9
    q.put((dev, tflops, gbs))


def run(devs, sec):
    q = mp.get_context("spawn").Queue()
    ps = [mp.get_context("spawn").Process(target=bench, args=(d, sec, q)) for d in devs]
    for p in ps:
        p.start()
    time.sleep(sec / 2)  # снимок шины во время нагрузки
    link = sh("nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current,"
              "utilization.gpu,power.draw,temperature.gpu --format=csv")
    for p in ps:
        p.join()
    res = {}
    while not q.empty():
        d, tf, gb = q.get()
        res[d] = (tf, gb)
    return res, link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=int, default=20, help="секунд счёта на один замер")
    sec = ap.parse_args().sec

    n = torch.cuda.device_count()
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, видеокарт: {n}\n")
    print(sh("nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.max,pcie.link.width.max,"
             "power.limit --format=csv"))
    print("\nТопология (как карты связаны между собой и с процессором):")
    print(sh("nvidia-smi topo -m"))

    alone = {}
    for d in range(n):
        r, link = run([d], sec)
        alone[d] = r[d]
        print(f"\nКарта {d} одна: {r[d][0]:.1f} TFLOPS, шина {r[d][1]:.1f} ГБ/с")
    print(f"\nВсе {n} карты одновременно:")
    together, link = run(list(range(n)), sec)
    print(link)

    print("\n=== Итог ===")
    print("карта | счёт одна → вместе | шина одна → вместе (ГБ/с)")
    for d in range(n):
        (t1, g1), (t2, g2) = alone[d], together.get(d, (0, 0))
        print(f"  {d}   | {t1:5.1f} → {t2:5.1f} TFLOPS ({t2 / t1:4.0%}) | {g1:5.1f} → {g2:5.1f} ({g2 / g1:4.0%})")
    worst = min(together.get(d, (0, 0))[0] / alone[d][0] for d in range(n))
    print("\nКак читать:")
    print("- счёт вместе >= 85 % от одиночного — три обучения (например, три разбиения nnU-Net) можно гнать параллельно;")
    print("- шина ~12–25 ГБ/с = x16, ~6–12 = x8, ~3 = x4, ~1 = x1. Для nnU-Net и MELD даже x4 обычно хватает,")
    print("  потому что данные на шаг маленькие; узкая шина мешает в основном обучению одной модели на нескольких картах;")
    print("- если карта вместе почти не считает или процесс падает — проблема в питании или слотах, а не в шине.")
    print(f"\nХудшая карта при совместной работе: {worst:.0%} от одиночной скорости.")


if __name__ == "__main__":
    main()
