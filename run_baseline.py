import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch

import DeepODModel.AE.run_ae as run_ae_module
from DeepODModel.AE.run_ae import run_AE
from EntropyEarlyStop import ModelEntropyEarlyStop
from exp_utils import dataLoading, set_seed

# 作者程式最後會呼叫 torch.cuda.max_memory_allocated('cuda')，沒有 NVIDIA GPU 時會出錯，這裡讓它回傳 0。
if not torch.cuda.is_available():
    torch.cuda.max_memory_allocated = lambda *args, **kwargs: 0


# 記錄 EntropyStop 停在哪一步：包一層作者的停止類別，把最後建立的物件存起來（不改變任何行為）。
class _RecordingEntropyStop(ModelEntropyEarlyStop):
    last = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _RecordingEntropyStop.last = self


run_ae_module.ModelEntropyEarlyStop = _RecordingEntropyStop

FIELDS = ["dataset", "method", "seed", "lr", "k", "R_down", "extra_param",
          "stop_iter", "total_iter", "AUC", "AP", "train_time_sec"]


def read_dataset_list(path):
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                names.append(line if line.endswith(".npz") else line + ".npz")
    return names


def load_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["dataset"], row["method"], row["seed"]))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=[os.path.join("configs", "datasets.txt")],
                   help="資料集清單檔，或直接列出資料集檔名")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--methods", nargs="+", default=["VanillaAE", "EntropyAE"],
                   choices=["VanillaAE", "EntropyAE"])
    p.add_argument("--epoch", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--Rdown", type=float, default=0.01)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--out", default=os.path.join("results", "baseline", "baseline_results.csv"))
    args = p.parse_args()

    if len(args.datasets) == 1 and args.datasets[0].endswith(".txt"):
        datasets = read_dataset_list(os.path.join(ROOT, args.datasets[0]))
    else:
        datasets = [d if d.endswith(".npz") else d + ".npz" for d in args.datasets]

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = load_done(out_path)
    new_file = not os.path.exists(out_path)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"裝置：{device}；資料集 {len(datasets)} 個；種子 {args.seeds}；方法 {args.methods}")

    total = len(datasets) * len(args.seeds) * len(args.methods)
    count = 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for name in datasets:
            x, y = dataLoading(os.path.join(ROOT, "data", name))
            for method in args.methods:
                for seed in args.seeds:
                    count += 1
                    if (name, method, str(seed)) in done:
                        print(f"[{count}/{total}] 已完成，跳過：{name} {method} seed={seed}")
                        continue
                    print(f"[{count}/{total}] {name} {method} seed={seed}")
                    set_seed(seed)
                    _RecordingEntropyStop.last = None
                    early = method == "EntropyAE"
                    auc, ap, rt, _mem = run_AE(
                        x, y, args.epoch, args.batch_size, args.lr,
                        entropyStop=early, k=args.k, R_down=args.Rdown, n_eval=args.n_eval)
                    es = _RecordingEntropyStop.last
                    writer.writerow({
                        "dataset": name, "method": method, "seed": seed, "lr": args.lr,
                        "k": args.k if early else "", "R_down": args.Rdown if early else "",
                        "extra_param": f"batch_size={args.batch_size};epoch={args.epoch}",
                        "stop_iter": es.best_epoch if (early and es) else "",
                        "total_iter": len(es.entropys) if (early and es) else "",
                        "AUC": round(float(auc), 6), "AP": round(float(ap), 6),
                        "train_time_sec": rt,
                    })
                    f.flush()
                    print(f"    AUC={auc:.4f}  AP={ap:.4f}  time={rt}s")
    print(f"完成，結果在 {out_path}")


if __name__ == "__main__":
    main()
