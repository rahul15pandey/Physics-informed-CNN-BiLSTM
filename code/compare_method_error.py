# -*- coding: utf-8 -*-
"""
Compare two method_error_txt logs (baseline vs. new physics loss).
Usage:
    python code/compare_method_error.py --new experiments_result/method_error_txt/grid_FD4_without_num_test100.txt --base <baseline_path>
"""
import argparse
import os
import re
import shutil
from typing import List, Dict, Optional


RUN_RE = re.compile(
    r"\((?P<run>\d+)\).*?RMSE:\s*(?P<rmse>[-+eE0-9\.]+).*?UPE:\s*(?P<upe>[-+eE0-9\.]+)"
    r".*?ER:\((?P<er_left>[-+eE0-9\.]+),(?P<er_right>[-+eE0-9\.]+)\)"
    r"(?:.*?MonoViolation:\s*(?P<mono>[-+eE0-9\.]+))?"
    r"(?:.*?SlopeRMSE:\s*(?P<slope>[-+eE0-9\.]+))?",
)


def parse_method_error(path: str) -> List[Dict[str, float]]:
    metrics = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = RUN_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            metrics.append(
                {
                    "run": int(d["run"]),
                    "rmse": float(d["rmse"]),
                    "upe": float(d["upe"]),
                    "er_left": float(d["er_left"]),
                    "er_right": float(d["er_right"]),
                    "mono": float(d["mono"]) if d.get("mono") else None,
                    "slope": float(d["slope"]) if d.get("slope") else None,
                }
            )
    return metrics


def mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(val: Optional[float]) -> str:
    return f"{val:.4f}" if val is not None else "-"


def compare(new_path: str, base_path: str):
    new = parse_method_error(new_path)
    base = parse_method_error(base_path)
    runs = max(len(new), len(base))
    print(f"New file: {new_path}")
    print(f"Base file: {base_path}")
    print("run  | new_rmse  base_rmse | new_upe   base_upe  | new_mono  base_mono | new_slope base_slope")
    print("-----|---------------------------------|---------------------------------|-----------------------")
    for i in range(runs):
        n = new[i] if i < len(new) else None
        b = base[i] if i < len(base) else None
        print(
            f"{i:>3}  | "
            f"{fmt(n['rmse'] if n else None):>8}  {fmt(b['rmse'] if b else None):>8} | "
            f"{fmt(n['upe'] if n else None):>7}  {fmt(b['upe'] if b else None):>7} | "
            f"{fmt(n['mono'] if n else None):>7}  {fmt(b['mono'] if b else None):>7} | "
            f"{fmt(n['slope'] if n else None):>8} {fmt(b['slope'] if b else None):>9}"
        )

    def summarize(label: str, items: List[Dict[str, float]]):
        print(f"\n{label} means (excluding missing):")
        print(f"  RMSE : {fmt(mean([m['rmse'] for m in items]))}")
        print(f"  UPE  : {fmt(mean([m['upe'] for m in items]))}")
        print(f"  Mono : {fmt(mean([m['mono'] for m in items]))}")
        print(f"  Slope: {fmt(mean([m['slope'] for m in items]))}")

    summarize("New", new)
    summarize("Base", base)


def copy_models(src_dir: Optional[str], dst_root: Optional[str], label: str):
    if not src_dir or not dst_root:
        return
    if not os.path.exists(src_dir):
        print(f"[{label}] model dir not found, skip copy: {src_dir}")
        return
    os.makedirs(dst_root, exist_ok=True)
    dst_dir = os.path.join(dst_root, os.path.basename(src_dir))
    if os.path.exists(dst_dir):
        print(f"[{label}] destination already exists, skip copy: {dst_dir}")
        return
    shutil.copytree(src_dir, dst_dir)
    print(f"[{label}] copied models to {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two method_error_txt logs")
    parser.add_argument("--new", required=True, help="Path to new (physics) method_error_txt")
    parser.add_argument("--base", required=True, help="Path to baseline method_error_txt")
    parser.add_argument("--new_model_dir", help="Directory containing new-model checkpoints to archive")
    parser.add_argument("--base_model_dir", help="Directory containing baseline-model checkpoints to archive")
    parser.add_argument("--save_models_to", help="Root directory to copy model dirs into for reruns")
    args = parser.parse_args()
    compare(args.new, args.base)
    copy_models(args.new_model_dir, args.save_models_to, "new")
    copy_models(args.base_model_dir, args.save_models_to, "base")
