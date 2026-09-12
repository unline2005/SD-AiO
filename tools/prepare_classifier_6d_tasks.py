"""Build the fixed six-degradation task file after FoundIR Blur is published."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path("/home/yhmi/data/complex-degradation-datasets")
FOUNDIR = ROOT / "FoundIR_512patch"
SPLITS = ROOT / "classifier_6d_split"
BASE_TASKS = Path("/home/yhmi/SD-AiO/configs/tasks_classifier_foundir_ggt_cdd11.yaml")
OUTPUT = Path("/home/yhmi/SD-AiO/configs/tasks_classifier_foundir_ggt_cdd11_6d.yaml")
NEW_TASKS = {
    "01Blur": ["blur"],
    "02Blur_Noise": ["blur", "noise"],
    "05Noise": ["noise"],
}


def paired_files(folder: Path) -> list[tuple[Path, Path]]:
    lq = {path.name: path for path in (folder / "lq").glob("*.png")}
    gt = {path.name: path for path in (folder / "gt").glob("*.png")}
    if not lq or lq.keys() != gt.keys():
        raise ValueError(f"Missing or mismatched LQ/GT pairs: {folder}")
    return [(lq[name], gt[name]) for name in sorted(lq)]


def write_manifest(path: Path, pairs: list[tuple[Path, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for lq, gt in pairs:
            stream.write(json.dumps({"lq": str(lq), "gt": str(gt)}) + "\n")


def task(dataset: str, labels: list[str], split: str) -> dict:
    common = {
        "name": f"{split}_FoundIR_512patch_{dataset}",
        "deg_type": labels,
        "repeat_ratio": 1,
        "prompt": "a high quality clean image",
        "source": "FoundIR",
        "sampling_weight": 0.125,
    }
    if split == "train":
        common.update(
            lq_path=str(FOUNDIR / "train" / dataset / "lq"),
            gt_path=str(FOUNDIR / "train" / dataset / "gt"),
        )
    else:
        common["manifest"] = str(SPLITS / f"{dataset}_{split}.jsonl")
    return common


def main() -> None:
    tasks = OmegaConf.to_container(OmegaConf.load(BASE_TASKS), resolve=True)
    for split in ("train", "val", "test"):
        for entry in tasks[split]:
            if entry.get("source") == "FoundIR":
                entry["sampling_weight"] = 0.125

    for dataset, labels in NEW_TASKS.items():
        tasks["train"].append(task(dataset, labels, "train"))
        tasks["val"].append(task(dataset, labels, "val"))
        tasks["test"].append(task(dataset, labels, "test"))

    OmegaConf.save(OmegaConf.create(tasks), OUTPUT)
    if "--tasks-only" in sys.argv[1:]:
        print(json.dumps({"tasks": str(OUTPUT), "status": "TASKS_ONLY"}))
        return
    if sys.argv[1:]:
        raise ValueError("Only --tasks-only is supported")

    for dataset in NEW_TASKS:
        pairs = paired_files(FOUNDIR / "test" / dataset)
        random.Random(f"classifier-6d:{dataset}:42").shuffle(pairs)
        midpoint = len(pairs) // 2
        if midpoint == 0 or midpoint == len(pairs):
            raise ValueError(f"Need at least two held-out pairs for {dataset}")
        write_manifest(SPLITS / f"{dataset}_val.jsonl", pairs[:midpoint])
        write_manifest(SPLITS / f"{dataset}_test.jsonl", pairs[midpoint:])
    print(json.dumps({"tasks": str(OUTPUT), "manifests": str(SPLITS), "status": "READY"}))


if __name__ == "__main__":
    main()
