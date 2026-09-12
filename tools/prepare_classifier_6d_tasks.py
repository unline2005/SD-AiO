"""Build the fixed six-degradation task file after FoundIR Blur is published."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path("/home/yhmi/data/complex-degradation-datasets")
FOUNDIR = ROOT / "FoundIR_512patch"
BASE_TASKS = Path("/home/yhmi/SD-AiO/configs/tasks_classifier_foundir_ggt_cdd11.yaml")
OUTPUT = Path("/home/yhmi/SD-AiO/configs/tasks_classifier_foundir_ggt_cdd11_6d.yaml")
FOUNDIR_TASKS = {
    "01Blur": ["blur"],
    "02Blur_Noise": ["blur", "noise"],
    "05Noise": ["noise"],
    "08Haze": ["haze"],
    "09Lowlight_Haze": ["haze", "lowlight"],
    "10Rain": ["rain"],
    "13Rain_Haze": ["haze", "rain"],
    "14Lowlight": ["lowlight"],
}
NEW_TRAIN_TASKS = ("01Blur", "02Blur_Noise", "05Noise")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def paired_files(folder: Path) -> list[tuple[Path, Path]]:
    lq = {path.stem: path for path in (folder / "lq").iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    gt = {path.stem: path for path in (folder / "gt").iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    if not lq or lq.keys() != gt.keys():
        raise ValueError(f"Missing or mismatched LQ/GT pairs: {folder}")
    return [(lq[stem], gt[stem]) for stem in sorted(lq)]


def task(dataset: str, labels: list[str], split: str) -> dict:
    common = {
        "name": f"{split}_FoundIR_512patch_{dataset}",
        "deg_type": labels,
        "repeat_ratio": 1,
        "prompt": "a high quality clean image",
        "source": "FoundIR",
        "sampling_weight": 0.125,
    }
    common.update(
        lq_path=str(FOUNDIR / split / dataset / "lq"),
        gt_path=str(FOUNDIR / split / dataset / "gt"),
    )
    return common


def main() -> None:
    tasks = OmegaConf.to_container(OmegaConf.load(BASE_TASKS), resolve=True)
    for entry in tasks["train"]:
        if entry.get("source") == "FoundIR":
            entry["sampling_weight"] = 0.125
    tasks["val"] = []
    tasks["test"] = []

    for dataset in NEW_TRAIN_TASKS:
        tasks["train"].append(task(dataset, FOUNDIR_TASKS[dataset], "train"))
    available_test_tasks = [dataset for dataset in FOUNDIR_TASKS if (FOUNDIR / "test" / dataset).is_dir()]
    for dataset in available_test_tasks:
        tasks["test"].append(task(dataset, FOUNDIR_TASKS[dataset], "test"))

    OmegaConf.save(OmegaConf.create(tasks), OUTPUT)
    if "--tasks-only" in sys.argv[1:]:
        print(json.dumps({"tasks": str(OUTPUT), "status": "TASKS_ONLY"}))
        return
    if sys.argv[1:]:
        raise ValueError("Only --tasks-only is supported")

    for dataset in FOUNDIR_TASKS:
        paired_files(FOUNDIR / "train" / dataset)
    if not available_test_tasks:
        raise ValueError(f"No test task under {FOUNDIR / 'test'} matches a train degradation")
    for dataset in available_test_tasks:
        paired_files(FOUNDIR / "test" / dataset)
    print(json.dumps({"tasks": str(OUTPUT), "test_tasks": available_test_tasks, "status": "READY"}))


if __name__ == "__main__":
    main()
