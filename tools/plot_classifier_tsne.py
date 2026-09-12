from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

COLORS = {
    "haze": "#4477AA",
    "rain": "#228833",
    "snow": "#66CCEE",
    "lowlight": "#AA3377",
    "haze+rain": "#CCBB44",
    "haze+lowlight": "#EE6677",
}
DISPLAY = {
    "haze": "Haze",
    "rain": "Rain",
    "snow": "Snow",
    "lowlight": "Low light",
    "haze+rain": "Haze + rain",
    "haze+lowlight": "Haze + low light",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("features")
    parser.add_argument("output_dir")
    parser.add_argument("--perplexity", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = np.load(args.features, allow_pickle=False)
    features = payload["features"].astype(np.float32)
    labels = payload["labels"].astype(np.int64)
    sources = payload["sources"].astype(str)
    class_names = payload["class_names"].astype(str)
    if not np.isfinite(features).all() or len(features) != len(labels):
        raise ValueError("Invalid or misaligned features")
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    output = Path(args.output_dir)
    metadata = {}
    for source in sorted(set(sources)):
        selected = sources == source
        x = features[selected]
        y = labels[selected]
        combinations = np.array(
            ["+".join(name for name, present in zip(class_names, row, strict=True) if present) for row in y]
        )
        perplexity = min(args.perplexity, len(x) - 1)
        with threadpool_limits(limits=4):
            model = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                max_iter=args.iterations,
                random_state=args.seed,
                n_jobs=4,
            )
            points = model.fit_transform(x)
        fig, ax = plt.subplots(figsize=(8, 8))
        for label in sorted(set(combinations)):
            mask = combinations == label
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=25,
                alpha=0.78,
                linewidths=0,
                color=COLORS.get(label, "#777777"),
                label=f"{DISPLAY.get(label, label)} (n={int(mask.sum())})",
                rasterized=True,
            )
        ax.set_title(f"{source} validation · step {output.name.split('_')[-1]}", fontsize=17)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
        fig.suptitle("DINOv2-B/14 degradation features", y=0.98, fontsize=19)
        fig.text(
            0.5,
            0.02,
            f"Independent {source} fit · perplexity {perplexity} · seed {args.seed}",
            ha="center",
            fontsize=9,
        )
        fig.subplots_adjust(top=0.90, bottom=0.18)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{source}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        with (output / f"{source}_coordinates.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["label", "x", "y"])
            writer.writerows(
                (label, float(point[0]), float(point[1]))
                for label, point in zip(combinations, points, strict=True)
            )
        metadata[source] = {
            "samples": int(selected.sum()),
            "perplexity": perplexity,
            "seed": args.seed,
            "iterations": args.iterations,
            "kl_divergence": float(model.kl_divergence_),
        }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
