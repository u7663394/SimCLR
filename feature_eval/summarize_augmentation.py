import argparse
import csv
import json
import os
import tempfile

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "simclr_matplotlib_cache")
os.environ.setdefault("MPLCONFIGDIR", _CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize data augmentation ablation results.")
    parser.add_argument(
        "--train-summary",
        action="append",
        required=True,
        help="Training summary in the form NAME=path/to/summary.json.")
    parser.add_argument(
        "--linear-result",
        action="append",
        required=True,
        help="Linear eval result in the form NAME=path/to/result.json.")
    parser.add_argument(
        "--output-dir",
        default="feature_eval/report/augmentation_ablation",
        help="Directory for report tables and figures.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_named_paths(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Expected NAME=path, got {}".format(item))
        name, path = item.split("=", 1)
        result[name] = path
    return result


def fmt(value):
    return "{:.2f}".format(value)


def display_name(name):
    labels = {
        "baseline": "Baseline",
        "no_blur": "Remove GaussianBlur",
        "no_color_jitter": "Remove ColorJitter",
        "no_grayscale": "Remove RandomGrayscale",
    }
    return labels.get(name, name)


def build_rows(train_summaries, linear_results):
    rows = []
    for name in ["baseline", "no_blur", "no_color_jitter", "no_grayscale"]:
        if name not in train_summaries or name not in linear_results:
            continue
        train_summary = load_json(train_summaries[name])
        linear_result = load_json(linear_results[name])
        rows.append({
            "name": name,
            "setting": display_name(name),
            "contrastive_top1": train_summary["final_top1"],
            "contrastive_loss": train_summary["final_loss"],
            "linear_best_acc": linear_result["best_test_acc"],
            "linear_final_acc": linear_result["final_test_acc"],
            "linear_best_epoch": linear_result["best_epoch"],
            "checkpoint": train_summary["checkpoint_path"],
            "train_summary": train_summaries[name],
            "linear_result": linear_results[name],
        })
    return rows


def write_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    lines = [
        "| Setting | Contrastive Top-1 (%) | Frozen linear best acc (%) | Frozen linear final acc (%) | Best epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {} | {} | {} | {} | {} |".format(
            row["setting"],
            fmt(row["contrastive_top1"]),
            fmt(row["linear_best_acc"]),
            fmt(row["linear_final_acc"]),
            row["linear_best_epoch"]))
    return "\n".join(lines)


def plot_contrastive_top1(path, rows):
    labels = [row["setting"] for row in rows]
    values = [row["contrastive_top1"] for row in rows]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color="#2A9D8F")
    plt.ylabel("Contrastive Top-1 (%)")
    plt.title("Contrastive Top-1 by Augmentation Setting")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.3,
                 fmt(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_linear_accuracy(path, rows):
    labels = [row["setting"] for row in rows]
    best_values = [row["linear_best_acc"] for row in rows]
    final_values = [row["linear_final_acc"] for row in rows]
    x_positions = list(range(len(rows)))
    width = 0.34

    plt.figure(figsize=(9, 5))
    plt.bar([x - width / 2 for x in x_positions], best_values,
            width=width, label="Best")
    plt.bar([x + width / 2 for x in x_positions], final_values,
            width=width, label="Final")
    plt.xticks(x_positions, labels, rotation=20, ha="right")
    plt.ylabel("Frozen Linear Evaluation Test Accuracy (%)")
    plt.title("Frozen Linear Evaluation by Augmentation Setting")
    plt.ylim(0, max(best_values + final_values) + 8)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    for x, value in zip([x - width / 2 for x in x_positions], best_values):
        plt.text(x, value + 0.3, fmt(value), ha="center", va="bottom")
    for x, value in zip([x + width / 2 for x in x_positions], final_values):
        plt.text(x, value + 0.3, fmt(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_contrastive_vs_linear(path, rows):
    plt.figure(figsize=(7, 5))
    for row in rows:
        plt.scatter(row["contrastive_top1"], row["linear_best_acc"], s=80)
        plt.text(row["contrastive_top1"], row["linear_best_acc"] + 0.25,
                 row["name"], ha="center", va="bottom")
    plt.xlabel("Contrastive Top-1 (%)")
    plt.ylabel("Frozen linear best test accuracy (%)")
    plt.title("Pretext Metric vs Downstream Accuracy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_linear_curve(path, rows):
    plt.figure(figsize=(9, 5))
    for row in rows:
        history = load_json(row["linear_result"])["history"]
        plt.plot(
            [item["epoch"] for item in history],
            [item["test_acc"] for item in history],
            label=row["setting"],
            linewidth=2,
        )
    plt.xlabel("Linear evaluation epoch")
    plt.ylabel("Frozen linear test accuracy (%)")
    plt.title("Linear Evaluation Accuracy Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_report(path, rows):
    best_linear = max(rows, key=lambda row: row["linear_best_acc"])
    best_contrastive = max(rows, key=lambda row: row["contrastive_top1"])

    report = """# Data Augmentation Ablation

## Result Table

{table}

## Figures

![Contrastive Top-1](contrastive_top1.png)

![Frozen linear evaluation](linear_accuracy.png)

![Linear evaluation curve](linear_accuracy_curve.png)

![Contrastive vs linear](contrastive_vs_linear.png)

## Report-Ready Interpretation

This ablation compares four augmentation settings: the full SimCLR baseline,
removing Gaussian blur, removing color jitter, and removing random grayscale.
The main downstream metric is frozen linear evaluation test accuracy. Contrastive
Top-1 is reported as the pretraining metric, but it should not be treated as a
direct substitute for downstream accuracy.

The best frozen linear result is obtained by `{best_linear_name}` with
{best_linear_acc}% best test accuracy. The highest contrastive Top-1 is obtained
by `{best_contrastive_name}` with {best_contrastive_top1}%.

If the setting with the highest contrastive Top-1 is not also the best setting
in frozen linear evaluation, this indicates that making the contrastive task
easier is not necessarily the same as learning more transferable features. This
point is especially important for data augmentation ablations, because removing
an augmentation can improve the pretext metric while weakening invariances that
matter for downstream classification.
""".format(
        table=markdown_table(rows),
        best_linear_name=best_linear["name"],
        best_linear_acc=fmt(best_linear["linear_best_acc"]),
        best_contrastive_name=best_contrastive["name"],
        best_contrastive_top1=fmt(best_contrastive["contrastive_top1"]),
    )
    with open(path, "w") as f:
        f.write(report)


def main():
    args = parse_args()
    train_summaries = parse_named_paths(args.train_summary)
    linear_results = parse_named_paths(args.linear_result)
    rows = build_rows(train_summaries, linear_results)
    if not rows:
        raise RuntimeError("No matching augmentation results were provided.")

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    report_md = os.path.join(args.output_dir, "augmentation_ablation_report.md")
    contrastive_plot = os.path.join(args.output_dir, "contrastive_top1.png")
    linear_plot = os.path.join(args.output_dir, "linear_accuracy.png")
    scatter_plot = os.path.join(args.output_dir, "contrastive_vs_linear.png")
    linear_curve = os.path.join(args.output_dir, "linear_accuracy_curve.png")

    write_summary_csv(summary_csv, rows)
    plot_contrastive_top1(contrastive_plot, rows)
    plot_linear_accuracy(linear_plot, rows)
    plot_contrastive_vs_linear(scatter_plot, rows)
    plot_linear_curve(linear_curve, rows)
    write_report(report_md, rows)

    print(markdown_table(rows))
    print("")
    print("Saved report artifacts:")
    print("- {}".format(report_md))
    print("- {}".format(summary_csv))
    print("- {}".format(contrastive_plot))
    print("- {}".format(linear_plot))
    print("- {}".format(linear_curve))
    print("- {}".format(scatter_plot))


if __name__ == "__main__":
    main()
