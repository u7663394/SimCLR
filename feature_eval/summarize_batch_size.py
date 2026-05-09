import argparse
import csv
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "simclr_matplotlib_cache"))

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize batch-size ablation linear-eval results.")
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="One result in the form BATCH_SIZE=path/to/result.json.")
    parser.add_argument(
        "--n-views",
        default=2,
        type=int,
        help="Number of contrastive views used during SimCLR pretraining.")
    parser.add_argument(
        "--baseline-batch-size",
        default=256,
        type=int,
        help="Batch size used as the baseline for delta reporting.")
    parser.add_argument(
        "--output-dir",
        default="feature_eval/report/batch_size_ablation",
        help="Directory for report tables and figures.")
    return parser.parse_args()


def fmt(value):
    return "{:.2f}".format(value)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_result_specs(specs):
    results = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                "--result must use BATCH_SIZE=path/to/result.json, got {}".format(spec))
        batch_text, path = spec.split("=", 1)
        batch_size = int(batch_text)
        result = load_json(path)
        result["batch_size"] = batch_size
        result["result_path"] = path
        results.append(result)
    return sorted(results, key=lambda row: row["batch_size"])


def negatives_per_anchor(batch_size, n_views):
    return n_views * batch_size - n_views


def make_rows(results, n_views):
    rows = []
    for result in results:
        batch_size = result["batch_size"]
        rows.append({
            "batch_size": batch_size,
            "negatives_per_anchor": negatives_per_anchor(batch_size, n_views),
            "feature_dim": result["feature_dim"],
            "best_test_acc": result["best_test_acc"],
            "best_epoch": result["best_epoch"],
            "final_test_acc": result["final_test_acc"],
            "result_path": result["result_path"],
        })
    return rows


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def write_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    lines = [
        "| Batch size | Negatives / anchor | Feature dim | Best acc (%) | Best epoch | Final acc (%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            row["batch_size"],
            row["negatives_per_anchor"],
            row["feature_dim"],
            fmt(row["best_test_acc"]),
            row["best_epoch"],
            fmt(row["final_test_acc"])))
    return "\n".join(lines)


def plot_accuracy_curve(path, results):
    plt.figure(figsize=(8, 5))
    for result in results:
        plt.plot(
            [row["epoch"] for row in result["history"]],
            [row["test_acc"] for row in result["history"]],
            label="Batch size {}".format(result["batch_size"]),
            linewidth=2.2,
        )
    plt.xlabel("Linear evaluation epoch")
    plt.ylabel("CIFAR-10 test accuracy (%)")
    plt.title("Linear Evaluation Accuracy by Pretraining Batch Size")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_loss_curve(path, results):
    plt.figure(figsize=(8, 5))
    for result in results:
        plt.plot(
            [row["epoch"] for row in result["history"]],
            [row["train_loss"] for row in result["history"]],
            label="Batch size {}".format(result["batch_size"]),
            linewidth=2.2,
        )
    plt.xlabel("Linear evaluation epoch")
    plt.ylabel("Linear classifier train loss")
    plt.title("Linear Evaluation Training Loss by Pretraining Batch Size")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_accuracy_bar(path, rows):
    labels = [str(row["batch_size"]) for row in rows]
    x_positions = list(range(len(rows)))
    width = 0.34
    best_values = [row["best_test_acc"] for row in rows]
    final_values = [row["final_test_acc"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(
        [x - width / 2 for x in x_positions],
        best_values,
        width=width,
        label="Best accuracy",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        final_values,
        width=width,
        label="Final accuracy",
    )
    plt.xticks(x_positions, labels)
    plt.xlabel("Pretraining batch size")
    plt.ylabel("CIFAR-10 test accuracy (%)")
    plt.title("Batch Size Ablation")
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


def plot_negatives_vs_accuracy(path, rows):
    x_values = [row["negatives_per_anchor"] for row in rows]
    y_values = [row["best_test_acc"] for row in rows]
    labels = [row["batch_size"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2.2)
    for x, y, label in zip(x_values, y_values, labels):
        plt.text(x, y + 0.25, "B={}".format(label), ha="center", va="bottom")
    plt.xlabel("Negatives per anchor")
    plt.ylabel("Best CIFAR-10 test accuracy (%)")
    plt.title("Negative Sample Count vs. Representation Quality")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def describe_trend(rows, baseline_batch_size):
    best_row = max(rows, key=lambda row: row["best_test_acc"])
    baseline = None
    for row in rows:
        if row["batch_size"] == baseline_batch_size:
            baseline = row
            break

    best_values = [row["best_test_acc"] for row in rows]
    monotonic_increase = all(
        later >= earlier for earlier, later in zip(best_values, best_values[1:]))

    if monotonic_increase:
        trend = (
            "Accuracy increases as batch size, and therefore the number of negative "
            "samples, increases.")
    else:
        trend = (
            "Accuracy does not increase monotonically with batch size in this run, "
            "so the largest batch is not automatically the best setting.")

    baseline_text = ""
    if baseline is not None:
        delta = best_row["best_test_acc"] - baseline["best_test_acc"]
        baseline_text = (
            "Compared with the batch-size-{} baseline, the best setting changes "
            "best accuracy by {} percentage points.".format(
                baseline_batch_size, fmt(delta)))

    return best_row, trend, baseline_text


def write_markdown_report(path, rows, results, n_views, baseline_batch_size):
    best_row, trend, baseline_text = describe_trend(rows, baseline_batch_size)

    report = """# Batch Size Ablation

## Result Table

{table}

## Figures

![Accuracy curve](accuracy_curve.png)

![Training loss curve](loss_curve.png)

![Accuracy comparison](accuracy_bar.png)

![Negatives versus accuracy](negatives_vs_accuracy.png)

## Report-Ready Interpretation

In this ablation, the intended variable is the SimCLR pretraining batch size.
With {n_views} augmented views per image, each anchor compares against
`{n_views} * batch_size - {n_views}` negative samples. Larger batches therefore
increase the number of in-batch negatives used by the NT-Xent objective.

The best linear-evaluation result is obtained with batch size {best_batch},
which provides {best_negatives} negatives per anchor and reaches {best_acc}%
best CIFAR-10 test accuracy.

{trend}
{baseline_text}

The linear-evaluation accuracy is the main metric for this ablation because it
measures the quality of the frozen encoder representation, while the contrastive
training loss only measures optimization of the pretext objective.
""".format(
        table=markdown_table(rows),
        n_views=n_views,
        best_batch=best_row["batch_size"],
        best_negatives=best_row["negatives_per_anchor"],
        best_acc=fmt(best_row["best_test_acc"]),
        trend=trend,
        baseline_text=baseline_text,
    )

    with open(path, "w") as f:
        f.write(report)


def main():
    args = parse_args()
    results = parse_result_specs(args.result)
    rows = make_rows(results, args.n_views)
    ensure_output_dir(args.output_dir)

    summary_csv = os.path.join(args.output_dir, "summary.csv")
    report_md = os.path.join(args.output_dir, "batch_size_ablation_report.md")
    accuracy_curve = os.path.join(args.output_dir, "accuracy_curve.png")
    loss_curve = os.path.join(args.output_dir, "loss_curve.png")
    accuracy_bar = os.path.join(args.output_dir, "accuracy_bar.png")
    negatives_plot = os.path.join(args.output_dir, "negatives_vs_accuracy.png")

    write_summary_csv(summary_csv, rows)
    plot_accuracy_curve(accuracy_curve, results)
    plot_loss_curve(loss_curve, results)
    plot_accuracy_bar(accuracy_bar, rows)
    plot_negatives_vs_accuracy(negatives_plot, rows)
    write_markdown_report(
        report_md, rows, results, args.n_views, args.baseline_batch_size)

    print(markdown_table(rows))
    best_row, trend, baseline_text = describe_trend(
        rows, args.baseline_batch_size)
    print("")
    print("Best setting: batch size {}, best acc {}%, negatives per anchor {}".format(
        best_row["batch_size"],
        fmt(best_row["best_test_acc"]),
        best_row["negatives_per_anchor"]))
    print(trend)
    if baseline_text:
        print(baseline_text)
    print("")
    print("Saved report artifacts:")
    print("- {}".format(report_md))
    print("- {}".format(summary_csv))
    print("- {}".format(accuracy_curve))
    print("- {}".format(loss_curve))
    print("- {}".format(accuracy_bar))
    print("- {}".format(negatives_plot))


if __name__ == "__main__":
    main()
