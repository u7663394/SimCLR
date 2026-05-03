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
        description="Summarize projection head ablation linear-eval results.")
    parser.add_argument("--with-head", required=True,
                        help="JSON result for the run with projection head.")
    parser.add_argument("--without-head", required=True,
                        help="JSON result for the run without projection head.")
    parser.add_argument("--output-dir",
                        default="feature_eval/report/projection_head_ablation",
                        help="Directory for report tables and figures.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def fmt(value):
    return "{:.2f}".format(value)


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def write_summary_csv(path, with_head, without_head):
    rows = [
        {
            "setting": "With projection head",
            "feature_dim": with_head["feature_dim"],
            "best_test_acc": with_head["best_test_acc"],
            "best_epoch": with_head["best_epoch"],
            "final_test_acc": with_head["final_test_acc"],
        },
        {
            "setting": "Without projection head",
            "feature_dim": without_head["feature_dim"],
            "best_test_acc": without_head["best_test_acc"],
            "best_epoch": without_head["best_epoch"],
            "final_test_acc": without_head["final_test_acc"],
        },
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy_curve(path, with_head, without_head):
    plt.figure(figsize=(8, 5))
    plt.plot(
        [row["epoch"] for row in with_head["history"]],
        [row["test_acc"] for row in with_head["history"]],
        label="With projection head",
        linewidth=2.2,
    )
    plt.plot(
        [row["epoch"] for row in without_head["history"]],
        [row["test_acc"] for row in without_head["history"]],
        label="Without projection head",
        linewidth=2.2,
    )
    plt.xlabel("Linear evaluation epoch")
    plt.ylabel("CIFAR-10 test accuracy (%)")
    plt.title("Linear Evaluation Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_loss_curve(path, with_head, without_head):
    plt.figure(figsize=(8, 5))
    plt.plot(
        [row["epoch"] for row in with_head["history"]],
        [row["train_loss"] for row in with_head["history"]],
        label="With projection head",
        linewidth=2.2,
    )
    plt.plot(
        [row["epoch"] for row in without_head["history"]],
        [row["train_loss"] for row in without_head["history"]],
        label="Without projection head",
        linewidth=2.2,
    )
    plt.xlabel("Linear evaluation epoch")
    plt.ylabel("Linear classifier train loss")
    plt.title("Linear Evaluation Training Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_bar_chart(path, with_head, without_head):
    labels = ["Best accuracy", "Final accuracy"]
    yes_values = [with_head["best_test_acc"], with_head["final_test_acc"]]
    no_values = [without_head["best_test_acc"], without_head["final_test_acc"]]

    x_positions = [0, 1]
    width = 0.34
    plt.figure(figsize=(7, 5))
    plt.bar(
        [x - width / 2 for x in x_positions],
        yes_values,
        width=width,
        label="With projection head",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        no_values,
        width=width,
        label="Without projection head",
    )
    plt.xticks(x_positions, labels)
    plt.ylabel("CIFAR-10 test accuracy (%)")
    plt.title("Projection Head Ablation")
    plt.ylim(0, max(yes_values + no_values) + 8)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    for x, value in zip([x - width / 2 for x in x_positions], yes_values):
        plt.text(x, value + 0.3, fmt(value), ha="center", va="bottom")
    for x, value in zip([x + width / 2 for x in x_positions], no_values):
        plt.text(x, value + 0.3, fmt(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def markdown_table(with_head, without_head):
    lines = [
        "| Setting | Feature dim | Best acc (%) | Best epoch | Final acc (%) |",
        "|---|---:|---:|---:|---:|",
        "| With projection head | {} | {} | {} | {} |".format(
            with_head["feature_dim"],
            fmt(with_head["best_test_acc"]),
            with_head["best_epoch"],
            fmt(with_head["final_test_acc"])),
        "| Without projection head | {} | {} | {} | {} |".format(
            without_head["feature_dim"],
            fmt(without_head["best_test_acc"]),
            without_head["best_epoch"],
            fmt(without_head["final_test_acc"])),
    ]
    return "\n".join(lines)


def write_markdown_report(path, with_head, without_head, output_dir):
    best_delta = with_head["best_test_acc"] - without_head["best_test_acc"]
    final_delta = with_head["final_test_acc"] - without_head["final_test_acc"]

    if best_delta > 0:
        conclusion = (
            "The model trained with a nonlinear projection head achieves higher "
            "linear-evaluation accuracy, suggesting that the projection head helps "
            "the encoder learn more transferable representations.")
    elif best_delta < 0:
        conclusion = (
            "The model trained without a projection head achieves higher "
            "linear-evaluation accuracy in this run, suggesting that the projection "
            "head did not improve the frozen encoder under this setting.")
    else:
        conclusion = (
            "Both settings reach the same best linear-evaluation accuracy in this run.")

    report = """# Projection Head Ablation

## Result Table

{table}

## Key Differences

- Best accuracy difference: {best_delta} percentage points.
- Final accuracy difference: {final_delta} percentage points.
- With projection head best epoch: {with_best_epoch}.
- Without projection head best epoch: {without_best_epoch}.

## Figures

![Accuracy curve](accuracy_curve.png)

![Training loss curve](loss_curve.png)

![Accuracy comparison](accuracy_bar.png)

## Report-Ready Interpretation

In this ablation, the only intended variable is whether the SimCLR model uses
the nonlinear projection head during contrastive pretraining. After pretraining,
the encoder is frozen and evaluated with the same linear classifier protocol on
CIFAR-10.

{conclusion}

This supports the common SimCLR interpretation that the projection head separates
the contrastive objective space from the representation space used by downstream
tasks. The NT-Xent loss can shape the projected vector while preserving more
useful semantic information in the encoder output, which is what the linear
classifier evaluates.
""".format(
        table=markdown_table(with_head, without_head),
        best_delta=fmt(best_delta),
        final_delta=fmt(final_delta),
        with_best_epoch=with_head["best_epoch"],
        without_best_epoch=without_head["best_epoch"],
        conclusion=conclusion,
    )

    with open(path, "w") as f:
        f.write(report)


def main():
    args = parse_args()
    with_head = load_json(args.with_head)
    without_head = load_json(args.without_head)
    ensure_output_dir(args.output_dir)

    best_delta = with_head["best_test_acc"] - without_head["best_test_acc"]
    final_delta = with_head["final_test_acc"] - without_head["final_test_acc"]

    summary_csv = os.path.join(args.output_dir, "summary.csv")
    accuracy_curve = os.path.join(args.output_dir, "accuracy_curve.png")
    loss_curve = os.path.join(args.output_dir, "loss_curve.png")
    accuracy_bar = os.path.join(args.output_dir, "accuracy_bar.png")
    report_md = os.path.join(args.output_dir, "projection_head_ablation_report.md")

    write_summary_csv(summary_csv, with_head, without_head)
    plot_accuracy_curve(accuracy_curve, with_head, without_head)
    plot_loss_curve(loss_curve, with_head, without_head)
    plot_bar_chart(accuracy_bar, with_head, without_head)
    write_markdown_report(report_md, with_head, without_head, args.output_dir)

    print(markdown_table(with_head, without_head))
    print("")
    print("Best accuracy delta, with - without: {} percentage points".format(
        fmt(best_delta)))
    print("Final accuracy delta, with - without: {} percentage points".format(
        fmt(final_delta)))

    if best_delta > 0:
        print("Conclusion: the nonlinear projection head improves the frozen encoder representation in this setting.")
    elif best_delta < 0:
        print("Conclusion: the run without projection head gives higher linear-eval accuracy in this setting.")
    else:
        print("Conclusion: both settings give the same best linear-eval accuracy in this setting.")
    print("")
    print("Saved report artifacts:")
    print("- {}".format(report_md))
    print("- {}".format(summary_csv))
    print("- {}".format(accuracy_curve))
    print("- {}".format(loss_curve))
    print("- {}".format(accuracy_bar))


if __name__ == "__main__":
    main()
