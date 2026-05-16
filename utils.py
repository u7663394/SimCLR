import json
import os
import shutil

import torch
import yaml


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def save_json(payload, filename):
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_history_plot(history, output_path, title, metric_groups):
    if not history:
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Warning: matplotlib is not installed, skipping plot generation for {output_path}.")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(len(metric_groups), 1, figsize=(8, 4 * len(metric_groups)), sharex=True)
    if len(metric_groups) == 1:
        axes = [axes]

    for axis, group in zip(axes, metric_groups):
        for metric_name, label in group["series"]:
            axis.plot(epochs, [item[metric_name] for item in history], marker="o", label=label)

        axis.set_ylabel(group["ylabel"])
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Epoch")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True


def save_simclr_history_plot(history, output_path, title):
    metric_groups = [
        {"ylabel": "Loss", "series": [("train_loss", "Train loss")]},
        {"ylabel": "Accuracy (%)", "series": [("train_top1", "Train top-1"), ("train_top5", "Train top-5")]},
        {"ylabel": "Learning rate", "series": [("learning_rate", "Learning rate")]},
    ]
    return save_history_plot(history, output_path, title, metric_groups)


def save_linear_eval_history_plot(history, output_path, title):
    metric_groups = [
        {"ylabel": "Loss", "series": [("train_loss", "Train loss"), ("test_loss", "Test loss")]},
        {"ylabel": "Accuracy (%)", "series": [("train_top1", "Train top-1"), ("test_top1", "Test top-1"), ("test_top5", "Test top-5")]},
    ]
    return save_history_plot(history, output_path, title, metric_groups)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
