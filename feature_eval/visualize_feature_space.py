import argparse
import csv
import inspect
import json
import os
import random
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "simclr_matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare SimCLR feature spaces with t-SNE. Defaults compare the "
            "baseline projection-head run and a random ResNet encoder."))
    parser.add_argument(
        "--baseline-result",
        default="feature_eval/results/projection_head_yes.json",
        help="Linear-eval JSON for the baseline model.")
    parser.add_argument(
        "--baseline-label",
        default="Baseline (with projection head)",
        help="Display label for the baseline model.")
    parser.add_argument(
        "--random-label",
        default="Random initialization",
        help="Display label for the random initialized encoder.")
    parser.add_argument(
        "--data",
        default="./datasets_local",
        help="Dataset root directory.")
    parser.add_argument(
        "--dataset-name",
        default="cifar10",
        choices=["cifar10"],
        help="Dataset to visualize.")
    parser.add_argument(
        "--arch",
        default=None,
        help="Backbone architecture. Defaults to the baseline result/checkpoint arch.")
    parser.add_argument(
        "--sample-size",
        default=1000,
        type=int,
        help="Number of CIFAR-10 test images used for t-SNE.")
    parser.add_argument(
        "--batch-size",
        default=256,
        type=int,
        help="Batch size for feature extraction.")
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        help="Number of dataloader workers.")
    parser.add_argument(
        "--perplexity",
        default=30.0,
        type=float,
        help="t-SNE perplexity. It is clipped automatically for small samples.")
    parser.add_argument(
        "--pca-dim",
        default=50,
        type=int,
        help="PCA dimension before t-SNE. Set 0 to disable PCA.")
    parser.add_argument(
        "--tsne-iter",
        default=1000,
        type=int,
        help="Number of t-SNE optimization iterations.")
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for sampling, random initialization, PCA, and t-SNE.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device used for feature extraction.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download CIFAR-10 if it is not present locally.")
    parser.add_argument(
        "--output-dir",
        default="feature_eval/report/feature_space_analysis",
        help="Directory for figures, CSV files, and the markdown report.")
    parser.add_argument(
        "--dpi",
        default=220,
        type=int,
        help="Figure resolution.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def slugify(label):
    chars = []
    for char in label.lower():
        if char.isalnum():
            chars.append(char)
        elif char in [" ", "-", "_", "(", ")"]:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def build_resnet(arch):
    model_fn = getattr(models, arch)
    try:
        return model_fn(weights=None)
    except TypeError:
        return model_fn(pretrained=False)


def load_checkpoint_encoder(checkpoint_path, arch, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = arch or checkpoint.get("arch", "resnet18")

    encoder = build_resnet(arch)
    feature_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()

    encoder_state = {}
    for key, value in checkpoint["state_dict"].items():
        if not key.startswith("backbone."):
            continue
        key = key[len("backbone."):]
        if key.startswith("fc."):
            continue
        encoder_state[key] = value

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise RuntimeError(
            "Unexpected encoder keys in {}: {}".format(checkpoint_path, unexpected))
    if missing:
        print("Warning: missing encoder keys in {}: {}".format(
            checkpoint_path, missing))

    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, feature_dim, arch


def load_random_encoder(arch, device, seed):
    set_seed(seed)
    encoder = build_resnet(arch)
    feature_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, feature_dim


def get_cifar10_test_dataset(data_root, download):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return datasets.CIFAR10(
        data_root, train=False, transform=transform, download=download)


def stratified_sample_indices(targets, sample_size, seed):
    targets = np.asarray(targets)
    unique_classes = sorted(np.unique(targets).tolist())
    sample_size = min(sample_size, len(targets))
    base_per_class = sample_size // len(unique_classes)
    remainder = sample_size % len(unique_classes)

    rng = np.random.RandomState(seed)
    selected = []
    for offset, class_id in enumerate(unique_classes):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        n_take = base_per_class + (1 if offset < remainder else 0)
        n_take = min(n_take, len(class_indices))
        selected.extend(class_indices[:n_take].tolist())

    rng.shuffle(selected)
    return selected


def make_loader(dataset, indices, batch_size, workers, device):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda")


def extract_features(encoder, loader, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            output = encoder(images).cpu().numpy()
            features.append(output)
            labels.append(targets.cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def tsne_kwargs(n_samples, args):
    perplexity = min(args.perplexity, max(2.0, (n_samples - 1) / 3.0))
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": 200.0,
        "init": "pca",
        "random_state": args.seed,
        "verbose": 0,
    }
    signature = inspect.signature(TSNE)
    if "max_iter" in signature.parameters:
        kwargs["max_iter"] = args.tsne_iter
    else:
        kwargs["n_iter"] = args.tsne_iter
    return kwargs


def compute_tsne(features, args):
    features = normalize_features(features)
    if args.pca_dim and features.shape[1] > args.pca_dim:
        pca = PCA(n_components=args.pca_dim, random_state=args.seed)
        features = pca.fit_transform(features)
    embedding = TSNE(**tsne_kwargs(features.shape[0], args)).fit_transform(features)
    return embedding


def plot_tsne_comparison(path, embeddings_by_slug, labels, model_specs, args):
    n_models = len(model_specs)
    fig, axes = plt.subplots(1, n_models, figsize=(5.3 * n_models, 4.8))
    if n_models == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10", 10)

    for ax, spec in zip(axes, model_specs):
        embedding = embeddings_by_slug[spec["slug"]]
        for class_id, class_name in enumerate(CIFAR10_CLASSES):
            mask = labels == class_id
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=10,
                alpha=0.72,
                color=cmap(class_id),
                label=class_name,
                linewidths=0)
        ax.set_title(spec["label"])
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.15)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("CIFAR-10 Encoder Feature Space t-SNE", y=1.03)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_single_tsne(path, embedding, labels, title, args):
    plt.figure(figsize=(6.2, 5.2))
    cmap = plt.get_cmap("tab10", 10)
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_id
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=11,
            alpha=0.74,
            color=cmap(class_id),
            label=class_name,
            linewidths=0)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.15)
    plt.legend(loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=args.dpi)
    plt.close()


def write_tsne_csv(path, embeddings_by_slug, labels, sample_indices, model_specs):
    fieldnames = [
        "model",
        "sample_index",
        "label_id",
        "label_name",
        "tsne_x",
        "tsne_y",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in model_specs:
            embedding = embeddings_by_slug[spec["slug"]]
            for idx, label, point in zip(sample_indices, labels, embedding):
                writer.writerow({
                    "model": spec["label"],
                    "sample_index": idx,
                    "label_id": int(label),
                    "label_name": CIFAR10_CLASSES[int(label)],
                    "tsne_x": float(point[0]),
                    "tsne_y": float(point[1]),
                })


def write_markdown_report(path, args):
    report = """# Feature Space Visualization Analysis

## Setup

- Dataset: CIFAR-10 test split.
- Sample size for t-SNE: `{sample_size}` images.
- t-SNE perplexity: `{perplexity}`.
- Random seed: `{seed}`.

## Figures

![t-SNE comparison](tsne_comparison.png)

## Report-Ready Interpretation

The t-SNE plots provide a qualitative view of the encoder feature space. If the
baseline SimCLR encoder forms more coherent same-class clusters than the random
encoder, this indicates that contrastive pretraining has learned semantic
structure beyond what is present in a randomly initialized ResNet.

The random initialization is an untrained control. It helps show whether visible
class structure is caused by learned representations rather than only by the
architecture or CIFAR-10 input statistics.

## How To Use These Outputs

- Use `tsne_comparison.png` as the qualitative overview of class clustering.
- Use `tsne_coordinates.csv` only if you need to remake or customize the scatter plots.

Note: t-SNE is a qualitative visualization and its axes are not directly
meaningful. Use it to discuss relative cluster separation and class mixing, not
as a precise quantitative metric.
""".format(
        sample_size=args.sample_size,
        perplexity=args.perplexity,
        seed=args.seed,
    )
    with open(path, "w") as f:
        f.write(report)


def make_model_specs(args, baseline_result, arch):
    baseline_checkpoint = resolve_path(baseline_result["checkpoint"])
    return [
        {
            "label": args.baseline_label,
            "slug": slugify(args.baseline_label),
            "checkpoint": baseline_checkpoint,
            "arch": arch,
            "best_acc": baseline_result.get("best_test_acc"),
            "final_acc": baseline_result.get("final_test_acc"),
            "kind": "checkpoint",
        },
        {
            "label": args.random_label,
            "slug": slugify(args.random_label),
            "checkpoint": None,
            "arch": arch,
            "best_acc": None,
            "final_acc": None,
            "kind": "random",
        },
    ]


def main():
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")

    set_seed(args.seed)
    device = get_device(args.device)
    ensure_output_dir(args.output_dir)

    baseline_result = load_json(args.baseline_result)
    arch = args.arch or baseline_result.get("arch")
    if arch is None:
        arch = "resnet18"

    model_specs = make_model_specs(args, baseline_result, arch)

    dataset = get_cifar10_test_dataset(args.data, args.download)
    sample_indices = stratified_sample_indices(
        dataset.targets, args.sample_size, args.seed)
    loader = make_loader(
        dataset, sample_indices, args.batch_size, args.workers, device)

    embeddings_by_slug = {}
    shared_labels = None

    print("Using device: {}".format(device))
    print("Using {} CIFAR-10 test images.".format(len(sample_indices)))

    for spec in model_specs:
        print("Extracting features for {}...".format(spec["label"]))
        if spec["kind"] == "checkpoint":
            encoder, _feature_dim, loaded_arch = load_checkpoint_encoder(
                spec["checkpoint"], spec["arch"], device)
            spec["arch"] = loaded_arch
        else:
            encoder, _feature_dim = load_random_encoder(
                spec["arch"], device, args.seed)

        features, labels = extract_features(encoder, loader, device)
        if shared_labels is None:
            shared_labels = labels
        elif not np.array_equal(shared_labels, labels):
            raise RuntimeError("Model feature extraction returned mismatched labels.")

        print("Computing t-SNE for {}...".format(spec["label"]))
        embeddings_by_slug[spec["slug"]] = compute_tsne(features, args)

    plot_tsne_comparison(
        os.path.join(args.output_dir, "tsne_comparison.png"),
        embeddings_by_slug,
        shared_labels,
        model_specs,
        args)

    for spec in model_specs:
        plot_single_tsne(
            os.path.join(args.output_dir, "tsne_{}.png".format(spec["slug"])),
            embeddings_by_slug[spec["slug"]],
            shared_labels,
            spec["label"],
            args)

    write_tsne_csv(
        os.path.join(args.output_dir, "tsne_coordinates.csv"),
        embeddings_by_slug,
        shared_labels,
        sample_indices,
        model_specs)
    write_markdown_report(
        os.path.join(args.output_dir, "feature_space_analysis_report.md"),
        args)

    print("Saved feature-space analysis to {}".format(args.output_dir))


if __name__ == "__main__":
    main()
