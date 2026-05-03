import argparse
import json
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Linear evaluation for SimCLR checkpoints")
    parser.add_argument("--data", default="./datasets", type=str, help="dataset root")
    parser.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--checkpoint-path", required=True, type=str, help="SimCLR checkpoint path")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--label-fraction", default=1.0, type=float, choices=[0.01, 0.1, 1.0])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--output-dir", default=None, type=str)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch):
    if arch == "resnet18":
        return torchvision.models.resnet18(weights=None, num_classes=10)
    return torchvision.models.resnet50(weights=None, num_classes=10)


def load_encoder_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state = checkpoint["state_dict"]
    encoder_state = {}

    for key, value in checkpoint_state.items():
        if key.startswith("backbone.") and not key.startswith("backbone.fc"):
            encoder_state[key[len("backbone."):]] = value

    load_result = model.load_state_dict(encoder_state, strict=False)
    expected_missing = {"fc.weight", "fc.bias"}
    if set(load_result.missing_keys) != expected_missing:
        raise RuntimeError(f"Unexpected missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {load_result.unexpected_keys}")

    return checkpoint


def build_subset(dataset, label_fraction, seed):
    if label_fraction >= 1.0:
        return dataset

    rng = random.Random(seed)
    class_to_indices = {}
    for index, label in enumerate(dataset.targets):
        class_to_indices.setdefault(label, []).append(index)

    selected_indices = []
    for label in sorted(class_to_indices):
        indices = class_to_indices[label]
        rng.shuffle(indices)
        count = max(1, int(len(indices) * label_fraction))
        selected_indices.extend(indices[:count])

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


def build_dataloaders(args):
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(args.data, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(args.data, train=False, transform=transform, download=True)
    train_dataset = build_subset(train_dataset, args.label_fraction, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=not args.disable_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=not args.disable_cuda,
    )
    return train_loader, test_loader


def freeze_encoder(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in {"fc.weight", "fc.bias"}


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            total_loss += loss.item()
            total_top1 += top1[0].item()
            total_top5 += top5[0].item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "top1": total_top1 / num_batches,
        "top5": total_top5 / num_batches,
    }


def train_linear_classifier(model, train_loader, test_loader, args, device):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_train_top1 = 0.0
        num_batches = 0

        for images, labels in tqdm(train_loader, desc=f"Linear eval epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            top1 = accuracy(logits, labels, topk=(1,))[0]
            total_train_top1 += top1[0].item()
            num_batches += 1

        train_top1 = total_train_top1 / num_batches
        test_metrics = evaluate(model, test_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_top1": train_top1,
            "test_top1": test_metrics["top1"],
            "test_top5": test_metrics["top5"],
            "test_loss": test_metrics["loss"],
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}\t"
            f"Train Top1 {train_top1:.2f}\t"
            f"Test Top1 {test_metrics['top1']:.2f}\t"
            f"Test Top5 {test_metrics['top5']:.2f}"
        )

    return history


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    train_loader, test_loader = build_dataloaders(args)
    model = build_model(args.arch).to(device)
    checkpoint = load_encoder_weights(model, args.checkpoint_path, device)
    freeze_encoder(model)

    history = train_linear_classifier(model, train_loader, test_loader, args, device)
    best_epoch = max(history, key=lambda item: item["test_top1"])

    result = {
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
        "label_fraction": args.label_fraction,
        "epochs": args.epochs,
        "arch": args.arch,
        "checkpoint_metadata": {
            "use_projection_head": checkpoint.get("use_projection_head"),
            "aug_strength": checkpoint.get("aug_strength"),
            "feature_dim": checkpoint.get("feature_dim"),
            "projection_dim": checkpoint.get("projection_dim"),
        },
        "best_epoch": best_epoch,
        "history": history,
    }

    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        args.output_dir = os.path.join(checkpoint_dir, f"linear_eval_{checkpoint_name}_labels_{args.label_fraction}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "results.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Best Test Top1: {best_epoch['test_top1']:.2f}")
    print("Saved linear evaluation results.")


if __name__ == "__main__":
    main()
