
import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "lab2_data"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("food11")


def load_data(dataset_name, batch_size):
    dataset_dir = DATA_ROOT / (
        "food11_processed_mini"
        if dataset_name == "mini"
        else "food11_processed"
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_data = datasets.ImageFolder(
        dataset_dir / "training", transform=transform
    )
    val_data = datasets.ImageFolder(
        dataset_dir / "validation", transform=transform
    )
    test_data = datasets.ImageFolder(
        dataset_dir / "evaluation", transform=transform
    )

    if len(train_data.classes) != 11:
        raise ValueError("Expected 11 Food-11 classes.")

    if not (
        train_data.class_to_idx
        == val_data.class_to_idx
        == test_data.class_to_idx
    ):
        raise ValueError("Class mappings do not match.")

    train_loader = DataLoader(
        train_data, batch_size=batch_size,
        shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def build_model():
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze pretrained layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the original 1000-class layer
    model.fc = nn.Linear(model.fc.in_features, 11)

    return model


def train_epoch(model, loader, criterion, optimizer):
    # Keep frozen backbone and batch normalization fixed
    model.eval()
    model.fc.train()

    total_loss = 0.0
    total_images = 0

    for images, labels in loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)

    return total_loss / total_images


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total_images += images.size(0)

    average_loss = total_loss / total_images
    accuracy = correct / total_images

    return average_loss, accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", choices=["mini", "processed"],
        default="mini"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")

    torch.manual_seed(42)
    device = torch.device("cpu")

    with mlflow.start_run() as run:

        mlflow.log_params({
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "architecture": "resnet18",
            "pretrained": True,
            "trainable_layers": "fc",
            "seed": 42,
        })

        train_loader, val_loader, test_loader = load_data(
            args.dataset, args.batch_size
        )

        model = build_model().to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.fc.parameters(),
            lr=args.lr
        )

        print("Run ID:", run.info.run_id)

        for epoch in range(1, args.epochs + 1):

            train_loss = train_epoch(
                model, train_loader, criterion, optimizer
            )

            val_loss, val_accuracy = evaluate(
                model, val_loader, criterion
            )

            mlflow.log_metric(
                "train_loss", train_loss, step=epoch
            )
            mlflow.log_metric(
                "val_loss", val_loss, step=epoch
            )
            mlflow.log_metric(
                "val_accuracy", val_accuracy, step=epoch
            )

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.2%}",
                flush=True,
            )

        _, test_accuracy = evaluate(
            model, test_loader, criterion
        )

        mlflow.log_metric("test_accuracy", test_accuracy)

        model.eval()
        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )

        print(f"Test Accuracy: {test_accuracy:.2%}")
        print("Training completed!")


if __name__ == "__main__":
    main()