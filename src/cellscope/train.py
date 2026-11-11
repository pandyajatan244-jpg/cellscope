"""Training loop for the CNN."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(model, loader: DataLoader, optimizer, criterion, device: str) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


def evaluate_epoch(model, loader: DataLoader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n
