# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 08:12:00 2024

This script contains functions to train and evaluate the multimodal model.

Author: Siva Siddharth
"""

import torch


def train_model(model, dataloader, criterion, optimizer, epochs):
    """
    Trains the multimodal model for a specified number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader providing batches of images, texts, and labels.
        criterion (nn.Module): Loss function to use.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        epochs (int): Number of training epochs.
    """
    for epoch in range(epochs):
        for images, texts, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


def evaluate_model(model, dataloader):
    """
    Evaluates the model on a dataset and prints accuracy.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader providing batches of images, texts, and labels.

    Returns:
        None: Prints accuracy of the model.
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, texts, labels in dataloader:
            outputs = model(images, texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")
