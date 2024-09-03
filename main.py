# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 08:12:54 2024

This script loads data, configures the model, trains it, and evaluates its performance on a 
multimodal dataset.

It handles both image and text data, combining the features from both to make predictions.

Author: Siva Siddharth
"""

import torch
import torch.nn as nn
import yaml
from preprocess import (
    load_image,
    load_text,
    load_data,
    augment_data,
    preprocess_images,
    preprocess_texts,
)
from dataset import create_dataloader
from model import MultiModalModel
from train import train_model, evaluate_model
from infer import inference

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

image_paths = (config["image_paths"],)
text_paths = (config["text_paths"],)
batch_size = (config["batch_size"],)
image_model_name = (config["image_model_name"],)
image_output_features = (config["image_output_features"],)
text_vocab_size = (config["text_vocab_size"],)
embedding_dim = (config["embedding_dim"],)
lstm_hidden_size = (config["lstm_hidden_size"],)
text_output_features = (config["text_output_features"],)
num_classes = (config["num_classes"],)
epochs = config["epochs"]
label_paths = config["label_paths"]
new_image_path = config["new_image_path"]
new_text_path = config["new_text_path"]

if __name__ == "__main__":
    images, texts, labels = load_data(image_paths, text_paths, label_paths)

    images, texts = augment_data(images, texts)

    processed_images = preprocess_images(images)
    processed_texts = preprocess_texts(texts)
    dataloader = create_dataloader(processed_images, processed_texts, labels, batch_size)

    model = MultiModalModel(
        image_model_name=config["image_model_name"],
        image_output_features=config["image_output_features"],
        text_vocab_size=config["text_vocab_size"],
        embedding_dim=config["embedding_dim"],
        lstm_hidden_size=config["lstm_hidden_size"],
        text_output_features=config["text_output_features"],
        num_classes=config["num_classes"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_model(model, dataloader, criterion, optimizer, epochs)
    evaluate_model(model, dataloader)

    # Example inference
    new_image = load_image(new_image_path)
    new_text = load_text(new_text_path)
    result = inference(model, new_image, new_text)
    print(f"Inference Result: {result}")
