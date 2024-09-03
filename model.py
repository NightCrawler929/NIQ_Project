# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 08:10:06 2024

This script defines the MultiModalModel class, which integrates both image and text processing models.

The model combines features extracted from images and texts to make predictions.

Author: Siva Siddharth
"""
import torch
import torch.nn as nn
from torchvision import models


class MultiModalModel(nn.Module):
    """
    A multimodal deep learning model that processes both images and text.

    Args:
        image_model_name (str): The name of the pre-trained image model to use (e.g., 'resnet18').
        image_output_features (int): The number of output features from the image model.
        text_vocab_size (int): The size of the vocabulary for the text model.
        embedding_dim (int): The dimensionality of the word embeddings.
        lstm_hidden_size (int): The number of hidden units in the LSTM.
        text_output_features (int): The number of output features from the text model.
        num_classes (int): The number of classes for the final classification.

    Returns:
        torch.Tensor: The model's output after processing the inputs.
    """

    def __init__(
        self,
        image_model_name,
        image_output_features,
        text_vocab_size,
        embedding_dim,
        lstm_hidden_size,
        text_output_features,
        num_classes,
    ):
        super(MultiModalModel, self).__init__()
        self.image_model = self.build_image_model(image_model_name, image_output_features)
        self.text_model = self.build_text_model(
            text_vocab_size, embedding_dim, lstm_hidden_size, text_output_features
        )
        self.fc = nn.Linear(image_output_features + text_output_features, num_classes)

    def build_image_model(self, model_name, output_features):
        """Builds a CNN model for image processing."""
        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=True)

        # Replace the final layer with a custom one that outputs the desired number of features
        model.fc = nn.Linear(model.fc.in_features, output_features)
        return model

    def build_text_model(self, vocab_size, embedding_dim, hidden_size, output_features):
        """Builds a RNN model for text processing."""
        model = nn.Sequential(
            nn.EmbeddingBag(vocab_size, embedding_dim),
            nn.LSTM(embedding_dim, hidden_size, batch_first=True),
            nn.Linear(hidden_size, output_features),
        )
        return model

    def forward(self, image, text):
        """
        Defines the forward pass of the model.

        Args:
            image (torch.Tensor): Input image tensor.
            text (torch.Tensor): Input text tensor.

        Returns:
            torch.Tensor: Output tensor after combining image and text features.
        """
        image_features = self.image_model(image)
        text_features = self.text_model(text)
        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.fc(combined_features)
        return output
