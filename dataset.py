# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 08:10:06 2024

This script contains functions to load, preprocess, and augment data for a multimodal 
deep learning model.

It includes functions for handling both image and text data.

Author: Siva Siddharth
"""

from torch.utils.data import DataLoader, Dataset


class NutritionDataset(Dataset):
    """
    Custom Dataset for handling image, text, and label data.

    Args:
        images (list): List of processed images.
        texts (list): List of tokenized texts.
        labels (list): List of labels corresponding to the images and texts.

    Returns:
        A tuple (image, text, label) for each item in the dataset.
    """

    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]
        label = self.labels[idx]
        return image, text, label


def create_dataloader(images, texts, labels, batch_size):
    """
    Creates a DataLoader for the NutritionDataset.

    Args:
        images (list): List of processed images.
        texts (list): List of tokenized texts.
        labels (list): List of labels corresponding to the images and texts.
        batch_size (int): The size of batches the DataLoader should return.

    Returns:
        DataLoader: An iterable over the dataset.
    """
    dataset = NutritionDataset(images, texts, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
