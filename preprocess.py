# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 08:05:47 2024

This script contains functions to load and preprocess image and text data.

Author: Siva Siddharth
"""

import random
from PIL import Image
import numpy as np
from nltk.tokenize import word_tokenize


def load_image(path):
    """Loads an image from a given path."""
    return Image.open(path)


def load_text(path):
    """Loads text data from a given path."""
    with open(path, "r") as file:
        return file.read()


def resize(image, size=(224, 224)):
    """Resizes the image to the specified size."""
    return image.resize(size)


def normalize(image):
    """Normalizes the image pixel values to be between 0 and 1."""
    return np.array(image) / 255.0


def tokenize(text):
    """Tokenizes the input text into words."""
    return word_tokenize(text)


def random_crop(image, crop_size=(200, 200)):
    """Randomly crops the image to the specified size."""
    pass


def random_insert(text, insertion="sample"):
    """Randomly inserts a word into the text."""
    words = text.split()
    insert_position = random.randint(0, len(words))
    words.insert(insert_position, insertion)
    return " ".join(words)


def load_data(image_paths, text_paths, label_paths):
    """
    Loads images, texts, and labels from the provided paths.

    Args:
        image_paths (list): List of paths to image files.
        text_paths (list): List of paths to text files.
        label_paths (str): Path to label file.

    Returns:
        tuple: (images, texts, labels) where images and texts are lists and labels 
        is a list of integers.
    """
    # Load images
    images = [load_image(path) for path in image_paths]
    # Load associated text
    texts = [load_text(path) for path in text_paths]
    # Load associated label
    with open(label_paths, "r") as file:
        labels = [int(line.strip()) for line in file]
    return images, texts, labels


def preprocess_images(images):
    """
    Preprocesses a list of images by resizing and normalizing them.

    Args:
        images (list): List of images to preprocess.

    Returns:
        list: List of processed images.
    """
    processed_images = []
    for image in images:
        image = resize(image)
        image = normalize(image)
        processed_images.append(image)
    return processed_images


def preprocess_texts(texts):
    """
    Tokenizes and processes a list of texts.

    Args:
        texts (list): List of texts to preprocess.

    Returns:
        list: List of tokenized texts.
    """
    processed_texts = []
    for text in texts:
        tokens = tokenize(text.lower())
        processed_texts.append(tokens)
    return processed_texts


def augment_data(images, texts):
    """
    Applies data augmentation to images and texts.

    Args:
        images (list): List of images to augment.
        texts (list): List of texts to augment.

    Returns:
        tuple: (augmented_images, augmented_texts)
    """
    augmented_images = []
    augmented_texts = []
    for image, text in zip(images, texts):
        augmented_images.append(random_crop(image))
        augmented_texts.append(random_insert(text))
    return augmented_images, augmented_texts
