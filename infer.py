# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 08:12:44 2024

This script defines the inference function for making predictions using the multimodal model.

Author: Siva Siddharth
"""

import torch
from preprocess import preprocess_images, preprocess_texts


def inference(model, image, text):
    """
    Performs inference using the trained model on a single image and text input.

    Args:
        model (nn.Module): The trained multimodal model.
        image (Image): The input image for inference.
        text (str): The input text for inference.

    Returns:
        torch.Tensor: The models output for the given image and text.
    """
    model.eval()
    with torch.no_grad():
        processed_image = preprocess_images([image])
        processed_text = preprocess_texts([text])
        output = model(processed_image, processed_text)
        return output
