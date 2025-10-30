# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 05:23:49 2024

@author: Peshraw
"""

import os
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image

# Set up augmentation parameters with increased ranges
seq = iaa.Sequential([
    iaa.Fliplr(1),  # horizontal flips with 50% probability
    iaa.Rotate((-45, 45)),  # random rotations (-45 to 45 degrees)
    iaa.GaussianBlur(sigma=(0, 2.0)),  # gaussian blur (sigma range: 0 to 2.0)
    iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),  # gaussian noise (scale range: 0 to 0.2*255)
    iaa.Dropout(p=(0, 0.2)),  # random dropout (probability range: 0 to 0.2)
    iaa.Resize({"height": (0.5, 1.5), "width": (0.5, 1.5)}),  # random scaling (height and width scale range: 0.5 to 1.5)
    iaa.Crop(percent=(0, 0.2)),  # random cropping (percent range: 0 to 0.2)
    iaa.ElasticTransformation(alpha=(0, 10.0), sigma=1.0),  # elastic transformation (alpha range: 0 to 10.0)
    iaa.PiecewiseAffine(scale=(0.02, 0.1)),  # piecewise affine transformation (scale range: 0.02 to 0.1)
    iaa.PerspectiveTransform(scale=(0.05, 0.15)),  # perspective transformation (scale range: 0.05 to 0.15)
    iaa.LinearContrast((0.2, 3.0)),  # contrast normalization (contrast range: 0.2 to 3.0)
    iaa.Multiply((0.5, 1.5), per_channel=0.5),  # random brightness adjustment (brightness scale range: 0.5 to 1.5)
], random_order=True)  # apply augmenters in random order

# Load original images
original_folder = r"D:\Cancer Dataset\Original Dataset\Non-Cancer"
augmented_folder = r"D:\Cancer Dataset\Original Dataset\augmented"

# Create augmented folder if it doesn't exist
os.makedirs(augmented_folder, exist_ok=True)

# Process each original image
for i, filename in enumerate(os.listdir(original_folder), start=1):
    img_path = os.path.join(original_folder, filename)
    original_image = np.array(Image.open(img_path))
    
    # Save original image
    original_name = filename
    original_path = os.path.join(augmented_folder, original_name)
    Image.fromarray(original_image).save(original_path)
    
    # Perform augmentation and save augmented images
    for j, augmenter in enumerate(seq, start=1):
        augmented_images = augmenter(images=[original_image])
        for k, augmented_image in enumerate(augmented_images, start=1):
            technique = augmenter.name
            if technique.startswith("Unnamed"):
                technique = technique.split("Unnamed")[1].strip()  # Remove "Unnamed" prefix
            if "rotate" in technique:  # If rotation, include the angle in the technique name
                angle = technique.split("rotate")[1].strip()
                augmented_name = f"{filename.split('.')[0]}_rotate{angle}.{filename.split('.')[-1]}"
            elif "dropout" in technique:  # If dropout, include the dropout percentage in the technique name
                dropout = int(technique.split("dropout")[1].strip()) * 10
                augmented_name = f"{filename.split('.')[0]}_dropout{dropout}.{filename.split('.')[-1]}"
            else:
                augmented_name = f"{filename.split('.')[0]}_{technique}.{filename.split('.')[-1]}"
            augmented_path = os.path.join(augmented_folder, augmented_name)
            Image.fromarray(augmented_image).save(augmented_path)
