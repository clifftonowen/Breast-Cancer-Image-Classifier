# Breast Cancer Image Classifier

This repository documents the development and exploration of deep learning models for classifying breast cancer images. The primary objective is to build robust models for identifying cancerous tissue from various medical imaging sources.

## Project Objectives

1.  **Develop High-Accuracy Classifiers:** To build and fine-tune various Convolutional Neural Network (CNN) architectures to accurately classify breast cancer.
2.  **Explore Different Data Modalities:** To investigate challenges related to different image types, starting with mammograms/ultrasound (JPG/PNG) and progressing to more complex DICOM-formatted images from larger clinical datasets.

## Repository Structure

This repository is organized into distinct projects:

* **`/1_binary_classification_mendeley/`**:
    * **Goal:** Train ResNet50 Model on a public Mendeley dataset of breast cancer images (JPG/PNG).
    * **Task:** Binary Classification (Cancer vs. Non-Cancer).
    * **Status:** *[ "Completed"]*

* **`/2_binary_classification_dicom/`**:
    * **Goal:** To build a data pipeline and train a model on a large-scale, clinical dataset from The Cancer Imaging Archive (TCIA) in DICOM format.
    * **Task:** Binary Classification (Benign vs. Malignant).
    * **Status:** *[ "In Progress"]*    
