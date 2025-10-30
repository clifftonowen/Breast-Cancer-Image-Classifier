# Breast Cancer Classification: Benign vs. Malignant (Mendeley)

This project trains a high-accuracy deep learning model to classify breast cancer images as either benign or malignant.

## Objective

The goal is to build a robust classifier using a pre-trained Convolutional Neural Network (CNN) on an augmented dataset of breast cancer images (mammograms).

## Dataset

* **Source:** The original images are from the "Breast Cancer (Ultrasound, Mammography, and Histopathology)" dataset available on Mendeley Data.
* **Link:** [https://data.mendeley.com/datasets/fvjhtskg93/1](https://data.mendeley.com/datasets/fvjhtskg93/1)
* **Methodology:**
    1.  The "Original Dataset" was held back to be used as our final, unseen test set.
    2.  The "Augmented Dataset" was split into an 80% training set and a 20% validation set.
    3.  The model was trained *only* on the augmented data, which helps it learn to be robust against variations in noise, rotation, and cropping.

## Model & Training

* **Model:** We used a **ResNet50** model, pre-trained on ImageNet, and fine-tuned it for this binary classification task.
* **Technique:** This is a **transfer learning** project. We replaced the final layer of the ResNet50 to output 2 classes ("Cancer" and "Non-Cancer") and fine-tuned the model.
* **Training:** The model was trained for 15 epochs. We monitored the validation accuracy at each epoch and saved the model weights that achieved the best performance.

## Results

The model achieved outstanding performance, demonstrating its ability to accurately and reliably classify the images.

* **Best Validation Accuracy:** **99.7%**
* **Accuracy on Full Augmented Set:** **99.6%**
* **Final Test Accuracy (on Original, Unseen Dataset):** **99.7%**

The near-perfect accuracy on the original, unseen dataset confirms that the model successfully learned the underlying features of cancerous vs. non-cancerous tissue and did not simply "memorize" the augmentation patterns.

## How to Use the Pre-trained Model

You can easily load the best-performing ResNet50 model weights to make new predictions.

1.  Make sure you have `torch`, `torchvision`, and `Pillow (PIL)` installed.
2.  The saved weights are in the `resnet50_model/` subfolder.
3.  Use the following code to load the model and predict on a new image:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Ensure these class names match your folder names 
class_names = ['Cancer', 'Non-Cancer'] 

model = models.resnet50(weights=None) # Start with an untrained shell
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Load the Best Weights
save_path = './resnet50_model/resnet50_best_weights.pth'
model.load_state_dict(torch.load(save_path, map_location=device))
model = model.to(device)
model.eval() # Set model to evaluation (inference) mode

# Define the Validation Transform 
# Must be the same transform used during validation
validation_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and Predict on a New Image
# --- Replace with the path to your image ---
img_path = "path/to/your/new_image.png" 

try:
    image = Image.open(img_path).convert('RGB')
    image_tensor = validation_transform(image).unsqueeze(0).to(device)

    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(image_tensor)
        _, pred_index = torch.max(outputs, 1)
        prediction = class_names[pred_index.item()]
    
    print(f"Image: {img_path}")
    print(f"Prediction: {prediction}")

except FileNotFoundError:
    print(f"Error: Image not found at {img_path}")
```
