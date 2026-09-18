# Breast Cancer Classification: Benign vs. Malignant (Mendeley)

This project trains a high-accuracy deep learning model to classify breast cancer images as either benign or malignant.

## Objective

The goal is to build a robust classifier using a pre-trained Convolutional Neural Network (CNN) on an augmented dataset of breast cancer images (mammograms).

## Dataset & Acknowledgements

This project uses the "Mammogram Mastery: A Robust Dataset for Breast Cancer Detection and Medical Education" dataset available on Mendeley Data.

**Citation:**

> Aqdar , Karzan Barzan; Abdalla, Peshraw Ahmed; Mustafa , Rawand Kawa; Abdulqadir, Zhiyar Hamid; Qadir, Abdalbasit Mohammed; Shali, Alla Abdulqader; Aziz, Nariman Muhamad (2024), “Mammogram Mastery: A Robust Dataset for Breast Cancer Detection and Medical Education”, Mendeley Data, V1, doi: 10.17632/fvjhtskg93.1

**Direct Link:**
[https://data.mendeley.com/datasets/fvjhtskg93/1](https://data.mendeley.com/datasets/fvjhtskg93/1)

* **Methodology (original, `train_resnet50.ipynb`):**
    1.  The "Original Dataset" was held back, intended as a final, unseen test set.
    2.  The "Augmented Dataset" was split into an 80% training set and a 20% validation set.
    3.  The model was trained *only* on the augmented data, which helps it learn to be robust against variations in noise, rotation, and cropping.

**⚠️ This split leaks the test set into training.** The Mendeley dataset's own augmentation script
(`Image_Augmentation.py`) saves an unmodified copy of every original image into the augmented
folder under the same filename, alongside 12 augmented variants. So the "Augmented Dataset" already
contains the "Original Dataset" — splitting it 80/20 and then testing on the originals means testing
on data the model trained on. Verified pixel-for-pixel and reproducible via `audit_leakage.py`
(full findings in the [Leakage Audit](#leakage-audit) section below).

**Corrected methodology (`train_resnet50_groupsplit.ipynb`, `make_group_split.py`):** every original
image and its 12 augmented variants are grouped as one lesion and kept together in exactly one of
train/val/test, so no test lesion's pixels appear in training in any form. Same architecture, same
hyperparameters.

## Model & Training

* **Model:** We used a **ResNet50** model, pre-trained on ImageNet, and fine-tuned it for this binary classification task.
* **Technique:** This is a **transfer learning** project. We replaced the final layer of the ResNet50 to output 2 classes ("Cancer" and "Non-Cancer") and fine-tuned the model.
* **Training:** The model was trained for 15 epochs. We monitored the validation accuracy at each epoch and saved the model weights that achieved the best performance.

## Results

* **Originally reported (leaked, `train_resnet50.ipynb`):** 99.7% validation, 99.6% full-augmented-set, 99.7% "unseen" test — all three numbers measured the same contaminated pool. Not a valid generalisation result; see the audit above.
* **Corrected result (`train_resnet50_groupsplit.ipynb`), on a genuinely held-out lesion-level test set (962 images / 74 lesions):**
    * **Test accuracy: 97.09%** (majority-class baseline on this test set: 83.78%)
    * **Cancer recall: 91.0%**, Non-Cancer recall: 98.3%
    * **ROC-AUC: 0.993**
    * Confusion matrix (rows=true, cols=pred; Cancer, Non-Cancer): `[[142, 14], [14, 792]]`

The corrected model clears the majority-class baseline by 13.3 points with zero lesion overlap
between train and test. This is the number to quote, not the 99.7%.

## Leakage Audit

**Confirmed 19 Sep 2026: the originally reported 99.7% test accuracy was data leakage, not
generalisation.** Three independent checks:

1. **The dataset's own generator writes the leak.** `Image_Augmentation.py` — the augmentation
   script shipped with the dataset — saves an unmodified copy of each original image into the
   augmented folder under the same filename, before any augmenter runs:
   ```python
   original_path = os.path.join(augmented_folder, original_name)
   Image.fromarray(original_image).save(original_path)
   ```
2. **Pixel comparison, all 745 pairs.** Every `Original Dataset/<class>/IMG (N).jpg` has a
   same-named counterpart in `Augmented Dataset/<class>/`. All 745 pairs match on pixel dimensions;
   16×16 perceptual-hash distance averages 0.33–1.87 bits out of 240 (a control of 300 random pairs
   of *different* originals averages 59.3).
3. **Exact overlap with the training data used.** Checking `train_resnet50.ipynb`'s 80/20 split
   against the "unseen" Original test set: 592/745 (79.5%) test images are verbatim inside the
   training folder, the other 153 (20.5%) inside validation — 100% of the 745 test lesions have a
   variant in training.

Two caveats from the earlier audit, resolved rather than exculpatory:
- *"Not byte-identical"* — true, but that's the JPEG re-encode from point 1 above (e.g.
  `IMG (1).jpg`: 237KB original vs 78KB augmented copy), not a genuine transform.
- *"Dataset removed from the repo"* — incorrect; it was never removed, it sits under
  `data/Breast Cancer Dataset Mendeley/` in `HEAD`.

**Whose mistake this is:** the dataset itself ships with the original folded into the augmented set
— that structure isn't something this repo's code generated. The error was treating the dataset's
`Original`/`Augmented` folder names as a valid train/test split.

Reproduce with `python audit_leakage.py`. The corrected, lesion-level split and retrain are in
`make_group_split.py` and `train_resnet50_groupsplit.ipynb`; results are in the Results section
above.

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
