# Pneumonia Detection Using CNN (K-Fold Cross-Validated)

This MATLAB project implements a Convolutional Neural Network (CNN) for classifying chest X-ray images as **Normal** or **Pneumonia**, using a custom architecture and **k-fold cross-validation**. The model is trained using the `trainNetwork` function with data augmentation and evaluated using confusion matrices, accuracy, precision, recall, and ROC curves.

## Features

- Custom CNN with batch normalization and ReLU activations
- 2-fold cross-validation
- Data augmentation with random rotation, reflection, and translation
- Confusion matrix, accuracy, precision, recall, and ROC curve evaluation
- Works on datasets with folder-based label organization

## File

- `KFCNN.m` — Full training, evaluation, and plotting script

## Dataset

This script assumes the dataset is structured like:

```
chest_xray/
├── PNEUMONIA/
│   ├── image1.jpeg
│   └── ...
├── NORMAL/
│   ├── image1.jpeg
│   └── ...
```

> Dataset must be local. You can use public datasets like [Kermany et al. (2018)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Instructions

- Open `KFCNN.m` in MATLAB
- Set the path in `dataPath = '/your/local/path/to/chest_xray/';`
- Run the script

## Outputs

- Console: accuracy and confusion matrix per fold + overall
- Plots:
  - CNN Architecture
  - Confusion Matrix
  - ROC Curve with AUC

## Author

**Ashwin Mohan**  
MSc Diagnostics, Data and Digital Health  
University of Warwick
