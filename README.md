# Cats and Dogs Classification using SVM and TensorFlow
This project uses a combination of Support Vector Machine (SVM) and Convolutional Neural Networks (CNN) to classify images of cats and dogs. It leverages the Cats and Dogs 40 Dataset from Kaggle, processes images into a uniform size, and trains models for classification. The SVM achieves an accuracy of 68.75%, while CNN implementation is designed for further exploration. Future improvements include data augmentation and hyperparameter tuning for enhanced performance.
# Project Workflow
## 1. Data Preparation
Dataset Source: Cats and Dogs 40 Dataset on Kaggle
Steps:
Download the dataset using the Kaggle API.
Extract and organize the dataset into train and test directories under categories cat and dog.
Resize images to (40x40x3) for uniform input.

## 2. Libraries Used
tensorflow
keras
numpy
pandas
matplotlib
sklearn
skimage
## 3. Data Loading and Preprocessing
Images are read using skimage.io.imread and resized using skimage.transform.resize.
Flattened image data is stored in a NumPy array.
Labels are encoded: cat=0, dog=1.
## 4. Model Building
Support Vector Machine (SVM)
Model: GridSearchCV with SVM.
Training and Testing:
Split the data using train_test_split (80% training, 20% testing).
GridSearchCV is used to optimize the hyperparameters.
Performance Metrics:
Accuracy: 68.75%
Precision, Recall, and F1-Score are reported for each class.
Convolutional Neural Network (CNN)
Architecture:
Input layer with images of shape (40, 40, 3)
Convolutional layers with max-pooling.
Dense layers for classification.
Training and evaluation will be further explored using TensorFlow.
## 5. Testing and Prediction
Predict the class of an input image (cat/dog) and visualize it using matplotlib.
Results
SVM Model Accuracy: 68.75%
Classification Report:
```
                precision    recall  f1-score   support

       cat       0.64      0.88      0.74         8
       dog       0.80      0.50      0.62         8

  accuracy                           0.69        16
 macro avg       0.72      0.69      0.68        16
weighted avg 0.72 0.69 0.68 16
```
