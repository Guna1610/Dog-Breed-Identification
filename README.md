# 🐶 Dog Vision: Dog Breed Classification

An end-to-end **deep learning project** that classifies dog breeds from images using **TensorFlow 2.x** and **transfer learning**.  
Given a dog photo, the model predicts one of **120 possible breeds** from the [Kaggle Dog Breed Identification dataset](https://www.kaggle.com/competitions/dog-breed-identification).

---

## 📌 Project Overview
This project tackles a **multi-class image classification** problem.  
The workflow includes:
- Loading and preprocessing a large image dataset.
- Using a pretrained **MobileNetV2** model from **TensorFlow Hub**.
- Adding a custom classification head for 120 breeds.
- Training and evaluating the model with **callbacks** (EarlyStopping, TensorBoard).
- Making predictions on unseen dog images.

---

## 📂 Dataset
- **Source:** [Kaggle Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/data)  
- **Size:** ~10,222 training images, 120 breeds, plus a test set (~10,000 images).  
- **Structure:** Each training image has a unique ID with a label in `labels.csv`.

---

## 🛠️ Approach
1. **Data Preparation**
   - Read images and labels.
   - Resize to `224x224` pixels and normalize to `[0,1]`.
   - One-hot encode labels for 120 breeds.
   - Build `tf.data` pipelines for efficient training.

2. **Model Architecture**
   - **Base Model:** MobileNetV2 pretrained on ImageNet (`mobilenet_v2_130_224/classification/5` from TF Hub).
   - **Classifier Head:** Dropout (0.2) + Dense layer with 120 units (`softmax`).
   - **Loss/Optimizer:** Categorical Crossentropy + Adam optimizer.

3. **Training**
   - Batch size = 32  
   - Up to 100 epochs with **EarlyStopping**.  
   - TensorBoard logging for monitoring.  
   - Validation split to track generalization.

4. **Evaluation**
   - Accuracy on validation set (~67% on small subset, higher on full dataset).  
   - Misclassifications mostly between visually similar breeds.  

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/dog-vision.git
cd dog-vision
```  
## 2. Install dependencies
    - pip install -r requirements.txt

## 3. Download the dataset
    - Register on Kaggle and download the Dog Breed Identification dataset
    - Place train/, test/, and labels.csv inside a data/ folder.
## 4. Run the notebook
   - Open Dog-Vision.ipynb in Google Colab or Jupyter.
   - Update file paths if needed.
   - Run all cells to preprocess data, train the model, and evaluate results.

  📊 Results
   - Validation Accuracy: ~67% (on 1k subset, improves with full data).
   - Training: Fast convergence thanks to transfer learning.
   - Future Improvements: Data augmentation, hyperparameter tuning, trying EfficientNet or ResNet models.
