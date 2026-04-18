# Multi-Modal Product Classifier Using Text and Images

## 📌 Project Overview
This project implements a multi-modal deep learning model that combines product images and textual metadata to classify fashion product categories. The model uses a CNN (ResNet18) for image processing and BERT for text processing, fusing both modalities to improve classification accuracy.

## 📊 Dataset
- **Source:** [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- **Size:** 44,419 products
- **Classes:** 43 subcategories (Topwear, Bottomwear, Shoes, Watches, etc.)
- **Features:**
  - Images: Product photos (`.jpg`)
  - Text: Gender, baseColour, season, usage, productDisplayName

## 🏗️ Model Architecture

### Image Pipeline (CNN)
- **Model:** ResNet18 (pretrained on ImageNet)
- **Output:** 256-dimensional feature vector
- **Preprocessing:** Resize to 224x224, normalization

### Text Pipeline (Transformer)
- **Model:** BERT-base-uncased
- **Output:** 256-dimensional feature vector
- **Max Length:** 128 tokens

### Fusion Strategy
- **Method:** Feature concatenation
- **Combined Vector:** 512 dimensions (256 + 256)
- **Classifier:** 2-layer MLP with Dropout
  - 512 → 256 (ReLU, Dropout 0.3)
  - 256 → 43 classes

## 📈 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 92% |
| Macro F1-Score | 0.616 |
| Weighted F1-Score | 0.92 |

### Performance by Category
- **Strong (F1 > 0.85):** Topwear, Shoes, Watches, Bags, Belts, Innerwear
- **Weak (F1 < 0.30):** Rare categories with limited samples (Bath and Body, Beauty Accessories, etc.)

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/rafiveyisov/Multi-Modal-Product-Classifier-Using-Text-and-Images.git
cd multimodal-product-classifier
pip install -r requirements.txt