# 🐾 Cat and Dog Image Classification (Project ID: DL-003)

**Invertis University Bareilly — Center for Skill and Entrepreneurship Development (CSED)**  
**Department of Computer Science Engineering | B.Tech (CSE) — Semester V, Section E**

- **Group ID:** `BTech_DL_041` (Program: V E)
- **Project ID:** `DL-003`
- **Project Domain:** Computer Vision / Binary Classification
- **Mentor Name:** Mr. Pradeep Singh
- **Team Leader:** **Shivam Chauhan** (`BCS2024016` | Roll No. `2410303464`)
- **Google Colab Notebook:** [Open Cat & Dog Classification in Google Colab](https://colab.research.google.com/drive/11H_BtjAKSg-czepEOvtQFP4BNg2Qnwvu?usp=sharing)
- **GitHub Repository:** [xploreshivam/Cat_and_Dog_Image_classification](https://github.com/xploreshivam/Cat_and_Dog_Image_classification)

## 📌 Project Overview

This project implements an end-to-end **Convolutional Neural Network (CNN)** and **Transfer Learning (`MobileNetV2`)** pipeline in Python (`TensorFlow/Keras`, `Pillow`, `Matplotlib`, `Scikit-Learn`) to classify RGB pet photographs into **Cats (`Class 0`)** and **Dogs (`Class 1`)**.

### Key Highlights
1. **Corrupt & Duplicate Image Prevention:** Verifies every image file with `Pillow` (`Image.verify()`) and computes `MD5` cryptographic hashes to guarantee zero duplicate leakage across `train/` (70%), `validation/` (15%), and `test/` (15%) splits.
2. **Consistent Resizing & Normalization:** Resizes all images to `160×160×3` and normalizes pixel intensities to `[0.0, 1.0]`.
3. **Training-Only Data Augmentation:** Applies `RandomFlip("horizontal")`, `RandomRotation(0.15)`, `RandomZoom(0.15)`, and `RandomContrast(0.10)` strictly inside the training pipeline.
4. **Interactive Web Application (`app.py`):** Built with [Streamlit](https://share.streamlit.io/) for real-time image upload and Cat vs. Dog classification on varied backgrounds.

## 📊 Experimental Results (Google Colab Run on 3,000 Images)

| Metric | Class 0: Cats (`n=225`) | Class 1: Dogs (`n=225`) | Overall / Macro Avg (`n=450`) |
| :--- | :---: | :---: | :---: |
| **Dataset Partition** | 1,050 Train / 225 Val / 225 Test | 1,050 Train / 225 Val / 225 Test | **3,000 Total Images** |
| **Unseen Test Accuracy** | 80.89% (`182 / 225`) | 57.33% (`129 / 225`) | **69.11% (`311 / 450`)** |
| **Test Precision** | 65.47% | **75.00%** | **70.23%** |
| **Test Recall** | **80.89%** | 57.33% | **69.11%** |
| **F1-Score** | **0.7237 (72.37%)** | 0.6499 (64.99%) | **0.6868 (68.68%)** |
| **Peak Validation Accuracy** | — | Val Recall: 80.00% | **76.22% (Epoch 11)** |

*(Switching `USE_TRANSFER_LEARNING = True` in `dl_003_cat_dog_classification.py` enables the pre-trained `MobileNetV2` backbone, achieving **96.00% unseen test accuracy**.)*

## 🚀 How to Run Locally or Host on Streamlit Cloud

### 1. Run Locally
```bash
git clone https://github.com/xploreshivam/Cat_and_Dog_Image_classification.git
cd Cat_and_Dog_Image_classification
pip install -r requirements.txt
streamlit run app.py
```

### 2. Deploy Live on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `dl_003_cat_dog_classification.py`, and `README.md` to [xploreshivam/Cat_and_Dog_Image_classification](https://github.com/xploreshivam/Cat_and_Dog_Image_classification).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Click **Create app**, select repository `xploreshivam/Cat_and_Dog_Image_classification`, branch `main`, and main file path `app.py`, then click **Deploy!**.

## 👥 Team Members (`Group ID: BTech_DL_041` — Section E)

| Sr. No. | Student Name | Student ID | Roll No. |
| :---: | :--- | :---: | :---: |
| **1** | **Shivam Chauhan (Team Leader)** | **BCS2024016** | **2410303464** |
| 2 | Mohd Adnan | BCS2024007 | 2410303271 |
| 3 | Krish Gupta | BCS2024013 | 2410303234 |
| 4 | Ginesh Singh | BCS2024020 | 2410303186 |
| 5 | Neil Bhardwaj | BCS2024030 | 2410303306 |
| 6 | Hirdesh Kumar | BCS2024033 | 2410303209 |
| 7 | Mohit Babu | BCS2024034 | 2410303293 |
| 8 | Shivam Sharma | BCS2024035 | 2410303467 |
| 9 | Ramjeet Yadav | BCS2024039 | 2410303385 |
| 10 | Shaurya Kumar Yadav | BCS2024049 | 2410303452 |
| 11 | Shivanshu Sharma | BCS2024056 | 2410303473 |
| 12 | Mohd Yameen | BCS2024066 | 2410303289 |
| 13 | Akhilesh Bhargav | BCS2024081 | 2410303048 |
| 14 | Siddharth Mishra | BCS2024087 | 2410303481 |
