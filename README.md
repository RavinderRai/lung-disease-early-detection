# Lung Disease Early Detection

A web interface to detect lung diseases given a chest x-ray image, using an end-to-end ML pipeline in the backend, pulling data from kaggle's [nih-chest-xrays](https://www.kaggle.com/competitions/nih-chest-xrays) dataset. The system detects multiple diseases based on the uploaded X-ray image, utilizing a machine learning model trained on the NIH Chest X-ray Dataset, and is designed with an interactive web interface, making it accessible for healthcare providers.

<div style="overflow: hidden; height: 400px;">
    <img src="lung_disease_demo.gif" alt="Lung Disease Demo" style="width: 100%; height: auto; object-fit: cover; transform: translateY(11%);">
</div>

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Tech Stack](#tech-stack)
6. [Architecture](#architecture)
7. [Future Improvements](#future-improvements)
8. [License](#license)

## Project Overview

The aim of this project is to build an AI-based tool to detect lung diseases in chest X-rays using a deep learning model. The system supports 15 disease classes, including pneumonia, emphysema, and fibrosis, along with a "No Findings" category when no disease is detected.

The project demonstrates the following:

- Backend: Machine learning pipeline to preprocess images, train a multi-label classification model, and serve predictions.
- Frontend: A simple web interface where users can upload chest X-rays and view the disease prediction results.
- Disease Classes:
  + Atelectasis
  + Consolidation
  + Infiltration
  + Pneumothorax
  + Edema
  + Emphysema
  + Fibrosis
  + Effusion
  + Pneumonia
  + Pleural Thickening
  + Cardiomegaly
  + Nodule Mass
  + Hernia
  + No Findings (No disease detected)

## Features

- Image Upload: Upload chest X-ray images through a web interface.
- Disease Detection: Detect multiple lung diseases with confidence scores for each disease.
- No Findings: If no disease is detected, the system returns "No Findings."
- Simple Frontend: A user-friendly interface built with Gradio for healthcare providers.

## Installation

### Requirements

- Python 3.10
- Conda
- Kaggle API (to download dataset)
- AWS and Docker (optional, for deployment)

### Step 1: Clone the repository

```bash
git clone https://github.com/ravinderrai/lung-disease-detection.git
cd lung-disease-detection
```

### Step 2: Set up the Conda environment

```bash
conda create --name chest-xray-venv python=3.10
conda activate chest-xray-venv
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the dataset from Kaggle

You will need to configure your Kaggle API to download the dataset:

```bash
kaggle competitions download -c nih-chest-xrays
unzip nih-chest-xrays.zip
```

### Step 5: Run the application

To start the web application run this from the root:

```bash
python main.py app
```

## Usage

1. Upload an X-ray: Go to the web interface and upload a chest X-ray image.
2. View Predictions: The system will display the detected diseases (if any) and the confidence score.

## Tech Stack

### Backend

- Python 3.10
- PyTorch (for CNN)
- Kaggle (for large dataset of real medical imaging dataset)

### Frontend

- Gradio (for web interface)

### Infrastructure

- AWS (optional for cloud-based deployment)
- Docker (optional for containerization)

## Architecture

1. Data Ingestion: Download and preprocess the NIH Chest X-ray dataset.
2. Model Training: A convolutional neural network (CNN) model is trained to detect if a disease is present or not. If it is, then another CNN predict which of the 15 lung diseases it is.
3. Web Interface: The frontend (Gradio) allows healthcare providers to interact with the system and view results.

## Future Improvements

1. Model Optimization: Fine-tune the model on larger datasets as we only used a subset for now. Also, the second model has not yet been implemented. 
2. Docker and Kubernetes: Containerize the feature engineering, training, and inference pipelines for scalability.
3. Advanced Frontend: Enhance the UI/UX of the web interface and add more features.
4. CI/CD Pipeline: Automate retraining and model updates using GitHub Actions or AWS CodePipeline.
5. Monitoring: Add logging and monitoring using AWS CloudWatch or Prometheus.

## License

This project is licensed under the MIT License.
