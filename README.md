# Lung Disease Early Detection

A web interface to detect lung diseases given a chest x-ray image, using an end-to-end ML pipeline in the backend, pulling data from kaggle's [nih-chest-xrays](https://www.kaggle.com/competitions/nih-chest-xrays) dataset. The system detects multiple diseases based on the uploaded X-ray image, utilizing a machine learning model trained on the NIH Chest X-ray Dataset, and is designed with an interactive web interface, making it accessible for healthcare providers.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Tech Stack](#tech-stack)
6. [Architecture](#architecture)
7. [Future Improvements](#future-improvements)
8. [Contributing](#contributing)
9. [License](#license)

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
- API Integration: Backend Flask/FastAPI to process requests and serve model predictions.
- Simple Frontend: A user-friendly interface built with Streamlit for healthcare providers.

## Installation

### Requirements

- Python 3.10
- Conda
- Kaggle API (to download dataset)
- Docker (optional, for containerization)
- AWS account (optional, for deployment)

### Step 1: Clone the repository

```bash
git clone https://github.com/ravinderrai/lung-disease-detection.git
cd lung-disease-detection
```

### Step 2: Set up the Conda environment

```bash
conda create --name chestdetect-ai python=3.10
conda activate chestdetect-ai
```

### Step 3: Install dependencies

```bash
conda install numpy pandas scikit-learn matplotlib seaborn tensorflow pytorch torchvision flask fastapi streamlit -c conda-forge
```

### Step 4: Download the dataset from Kaggle

You will need to configure your Kaggle API to download the dataset:

```bash
kaggle competitions download -c nih-chest-xrays
unzip nih-chest-xrays.zip
```

### Step 5: Run the application

To start the web application:

```bash
streamlit run app.py
```

## Usage

1. Upload an X-ray: Go to the web interface and upload a chest X-ray image.
2. View Predictions: The system will display the detected diseases (if any) and the confidence score for each class.
3. Check Logs: Check the terminal or the logs for detailed information on predictions.

## Tech Stack

### Backend:
- Python 3.10
- TensorFlow/PyTorch (for deep learning)
- Flask/FastAPI (for serving predictions via API)

### Frontend:
- Streamlit (for web interface)

### Infrastructure:
- AWS (optional for cloud-based deployment)
- Docker (optional for containerization)
- GitHub Actions (optional for CI/CD)

## Architecture

1. Data Ingestion: Download and preprocess the NIH Chest X-ray dataset.
2. Model Training: A convolutional neural network (CNN) model is trained to classify 15 lung diseases based on X-ray images.
3. API Layer: Flask/FastAPI handles image uploads and runs inference with the trained model.
4. Web Interface: The frontend (Streamlit) allows healthcare providers to interact with the system and view results.
5. CI/CD: Basic pipeline set up for model retraining when new data is added (optional feature for future improvements).

## Future Improvements

1. Model Optimization: Fine-tune the model on larger datasets and experiment with different architectures (e.g., EfficientNet).
2. Docker and Kubernetes: Containerize the feature engineering, training, and inference pipelines for scalability.
3. Advanced Frontend: Enhance the UI/UX of the web interface and add more features (e.g., historical patient data, performance tracking).
4. CI/CD Pipeline: Automate retraining and model updates using GitHub Actions or AWS CodePipeline.
5. Monitoring: Add logging and monitoring using AWS CloudWatch or Prometheus.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your changes pass all tests.

## License

This project is licensed under the MIT License.
