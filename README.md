# Customer Churn Prediction Model

A deep learning model to predict which customers are at risk of leaving using Artificial Neural Networks (ANN).

## Overview

This project implements a customer churn prediction system using TensorFlow/Keras. It analyzes various customer attributes to predict the likelihood of a customer leaving the service.

## Project Structure

- `train.py` - Training script for the ANN model
- `predict.py` - Prediction script for making inferences
- `model.keras` - Trained model file
- `*.pkl` - Saved encoders and scalers

## Features Used

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

## Requirements

- TensorFlow
- pandas
- scikit-learn
- numpy

## Usage

### Training

```bash
python train.py
```

### Prediction

```bash
python predict.py
```

Example input format:
```python
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
```

## Model Details

- Architecture: Multi-layer Neural Network
- Input Layer: Based on feature dimensions
- Hidden Layers: 64 units (ReLU), 32 units (ReLU)
- Output Layer: 1 unit (Sigmoid)
- Optimizer: Adam
- Loss: Binary Cross-Entropy