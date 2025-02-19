# Customer Churn Prediction Model

A deep learning model to predict which customers are at risk of leaving using Artificial Neural Networks (ANN).

## Overview

This project implements a customer churn prediction system using TensorFlow/Keras. It analyzes various customer attributes to predict the likelihood of a customer leaving the service.


## What I Did?

I developed this customer churn prediction model from scratch using the following approach:


1. **Model Development**
   - Built a custom Neural Network using TensorFlow/Keras
   - Implemented a 4-layer architecture:
     * Input Layer: Matches feature dimensions
     * First Hidden Layer: 64 neurons with ReLU activation
     * Second Hidden Layer: 32 neurons with ReLU activation
     * Output Layer: Single neuron with Sigmoid activation for binary classification

2. **Model Training Process**
   - Split data into training (80%) and testing (20%) sets
   - Implemented Early Stopping to prevent overfitting
   - Used Adam optimizer with Binary Cross-Entropy loss
   - Monitored training progress using TensorBoard
   - Achieved good accuracy in predicting customer churn

3. **Model Deployment**
   - Saved trained model and preprocessing components
   - Created prediction pipeline for real-time inference
   - Implemented JSON output format for easy integration


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