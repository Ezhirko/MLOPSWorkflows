[![CI/CD Pipeline](https://github.com/Ezhirko/MLOPSWorkflows/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/Ezhirko/MLOPSWorkflows/actions/workflows/ci_cd.yml)

# Simple CNN Model CICD Pipeline 🚀

This repository contains a lightweight PyTorch-based Convolutional Neural Network (CNN) for MNIST digit classification. The model achieves over **95% accuracy in just 1 epoch**, has fewer than **25,000 parameters**, and includes **image augmentation** for better generalization.

The project is integrated with GitHub Actions for continuous integration, ensuring the following conditions are tested and verified:

1. The model has fewer than 25,000 parameters.
2. The model achieves over 95% accuracy in 1 epoch.
3. Additional relevant tests ensure robustness.

## Features

- **Lightweight Model:** The CNN has fewer than 25,000 parameters, making it efficient for resource-constrained environments.
- **High Accuracy:** Achieves over 95% training accuracy in just one epoch.
- **Image Augmentation:** Techniques like random rotation, shifts, and flips are applied to enhance dataset variability.
- **Continuous Integration:** GitHub Actions pipeline automatically tests the model's compliance with requirements.
- **Automated Testing:** Includes unit tests for parameters, output probabilities, and input compatibility.

---

## Requirements

To run the model and tests, install the required dependencies:
```bash
pip install -r requirements.txt
```
---

## Model Architecture

The model consists of a simple CNN with:

- **2 convolutional layers**
- **2 fully connected layers**
- **ReLU activations**
- **Softmax output for classification**

### Parameters Count
The model has fewer than **25,000 parameters**, ensuring lightweight computation.

---

## Training and Augmentation

The training pipeline uses:

- **Adam optimizer**
- **CrossEntropy loss**
- **Batch size of 512**
- **Image Augmentation:** Random rotation, translation, croping and scaling
Example of augmented images (5x5 grid with original and augmented samples):
![](./Screenshot/Image%20Augmentation.png)

---

## GitHub Actions Workflow
The repository is integrated with GitHub Actions. The workflow tests:

1. **Model parameters:** Ensures the total number of parameters is below the limit.
2. **Training accuracy:** Confirms training accuracy is 95% or higher in one epoch.
3. **Additional tests:** Includes unique tests for model output probabilities, input compatibility, and more.
### Unique Tests
- **Parameter Count Test:** Verifies the model has fewer than 25,000 parameters.
- **Training Accuracy Test:** Ensures the model achieves over 95% training accuracy in one epoch.
- **Output Probability Test:** Confirms the model outputs valid probability distributions (sum to 1, non-negative).
- **Input Compatibility Test:** Validates that the model handles MNIST-sized inputs (28x28 grayscale images).
- **Output Shape Test:** Checks that the final layer has 10 units (one for each digit).

---

## Repository Structure

├── train.py             # Training pipeline with augmentation
├── test_pipeline.py     # Pytest-based unit tests
├── requirements.txt     # Python dependencies
├── screenshots/         # Folder containing screenshots (e.g., augmented images)
└── .github/
    └── workflows/
        └── ci_cd.yml     # GitHub Actions configuration

---
## How to Run
1. **Train the Model:**
```bash
python train.py
```
2. **Run Tests:**
```bash
pytest test_pipeline.py
```
---

## CI/CD Pipeline
The GitHub Actions workflow automatically runs the following:

- **Install Dependencies:** Installs required Python packages.
- **Run Unit Tests:** Executes all test cases to ensure compliance.
- **Validate Accuracy:** Confirms training accuracy exceeds 95% in one epoch.



