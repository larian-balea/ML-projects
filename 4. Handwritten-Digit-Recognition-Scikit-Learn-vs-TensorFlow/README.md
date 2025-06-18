# Handwritten Digit Recognition: TensorFlow vs. Scikit-Learn

This project implements and compares neural network models for handwritten digit recognition using the MNIST dataset. The primary goal was to achieve over 96% test accuracy while analyzing the trade-offs between a high-level library (`Scikit-Learn`) and a dedicated deep learning framework (`TensorFlow`).

<!-- Add a screenshot of your best model's confusion matrix here -->

## Table of Contents
* [Project Goal](#project-goal)
* [Tech Stack](#tech-stack)
* [Methodology](#methodology)
* [Performance Comparison](#performance-comparison)
* [Key Insight](#key-insight)
* [How to Run](#how-to-run)

## Project Goal
To build a robust digit recognition system by:
1.  Implementing a Multi-Layer Perceptron (MLP) using Scikit-Learn's `MLPClassifier` and optimizing it with `GridSearchCV`.
2.  Implementing a similar MLP architecture in TensorFlow/Keras and manually tuning hyperparameters.
3.  Comparing the two approaches on performance, flexibility, and ease of implementation.

## Tech Stack
- TensorFlow & Keras
- Scikit-learn
- NumPy
- Plotly

## Methodology

### 1. Scikit-Learn MLP
- An `MLPClassifier` was systematically tuned using `GridSearchCV` to explore a wide range of hyperparameters, including hidden layer sizes, activation functions, regularization (`alpha`), and learning rate schemes.
- This approach automated the search for the optimal model configuration.

### 2. TensorFlow MLP
- A `Sequential` model was built in Keras with a similar architecture to the best Scikit-Learn model.
- Several training runs were conducted with manual adjustments to learning rate, batch size, and epoch count to observe their impact on the training and validation curves.

## Performance Comparison

| Model Approach          | Best Test Accuracy | Key Advantage                                       |
| ----------------------- | ------------------ | --------------------------------------------------- |
| Scikit-Learn MLP        | **97.63%**         | Ease of use and systematic tuning with `GridSearchCV`. |
| TensorFlow MLP          | **98.06%**         | Higher accuracy, finer control over training loop.   |

## Key Insight
The core takeaway is the trade-off between convenience and control.
- **Scikit-Learn** is ideal for rapid prototyping and systematic tuning of standard models.
- **TensorFlow** offers superior performance and the flexibility needed to build custom, state-of-the-art architectures like CNNs, which ultimately yield the best results for computer vision tasks.
