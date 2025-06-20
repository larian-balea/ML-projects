# Custom CNN vs. AlexNet for Image Classification

This project implements and compares two distinct Convolutional Neural Network (CNN) architectures for image classification on the CIFAR-10 and MNIST datasets. The primary goals were to design a custom CNN, adapt a well-known architecture (AlexNet), systematically compare their performance, and improve the best-performing model using data augmentation.

![Image](https://github.com/user-attachments/assets/02d25bf5-d315-4cc2-9bcf-3d27d17a9fd9)

## Table of Contents
* [Project Goal](#project-goal)
* [Tech Stack](#tech-stack)
* [Methodology](#methodology)
* [Performance Showdown: Custom CNN vs. AlexNet](#performance-showdown-custom-cnn-vs-alexnet)
* [Improving Performance with Data Augmentation](#improving-performance-with-data-augmentation)
* [Key Takeaways](#key-takeaways)
* [How to Run](#how-to-run)

## Project Goal
1.  **Design & Build:** Create a custom CNN architecture from scratch.
2.  **Adapt & Implement:** Recreate an AlexNet-inspired architecture adapted for smaller 32x32 images.
3.  **Compare & Analyze:** Evaluate both models on CIFAR-10 based on accuracy, model size, and training time.
4.  **Diagnose & Improve:** Identify weakly performing classes and apply data augmentation to boost model robustness and accuracy.

## Tech Stack
- TensorFlow & Keras
- Scikit-learn
- NumPy
- Plotly

## Methodology
- **Data:** The CIFAR-10 and MNIST datasets were used. To enable rapid experimentation, a 25% stratified sample was taken from each class.
- **Custom CNN Design:** An iterative approach was used, testing three variations with different filter counts, kernel sizes, and dropout rates to find an optimal configuration.
- **AlexNet Adaptation:** The core principles of AlexNet (deep convolutional stacks, max pooling, heavy dropout in dense layers) were scaled down to fit the 32x32 input of CIFAR-10.
- **Data Augmentation:** For the best-performing model, `ImageDataGenerator` was used to apply random rotations, shifts, flips, and zooms, creating a more diverse training set.

## Performance Showdown: Custom CNN vs. AlexNet
The comparison was focused on the more challenging CIFAR-10 dataset.

| Metric                | Best Custom CNN (V2)  | AlexNet-inspired      |
| --------------------- | --------------------- | --------------------- |
| **Test Accuracy**     | `66.28%`              | **`62.92%`**          |

## Improving Performance with Data Augmentation
The Best Custom CNN model was selected for further improvement.
- **Diagnosis:** The initial classification report showed lower F1-scores for classes like 'cat' and 'dog', indicating confusion with other animal classes.
- **Solution:** `ImageDataGenerator` was applied to the training data.
- **Result:**
  - Accuracy Before Augmentation: **`66.28%`**
  - Accuracy After Augmentation: **`70.28%`**
- The augmented model demonstrated improved accuracy and a more balanced performance across all classes, confirming the strategy's success.

## Key Takeaways
- **Iterative Design is Key:** Systematically testing variations of the custom CNN was crucial to finding a competitive baseline.
- **Data Augmentation is a Powerful Tool:** Augmentation proved to be highly effective in improving model robustness and overcoming challenges with confusing classes.
