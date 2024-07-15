# Fake News Detection Using Deep Learning

This repository contains the implementation and evaluation of various Artificial Neural Network (ANN) models aimed at automatically classifying Twitter posts as True or Fake based on their veracity. The project leverages the Python Keras library, scikit-learn, and GloVe word embeddings to build and test different model architectures.

## Project Overview

Fake news detection is a critical task in today's digital age, where misinformation can spread rapidly through social media. This project explores the use of deep learning models to tackle this challenge, focusing on the classification of Twitter posts.

## Models Implemented

The project experiments with several ANN architectures, each designed to improve the accuracy and generalizability of fake news detection:

- **Multi-Layer Perceptron (MLP) with Single Flattened Word Embedding Vector**
- **MLP with Keras Embedding Layer**
- **Convolutional Neural Network (CNN)**
- **Recurrent Neural Network (RNN)**

### MLP with Single Flattened Word Embedding Vector

- **Initial Accuracy**: 0.78
- **Final Accuracy**: 0.81
- **Training Time**: 6.4 seconds per fold
- **Key Adjustments**: Lowered learning rate, reduced number of neurons, decreased batch size, and applied Lasso regularization.

### MLP with Keras Embedding Layer

- **Accuracy**: 0.85
- **Training Time**: 16.4 seconds per fold
- **Key Adjustments**: Utilized Keras Embedding layer to simplify the embedding process for the user, resulting in a performance improvement over the previous MLP model.

### Convolutional Neural Network (CNN)

- **Accuracy**: 0.87
- **Training Time**: 74.56 seconds per fold
- **Key Adjustments**: Reduced the number of filters and increased filter size to improve generalizability, applied dropout layers and Lasso regularization.

### Recurrent Neural Network (RNN)

- **Accuracy**: 0.88
- **Training Time**: 77 seconds per fold
- **Key Adjustments**: Simple LSTM architecture with 4 units and ReLU activation for effective long-range dependency capture.

## Model Comparison and Selection

Among the models investigated, the MLP with the Keras Embedding layer was chosen for its balance between accuracy, training efficiency, and stability. Despite CNN and RNN models achieving higher accuracy, their longer training times and complexity present significant trade-offs.

### Final Model Metrics

- **Accuracy**: 0.88
- **Precision**: 0.90
- **Recall**: 0.87
- **F1 Score**: 0.88

## Challenges and Future Work

### Challenges

- **Hardware Limitations**: Reliance on CPU for model training resulted in prolonged training times.
- **Limited Dataset**: The training set consisted of only 2000 samples.

### Future Work

- **Parameter Tuning**: Implementing a grid search algorithm for parameter tuning.
- **Hardware Upgrade**: Leveraging faster hardware or cloud computing for model training.
- **Dataset Augmentation**: Expanding the training dataset and exploring advanced preprocessing techniques.
- **Confidence Thresholds**: Establishing confidence thresholds for automated classifications to ensure high accuracy with human oversight.

## Conclusion

This project demonstrates the potential of using deep learning models for fake news detection on social media platforms. The selected MLP model with the Keras Embedding layer strikes a balance between performance and practicality, making it a viable solution for real-time applications with constrained computational resources.

## Acknowledgements

This project was completed as part of the CSC3066 Deep Learning course at the University, under the guidance of Professor [Name]. Special thanks to the creators of the GloVe word embeddings and the Keras library for their valuable tools and resources.

For further details, refer to the project report included in the repository.
