# Extra Work of Introduction to Programming: A very first inmersion in Neural Networks

## Overview
This project, developed by Alfonso Mateos Vicente at École des Ponts ParisTech, focuses on implementing neural networks for image recognition and natural language processing tasks. Inspired by mathematical transformations in biological neurons, the project delves into the architecture of neural networks, their implementation in C++, and their application in practical scenarios.

## Implementation
The neural network model incorporates various layers - Dense, Dropout, Batch Normalization - each playing a unique role in the network's functionality. Key features include:
- **Layer Types**: Dense for linear transformation, Dropout for overfitting prevention, and Batch Normalization for input normalization.
- **Training Methodology**: Utilizes Stochastic Gradient Descent (SGD) for optimizing network parameters.
- **Activation Functions**: Includes Sigmoid, ReLU, and TanH functions to introduce non-linearity.
- **Architecture**: A sequence of layers forming the neural network, managed and trained using specific functions like `addLayer`, `train`, and `predict`.

## Experiment
The model was tested using the "Adult dataset" from UCI Machine Learning Repository, aimed at predicting income levels based on census data. It involved preprocessing steps like normalizing continuous features and encoding categorical data. The network architecture for this experiment included various layers with a focus on dropout and batch normalization, trained over 100 epochs.

## Results
The neural network demonstrated effective learning with a consistent decrease in loss over epochs, indicating its ability to generalize well to unseen data.

## Usage
To use this project:
1. Clone the repository.
2. Install required dependencies (list dependencies here).
3. Run the main program (specify the command).

## Dependencies
- C++ Compiler
- Additional Libraries (if any)

## Acknowledgements
Special thanks to Prof. Pascal Monasse for his guidance and support in this project.
