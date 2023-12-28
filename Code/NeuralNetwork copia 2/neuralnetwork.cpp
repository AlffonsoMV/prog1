#include "neuralnetwork.h"
#include <cmath>
#include <iostream>

void NeuralNetwork::addLayer(Layer l) {
    layers.emplace_back(l);
}

void NeuralNetwork::train(std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset, int epochs, const double learningRate = 0.01) {
// const std::vector<double>& input, const std::vector<double>& target
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (auto& v : dataset) {
            const std::vector<double>& input = v.first;
            const std::vector<double>& target = v.second;
            // Forward pass
            std::vector<double> currentInput = input;
            for (auto& layer : layers) {
                layer.forward(currentInput);
                currentInput = layer.outputs;
            }

            // Backward pass (backpropagation)
            for (int i = layers.size() - 1; i >= 0; --i) {
                if (i == layers.size() - 1) {
                    // Output layer deltas
                    layers[i].deltas.resize(target.size());
                    for (size_t j = 0; j < target.size(); ++j) {
                        layers[i].deltas[j] = (layers[i].outputs[j] - target[j]) * layers[i].outputs[j] * (1 - layers[i].outputs[j]);
                    }
                } else {
                    // Hidden layer deltas
                    layers[i].computeDeltas(layers[i + 1].deltas, layers[i + 1].weights);
                }
            }

            // Update weights and biases
            for (size_t i = 0; i < layers.size(); ++i) {
                const std::vector<double>& prevOutputs = (i == 0) ? input : layers[i - 1].outputs;
                for (size_t j = 0; j < layers[i].weights.size(); ++j) {
                    for (size_t k = 0; k < layers[i].weights[j].size(); ++k) {
                        layers[i].weights[j][k] -= learningRate * layers[i].deltas[j] * prevOutputs[k];
                    }
                    layers[i].biases[j] -= learningRate * layers[i].deltas[j];
                }
            }


            // Calculate loss MSE
            for (size_t j = 0; j < target.size(); ++j) {
                double error = layers.back().outputs[j] - target[j];
                totalLoss += error * error;
            }
        }

        totalLoss /= dataset.size();
        std::cout << "Epoch " << (epoch + 1) << " / " << epochs << ", Loss: " << totalLoss << std::endl;
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> currentInput = input;
    for (auto& layer : layers) {
        layer.forward(currentInput);
        currentInput = layer.outputs;
    }
    return layers.back().outputs;
}
