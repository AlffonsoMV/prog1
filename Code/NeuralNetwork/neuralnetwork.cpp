#include "NeuralNetwork.h"
#include <iostream>

void NeuralNetwork::addLayer(Layer* l) {
    layers.emplace_back(l);
}

void NeuralNetwork::initializeGradientAccumulators() {
    accumulatedWeightGradients.clear();
    accumulatedBiasGradients.clear();
    for (auto& layer : layers) {
        std::vector<std::vector<double>> layerWeightGradients(layer->weights.size(), std::vector<double>(layer->weights[0].size(), 0.0));
        std::vector<double> layerBiasGradients(layer->biases.size(), 0.0);
        accumulatedWeightGradients.push_back(layerWeightGradients);
        accumulatedBiasGradients.push_back(layerBiasGradients);
    }
}

void NeuralNetwork::accumulateGradients() {
    for (size_t i = 0; i < layers.size(); ++i) {
        const std::vector<double>& prevOutputs = (i == 0) ? std::vector<double>(layers[i]->inputs.size(), 1.0) : layers[i - 1]->outputs;
        for (size_t j = 0; j < layers[i]->weights.size(); ++j) {
            for (size_t k = 0; k < layers[i]->weights[j].size(); ++k) {
                accumulatedWeightGradients[i][j][k] += layers[i]->deltas[j] * prevOutputs[k];
            }
            accumulatedBiasGradients[i][j] += layers[i]->deltas[j];
        }
    }
}

void NeuralNetwork::applyAccumulatedGradients(double learningRate) {
    for (size_t i = 0; i < layers.size(); ++i) {
        for (size_t j = 0; j < layers[i]->weights.size(); ++j) {
            for (size_t k = 0; k < layers[i]->weights[j].size(); ++k) {
                layers[i]->weights[j][k] -= learningRate * accumulatedWeightGradients[i][j][k];
            }
            layers[i]->biases[j] -= learningRate * accumulatedBiasGradients[i][j];
        }
    }
}

void NeuralNetwork::train(std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset, int epochs, TrainingMode mode, double learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        if (mode == TrainingMode::BATCH) {
            initializeGradientAccumulators();
        }

        for (auto& v : dataset) {
            const std::vector<double>& input = v.first;
            const std::vector<double>& target = v.second;

            // Forward pass
            std::vector<double> currentInput = input;
            for (auto& layer : layers) {
                layer->forward(currentInput);
                currentInput = layer->outputs;
            }

            // Backward pass (backpropagation)
            for (int i = layers.size() - 1; i >= 0; --i) {
                if (i == layers.size() - 1) {
                    layers[i]->deltas.resize(target.size());
                    for (size_t j = 0; j < target.size(); ++j) {
                        layers[i]->deltas[j] = (layers[i]->outputs[j] - target[j]) * layers[i]->outputs[j] * (1 - layers[i]->outputs[j]);
                    }
                } else {
                    layers[i]->computeDeltas(layers[i + 1]->deltas, layers[i + 1]->weights);
                }
            }

            if (mode == TrainingMode::BATCH) {
                accumulateGradients();
            } else {
                updateWeightsAndBiases(input, learningRate);
            }

            // Calculate loss (Mean Squared Error)
            for (size_t j = 0; j < target.size(); ++j) {
                double error = layers.back()->outputs[j] - target[j];
                totalLoss += error * error;
            }
        }

        if (mode == TrainingMode::BATCH) {
            applyAccumulatedGradients(learningRate);
        }

        totalLoss /= dataset.size();
        std::cout << "Epoch " << (epoch + 1) << " / " << epochs << ", Loss: " << totalLoss << std::endl;
    }
}

void NeuralNetwork::updateWeightsAndBiases(const std::vector<double>& input, double learningRate) {
    for (size_t i = 0; i < layers.size(); ++i) {
        const std::vector<double>& prevOutputs = (i == 0) ? input : layers[i - 1]->outputs;
        for (size_t j = 0; j < layers[i]->weights.size(); ++j) {
            for (size_t k = 0; k < layers[i]->weights[j].size(); ++k) {
                layers[i]->weights[j][k] -= learningRate * layers[i]->deltas[j] * prevOutputs[k];
            }
            layers[i]->biases[j] -= learningRate * layers[i]->deltas[j];
        }
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> currentInput = input;
    for (auto& layer : layers) {
        layer->forward(currentInput);
        currentInput = layer->outputs;
    }
    return layers.back()->outputs;
}
