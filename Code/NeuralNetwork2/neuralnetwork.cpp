#include "neuralnetwork.h"
#include <cmath>

#include "FullyConnectedLayer.h"

NeuralNetwork::NeuralNetwork() {
}


void NeuralNetwork::addLayer(Layer* layer) {
    layers.emplace_back(layer);
}

void NeuralNetwork::forwardBackwardPass(const std::vector<double>& input, const std::vector<double>& target) {
    // Forward pass
    std::vector<double> currentInput = input;
    for (auto& layer: layers) {
        layer->forward(currentInput);
        currentInput = layer->getInputs();
    }

    // Backward pass (backpropagation)
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (i == layers.size() - 1) {
            // Output layer deltas
            for (size_t j = 0; j < target.size(); ++j) {
                double output = layers[i]->getOutputs()[j];
                layers[i]->getDeltas()[j] = (output - target[j]) * output * (1 - output);
            }
        } else {
            // Hidden layer deltas
            layers[i]->computeDeltas(layers[i + 1]->getDeltas(), layers[i + 1]->getWeights());
        }
    }
}


void NeuralNetwork::updateWeightsBiases(std::vector<std::vector<double > > & weightGradients, std::vector<std::vector<double> > & biasGradients, double learningRate, size_t batchSize) {
    for (size_t i = 0; i < layers.size(); ++i) {
        for (size_t j = 0; j < layers[i]->getWeights().size(); ++j) {
            for (size_t k = 0; k < layers[i]->getWeights()[j].size(); ++k) {
                layers[i]->getWeights()[j][k] -= learningRate * weightGradients[i][j] / batchSize;
            }
            layers[i]->getBiases()[j] -= learningRate * biasGradients[i][j] / batchSize;
        }
    }
}

void NeuralNetwork::updateWeightsBiasesStochastic(const std::vector<double>& input, double learningRate) {
    for (size_t i = 0; i < layers.size(); ++i) {
        const std::vector<double>& prevOutputs = (i == 0) ? input : layers[i - 1]->getOutputs();
        for (size_t j = 0; j < layers[i]->getWeights().size(); ++j) {
            for (size_t k = 0; k < layers[i]->getWeights()[j].size(); ++k) {
                layers[i]->getWeights()[j][k] -= learningRate * layers[i]->getDeltas()[j] * prevOutputs[k];
            }
            layers[i]->getBiases()[j] -= learningRate * layers[i]->getDeltas()[j];
        }
    }
}


void NeuralNetwork::train(const std::vector<std::pair<std::vector<double>, std::vector<double> > >& dataset, int epochs, double learningRate, TrainingMode mode) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (mode == BATCH) {
            // Initialize accumulators for gradients
            std::vector<std::vector<double> > weightGradients;
            std::vector<std::vector<double> > biasGradients;

            // Initialize gradients for each layer
            for (auto& layer : layers) {
                weightGradients.push_back(std::vector<double>(layer->getWeights().size(), 0.0));
                biasGradients.push_back(std::vector<double>(layer->getBiases().size(), 0.0));
            }

            // Accumulate gradients over the dataset
            for (const auto& data : dataset) {
                forwardBackwardPass(data.first, data.second);

                // Accumulate gradients
                for (size_t i = 0; i < layers.size(); ++i) {
                    for (size_t j = 0; j < layers[i]->getWeights().size(); ++j) {
                        for (size_t k = 0; k < layers[i]->getWeights()[j].size(); ++k) {
                            weightGradients[i][j] += layers[i]->getDeltas()[j] * ((i == 0) ? data.first[k] : layers[i - 1]->getOutputs()[k]);
                        }
                        biasGradients[i][j] += layers[i]->getDeltas()[j];
                    }
                }
            }

            // Update weights and biases using the accumulated gradients
            updateWeightsBiases(weightGradients, biasGradients, learningRate, dataset.size());

        } else { // STOCHASTIC
            for (const auto& data : dataset) {
                forwardBackwardPass(data.first, data.second);
                updateWeightsBiasesStochastic(data.first, learningRate);
            }
        }
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> currentInput = input;
    for (auto& layer : layers) {
        layer->forward(currentInput);
        currentInput = layer->getOutputs();
    }
    return layers.back()->getOutputs();
}
