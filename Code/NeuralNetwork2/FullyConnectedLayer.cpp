#include "FullyConnectedLayer.h"
#include <cmath>
#include <random>

FullyConnectedLayer::FullyConnectedLayer(int inputSize, int outputSize) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.1);

    inputs.resize(inputSize);
    outputs.resize(outputSize);
    weights.resize(outputSize, std::vector<double>(inputSize));
    biases.resize(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = distribution(generator);
        }
        biases[i] = distribution(generator);
    }
}


void FullyConnectedLayer::forward(const std::vector<double>& input) {
    for (int i = 0; i < outputs.size(); ++i) {
        outputs[i] = 0;
        for (int j = 0; j < inputs.size(); ++j) {
            outputs[i] += input[j] * weights[i][j];
        }
        outputs[i] += biases[i];
        outputs[i] = 1.0 / (1.0 + exp(-outputs[i])); // Sigmoid activation
    }
}

void FullyConnectedLayer::computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double > > & nextWeights) {
    deltas.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        double d = 0.0;
        for (size_t j = 0; j < nextDeltas.size(); ++j) {
            d += nextDeltas[j] * nextWeights[j][i];
        }
        deltas[i] = d * outputs[i] * (1 - outputs[i]); // Assuming sigmoid activation
    }
}
