#include "Layer.h"
#include <cmath>

Layer::Layer(int inputSize, int outputSize, Activation * ac) {
    inputs.resize(inputSize);
    outputs.resize(outputSize);
    weights.resize(outputSize, std::vector<double>(inputSize));
    biases.resize(outputSize);
    act = ac;

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = 0.5;
        }
        biases[i] = 0.5;
    }
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

void Layer::forward(const std::vector<double>& input) {
    for (int i = 0; i < outputs.size(); ++i) {
        outputs[i] = 0;
        for (int j = 0; j < inputs.size(); ++j) {
            outputs[i] += input[j] * weights[i][j];
        }
        outputs[i] += biases[i];
        outputs[i] = act->f(outputs[i]);
    }
}

void Layer::computeDeltas(const std::vector<double>& nextDeltas, const std::vector<std::vector<double>>& nextWeights) {
    deltas.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        double d = 0.0;
        for (size_t j = 0; j < nextDeltas.size(); ++j) {
            d += nextDeltas[j] * nextWeights[j][i];
        }
        deltas[i] = d * act->df(outputs[i]);
    }
}
